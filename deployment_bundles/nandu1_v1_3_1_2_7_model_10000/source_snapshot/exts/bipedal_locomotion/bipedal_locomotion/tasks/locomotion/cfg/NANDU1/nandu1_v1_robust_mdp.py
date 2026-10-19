"""Shared robustness components for NANDU1 absolute-height tasks.

This module deliberately does not define or inherit a command term.  Every v1
task keeps its own command generator and curriculum; the components below only
consume the task's command name and current stage.
"""

from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from . import nandu1_mdp as base
from . import nandu1_v0_2_math as robust_math


SCHEMA_VERSION = "nandu1_height_robustness_v1"
URDF = Path(__file__).resolve().parents[4] / "assets/usd/NANDU1/urdf/robot.urdf"


def _ids(env_ids: Sequence[int] | slice | None, count: int, device: str) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(count, device=device)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def _measurement_cache(env) -> dict:
    """One random measurement per simulator step, shared by policy and history."""
    cache = getattr(env, "_nandu_v1_measurement_cache", None)
    if cache is None or cache.get("step") != env.common_step_counter:
        cache = {"step": env.common_step_counter}
        env._nandu_v1_measurement_cache = cache
    return cache


def _noise_amplitude(env, command_name: str, key: str) -> float:
    stage = int(env.command_manager.get_term(command_name).stage)
    return float(robust_math.STAGES[stage][key])


def cached_noisy_joint_position(env, command_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    if not env.cfg.observations.policy.enable_corruption:
        return robot.data.joint_pos
    cache = _measurement_cache(env)
    if "joint_pos_all" not in cache:
        cache["joint_pos_all"] = robust_math.uniform_noise(
            robot.data.joint_pos, _noise_amplitude(env, command_name, "joint_pos_noise")
        )
    return cache["joint_pos_all"]


def noisy_observation(env, kind: str, command_name: str, asset_cfg=None, sensor_cfg=None):
    """Return noisy deployable measurements without contaminating critic labels."""
    robot = env.scene["robot"]
    if kind == "joint_pos":
        value = cached_noisy_joint_position(env, command_name)
        return value[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    if kind == "joint_vel":
        if not env.cfg.observations.policy.enable_corruption:
            return robot.data.joint_vel[:, asset_cfg.joint_ids]
        cache = _measurement_cache(env)
        if "joint_vel_all" not in cache:
            cache["joint_vel_all"] = robust_math.uniform_noise(
                robot.data.joint_vel, _noise_amplitude(env, command_name, "joint_vel_noise")
            )
        return cache["joint_vel_all"][:, asset_cfg.joint_ids]

    sensor = env.scene.sensors[sensor_cfg.name]
    value = sensor.data.ang_vel_b if kind == "gyro" else sensor.data.projected_gravity_b
    installation = getattr(env, "_nandu_v1_imu_measurements", None)
    if installation is not None:
        amplitude_key = "gyro_noise" if kind == "gyro" else "gravity_noise"
        amplitude = (_noise_amplitude(env, command_name, amplitude_key)
                     if env.cfg.observations.policy.enable_corruption else 0.)
        # Opt-in v1.2.6 only. Same path is used by policy/history and FK helpers;
        # critic/rewards continue reading the original, unmodified sensor data.
        return installation.observe(sensor_cfg.name, kind, value, env.common_step_counter, amplitude)
    if not env.cfg.observations.policy.enable_corruption:
        return value
    cache = _measurement_cache(env)
    key = f"{kind}:{sensor_cfg.name}"
    if key not in cache:
        amplitude_key = "gyro_noise" if kind == "gyro" else "gravity_noise"
        # Projected gravity is intentionally not re-normalized: component noise
        # matches the deployable sensor preprocessing contract.
        cache[key] = robust_math.uniform_noise(value, _noise_amplitude(env, command_name, amplitude_key))
    return cache[key]


def kinematic_measurements(env, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint angles and base gravity from the exact policy-noise cache."""
    joints = cached_noisy_joint_position(env, command_name)
    gravity = noisy_observation(
        env, "gravity", command_name, sensor_cfg=SceneEntityCfg("base_imu")
    )
    return joints, gravity


def ensure_robust_metrics(term) -> None:
    names = (
        "Reset/rejection_rate", "Reset/fallback", "Reset/position_offset_max", "Reset/joint_speed_max",
        "Disturbance/active", "Disturbance/force_norm", "Disturbance/torque_norm",
        "Disturbance/recovery_seconds", "Disturbance/recovery_timeout",
        "Tracking/clean_error", "Tracking/disturbed_error", "Tracking/recovery_error",
        "Tracking/clean_fraction", "Tracking/disturbed_fraction", "Tracking/recovery_fraction",
    )
    for name in names:
        if name not in term.metrics:
            term.metrics[name] = torch.zeros(term.num_envs, device=term.device)
    for key in robust_math.STAGES[0]:
        name = f"Robustness/{key}_limit"
        if name not in term.metrics:
            term.metrics[name] = torch.zeros(term.num_envs, device=term.device)


def update_robust_metrics(term) -> None:
    ensure_robust_metrics(term)
    scale = 1.0 / float(term._env.max_episode_length)
    for key, value in robust_math.STAGES[term.stage].items():
        term.metrics[f"Robustness/{key}_limit"] += value * scale
    scheduler = getattr(term._env, "_nandu_v1_robust_disturbance", None)
    if scheduler is None:
        return
    active = scheduler.active
    recovery = term.disturbance_recovery_pending & ~active
    clean = ~(active | recovery)
    robot = term._robot
    actual_vx = term._up3_velocity_w()
    from isaaclab.utils.math import quat_apply_inverse
    actual_vx = quat_apply_inverse(robot.data.root_quat_w, actual_vx)[:, 0]
    height = term._up3_position_w()[:, 2]
    tilt = torch.acos((-term._env.scene.sensors["up3_imu"].data.projected_gravity_b[:, 2]).clamp(-1, 1))
    error = torch.stack((
        (robot.data.root_ang_vel_w[:, 2] - term.command[:, 0]).abs() / .25,
        (actual_vx - term.command[:, 1]).abs() / .14,
        (height - term.command[:, 2]).abs() / .10,
        tilt / math.radians(10),
    ), -1).mean(-1)
    for label, mask in (("clean", clean), ("disturbed", active), ("recovery", recovery)):
        term.metrics[f"Tracking/{label}_error"] += error * mask * scale
        term.metrics[f"Tracking/{label}_fraction"] += mask * scale
    term.metrics["Disturbance/active"] += active * scale
    term.metrics["Disturbance/force_norm"] += scheduler.applied_force.norm(dim=-1) * scale
    term.metrics["Disturbance/torque_norm"] += scheduler.applied_torque.norm(dim=-1) * scale


class HeightRobustDomainRandomization(base.StageDomainRandomization):
    """Reuse material buckets; randomize every body's centroidal mass properties."""

    def __call__(self, env, env_ids, command_name, asset_cfg, material_buckets_per_stage=48, excluded_body_ids=()):
        super().__call__(env, env_ids, command_name, asset_cfg, material_buckets_per_stage)
        cpu_ids = env_ids.to(device="cpu", dtype=torch.long)
        coms = self._robot.root_physx_view.get_coms()
        coms[cpu_ids] = self._default_com[cpu_ids]
        amplitude = robust_math.STAGES[int(env.command_manager.get_term(command_name).stage)]["com"]
        if amplitude:
            coms[cpu_ids, :, :3] += torch.empty(len(cpu_ids), self._robot.num_bodies, 3).uniform_(
                -amplitude, amplitude
            )
        if self._excluded_body_ids:
            excluded = list(self._excluded_body_ids)
            coms[cpu_ids[:, None], excluded] = self._default_com[cpu_ids[:, None], excluded]
        self._robot.root_physx_view.set_coms(coms, cpu_ids)


class HeightRobustReset(ManagerTermBase):
    """Soft-limit reset with mesh collision and wheel-support rejection."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.command_name = cfg.params["command_name"]
        self.geometry = robust_math.ResetGeometry(URDF, self.robot.joint_names, env.device)
        legal, height = self.geometry.evaluate(
            self.robot.data.default_joint_pos[:1], torch.eye(3, device=env.device)[None]
        )
        if not bool(legal.all()):
            raise ValueError("NANDU default reset fails mesh geometry audit")
        self.default_z = float(height[0])
        env._nandu_v1_robust_reset = self

    def __call__(self, env, env_ids, command_name):
        ids = env_ids.to(env.device, dtype=torch.long)
        if not len(ids):
            return
        term = env.command_manager.get_term(command_name)
        ensure_robust_metrics(term)
        stage = robust_math.STAGES[int(term.stage)]
        enabled = bool(getattr(term.cfg, "robustness_enabled", True))
        robot = self.robot
        defaults = robot.data.default_joint_pos[ids]
        limits = robot.data.soft_joint_pos_limits[ids]
        severity = torch.rand(len(ids), device=env.device)
        severity = torch.where(severity < .6, .25, torch.where(severity < .9, .6, 1.0))
        if not enabled:
            severity.zero_()
        q = defaults.clone()
        rpy = torch.zeros(len(ids), 3, device=env.device)
        z = torch.full((len(ids),), self.default_z, device=env.device)
        remaining = torch.arange(len(ids), device=env.device)
        attempts = torch.zeros(len(ids), device=env.device)
        for _ in range(16):
            if not len(remaining):
                break
            amp = severity[remaining, None] * stage["joint_offset"]
            low = torch.maximum(defaults[remaining] - amp, limits[remaining, :, 0])
            high = torch.minimum(defaults[remaining] + amp, limits[remaining, :, 1])
            candidate = low + torch.rand_like(low) * (high - low)
            candidate_rpy = torch.zeros(len(remaining), 3, device=env.device)
            candidate_rpy[:, :2] = (
                torch.empty(len(remaining), 2, device=env.device).uniform_(-1, 1)
                * severity[remaining, None]
                * math.radians(stage["tilt_deg"])
            )
            if enabled:
                candidate_rpy[:, 2].uniform_(-math.pi, math.pi)
            legal, candidate_z = self.geometry.evaluate(
                candidate, robust_math.rpy_rotation(candidate_rpy)
            )
            attempts[remaining] += 1
            accepted = remaining[legal]
            q[accepted] = candidate[legal]
            rpy[accepted] = candidate_rpy[legal]
            z[accepted] = candidate_z[legal]
            remaining = remaining[~legal]

        root = robot.data.default_root_state[ids].clone()
        root[:, :3] = env.scene.env_origins[ids]
        if enabled:
            root[:, :2] += torch.empty(len(ids), 2, device=env.device).uniform_(-.1, .1)
        root[:, 2] += z
        root[:, 3:7] = quat_from_euler_xyz(*rpy.unbind(-1))
        root[:, 7:10] = torch.empty(len(ids), 3, device=env.device).uniform_(-1, 1) * severity[:, None] * stage["root_v"]
        root[:, 10:13] = torch.empty(len(ids), 3, device=env.device).uniform_(-1, 1) * severity[:, None] * stage["root_w"]
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        robot.write_root_pose_to_sim(root[:, :7], env_ids=ids)
        robot.write_root_velocity_to_sim(root[:, 7:], env_ids=ids)

        term.metrics["Reset/rejection_rate"][ids] = (attempts - 1) / attempts.clamp_min(1)
        term.metrics["Reset/rejection_rate"][ids[remaining]] = 1
        term.metrics["Reset/fallback"][ids] = 0
        term.metrics["Reset/fallback"][ids[remaining]] = 1
        term.metrics["Reset/position_offset_max"][ids] = (q - defaults).abs().amax(-1)
        term.metrics["Reset/joint_speed_max"][ids] = 0
        env._nandu_v1_measurement_cache = {}


class HeightRobustDisturbance(ManagerTermBase):
    """Height-task scheduler without the horizontal root push disabled by v1."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.command_name = cfg.params["command_name"]
        self.up3 = self.robot.find_bodies("up3")[0][0]
        sensor = env.scene.sensors["contact_forces"]
        self.wheel_contact_ids = robust_math.wheel_sensor_indices(sensor.body_names)
        count, device = env.num_envs, env.device
        self.wait = torch.empty(count, device=device).uniform_(8, 12)
        self.enabled = torch.rand(count, device=device) > .2
        self.duration = torch.ones(count, device=device)
        self.elapsed = torch.ones(count, device=device)
        self.force = torch.zeros(count, 3, device=device)
        self.torque_local = torch.zeros_like(self.force)
        self.applied_force = torch.zeros_like(self.force)
        self.applied_torque = torch.zeros_like(self.force)
        self.active = torch.zeros(count, dtype=torch.bool, device=device)
        self.mode = torch.full((count,), -1, dtype=torch.long, device=device)
        self.events = 0
        env._nandu_v1_robust_disturbance = self

    def __call__(self, env, env_ids, command_name):
        del env_ids
        term = env.command_manager.get_term(command_name)
        ensure_robust_metrics(term)
        stage = robust_math.STAGES[int(term.stage)]
        dt = env.step_dt
        self.wait -= dt
        self.elapsed += dt
        starting = ((self.wait <= 0) & self.enabled & (env.episode_length_buf * dt > 1) & (term.stage > 0)).nonzero().flatten()
        if len(starting):
            self.events += len(starting)
            modes = torch.randint(0, 4, (len(starting),), device=env.device)
            self.mode[starting] = modes
            self.force[starting] = 0
            self.torque_local[starting] = 0
            force_ids = starting[modes == 0]
            torque_ids = starting[modes == 1]
            self.force[force_ids] = torch.empty(len(force_ids), 3, device=env.device).uniform_(-stage["force"], stage["force"])
            self.torque_local[torque_ids] = torch.empty(len(torque_ids), 3, device=env.device).uniform_(-stage["torque"], stage["torque"])
            if int(term.stage) == 4:
                combined = starting[torch.rand(len(starting), device=env.device) < .1]
                self.force[combined] = torch.empty(len(combined), 3, device=env.device).uniform_(-.7 * stage["force"], .7 * stage["force"])
                self.torque_local[combined] = torch.empty(len(combined), 3, device=env.device).uniform_(-.7 * stage["torque"], .7 * stage["torque"])
            velocity = self.robot.data.root_vel_w[starting].clone()
            z_ids = modes == 2
            angular_ids = modes == 3
            velocity[z_ids, 2] += torch.empty(int(z_ids.sum()), device=env.device).uniform_(-stage["z_kick"], stage["z_kick"])
            velocity[angular_ids, 3:] += torch.empty(int(angular_ids.sum()), 3, device=env.device).uniform_(-stage["angular_kick"], stage["angular_kick"])
            self.robot.write_root_velocity_to_sim(velocity, env_ids=starting)
            self.duration[starting] = torch.empty(len(starting), device=env.device).uniform_(.10, .25)
            self.elapsed[starting] = 0
            self.wait[starting] = torch.empty(len(starting), device=env.device).uniform_(8, 12)

        self.active[:] = self.elapsed < self.duration
        gain = robust_math.pulse_gain(self.elapsed, self.duration)[:, None]
        self.applied_force[:] = self.force * gain
        self.applied_torque[:] = quat_apply(
            self.robot.data.body_quat_w[:, self.up3], self.torque_local
        ) * gain
        self.robot.set_external_force_and_torque(
            self.applied_force[:, None], self.applied_torque[:, None],
            body_ids=[self.up3], is_global=True,
        )
        self.force[~self.active] = 0
        self.torque_local[~self.active] = 0
        term.update_disturbance_state(self.active)

    def reset(self, env_ids=None):
        ids = _ids(env_ids, len(self.wait), self.wait.device)
        self.wait[ids] = torch.empty(len(ids), device=self.wait.device).uniform_(8, 12)
        self.enabled[ids] = torch.rand(len(ids), device=self.wait.device) > .2
        self.elapsed[ids] = 1
        self.duration[ids] = 1
        self.force[ids] = 0
        self.torque_local[ids] = 0
        self.applied_force[ids] = 0
        self.applied_torque[ids] = 0
        self.active[ids] = False
        self.mode[ids] = -1
        zero = torch.zeros(len(ids), 1, 3, device=self.wait.device)
        self.robot.set_external_force_and_torque(
            zero, zero, body_ids=[self.up3], env_ids=ids, is_global=True
        )
        # A physical pulse is deliberately not checkpointed.  Clear the command
        # term's matching transient state as well, so a normal resume cannot
        # manufacture a recovery interval for a force that is no longer active.
        term = self._env.command_manager.get_term(self.command_name)
        term.disturbance_active[ids] = False
        term.disturbance_recovery_pending[ids] = False
        term.disturbance_recovery_elapsed[ids] = 0.0
        term.disturbance_recovery_stable[ids] = 0.0

    def state_dict(self):
        # Transient physics is never replayed; only cumulative provenance is saved.
        return {"schema": SCHEMA_VERSION, "events": self.events}

    def load_state_dict(self, state):
        if state.get("schema") not in (None, SCHEMA_VERSION):
            raise ValueError("incompatible NANDU1 height disturbance checkpoint")
        self.events = int(state.get("events", 0))
        self.reset()
