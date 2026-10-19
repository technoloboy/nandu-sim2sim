"""Task-local MDP terms for NANDU1.

This module intentionally lives beside the NANDU1 configuration so the task
does not inherit behavior or state semantics from WF_TRON2A.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, Imu
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from . import nandu1_math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _ids(env_ids: Sequence[int] | slice, num_envs: int, device: str) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def _heading_velocity(root_quat_w: torch.Tensor, velocity_w: torch.Tensor) -> torch.Tensor:
    return quat_apply_inverse(yaw_quat(root_quat_w), velocity_w)


def base_heading_linear_velocity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return _heading_velocity(robot.data.root_quat_w, robot.data.root_lin_vel_w)


def body_heading_linear_velocity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    # SceneEntityCfg terms are normally resolved by a manager.  Some task-local
    # rewards construct the entity config at call time, in which case the
    # unresolved sentinel is ``slice(None)`` rather than ``None``.
    if body_ids is None or isinstance(body_ids, slice):
        body_ids = robot.find_bodies(asset_cfg.body_names)[0]
    velocity_w = robot.data.body_lin_vel_w[:, body_ids[0]]
    return _heading_velocity(robot.data.root_quat_w, velocity_w)


def imu_angular_velocity(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: Imu = env.scene.sensors[sensor_cfg.name]
    return sensor.data.ang_vel_b


def imu_projected_gravity(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: Imu = env.scene.sensors[sensor_cfg.name]
    return sensor.data.projected_gravity_b


def _observation_corruption_enabled(env: ManagerBasedEnv) -> bool:
    return bool(env.cfg.observations.policy.enable_corruption)


def stage_noisy_imu_angular_velocity(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    final_noise: float,
) -> torch.Tensor:
    value = imu_angular_velocity(env, sensor_cfg)
    stage = int(env.command_manager.get_term(command_name).stage)
    if stage < 3 or not _observation_corruption_enabled(env):
        return value
    return value + torch.empty_like(value).uniform_(-final_noise, final_noise)


def stage_noisy_joint_position(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    noise: float,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    value = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    stage = int(env.command_manager.get_term(command_name).stage)
    if stage < 2 or not _observation_corruption_enabled(env):
        return value
    return value + torch.empty_like(value).uniform_(-noise, noise)


def stage_noisy_joint_velocity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    noise: float,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    value = robot.data.joint_vel[:, asset_cfg.joint_ids]
    stage = int(env.command_manager.get_term(command_name).stage)
    if stage < 2 or not _observation_corruption_enabled(env):
        return value
    return value + torch.empty_like(value).uniform_(-noise, noise)


def non_finite_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    finite = torch.isfinite(robot.data.root_state_w).all(dim=1)
    finite &= torch.isfinite(robot.data.joint_pos).all(dim=1)
    finite &= torch.isfinite(robot.data.joint_vel).all(dim=1)
    return ~finite


class SustainedBadOrientation(ManagerTermBase):
    """Terminate only after the base remains beyond the tilt limit.

    A stateful debounce avoids treating a recoverable, one-step inverted
    pendulum excursion as a terminal failure.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._elapsed = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        limit_angle: float,
        duration_s: float,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        sensor: Imu = env.scene.sensors[sensor_cfg.name]
        tilt = torch.acos(torch.clamp(-sensor.data.projected_gravity_b[:, 2], -1.0, 1.0))
        outside = tilt > limit_angle
        self._elapsed = torch.where(outside, self._elapsed + env.step_dt, torch.zeros_like(self._elapsed))
        return self._elapsed >= duration_s

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        if env_ids is None:
            self._elapsed.zero_()
        else:
            self._elapsed[_ids(env_ids, len(self._elapsed), self._elapsed.device)] = 0.0


@configclass
class Nandu1CommandCfg(CommandTermCfg):
    class_type: type = None

    @configclass
    class Ranges:
        yaw_rate: tuple[float, float] = MISSING
        ee_vx: tuple[float, float] = MISSING
        ee_vz: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    acceleration_limits: tuple[float, float, float] = (2.0, 0.8, 0.5)
    nominal_height_range: tuple[float, float] = (0.55, 2.00)
    height_taper_buffer: float = 0.10
    vertical_pulse_time_range: tuple[float, float] = (0.8, 2.0)
    vertical_pulse_safety_fraction: float = 0.8
    evaluation_steps: int = 6000
    max_reach_resamples: int = 8
    reach_resample_limit: float = 0.66


class Nandu1Command(CommandTerm):
    """Slew-limited yaw and end-effector velocity command."""

    cfg: Nandu1CommandCfg
    STAGE_LIMITS = (
        (0.20, 0.08, 0.05),
        (0.50, 0.20, 0.10),
        (0.85, 0.40, 0.20),
        (1.20, 0.70, 0.35),
        (1.57, 1.00, 0.50),
    )
    STAGE_ACCELERATION_LIMITS = (
        (0.50, 0.20, 0.12),
        (0.80, 0.35, 0.20),
        (1.20, 0.50, 0.30),
        (1.60, 0.65, 0.40),
        (2.00, 0.80, 0.50),
    )
    # Probabilities for zero-, single-, dual- and triple-axis commands.
    STAGE_AXIS_COUNT_PROBABILITIES = (
        (0.30, 0.50, 0.20, 0.00),
        (0.10, 0.70, 0.20, 0.00),
        (0.00, 0.30, 0.60, 0.10),
        (0.00, 0.20, 0.30, 0.50),
        (0.00, 0.10, 0.20, 0.70),
    )
    SAMPLE_MODE_PROBABILITIES = (0.60, 0.25, 0.15)  # core, boundary/hard, recovery
    STAGE_COMPLETION_THRESHOLDS = (0.75, 0.80, 0.85, 0.90)
    STAGE_SAFETY_THRESHOLDS = (0.10, 0.08, 0.06, 0.05)
    STAGE_WHEEL_SPACING_THRESHOLDS = (0.45, 0.30, 0.15, 0.05)
    STAGE_KNEE_SPACING_THRESHOLDS = (0.10, 0.07, 0.05, 0.02)
    STAGE_CANCELLATION_THRESHOLDS = (0.30, 0.20, 0.10, 0.05)
    STAGE_REACH_SOFT_THRESHOLDS = (0.06, 0.04, 0.02, 0.01)
    STAGE_REACH_068_THRESHOLDS = (0.05, 0.04, 0.03, 0.02)
    STAGE_REACH_070_THRESHOLDS = (0.02, 0.015, 0.01, 0.005)
    # Auxiliary geometry uses a soft aggregate score instead of eight brittle
    # logical-AND gates.  The original limits remain the full-credit targets;
    # the wider limits below are non-negotiable safety ceilings.
    STAGE_COORDINATION_SCORE_THRESHOLDS = (0.60, 0.65, 0.70, 0.75)
    STAGE_COORDINATION_WEIGHTS = (0.10, 0.20, 0.10, 0.15, 0.10, 0.05, 0.05, 0.25)
    STAGE_COORDINATION_HARD_LIMITS = (
        (0.18, 0.14, 0.70, 0.50, 0.10, 0.12, 0.06, 0.35),
        (0.12, 0.12, 0.50, 0.40, 0.08, 0.10, 0.05, 0.30),
        (0.08, 0.11, 0.35, 0.30, 0.07, 0.08, 0.04, 0.25),
        (0.05, 0.10, 0.20, 0.20, 0.06, 0.06, 0.03, 0.18),
    )
    STAGE_TRACKING_RMSE_THRESHOLDS = (
        (0.20, 0.10, 0.07),
        (0.24, 0.14, 0.09),
        (0.28, 0.20, 0.13),
        (0.32, 0.28, 0.18),
    )
    STAGE_TILT_P95_THRESHOLDS = tuple(math.radians(value) for value in (15.0, 13.0, 11.0, 10.0))
    STAGE_HEIGHT_RANGES = (
        (0.78, 1.04),
        (0.68, 1.10),
        (0.55, 1.15),
        (0.45, 1.18),
        (0.40, 1.20),
    )
    STAGE_BASE_HEIGHT_RANGES = (
        (0.61, 0.70),
        (0.56, 0.71),
        (0.50, 0.71),
        (0.44, 0.71),
        (0.38, 0.71),
    )
    STAGE_ARM_HEIGHT_RANGES = (
        (0.12, 0.35),
        (0.02, 0.50),
        (-0.05, 0.65),
        (-0.10, 0.78),
        (-0.20, 0.95),
    )

    def __init__(self, cfg: Nandu1CommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene["robot"]
        self._base_id = self._robot.find_bodies("base_Link")[0][0]
        self._up3_id = self._robot.find_bodies("up3")[0][0]
        self._wheel_ids = self._robot.find_bodies(["wheel_L_Link", "wheel_R_Link"], preserve_order=True)[0]
        self._knee_body_ids = self._robot.find_bodies(["knee_L_Link", "knee_R_Link"], preserve_order=True)[0]
        self._left_leg_ids = self._robot.find_joints(
            ["proximal_pitch_L_Joint", "proximal_roll_L_Joint", "proximal_yaw_L_Joint", "knee_L_Joint"],
            preserve_order=True,
        )[0]
        self._right_leg_ids = self._robot.find_joints(
            ["proximal_pitch_R_Joint", "proximal_roll_R_Joint", "proximal_yaw_R_Joint", "knee_R_Joint"],
            preserve_order=True,
        )[0]
        self._leg_ids = self._left_leg_ids + self._right_leg_ids
        self._arm_ids = self._robot.find_joints(["arm1", "arm2", "arm3"], preserve_order=True)[0]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_command = torch.zeros_like(self._command)
        self.command_acceleration = torch.zeros_like(self._command)
        self.vertical_pulse_time_left = torch.zeros(self.num_envs, device=self.device)
        self.nominal_height = torch.ones(self.num_envs, device=self.device)
        self.filtered_base_vx_target = torch.zeros(self.num_envs, device=self.device)
        self.alpha_base = torch.full((self.num_envs,), 0.5, device=self.device)
        self.horizontal_alpha_base = torch.full((self.num_envs,), 0.65, device=self.device)
        self.base_height = torch.zeros(self.num_envs, device=self.device)
        self.arm_relative_height = torch.zeros(self.num_envs, device=self.device)
        self.horizontal_reach = torch.zeros(self.num_envs, device=self.device)
        self.com_support_offset = torch.zeros(self.num_envs, device=self.device)
        self.actual_base_vz = torch.zeros(self.num_envs, device=self.device)
        self.actual_arm_vz = torch.zeros(self.num_envs, device=self.device)
        self.target_base_vz = torch.zeros(self.num_envs, device=self.device)
        self.target_arm_vz = torch.zeros(self.num_envs, device=self.device)
        self.actual_base_vx = torch.zeros(self.num_envs, device=self.device)
        self.actual_arm_vx = torch.zeros(self.num_envs, device=self.device)
        self.target_base_vx = torch.zeros(self.num_envs, device=self.device)
        self.target_arm_vx = torch.zeros(self.num_envs, device=self.device)
        self.horizontal_reach_rate = torch.zeros(self.num_envs, device=self.device)
        self.tracking_gate = torch.zeros(self.num_envs, device=self.device)
        self.symmetry_gate = torch.zeros(self.num_envs, device=self.device)
        self.base_directional_margin = torch.ones(self.num_envs, device=self.device)
        self.arm_directional_margin = torch.ones(self.num_envs, device=self.device)
        self.leg_position_residual = torch.zeros(self.num_envs, 4, device=self.device)
        self.leg_velocity_residual = torch.zeros(self.num_envs, 4, device=self.device)
        self.wheel_height_difference = torch.zeros(self.num_envs, device=self.device)
        self.wheel_spacing = torch.zeros(self.num_envs, device=self.device)
        self.wheel_load_difference = torch.zeros(self.num_envs, device=self.device)
        self.knee_delta_h = torch.zeros(self.num_envs, 3, device=self.device)
        self.knee_spacing = torch.zeros(self.num_envs, device=self.device)
        self.cancellation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.relative_up3_h = torch.zeros(self.num_envs, 3, device=self.device)
        self.stage = 0
        self.qualifying_windows = 0
        self.last_evaluation_step = 0
        self.window_error_sum = torch.zeros(4, device=self.device)
        self.window_error_square_sum = torch.zeros(4, device=self.device)
        self.window_samples = 0
        self.window_resets = 0
        self.window_timeouts = 0
        self.window_coordination_sum = torch.zeros(9, device=self.device)
        self.window_reach_samples = 0
        self.window_reach_rejections = 0
        self.window_reach_attempts = 0
        self.metrics = {
            "stage": torch.zeros(self.num_envs, device=self.device),
            "yaw_abs_error": torch.zeros(self.num_envs, device=self.device),
            "ee_vx_abs_error": torch.zeros(self.num_envs, device=self.device),
            "ee_vz_abs_error": torch.zeros(self.num_envs, device=self.device),
            "up3_tilt": torch.zeros(self.num_envs, device=self.device),
            "Command/target_abs_yaw": torch.zeros(self.num_envs, device=self.device),
            "Command/target_abs_vx": torch.zeros(self.num_envs, device=self.device),
            "Command/target_abs_vz": torch.zeros(self.num_envs, device=self.device),
            "Command/vertical_pulse_time": torch.zeros(self.num_envs, device=self.device),
            "Geometry/base_height": torch.zeros(self.num_envs, device=self.device),
            "Geometry/arm_relative_height": torch.zeros(self.num_envs, device=self.device),
            "Geometry/up3_base_horizontal_distance": torch.zeros(self.num_envs, device=self.device),
            "Geometry/COM_support_offset": torch.zeros(self.num_envs, device=self.device),
            "Allocation/alpha_base": torch.zeros(self.num_envs, device=self.device),
            "Allocation/actual_base_vz": torch.zeros(self.num_envs, device=self.device),
            "Allocation/actual_arm_vz": torch.zeros(self.num_envs, device=self.device),
            "Allocation/target_base_vz": torch.zeros(self.num_envs, device=self.device),
            "Allocation/target_arm_vz": torch.zeros(self.num_envs, device=self.device),
            "Allocation/cancellation_rate": torch.zeros(self.num_envs, device=self.device),
            "Allocation/horizontal_alpha_base": torch.zeros(self.num_envs, device=self.device),
            "Allocation/actual_base_vx": torch.zeros(self.num_envs, device=self.device),
            "Allocation/actual_arm_vx": torch.zeros(self.num_envs, device=self.device),
            "Allocation/target_base_vx": torch.zeros(self.num_envs, device=self.device),
            "Allocation/target_arm_vx": torch.zeros(self.num_envs, device=self.device),
            "Allocation/horizontal_reach_rate": torch.zeros(self.num_envs, device=self.device),
            "Symmetry/pitch_error": torch.zeros(self.num_envs, device=self.device),
            "Symmetry/roll_mirror_error": torch.zeros(self.num_envs, device=self.device),
            "Symmetry/yaw_mirror_error": torch.zeros(self.num_envs, device=self.device),
            "Symmetry/knee_error": torch.zeros(self.num_envs, device=self.device),
            "Symmetry/velocity_mirror_error": torch.zeros(self.num_envs, device=self.device),
            "Symmetry/wheel_height_difference": torch.zeros(self.num_envs, device=self.device),
            "Symmetry/wheel_spacing": torch.zeros(self.num_envs, device=self.device),
            "Symmetry/wheel_load_difference": torch.zeros(self.num_envs, device=self.device),
            "Symmetry/knee_spacing": torch.zeros(self.num_envs, device=self.device),
        }

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _sample_stage_command(self, ids: torch.Tensor) -> torch.Tensor:
        count = len(ids)
        maximum = torch.tensor(self.STAGE_LIMITS[self.stage], device=self.device)
        axis_count = nandu1_math.sample_axis_counts(
            self.STAGE_AXIS_COUNT_PROBABILITIES[self.stage], count, self.device
        )
        axis_order = torch.argsort(torch.rand(count, 3, device=self.device), dim=1)
        active_mask = torch.zeros(count, 3, dtype=torch.bool, device=self.device)
        active_mask.scatter_(1, axis_order, torch.arange(3, device=self.device)[None, :] < axis_count[:, None])

        sample_mode = torch.multinomial(
            torch.tensor(self.SAMPLE_MODE_PROBABILITIES, device=self.device), count, replacement=True
        )
        core = (2.0 * torch.rand(count, 3, device=self.device) - 1.0) * (0.70 * maximum)
        hard = torch.where(
            torch.rand(count, 3, device=self.device) < 0.5,
            -torch.ones(count, 3, device=self.device),
            torch.ones(count, 3, device=self.device),
        )
        hard *= (0.80 + 0.20 * torch.rand(count, 3, device=self.device)) * maximum
        sampled = torch.where((sample_mode == 1)[:, None], hard, core)
        sampled *= active_mask

        # Recovery samples point toward nominal geometric reserve without using
        # the actual up3 position to close the command loop.
        recovery = sample_mode == 2
        if recovery.any():
            recovery_ids = ids[recovery]
            sampled[recovery, 0] *= 0.5
            sampled[recovery, 1] = -torch.sign(self.relative_up3_h[recovery_ids, 0]) * sampled[
                recovery, 1
            ].abs()
            low, high = self.STAGE_HEIGHT_RANGES[self.stage]
            height_direction = torch.sign((low + high) * 0.5 - self.nominal_height[recovery_ids])
            sampled[recovery, 2] = height_direction * sampled[recovery, 2].abs()

        # Stage 0 dual-axis samples are deliberately low amplitude while the
        # balance and online velocity estimator are still bootstrapping.
        sampled[axis_count == 2] *= 0.5 if self.stage == 0 else 1.0
        return sampled

    def _apply_height_preview(self, ids: torch.Tensor, sampled: torch.Tensor) -> torch.Tensor:
        low, high = self.STAGE_HEIGHT_RANGES[self.stage]
        center = 0.5 * (low + high)
        minimum_pulse = self.cfg.vertical_pulse_time_range[0]
        margin = torch.where(
            sampled[:, 2] >= 0.0,
            high - self.nominal_height[ids],
            self.nominal_height[ids] - low,
        )
        insufficient = margin < sampled[:, 2].abs() * minimum_pulse / self.cfg.vertical_pulse_safety_fraction
        inward = torch.sign(center - self.nominal_height[ids]) * sampled[:, 2].abs()
        sampled[:, 2] = torch.where(insufficient, inward, sampled[:, 2])
        return sampled

    def _schedule_vertical_pulse(self, ids: torch.Tensor):
        if len(ids) == 0:
            return
        low, high = self.STAGE_HEIGHT_RANGES[self.stage]
        requested = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.vertical_pulse_time_range)
        duration = nandu1_math.safe_vertical_pulse_duration(
            self.target_command[ids, 2],
            self.nominal_height[ids],
            low,
            high,
            requested,
            self.cfg.vertical_pulse_safety_fraction,
        )
        inactive = self.target_command[ids, 2].abs() <= 1.0e-6
        duration = torch.where(inactive, self.time_left[ids].clamp_min(self._env.step_dt), duration)
        self.vertical_pulse_time_left[ids] = duration.clamp_min(self._env.step_dt)

    def set_manual_velocity_command(self, values):
        """Play-only target override; no changes to the applied command or slew."""
        target = torch.as_tensor(values, device=self.device, dtype=self.target_command.dtype)
        if target.shape != (3,) or not torch.isfinite(target).all():
            raise ValueError("Manual velocity must be finite [yaw, vx, vz].")
        limits = torch.tensor(self.STAGE_LIMITS[self.stage], device=self.device)
        self._manual_velocity_target = target.clamp(-limits, limits)
        self.target_command[:] = self._manual_velocity_target

    def _resample_command(self, env_ids: Sequence[int]):
        ids = _ids(env_ids, self.num_envs, self.device)
        if getattr(self, "_manual_velocity_target", None) is not None:
            self.target_command[ids] = self._manual_velocity_target
            return
        sampled = torch.zeros(len(ids), 3, device=self.device)
        remaining = torch.arange(len(ids), device=self.device)
        for _ in range(self.cfg.max_reach_resamples):
            if len(remaining) == 0:
                break
            global_ids = ids[remaining]
            candidate = self._sample_stage_command(global_ids)
            candidate = self._apply_height_preview(global_ids, candidate)
            rejected = nandu1_math.outward_reach_command(
                self.relative_up3_h[global_ids, 0],
                candidate[:, 1],
                self.horizontal_reach[global_ids],
                self.cfg.reach_resample_limit,
            )
            self.window_reach_attempts += len(remaining)
            self.window_reach_rejections += int(rejected.sum().item())
            sampled[remaining[~rejected]] = candidate[~rejected]
            remaining = remaining[rejected]
        if len(remaining) > 0:
            global_ids = ids[remaining]
            fallback = self._apply_height_preview(global_ids, self._sample_stage_command(global_ids))
            fallback[:, 1] = 0.0
            sampled[remaining] = fallback
        self.target_command[ids] = sampled
        self._schedule_vertical_pulse(ids)

    def _update_command(self):
        self.vertical_pulse_time_left -= self._env.step_dt
        expired = (self.vertical_pulse_time_left <= 0.0) & (self.target_command[:, 2].abs() > 1.0e-6)
        if getattr(self, "_manual_velocity_target", None) is None and expired.any():
            expired_ids = expired.nonzero().flatten()
            self.target_command[expired_ids, 2] *= -1.0
            flipped = self._apply_height_preview(expired_ids, self.target_command[expired_ids].clone())
            self.target_command[expired_ids, 2] = flipped[:, 2]
            self._schedule_vertical_pulse(expired_ids)
        previous = self._command.clone()
        limits = torch.tensor(self.STAGE_ACCELERATION_LIMITS[self.stage], device=self.device) * self._env.step_dt
        delta = torch.clamp(self.target_command - self._command, min=-limits, max=limits)
        self._command += delta
        height_range = self.STAGE_HEIGHT_RANGES[self.stage]
        self._command[:, 2] = nandu1_math.limit_vertical_velocity(
            self._command[:, 2],
            self.nominal_height,
            *height_range,
            self.cfg.height_taper_buffer,
        )
        self.command_acceleration = (self._command - previous) / self._env.step_dt
        self.nominal_height.add_(self._command[:, 2] * self._env.step_dt)
        self.nominal_height.clamp_(*height_range)
        alpha = self._env.step_dt / (0.8 + self._env.step_dt)
        self.filtered_base_vx_target += alpha * (self._command[:, 1] - self.filtered_base_vx_target)

    REACH_SOFT_LIMIT = 0.60
    REACH_HARD_LIMIT = 0.70
    REACH_DIAGNOSTIC_LIMITS = (0.67, 0.68, 0.70)

    def _up3_position_w(self):
        return self._robot.data.body_pos_w[:, self._up3_id]

    def _up3_velocity_w(self):
        return self._robot.data.body_lin_vel_w[:, self._up3_id]

    def _reach_state_world(self):
        # Default retains every existing task's point/velocity semantics.
        return self._up3_position_w(), self._up3_velocity_w()

    def _update_reach_geometry(self):
        position, velocity = self._reach_state_world()
        robot = self._robot
        relative = position - robot.data.body_pos_w[:, self._base_id]
        self.relative_up3_h[:] = _heading_velocity(robot.data.root_quat_w, relative)
        self.horizontal_reach[:] = torch.linalg.norm(self.relative_up3_h[:, :2], dim=1)
        relative_velocity_h = _heading_velocity(
            robot.data.root_quat_w, velocity - self._base_point_velocity_w(position)
        )
        self.horizontal_reach_rate[:] = torch.sum(
            self.relative_up3_h[:, :2] * relative_velocity_h[:, :2], dim=1
        ) / self.horizontal_reach.clamp_min(1.0e-6)

    def _base_point_velocity_w(self, up3_position):
        # Legacy defaults intentionally retain the original task semantics.
        relative = up3_position - self._robot.data.body_pos_w[:, self._base_id]
        return self._robot.data.body_lin_vel_w[:, self._base_id] + torch.linalg.cross(
            self._robot.data.body_ang_vel_w[:, self._base_id], relative, dim=1
        )

    def _vertical_velocity_components(self, up3_velocity, base_point_velocity):
        base = self._robot.data.body_lin_vel_w[:, self._base_id, 2]
        return base, up3_velocity[:, 2] - base

    def _com_quality(self):
        """Default-preserving extension point for isolated balance-cost children."""
        return torch.exp(-torch.square(self.com_support_offset / 0.08))

    def _update_coordination_state(self):
        robot = self._robot
        base_position = robot.data.body_pos_w[:, self._base_id]
        up3_position = self._up3_position_w()
        wheel_position = robot.data.body_pos_w[:, self._wheel_ids]
        self._update_reach_geometry()
        wheel_contact_plane = wheel_position[:, :, 2].mean(dim=1) - 0.10
        self.base_height[:] = base_position[:, 2] - wheel_contact_plane
        self.arm_relative_height[:] = up3_position[:, 2] - base_position[:, 2]
        up3_velocity = self._up3_velocity_w()
        base_point_velocity_w = self._base_point_velocity_w(up3_position)
        self.actual_base_vz[:], self.actual_arm_vz[:] = self._vertical_velocity_components(
            up3_velocity, base_point_velocity_w
        )
        relative_velocity_w = up3_velocity - base_point_velocity_w
        base_point_velocity_h = _heading_velocity(robot.data.root_quat_w, base_point_velocity_w)
        relative_velocity_h = _heading_velocity(robot.data.root_quat_w, relative_velocity_w)
        self.actual_base_vx[:] = base_point_velocity_h[:, 0]
        self.actual_arm_vx[:] = relative_velocity_h[:, 0]
        masses = getattr(self._env, "_nandu1_randomized_body_masses", robot.data.default_mass)
        masses = masses.to(device=self.device, dtype=robot.data.body_com_pos_w.dtype)
        total_com = torch.sum(robot.data.body_com_pos_w * masses.unsqueeze(-1), dim=1) / masses.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        support = wheel_position.mean(dim=1)
        self.com_support_offset[:] = torch.linalg.norm((total_com - support)[:, :2], dim=1)

        torque_utilization = robot.data.applied_torque.abs() / robot.data.joint_effort_limits.clamp_min(1.0e-6)
        base_torque_quality = 1.0 - torch.clamp(torque_utilization[:, self._leg_ids].amax(dim=1), 0.0, 1.0)
        arm_torque_quality = 1.0 - torch.clamp(torque_utilization[:, self._arm_ids].amax(dim=1), 0.0, 1.0)
        com_quality = self._com_quality()

        sensor: ContactSensor = self._env.scene.sensors["contact_forces"]
        loads = torch.clamp(sensor.data.net_forces_w[:, getattr(self, "_wheel_contact_ids", self._wheel_ids), 2], min=0.0)
        self.wheel_load_difference[:] = torch.abs(loads[:, 0] - loads[:, 1]) / loads.sum(dim=1).clamp_min(1.0)
        load_quality = 1.0 - torch.clamp(self.wheel_load_difference, 0.0, 1.0)
        # Preserve authority around the 0.533 m neutral reach and taper only
        # through the final 0.10 m before the 0.70 m hard envelope.
        reach_quality = torch.clamp((self.REACH_HARD_LIMIT - self.horizontal_reach) / 0.10, 0.0, 1.0)

        base_range = self.STAGE_BASE_HEIGHT_RANGES[self.stage]
        arm_range = self.STAGE_ARM_HEIGHT_RANGES[self.stage]
        self.base_directional_margin[:] = nandu1_math.directional_margin(
            self._command[:, 2], self.base_height, base_range
        )
        self.arm_directional_margin[:] = nandu1_math.directional_margin(
            self._command[:, 2], self.arm_relative_height, arm_range
        )
        raw_alpha = nandu1_math.height_allocation_ratio(
            self._command[:, 2],
            self.base_height,
            self.arm_relative_height,
            base_range,
            arm_range,
            base_torque_quality * com_quality * load_quality,
            arm_torque_quality * reach_quality,
        )
        active = self._command[:, 2].abs() > 0.02
        filter_gain = self._env.step_dt / (0.30 + self._env.step_dt)
        filtered = self.alpha_base + filter_gain * (raw_alpha - self.alpha_base)
        self.alpha_base[:] = torch.where(active, filtered, self.alpha_base)
        self.target_base_vz[:] = self.alpha_base * self._command[:, 2]
        self.target_arm_vz[:] = (1.0 - self.alpha_base) * self._command[:, 2]

        raw_horizontal_alpha = nandu1_math.horizontal_allocation_ratio(
            self.filtered_base_vx_target,
            self.relative_up3_h[:, 0],
            self.horizontal_reach,
            base_torque_quality * com_quality * load_quality,
            arm_torque_quality * reach_quality,
            soft_limit=self.REACH_SOFT_LIMIT, hard_limit=self.REACH_HARD_LIMIT,
        )
        horizontal_active = torch.maximum(
            self._command[:, 1].abs(), self.filtered_base_vx_target.abs()
        ) > 0.02
        horizontal_filter_gain = self._env.step_dt / (0.30 + self._env.step_dt)
        horizontal_filtered = self.horizontal_alpha_base + horizontal_filter_gain * (
            raw_horizontal_alpha - self.horizontal_alpha_base
        )
        self.horizontal_alpha_base[:] = torch.where(
            horizontal_active, horizontal_filtered, self.horizontal_alpha_base
        )
        self.target_base_vx[:] = self.horizontal_alpha_base * self.filtered_base_vx_target
        self.target_arm_vx[:] = self._command[:, 1] - self.target_base_vx

        left_pos = robot.data.joint_pos[:, self._left_leg_ids]
        right_pos = robot.data.joint_pos[:, self._right_leg_ids]
        left_vel = robot.data.joint_vel[:, self._left_leg_ids]
        right_vel = robot.data.joint_vel[:, self._right_leg_ids]
        self.leg_position_residual[:] = nandu1_math.leg_mirror_residual(left_pos, right_pos)
        self.leg_velocity_residual[:] = nandu1_math.leg_mirror_residual(left_vel, right_vel)
        wheel_delta_h = _heading_velocity(
            robot.data.root_quat_w, wheel_position[:, 0] - wheel_position[:, 1]
        )
        self.wheel_height_difference[:] = wheel_delta_h[:, 2].abs()
        self.wheel_spacing[:] = wheel_delta_h[:, 1].abs()
        knee_position = robot.data.body_pos_w[:, self._knee_body_ids]
        self.knee_delta_h[:] = _heading_velocity(
            robot.data.root_quat_w, knee_position[:, 0] - knee_position[:, 1]
        )
        self.knee_spacing[:] = self.knee_delta_h[:, 1].abs()
        direction = torch.sign(self._command[:, 2])
        self.cancellation[:] = (self._command[:, 2].abs() > 0.02) & (
            (self.actual_base_vz * direction < 0.0) | (self.actual_arm_vz * direction < 0.0)
        )

    def _update_metrics(self):
        self._update_coordination_state()
        robot = self._robot
        up3_velocity = _heading_velocity(robot.data.root_quat_w, robot.data.body_lin_vel_w[:, self._up3_id])
        base_yaw_rate = robot.data.root_ang_vel_w[:, 2]
        up3_imu: Imu = self._env.scene.sensors["up3_imu"]
        tilt = torch.acos(torch.clamp(-up3_imu.data.projected_gravity_b[:, 2], -1.0, 1.0))
        errors = torch.stack(
            (
                torch.abs(base_yaw_rate - self._command[:, 0]),
                torch.abs(up3_velocity[:, 0] - self._command[:, 1]),
                torch.abs(up3_velocity[:, 2] - self._command[:, 2]),
                tilt,
            ),
            dim=1,
        )
        metric_scale = 1.0 / float(self._env.max_episode_length)
        self.metrics["stage"][:] = float(self.stage)
        self.metrics["yaw_abs_error"] += errors[:, 0] * metric_scale
        self.metrics["ee_vx_abs_error"] += errors[:, 1] * metric_scale
        self.metrics["ee_vz_abs_error"] += errors[:, 2] * metric_scale
        self.metrics["up3_tilt"] += errors[:, 3] * metric_scale
        self.metrics["Command/target_abs_yaw"] += self.target_command[:, 0].abs() * metric_scale
        self.metrics["Command/target_abs_vx"] += self.target_command[:, 1].abs() * metric_scale
        self.metrics["Command/target_abs_vz"] += self.target_command[:, 2].abs() * metric_scale
        self.metrics["Command/vertical_pulse_time"] += self.vertical_pulse_time_left.clamp_min(0.0) * metric_scale
        self.tracking_gate[:] = nandu1_math.vertical_command_gate(self._command[:, 2])
        self.tracking_gate *= torch.exp(-torch.square(errors[:, 2] / 0.08) - torch.square(errors[:, 3] / 0.12))
        self.symmetry_gate[:] = nandu1_math.tracking_quality_gate(
            errors, (0.25, 0.14, 0.10, math.radians(15.0))
        )
        self.metrics["Geometry/base_height"] += self.base_height * metric_scale
        self.metrics["Geometry/arm_relative_height"] += self.arm_relative_height * metric_scale
        self.metrics["Geometry/up3_base_horizontal_distance"] += self.horizontal_reach * metric_scale
        self.metrics["Geometry/COM_support_offset"] += self.com_support_offset * metric_scale
        self.metrics["Allocation/alpha_base"] += self.alpha_base * metric_scale
        self.metrics["Allocation/actual_base_vz"] += self.actual_base_vz * metric_scale
        self.metrics["Allocation/actual_arm_vz"] += self.actual_arm_vz * metric_scale
        self.metrics["Allocation/target_base_vz"] += self.target_base_vz * metric_scale
        self.metrics["Allocation/target_arm_vz"] += self.target_arm_vz * metric_scale
        self.metrics["Allocation/cancellation_rate"] += self.cancellation.float() * metric_scale
        self.metrics["Allocation/horizontal_alpha_base"] += self.horizontal_alpha_base * metric_scale
        self.metrics["Allocation/actual_base_vx"] += self.actual_base_vx * metric_scale
        self.metrics["Allocation/actual_arm_vx"] += self.actual_arm_vx * metric_scale
        self.metrics["Allocation/target_base_vx"] += self.target_base_vx * metric_scale
        self.metrics["Allocation/target_arm_vx"] += self.target_arm_vx * metric_scale
        self.metrics["Allocation/horizontal_reach_rate"] += self.horizontal_reach_rate * metric_scale
        for index, name in enumerate(("pitch_error", "roll_mirror_error", "yaw_mirror_error", "knee_error")):
            self.metrics[f"Symmetry/{name}"] += self.leg_position_residual[:, index].abs() * metric_scale
        self.metrics["Symmetry/velocity_mirror_error"] += torch.sqrt(
            torch.mean(torch.square(self.leg_velocity_residual), dim=1)
        ) * metric_scale
        self.metrics["Symmetry/wheel_height_difference"] += self.wheel_height_difference * metric_scale
        self.metrics["Symmetry/wheel_spacing"] += self.wheel_spacing * metric_scale
        self.metrics["Symmetry/wheel_load_difference"] += self.wheel_load_difference * metric_scale
        self.metrics["Symmetry/knee_spacing"] += self.knee_spacing * metric_scale
        self.window_error_sum += errors.mean(dim=0)
        self.window_error_square_sum += torch.square(errors).mean(dim=0)
        yaw_relaxation = 1.0 + 1.5 * self._command[:, 0].abs() / self.STAGE_LIMITS[self.stage][0]
        yaw_relaxation += 0.5 * self.command_acceleration[:, 0].abs() / self.STAGE_ACCELERATION_LIMITS[self.stage][0]
        curriculum_residual = self.leg_position_residual.clone()
        curriculum_residual[:, 1:3] /= yaw_relaxation[:, None]
        coordination = torch.stack(
            (
                (self.horizontal_reach > 0.67).float().mean(),
                torch.mean(torch.square(curriculum_residual)),
                ((self.wheel_spacing < 0.42) | (self.wheel_spacing > 0.49)).float().mean(),
                self.cancellation.float().mean(),
                (self.wheel_height_difference > 0.03).float().mean(),
                (self.horizontal_reach > 0.68).float().mean(),
                (self.horizontal_reach > 0.70).float().mean(),
                ((self.knee_spacing < 0.35) | (self.knee_spacing > 0.44)).float().mean(),
                (errors[:, 3] > self.STAGE_TILT_P95_THRESHOLDS[min(self.stage, 3)]).float().mean(),
            )
        )
        self.window_coordination_sum += coordination
        self.window_reach_samples += self.num_envs
        self.window_samples += 1

    def record_resets(self, env_ids: Sequence[int], timeouts: torch.Tensor):
        ids = _ids(env_ids, self.num_envs, self.device)
        self.window_resets += len(ids)
        self.window_timeouts += int(timeouts[ids].sum().item())

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, float]:
        """Align the new nominal path with the *post-reset* arm pose."""
        ids = _ids(slice(None) if env_ids is None else env_ids, self.num_envs, self.device)
        self._update_coordination_state()
        height_range = self.STAGE_HEIGHT_RANGES[self.stage]
        self.nominal_height[ids] = self._robot.data.body_pos_w[ids, self._up3_id, 2].clamp(*height_range)
        self._command[ids] = 0.0
        self.target_command[ids] = 0.0
        self.command_acceleration[ids] = 0.0
        self.vertical_pulse_time_left[ids] = 0.0
        self.filtered_base_vx_target[ids] = 0.0
        self.alpha_base[ids] = 0.5
        self.target_base_vz[ids] = 0.0
        self.target_arm_vz[ids] = 0.0
        self.horizontal_alpha_base[ids] = 0.65
        self.actual_base_vx[ids] = 0.0
        self.actual_arm_vx[ids] = 0.0
        self.target_base_vx[ids] = 0.0
        self.target_arm_vx[ids] = 0.0
        self.horizontal_reach_rate[ids] = 0.0
        return super().reset(env_ids)

    def evaluate_curriculum(self, common_step_counter: int) -> dict[str, float]:
        if common_step_counter - self.last_evaluation_step < self.cfg.evaluation_steps or self.window_samples == 0:
            return self.curriculum_state()
        error_rmse = torch.sqrt(self.window_error_square_sum / self.window_samples)
        coordination = self.window_coordination_sum / self.window_samples
        completion = self.window_timeouts / max(self.window_resets, 1)
        safety_failure = 1.0 - completion
        if self.stage < 4:
            yaw, vx, vz = self.STAGE_TRACKING_RMSE_THRESHOLDS[self.stage]
            tilt_p95_qualification = coordination[8] <= 0.05
            coordination_score, coordination_hard_safe = self._coordination_curriculum_state(coordination)
            qualified = bool(
                error_rmse[0] <= yaw
                and error_rmse[1] <= vx
                and error_rmse[2] <= vz
                and tilt_p95_qualification
                and completion >= self.STAGE_COMPLETION_THRESHOLDS[self.stage]
                and safety_failure <= self.STAGE_SAFETY_THRESHOLDS[self.stage]
                and coordination_hard_safe
                and coordination_score >= self.STAGE_COORDINATION_SCORE_THRESHOLDS[self.stage]
            )
            self.qualifying_windows = self.qualifying_windows + 1 if qualified else 0
            if self.qualifying_windows >= 3:
                self.stage += 1
                self.qualifying_windows = 0
        self.last_evaluation_step = common_step_counter
        # Preserve the just-completed window for TensorBoard.  Building this
        # dictionary after zeroing the accumulators silently reported zeros at
        # every curriculum boundary and made failed gates impossible to audit.
        result = self.curriculum_state(completion, safety_failure)
        self.window_error_sum.zero_()
        self.window_error_square_sum.zero_()
        self.window_samples = 0
        self.window_resets = 0
        self.window_timeouts = 0
        self.window_coordination_sum.zero_()
        self.window_reach_samples = 0
        self.window_reach_rejections = 0
        self.window_reach_attempts = 0
        return result

    def _coordination_curriculum_state(self, coordination: torch.Tensor) -> tuple[float, bool]:
        stage = min(self.stage, 3)
        metrics = torch.stack(
            (
                coordination[0],
                torch.sqrt(coordination[1]),
                coordination[2],
                coordination[3],
                coordination[4],
                coordination[5],
                coordination[6],
                coordination[7],
            )
        )
        targets = metrics.new_tensor(
            (
                self.STAGE_REACH_SOFT_THRESHOLDS[stage],
                0.08,
                self.STAGE_WHEEL_SPACING_THRESHOLDS[stage],
                self.STAGE_CANCELLATION_THRESHOLDS[stage],
                0.05,
                self.STAGE_REACH_068_THRESHOLDS[stage],
                self.STAGE_REACH_070_THRESHOLDS[stage],
                self.STAGE_KNEE_SPACING_THRESHOLDS[stage],
            )
        )
        hard_limits = metrics.new_tensor(self.STAGE_COORDINATION_HARD_LIMITS[stage])
        weights = metrics.new_tensor(self.STAGE_COORDINATION_WEIGHTS)
        score, hard_safe = nandu1_math.coordination_curriculum_score(
            metrics, targets, hard_limits, weights
        )
        return float(score), bool(hard_safe)

    def curriculum_state(self, completion: float = 0.0, safety_failure: float = 0.0) -> dict[str, float]:
        coordination = self.window_coordination_sum / max(self.window_samples, 1)
        error_rmse = torch.sqrt(self.window_error_square_sum / max(self.window_samples, 1))
        coordination_score, coordination_hard_safe = self._coordination_curriculum_state(coordination)
        return {
            "stage": float(self.stage),
            "qualifying_windows": float(self.qualifying_windows),
            "completion_rate": completion,
            "safety_failure_rate": safety_failure,
            "reach_soft_violation_rate": float(coordination[0]),
            "reach_over_068_rate": float(coordination[5]),
            "reach_over_070_rate": float(coordination[6]),
            "knee_spacing_violation_rate": float(coordination[7]),
            "leg_mirror_rms": float(torch.sqrt(coordination[1])),
            "wheel_spacing_violation_rate": float(coordination[2]),
            "vertical_cancellation_rate": float(coordination[3]),
            "wheel_height_violation_rate": float(coordination[4]),
            "reach_resample_rate": self.window_reach_rejections / max(self.window_reach_attempts, 1),
            "symmetry_qualification": float(torch.sqrt(coordination[1]) < 0.08),
            "allocation_qualification": float(coordination[3] < 0.02),
            "coordination_score": coordination_score,
            "coordination_hard_safe": float(coordination_hard_safe),
            "yaw_rmse": float(error_rmse[0]),
            "ee_vx_rmse": float(error_rmse[1]),
            "ee_vz_rmse": float(error_rmse[2]),
            "up3_tilt_mean": float(self.window_error_sum[3] / max(self.window_samples, 1)),
            "up3_tilt_over_stage_limit_rate": float(coordination[8]),
            "tilt_p95_qualification": float(coordination[8] <= 0.05),
            "command_yaw_limit": float(self.STAGE_LIMITS[self.stage][0]),
            "command_vx_limit": float(self.STAGE_LIMITS[self.stage][1]),
            "command_vz_limit": float(self.STAGE_LIMITS[self.stage][2]),
        }

    def state_dict(self) -> dict:
        """Serialize the performance gate and command-filter state."""
        return {
            "stage": self.stage,
            "qualifying_windows": self.qualifying_windows,
            "last_evaluation_step": self.last_evaluation_step,
            "elapsed_since_last_evaluation": max(0, int(self._env.common_step_counter) - self.last_evaluation_step),
            "window_error_sum": self.window_error_sum.detach().cpu(),
            "window_error_square_sum": self.window_error_square_sum.detach().cpu(),
            "window_samples": self.window_samples,
            "window_resets": self.window_resets,
            "window_timeouts": self.window_timeouts,
            "window_coordination_sum": self.window_coordination_sum.detach().cpu(),
            "window_reach_samples": self.window_reach_samples,
            "window_reach_rejections": self.window_reach_rejections,
            "window_reach_attempts": self.window_reach_attempts,
            "alpha_base": self.alpha_base.detach().cpu(),
            "horizontal_alpha_base": self.horizontal_alpha_base.detach().cpu(),
            "command": self._command.detach().cpu(),
            "target_command": self.target_command.detach().cpu(),
            "command_acceleration": self.command_acceleration.detach().cpu(),
            "nominal_height": self.nominal_height.detach().cpu(),
            "filtered_base_vx_target": self.filtered_base_vx_target.detach().cpu(),
        }

    def load_state_dict(self, state):
        """Restore global curriculum state and compatible per-env buffers."""
        self.stage = int(state.get("stage", 0))
        self.qualifying_windows = int(state.get("qualifying_windows", 0))
        self.window_samples = int(state.get("window_samples", 0))
        elapsed = int(state.get("elapsed_since_last_evaluation", self.window_samples))
        self.last_evaluation_step = nandu1_math.resumed_evaluation_anchor(
            self._env.common_step_counter, elapsed, self.cfg.evaluation_steps
        )
        self.window_resets = int(state.get("window_resets", 0))
        self.window_timeouts = int(state.get("window_timeouts", 0))
        if "window_error_sum" in state and state["window_error_sum"].shape == self.window_error_sum.shape:
            self.window_error_sum.copy_(state["window_error_sum"].to(self.device))
        if (
            "window_error_square_sum" in state
            and state["window_error_square_sum"].shape == self.window_error_square_sum.shape
        ):
            self.window_error_square_sum.copy_(state["window_error_square_sum"].to(self.device))
        if (
            "window_coordination_sum" in state
            and state["window_coordination_sum"].shape == self.window_coordination_sum.shape
        ):
            self.window_coordination_sum.copy_(state["window_coordination_sum"].to(self.device))
        self.window_reach_samples = int(state.get("window_reach_samples", 0))
        self.window_reach_rejections = int(state.get("window_reach_rejections", 0))
        self.window_reach_attempts = int(state.get("window_reach_attempts", 0))
        if "alpha_base" in state and state["alpha_base"].shape == self.alpha_base.shape:
            self.alpha_base.copy_(state["alpha_base"].to(self.device))
        if (
            "horizontal_alpha_base" in state
            and state["horizontal_alpha_base"].shape == self.horizontal_alpha_base.shape
        ):
            self.horizontal_alpha_base.copy_(state["horizontal_alpha_base"].to(self.device))
        # A checkpoint does not restore physical episode states.  Keep the
        # fresh reset's per-env command and height buffers instead of copying
        # stale command/height values from unrelated old episodes.
        self.time_left.zero_()
        self.metrics["stage"].fill_(float(self.stage))

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError

    def _debug_vis_callback(self, event):
        raise NotImplementedError


Nandu1CommandCfg.class_type = Nandu1Command


class StageDomainRandomization(ManagerTermBase):
    """Apply stage-dependent mass, COM, material and actuator variation.

    Defaults are restored before every sample, so repeated resets cannot
    accumulate scale error.  The curriculum stage is owned by the command
    term and therefore also follows checkpoint resume.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._robot: Articulation = env.scene[asset_cfg.name]
        self._default_com = self._robot.root_physx_view.get_coms().clone()
        self._base_id = self._robot.find_bodies("base_Link")[0][0]
        self._excluded_body_ids = tuple(cfg.params.get("excluded_body_ids", ()))
        self.material_buckets_per_stage = int(cfg.params.get("material_buckets_per_stage", 48))
        self._material_buckets = nandu1_math.build_stage_material_buckets(self.material_buckets_per_stage)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        material_buckets_per_stage: int = 48,
        excluded_body_ids: tuple[int, ...] = (),
    ):
        del asset_cfg, material_buckets_per_stage, excluded_body_ids  # resolved during construction
        stage = int(env.command_manager.get_term(command_name).stage)
        if stage < 2:
            mass_range, gain_range, com_radius = (0.98, 1.02), (0.97, 1.03), 0.0
        elif stage == 2:
            mass_range, gain_range, com_radius = (0.95, 1.05), (0.95, 1.05), 0.005
        else:
            mass_range, gain_range, com_radius = (0.93, 1.07), (0.90, 1.10), 0.010

        cpu_ids = env_ids.to(device="cpu", dtype=torch.long)
        device_ids = env_ids.to(self._robot.device, dtype=torch.long)
        num_ids = len(cpu_ids)
        body_ids = torch.arange(self._robot.num_bodies, device="cpu")

        masses = self._robot.root_physx_view.get_masses()
        defaults = self._robot.data.default_mass[cpu_ids[:, None], body_ids].cpu()
        mass_scale = torch.empty_like(defaults).uniform_(*mass_range)
        masses[cpu_ids[:, None], body_ids] = defaults * mass_scale
        if self._excluded_body_ids:
            excluded = list(self._excluded_body_ids)
            masses[cpu_ids[:, None], excluded] = self._robot.data.default_mass[cpu_ids[:, None], excluded].cpu()
        self._robot.root_physx_view.set_masses(masses, cpu_ids)
        if not hasattr(env, "_nandu1_randomized_body_masses"):
            env._nandu1_randomized_body_masses = self._robot.data.default_mass.to(self._robot.device).clone()
        env._nandu1_randomized_body_masses[device_ids] = masses[cpu_ids].to(self._robot.device)

        inertias = self._robot.root_physx_view.get_inertias()
        inertias[cpu_ids[:, None], body_ids] = (
            self._robot.data.default_inertia[cpu_ids[:, None], body_ids].cpu() * mass_scale[..., None]
        )
        if self._excluded_body_ids:
            inertias[cpu_ids[:, None], excluded] = self._robot.data.default_inertia[cpu_ids[:, None], excluded].cpu()
        self._robot.root_physx_view.set_inertias(inertias, cpu_ids)

        coms = self._robot.root_physx_view.get_coms()
        coms[cpu_ids, self._base_id] = self._default_com[cpu_ids, self._base_id]
        if com_radius > 0.0 and self._base_id not in self._excluded_body_ids:
            coms[cpu_ids, self._base_id, :3] += torch.empty(num_ids, 3).uniform_(-com_radius, com_radius)
        self._robot.root_physx_view.set_coms(coms, cpu_ids)

        materials = self._robot.root_physx_view.get_material_properties()
        shape_count = materials.shape[1]
        materials[cpu_ids] = nandu1_math.sample_stage_material_buckets(
            self._material_buckets,
            stage=stage,
            num_envs=num_ids,
            num_shapes=shape_count,
        )
        self._robot.root_physx_view.set_material_properties(materials, cpu_ids)

        for actuator in self._robot.actuators.values():
            global_ids = actuator.joint_indices
            if isinstance(global_ids, slice):
                global_ids = torch.arange(self._robot.num_joints, device=self._robot.device)
            elif not isinstance(global_ids, torch.Tensor):
                global_ids = torch.as_tensor(global_ids, device=self._robot.device)
            else:
                global_ids = global_ids.to(self._robot.device)
            scale = torch.empty(len(device_ids), len(global_ids), device=self._robot.device).uniform_(*gain_range)
            stiffness = self._robot.data.default_joint_stiffness[device_ids][:, global_ids] * scale
            damping = self._robot.data.default_joint_damping[device_ids][:, global_ids] * scale
            actuator.stiffness[device_ids] = stiffness
            actuator.damping[device_ids] = damping
            self._robot.write_joint_stiffness_to_sim(stiffness, joint_ids=global_ids, env_ids=device_ids)
            self._robot.write_joint_damping_to_sim(damping, joint_ids=global_ids, env_ids=device_ids)


def stage_gated_push(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Enable external push disturbances only in curriculum stages 3 and 4."""
    if int(env.command_manager.get_term(command_name).stage) < 3:
        return
    robot: Articulation = env.scene[asset_cfg.name]
    velocity = robot.data.root_vel_w[env_ids].clone()
    ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=robot.device,
    )
    velocity += ranges[:, 0] + torch.rand_like(velocity) * (ranges[:, 1] - ranges[:, 0])
    robot.write_root_velocity_to_sim(velocity, env_ids=env_ids)


def performance_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str) -> dict[str, float]:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    term.record_resets(env_ids, env.termination_manager.time_outs)
    return term.evaluate_curriculum(env.common_step_counter)


def _command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def yaw_rate_tracking(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    error = robot.data.root_ang_vel_w[:, 2] - _command(env, command_name)[:, 0]
    return torch.exp(-torch.square(error / std))


def ee_velocity_tracking(
    env: ManagerBasedRLEnv, command_name: str, axis: int, command_index: int, std: float
) -> torch.Tensor:
    actual = body_heading_linear_velocity(env, SceneEntityCfg("robot", body_names="up3"))[:, axis]
    error = actual - _command(env, command_name)[:, command_index]
    return torch.exp(-torch.square(error / std))


def up3_horizontal(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    sensor: Imu = env.scene.sensors["up3_imu"]
    tilt = torch.acos(torch.clamp(-sensor.data.projected_gravity_b[:, 2], -1.0, 1.0))
    return torch.exp(-torch.square(tilt / std))


def arm3_height_envelope(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Training-only penalty on arm3's world height (the up3 link origin)."""
    robot: Articulation = env.scene["robot"]
    up3_id = robot.find_bodies("up3")[0][0]
    return nandu1_math.height_envelope_penalty(robot.data.body_pos_w[:, up3_id, 2])


def horizontal_reach_barrier(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    return nandu1_math.horizontal_reach_penalty(term.horizontal_reach)


def vertical_allocation_tracking(
    env: ManagerBasedRLEnv, command_name: str, base_std: float, arm_std: float
) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    error = torch.square((term.actual_base_vz - term.target_base_vz) / base_std)
    error += torch.square((term.actual_arm_vz - term.target_arm_vz) / arm_std)
    return term.tracking_gate * torch.exp(-error)


def opposite_vertical_cancellation(env: ManagerBasedRLEnv, command_name: str, scale: float) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    direction = torch.sign(term.command[:, 2])
    base_wrong_way = torch.relu(-direction * term.actual_base_vz)
    arm_wrong_way = torch.relu(-direction * term.actual_arm_vz)
    active = nandu1_math.vertical_command_gate(term.command[:, 2])
    return active * (torch.square(base_wrong_way / scale) + torch.square(arm_wrong_way / scale))


def height_directional_reserve(env: ManagerBasedRLEnv, command_name: str, minimum: float) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    base = torch.square(torch.relu(minimum - term.base_directional_margin) / minimum)
    arm = torch.square(torch.relu(minimum - term.arm_directional_margin) / minimum)
    return nandu1_math.vertical_command_gate(term.command[:, 2]) * (base + arm)


def com_to_wheel_support(env: ManagerBasedRLEnv, command_name: str, scale: float) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    return torch.square(term.com_support_offset / scale)


def _wheel_stance_penalties(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[asset_cfg.name]
    wheel_delta_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]] - robot.data.body_pos_w[:, asset_cfg.body_ids[1]]
    wheel_delta_h = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), wheel_delta_w)
    return nandu1_math.wheel_stance_penalties(wheel_delta_h)


def wheel_spacing_envelope(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _wheel_stance_penalties(env, asset_cfg)[0]


def wheel_fore_aft_alignment(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _wheel_stance_penalties(env, asset_cfg)[1]


def wheel_spacing_centering(env: ManagerBasedRLEnv, command_name: str, target: float, scale: float) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    return torch.square((term.wheel_spacing - target) / scale)


def wheel_height_difference(env: ManagerBasedRLEnv, command_name: str, scale: float) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    return torch.square(term.wheel_height_difference / scale)


def knee_spacing_envelope(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    return nandu1_math.knee_spacing_penalties(term.knee_delta_h)[0]


def knee_spacing_centering(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    return nandu1_math.knee_spacing_penalties(term.knee_delta_h)[1]


def base_roll_stability(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    sensor: Imu = env.scene.sensors["base_imu"]
    gravity = sensor.data.projected_gravity_b
    roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
    return torch.exp(-torch.square(roll / std))


def base_pitch_soft_envelope(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    sensor: Imu = env.scene.sensors["base_imu"]
    gravity = sensor.data.projected_gravity_b
    pitch = torch.atan2(-gravity[:, 0], -gravity[:, 2]).abs()
    term: Nandu1Command = env.command_manager.get_term(command_name)
    normalized_acceleration = torch.clamp(term.command_acceleration[:, 1].abs() / 0.8, 0.0, 1.0)
    smooth = normalized_acceleration * normalized_acceleration * (3.0 - 2.0 * normalized_acceleration)
    free_pitch = math.radians(6.0) + math.radians(8.0) * smooth
    return torch.square(torch.relu(pitch - free_pitch))


def up3_roll_pitch_angular_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    sensor: Imu = env.scene.sensors["up3_imu"]
    return torch.sum(torch.square(sensor.data.ang_vel_b[:, :2]), dim=1)


def horizontal_allocation_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    base_std: float,
    arm_std: float,
    tracking_std: float,
) -> torch.Tensor:
    """Track the reach-aware low/high-frequency split without replacing EE tracking."""
    term: Nandu1Command = env.command_manager.get_term(command_name)
    base_error = torch.square((term.actual_base_vx - term.target_base_vx) / base_std)
    arm_error = torch.square((term.actual_arm_vx - term.target_arm_vx) / arm_std)
    ee_error = term.actual_base_vx + term.actual_arm_vx - term.command[:, 1]
    tracking_gate = torch.exp(-torch.square(ee_error / tracking_std))
    active = nandu1_math.vertical_command_gate(term.command[:, 1], deadband=0.02, full_scale=0.08)
    return active * tracking_gate * torch.exp(-(base_error + arm_error))


def outward_reach_velocity(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Discourage configuration-margin loss without penalizing arm retraction."""
    term: Nandu1Command = env.command_manager.get_term(command_name)
    return nandu1_math.outward_reach_velocity_penalty(
        term.horizontal_reach, term.horizontal_reach_rate
    )


def arm_default_deviation_gated(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    deviation = torch.sum(
        torch.square(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    command = _command(env, command_name)
    up3_velocity = body_heading_linear_velocity(env, SceneEntityCfg("robot", body_names="up3"))
    tracking_error = torch.abs(up3_velocity[:, 0] - command[:, 1]) + torch.abs(up3_velocity[:, 2] - command[:, 2])
    return deviation * (tracking_error < 0.12)


def wheel_lateral_slip(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    velocity_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids]
    velocity_heading = _heading_velocity(
        robot.data.root_quat_w[:, None, :].expand(-1, len(asset_cfg.body_ids), -1).reshape(-1, 4),
        velocity_w.reshape(-1, 3),
    ).reshape(velocity_w.shape)
    contact = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > 1.0
    return torch.sum(torch.square(velocity_heading[:, :, 1]) * contact, dim=1)


def wheel_load_imbalance(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    loads = torch.clamp(sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2], min=0.0)
    return torch.square(loads[:, 0] - loads[:, 1]) / torch.square(loads.sum(dim=1).clamp_min(1.0))


def normalized_torque_utilization(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    utilization = robot.data.applied_torque.abs() / robot.data.joint_effort_limits.clamp_min(1.0e-6)
    return torch.mean(torch.square(utilization), dim=1)


def mechanical_power(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1)


def _task_aware_leg_residual(term: Nandu1Command, residual: torch.Tensor) -> torch.Tensor:
    yaw_relaxation = 1.0 + 1.5 * term.command[:, 0].abs() / term.STAGE_LIMITS[term.stage][0]
    yaw_relaxation += 0.5 * term.command_acceleration[:, 0].abs() / term.STAGE_ACCELERATION_LIMITS[term.stage][0]
    adjusted = residual.clone()
    adjusted[:, 1:3] /= yaw_relaxation[:, None]
    return adjusted


def leg_mirror_position(env: ManagerBasedRLEnv, command_name: str, beta: float) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    residual = _task_aware_leg_residual(term, term.leg_position_residual)
    penalty = torch.nn.functional.smooth_l1_loss(
        residual, torch.zeros_like(residual), beta=beta, reduction="none"
    ).sum(dim=1)
    return term.symmetry_gate * penalty


def leg_mirror_velocity(env: ManagerBasedRLEnv, command_name: str, beta: float) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    residual = _task_aware_leg_residual(term, term.leg_velocity_residual)
    return torch.nn.functional.smooth_l1_loss(
        residual, torch.zeros_like(residual), beta=beta, reduction="none"
    ).sum(dim=1)


def gross_leg_asymmetry(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    term: Nandu1Command = env.command_manager.get_term(command_name)
    yaw_relaxation = 1.0 + 1.5 * term.command[:, 0].abs() / term.STAGE_LIMITS[term.stage][0]
    yaw_relaxation += 0.5 * term.command_acceleration[:, 0].abs() / term.STAGE_ACCELERATION_LIMITS[term.stage][0]
    threshold = term.leg_position_residual.new_tensor((0.18, 0.12, 0.12, 0.18)).expand_as(
        term.leg_position_residual
    ).clone()
    threshold[:, 1:3] *= yaw_relaxation[:, None]
    excess = torch.relu(term.leg_position_residual.abs() - threshold) / threshold
    return torch.sum(torch.square(excess), dim=1)
