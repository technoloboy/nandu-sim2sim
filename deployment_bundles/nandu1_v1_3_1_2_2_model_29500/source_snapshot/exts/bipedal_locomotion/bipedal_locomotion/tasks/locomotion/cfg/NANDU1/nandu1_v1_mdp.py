"""NANDU1 v1 absolute-height command, rewards and normalized curriculum.

The module is deliberately additive: v0 keeps its velocity-command semantics,
while v1 owns all state whose meaning changed (height commands, body-frame
velocity targets, wrench curriculum and normalized curriculum progress).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, Imu
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from . import nandu1_math
from . import nandu1_mdp as v0
from . import nandu1_v1_robust_mdp as robust
from . import nandu1_v1_statistics as statistics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


COMMAND_NAME = "nandu1_height"


def _ids(env_ids: Sequence[int] | slice | None, count: int, device: str) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(count, device=device)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def _body_velocity(root_quat_w: torch.Tensor, velocity_w: torch.Tensor) -> torch.Tensor:
    return quat_apply_inverse(root_quat_w, velocity_w)


def base_body_linear_velocity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return _body_velocity(robot.data.root_quat_w, robot.data.root_lin_vel_w)


def body_linear_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if body_ids is None or isinstance(body_ids, slice):
        body_ids = robot.find_bodies(asset_cfg.body_names)[0]
    velocity_w = robot.data.body_lin_vel_w[:, body_ids[0]]
    return _body_velocity(robot.data.root_quat_w, velocity_w)


def true_up3_height(env: ManagerBasedEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    up3_id = robot.find_bodies("up3")[0][0]
    return robot.data.body_pos_w[:, up3_id, 2:3]


def _rpy_z_matrix(yaw: float, reference: torch.Tensor) -> torch.Tensor:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return reference.new_tensor(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _axis_rotation(axis: tuple[float, float, float], angle: torch.Tensor) -> torch.Tensor:
    axis_t = angle.new_tensor(axis)
    axis_t = axis_t / torch.linalg.norm(axis_t)
    x, y, z = axis_t.unbind()
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    identity = torch.eye(3, device=angle.device, dtype=angle.dtype)
    outer = axis_t[:, None] * axis_t[None, :]
    c = torch.cos(angle)[:, None, None]
    s = torch.sin(angle)[:, None, None]
    return c * identity + (1.0 - c) * outer + s * skew


def _advance_joint(
    rotation: torch.Tensor,
    position: torch.Tensor,
    origin: tuple[float, float, float],
    origin_yaw: float,
    axis: tuple[float, float, float],
    angle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = angle.new_tensor(origin)
    position = position + torch.matmul(rotation, offset)
    fixed_rotation = _rpy_z_matrix(origin_yaw, angle)
    rotation = torch.matmul(torch.matmul(rotation, fixed_rotation), _axis_rotation(axis, angle))
    return rotation, position


def _branch_position(
    q: torch.Tensor,
    joints: tuple[tuple[tuple[float, float, float], float, tuple[float, float, float]], ...],
    terminal_origin: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    rotation = torch.eye(3, device=q.device, dtype=q.dtype).expand(len(q), 3, 3).clone()
    position = torch.zeros(len(q), 3, device=q.device, dtype=q.dtype)
    for index, (origin, origin_yaw, axis) in enumerate(joints):
        rotation, position = _advance_joint(rotation, position, origin, origin_yaw, axis, q[:, index])
    if terminal_origin is not None:
        position = position + torch.matmul(rotation, q.new_tensor(terminal_origin))
    return position


_LEFT_LEG_CHAIN = (
    ((-0.092607, 0.10508, 0.078903), -1.57078898038469, (0.99619, 0.0, 0.087156)),
    ((-0.063116, -0.042475, -0.0055219), 0.0, (0.0, 1.0, 0.0)),
    ((0.0, 0.042475, -0.1155), 0.0, (0.0, 0.0, 1.0)),
    ((-0.026791, 0.004, -0.2345), 0.0, (1.0, 0.0, 0.0)),
)
_RIGHT_LEG_CHAIN = (
    ((-0.092607, -0.10406, 0.078871), -1.57078898038469, (-0.99619, 0.0, 0.087156)),
    ((0.062749, -0.042475, -0.0054898), 0.0, (0.0, 1.0, 0.0)),
    ((0.0, 0.042475, -0.1152), 0.0, (0.0, 0.0, -1.0)),
    ((0.026791, 0.004, -0.2348), 0.0, (1.0, 0.0, 0.0)),
)
_ARM_CHAIN = (
    ((0.12997, -0.057104, 0.34442), -1.57078898038469, (1.0, 0.0, 0.0)),
    ((-0.00075, 0.22981, 0.22981), 0.0, (1.0, 0.0, 0.0)),
    ((0.001, 0.23688, 0.23688), 0.0, (1.0, 0.0, 0.0)),
)


def kinematic_up3_height(env: ManagerBasedEnv, terminal_origin=None) -> torch.Tensor:
    """Deployable flat-ground height estimate from encoders, IMU and wheel radius."""
    robot: Articulation = env.scene["robot"]
    command_name = getattr(env, "_nandu_v1_robust_command_name", None)
    if command_name is not None:
        measured_joint_pos, measured_gravity = robust.kinematic_measurements(env, command_name)
    else:
        measured_joint_pos = robot.data.joint_pos
        measured_gravity = env.scene.sensors["base_imu"].data.projected_gravity_b
    left_ids = robot.find_joints(
        ["proximal_pitch_L_Joint", "proximal_roll_L_Joint", "proximal_yaw_L_Joint", "knee_L_Joint"],
        preserve_order=True,
    )[0]
    right_ids = robot.find_joints(
        ["proximal_pitch_R_Joint", "proximal_roll_R_Joint", "proximal_yaw_R_Joint", "knee_R_Joint"],
        preserve_order=True,
    )[0]
    arm_ids = robot.find_joints(["arm1", "arm2", "arm3"], preserve_order=True)[0]
    left_wheel = _branch_position(
        measured_joint_pos[:, left_ids], _LEFT_LEG_CHAIN, (-0.037909, -0.004, -0.35)
    )
    right_wheel = _branch_position(
        measured_joint_pos[:, right_ids], _RIGHT_LEG_CHAIN, (0.037909, -0.004, -0.35)
    )
    up3 = _branch_position(measured_joint_pos[:, arm_ids], _ARM_CHAIN, terminal_origin)
    world_up_b = -measured_gravity
    world_up_b = world_up_b / torch.linalg.norm(world_up_b, dim=1, keepdim=True).clamp_min(1.0e-6)
    wheel_center = 0.5 * (left_wheel + right_wheel)
    height = 0.10 + torch.sum((up3 - wheel_center) * world_up_b, dim=1)
    return height.unsqueeze(-1)


@configclass
class Nandu1HeightCommandCfg(v0.Nandu1CommandCfg):
    @configclass
    class Ranges:
        yaw_rate: tuple[float, float] = MISSING
        ee_vx: tuple[float, float] = MISSING
        up3_height: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    height_rate_limits: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50)
    height_step_probabilities: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20)
    height_step_limits: tuple[float, ...] = (0.0, 0.05, 0.08, 0.12, 0.18)
    robustness_enabled: bool = True
    measurement_command_name: str = COMMAND_NAME
    checkpoint_schema: str = "nandu1_height_v1_robust_v2"


class Nandu1HeightCommand(v0.Nandu1Command):
    """Yaw/body-x velocity plus direct absolute-height commands."""

    cfg: Nandu1HeightCommandCfg
    STAGE_LIMITS = ((0.20, 0.08), (0.50, 0.20), (0.85, 0.40), (1.20, 0.70), (1.57, 1.00))
    STAGE_HEIGHT_RANGES = ((0.78, 1.04), (0.68, 1.12), (0.58, 1.22), (0.48, 1.32), (0.40, 1.40))
    STAGE_BASE_HEIGHT_RANGES = v0.Nandu1Command.STAGE_BASE_HEIGHT_RANGES
    STAGE_ARM_HEIGHT_RANGES = ((0.12, 0.35), (0.02, 0.50), (-0.05, 0.65), (-0.10, 0.85), (-0.20, 1.05))
    STAGE_VREF_LIMITS = (0.08, 0.15, 0.25, 0.38, 0.50)
    STAGE_SCORE_THRESHOLDS = (0.65, 0.70, 0.75, 0.80)
    STAGE_SAFETY_THRESHOLDS = (0.10, 0.08, 0.06, 0.05)
    SCORE_GOOD_BAD = {
        "yaw": ((0.12, 0.30), (0.14, 0.34), (0.16, 0.38), (0.18, 0.42)),
        "vx": ((0.06, 0.16), (0.08, 0.22), (0.12, 0.30), (0.16, 0.40)),
        # Promotion thresholds deliberately relax with the expanding height range,
        # command slew rate, disturbances and domain randomization.  Final policy
        # acceptance remains a separate full-range Stage-4 evaluation.
        "height_rmse": ((0.05, 0.14), (0.06, 0.16), (0.075, 0.19), (0.09, 0.22)),
        "height_p95": ((0.10, 0.24), (0.12, 0.28), (0.15, 0.32), (0.18, 0.38)),
        "completion": ((0.80, 0.50), (0.85, 0.55), (0.90, 0.60), (0.93, 0.65)),
        "tilt": tuple((math.radians(g), math.radians(b)) for g, b in ((8, 20), (7, 18), (6, 16), (5, 14))),
        "settled_vz": ((0.04, 0.14), (0.035, 0.12), (0.03, 0.10), (0.025, 0.08)),
        "static_drift": ((0.12, 0.35), (0.10, 0.30), (0.08, 0.25), (0.07, 0.20)),
    }

    def __init__(self, cfg: Nandu1HeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        env._nandu_v1_robust_command_name = cfg.measurement_command_name
        self._wheel_contact_ids = robust.robust_math.wheel_sensor_indices(
            env.scene.sensors["contact_forces"].body_names
        )
        self._evaluation_stage: int | None = None
        self._external_control = False
        self.sampled_height_target = torch.ones(self.num_envs, device=self.device)
        self.commanded_height_rate = torch.zeros(self.num_envs, device=self.device)
        self.height_rate_limited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pending_height_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.height_step_segment = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.height_reference_velocity = torch.zeros(self.num_envs, device=self.device)
        self.height_low_frequency_velocity = torch.zeros(self.num_envs, device=self.device)
        self.height_high_frequency_velocity = torch.zeros(self.num_envs, device=self.device)
        self.static_gate = torch.zeros(self.num_envs, device=self.device)
        self.curriculum_progress = torch.tensor(0.0, device=self.device)
        self.score_tracking = torch.tensor(0.0, device=self.device)
        self.score_stability = torch.tensor(0.0, device=self.device)
        self.score_coordination = torch.tensor(0.0, device=self.device)
        self.score_coverage = torch.tensor(0.0, device=self.device)
        self.score_tracking_ema = torch.tensor(0.0, device=self.device)
        self.score_stability_ema = torch.tensor(0.0, device=self.device)
        self.score_coordination_ema = torch.tensor(0.0, device=self.device)
        self.score_coverage_ema = torch.tensor(0.0, device=self.device)
        self.score_total = torch.tensor(0.0, device=self.device)
        self.score_ema = torch.tensor(0.0, device=self.device)
        self.safety_veto = False
        self.window_non_finite = 0
        self.window_forbidden_contact = 0
        self.window_catastrophic_tilt = 0
        self.disturbance_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.disturbance_recovery_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.disturbance_recovery_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.disturbance_recovery_stable = torch.zeros(self.num_envs, device=self.device)
        self.window_recovery_time_sum = 0.0
        self.window_recovery_count = 0
        self.normalized_metrics: dict[str, float] = {}
        self.height_error_histogram = statistics.HeightErrorHistogram(self.device)
        self.last_height_p95 = 0.0
        self.last_completion_rate = 0.0
        self.last_safety_failure_rate = 0.0
        self.statistics_audit_valid = False
        self.window_settled_vz_square_sum = 0.0
        self.window_settled_samples = 0
        self.window_static_drift_square_sum = 0.0
        self.window_static_samples = 0
        self.window_torque_high_sum = 0.0
        self.window_coverage = torch.zeros(5, 10, device=self.device)
        self._coverage_stride = 0
        self.metrics = {
            "stage": torch.zeros(self.num_envs, device=self.device),
            "yaw_abs_error": torch.zeros(self.num_envs, device=self.device),
            "ee_vx_abs_error": torch.zeros(self.num_envs, device=self.device),
            "height_abs_error": torch.zeros(self.num_envs, device=self.device),
            "up3_tilt": torch.zeros(self.num_envs, device=self.device),
            "Command/sample_height_target": torch.zeros(self.num_envs, device=self.device),
            "Command/applied_height": torch.zeros(self.num_envs, device=self.device),
            "Command/height_rate": torch.zeros(self.num_envs, device=self.device),
            "Command/is_step": torch.zeros(self.num_envs, device=self.device),
            "Command/rate_limit_active": torch.zeros(self.num_envs, device=self.device),
            "Height/kinematic_estimate": torch.zeros(self.num_envs, device=self.device),
            "Height/sim_true": torch.zeros(self.num_envs, device=self.device),
            "Height/estimate_error": torch.zeros(self.num_envs, device=self.device),
            "Allocation/vref": torch.zeros(self.num_envs, device=self.device),
            "Allocation/vlow": torch.zeros(self.num_envs, device=self.device),
            "Allocation/vhigh": torch.zeros(self.num_envs, device=self.device),
            "Allocation/alpha_base": torch.zeros(self.num_envs, device=self.device),
            "Static/gate": torch.zeros(self.num_envs, device=self.device),
        }
        robust.ensure_robust_metrics(self)

    @property
    def active_stage(self) -> int:
        """Curriculum stage used by the command path during training or Play."""
        evaluation_stage = getattr(self, "_evaluation_stage", None)
        return self.stage if evaluation_stage is None else evaluation_stage

    def set_evaluation_stage(self, stage: int) -> None:
        """Enable evaluation-only command limits without mutating learned curriculum state."""
        stage = int(stage)
        if stage not in range(len(self.STAGE_LIMITS)):
            raise ValueError("NANDU1 v1 evaluation stage must be 0..4")
        self._evaluation_stage = stage

    def set_external_command(self, command: torch.Tensor) -> None:
        """Apply a Play command while retaining the environment's height-rate limiter.

        Yaw and vx are expected to have already passed the UI watchdog and physical
        acceleration filter.  They are latched at the action boundary.  Height is
        an absolute target and continues through the task's stage-specific slew
        limiter, matching the deployment command semantics.
        """
        if self._evaluation_stage is None:
            raise RuntimeError("external NANDU1 commands require set_evaluation_stage()")
        value = torch.as_tensor(command, device=self.device, dtype=self._command.dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape not in ((1, 3), (self.num_envs, 3)):
            raise ValueError(f"external command must have shape (3,), (1,3), or ({self.num_envs},3)")
        if not torch.isfinite(value).all():
            raise ValueError("external command must be finite")
        if value.shape[0] == 1 and self.num_envs != 1:
            value = value.expand(self.num_envs, -1)
        stage = self.active_stage
        yaw_limit, vx_limit = self.STAGE_LIMITS[stage]
        height_low, height_high = self.STAGE_HEIGHT_RANGES[stage]
        clamped = value.clone()
        clamped[:, 0].clamp_(-yaw_limit, yaw_limit)
        clamped[:, 1].clamp_(-vx_limit, vx_limit)
        clamped[:, 2].clamp_(height_low, height_high)
        self._external_control = True
        self.time_left.fill_(1.0e9)
        self.target_command.copy_(clamped)
        self.sampled_height_target.copy_(clamped[:, 2])
        # Motion axes are already acceleration limited by the UI process.  Latch
        # them directly so the policy and command observation see the same sample.
        self._command[:, :2].copy_(clamped[:, :2])

    def disable_external_control(self) -> None:
        """Return the term to its training sampler (not used by normal Play)."""
        self._external_control = False
        self._evaluation_stage = None
        self.time_left.zero_()

    def _resample_command(self, env_ids: Sequence[int]):
        if getattr(self, "_external_control", False):
            return
        ids = _ids(env_ids, self.num_envs, self.device)
        stage = self.active_stage
        yaw_limit, vx_limit = self.STAGE_LIMITS[stage]
        height_low, height_high = self.STAGE_HEIGHT_RANGES[stage]
        sampled = nandu1_math.sample_uniform_stage_commands(
            len(ids),
            yaw_limit=yaw_limit,
            vx_limit=vx_limit,
            height_low=height_low,
            height_high=height_high,
            device=self.device,
            dtype=self._command.dtype,
        )
        self.target_command[ids] = sampled
        self.sampled_height_target[ids] = sampled[:, 2]
        self.height_step_segment[ids] = False
        self.pending_height_step[ids] = False

    def _update_command(self):
        previous = self._command.clone()
        self._command.copy_(self.target_command)
        delta = (self._command - previous) / self._env.step_dt
        self.commanded_height_rate.copy_(delta[:, 2])
        self.height_rate_limited.zero_()
        self.pending_height_step.zero_()
        self.command_acceleration.copy_(delta)
        alpha = self._env.step_dt / (0.8 + self._env.step_dt)
        self.filtered_base_vx_target += alpha * (self._command[:, 1] - self.filtered_base_vx_target)

    def _kinematic_up3_height(self):
        return kinematic_up3_height(self._env)

    def _update_coordination_state(self):
        super()._update_coordination_state()
        robot = self._robot
        up3_position = self._up3_position_w()
        base_point_velocity_w = self._base_point_velocity_w(up3_position)
        relative_velocity_w = self._up3_velocity_w() - base_point_velocity_w
        base_point_velocity_b = _body_velocity(robot.data.root_quat_w, base_point_velocity_w)
        relative_velocity_b = _body_velocity(robot.data.root_quat_w, relative_velocity_w)
        self.actual_base_vx[:] = base_point_velocity_b[:, 0]
        self.actual_arm_vx[:] = relative_velocity_b[:, 0]

        height = up3_position[:, 2]
        height_error = self._command[:, 2] - height
        vref = 1.5 * height_error - 0.25 * (self._up3_velocity_w()[:, 2] - self.commanded_height_rate)
        self.height_reference_velocity[:] = torch.clamp(
            vref, -self.STAGE_VREF_LIMITS[self.stage], self.STAGE_VREF_LIMITS[self.stage]
        )
        low_pass_gain = self._env.step_dt / (0.8 + self._env.step_dt)
        self.height_low_frequency_velocity += low_pass_gain * (
            self.height_reference_velocity - self.height_low_frequency_velocity
        )
        self.height_high_frequency_velocity[:] = (
            self.height_reference_velocity - self.height_low_frequency_velocity
        )

        torque = robot.data.applied_torque.abs() / robot.data.joint_effort_limits.clamp_min(1.0e-6)
        base_quality = 1.0 - torch.clamp(torque[:, self._leg_ids].amax(dim=1), 0.0, 1.0)
        arm_quality = 1.0 - torch.clamp(torque[:, self._arm_ids].amax(dim=1), 0.0, 1.0)
        com_quality = self._com_quality()
        load_quality = 1.0 - torch.clamp(self.wheel_load_difference, 0.0, 1.0)
        reach_quality = torch.clamp((self.REACH_HARD_LIMIT - self.horizontal_reach) / 0.10, 0.0, 1.0)
        base_range = self.STAGE_BASE_HEIGHT_RANGES[self.stage]
        arm_range = self.STAGE_ARM_HEIGHT_RANGES[self.stage]
        self.base_directional_margin[:] = nandu1_math.directional_margin(
            self.height_reference_velocity, self.base_height, base_range
        )
        self.arm_directional_margin[:] = nandu1_math.directional_margin(
            self.height_reference_velocity, self.arm_relative_height, arm_range
        )
        raw_alpha = nandu1_math.height_allocation_ratio(
            self.height_reference_velocity,
            self.base_height,
            self.arm_relative_height,
            base_range,
            arm_range,
            base_quality * com_quality * load_quality,
            arm_quality * reach_quality,
        )
        gain = self._env.step_dt / (0.30 + self._env.step_dt)
        active = self.height_reference_velocity.abs() > 0.02
        self.alpha_base[:] = torch.where(active, self.alpha_base + gain * (raw_alpha - self.alpha_base), self.alpha_base)
        self.target_base_vz[:] = self.alpha_base * self.height_low_frequency_velocity
        self.target_arm_vz[:] = self.height_high_frequency_velocity + (
            1.0 - self.alpha_base
        ) * self.height_low_frequency_velocity
        direction = torch.sign(self.height_reference_velocity)
        self.cancellation[:] = active & (
            (self.actual_base_vz * direction < 0.0) | (self.actual_arm_vz * direction < 0.0)
        )
        self.static_gate[:] = nandu1_math.static_hold_gate(
            self._command[:, 0], self._command[:, 1], height_error, self.commanded_height_rate
        )

    def _update_metrics(self):
        self._update_coordination_state()
        robot = self._robot
        up3_velocity_b = _body_velocity(robot.data.root_quat_w, self._up3_velocity_w())
        height = self._up3_position_w()[:, 2]
        height_estimate = self._kinematic_up3_height().squeeze(-1)
        up3_imu: Imu = self._env.scene.sensors["up3_imu"]
        tilt = torch.acos(torch.clamp(-up3_imu.data.projected_gravity_b[:, 2], -1.0, 1.0))
        errors = torch.stack(
            (
                torch.abs(robot.data.root_ang_vel_w[:, 2] - self._command[:, 0]),
                torch.abs(up3_velocity_b[:, 0] - self._command[:, 1]),
                torch.abs(height - self._command[:, 2]),
                tilt,
            ),
            dim=1,
        )
        self.tracking_gate[:] = torch.exp(
            -torch.square(errors[:, 2] / 0.10) - torch.square(errors[:, 3] / 0.15)
        )
        self.symmetry_gate[:] = nandu1_math.tracking_quality_gate(
            errors, (0.25, 0.14, 0.10, math.radians(15.0))
        )
        scale = 1.0 / float(self._env.max_episode_length)
        self.metrics["stage"][:] = float(self.stage)
        self.metrics["yaw_abs_error"] += errors[:, 0] * scale
        self.metrics["ee_vx_abs_error"] += errors[:, 1] * scale
        self.metrics["height_abs_error"] += errors[:, 2] * scale
        self.metrics["up3_tilt"] += errors[:, 3] * scale
        self.metrics["Command/sample_height_target"] += self.sampled_height_target * scale
        self.metrics["Command/applied_height"] += self._command[:, 2] * scale
        self.metrics["Command/height_rate"] += self.commanded_height_rate * scale
        self.metrics["Command/is_step"] += self.height_step_segment.float() * scale
        self.metrics["Command/rate_limit_active"] += self.height_rate_limited.float() * scale
        self.metrics["Height/kinematic_estimate"] += height_estimate * scale
        self.metrics["Height/sim_true"] += height * scale
        self.metrics["Height/estimate_error"] += torch.abs(height_estimate - height) * scale
        self.metrics["Allocation/vref"] += self.height_reference_velocity * scale
        self.metrics["Allocation/vlow"] += self.height_low_frequency_velocity * scale
        self.metrics["Allocation/vhigh"] += self.height_high_frequency_velocity * scale
        self.metrics["Allocation/alpha_base"] += self.alpha_base * scale
        self.metrics["Static/gate"] += self.static_gate * scale

        statistics_mask = getattr(self, "tracking_statistics_mask", None)
        statistics_errors = errors if statistics_mask is None else errors[statistics_mask]
        if statistics_errors.shape[0]:
            self.window_error_sum += statistics_errors.mean(dim=0)
            self.window_error_square_sum += torch.square(statistics_errors).mean(dim=0)
            self.window_samples += 1
            self.height_error_histogram.update(statistics_errors[:, 2])
        # Every environment at every statistics update, not a mean of slice P95s.
        self._coverage_stride += 1
        if self._coverage_stride % 10 == 0:
            low_h, high_h = self.STAGE_HEIGHT_RANGES[self.stage]
            values = (
                self._command[:, 0],
                self._command[:, 1],
                self._command[:, 2],
            )
            ranges = (
                (-self.STAGE_LIMITS[self.stage][0], self.STAGE_LIMITS[self.stage][0]),
                (-self.STAGE_LIMITS[self.stage][1], self.STAGE_LIMITS[self.stage][1]),
                (low_h, high_h),
            )
            for row, (value, limits) in enumerate(zip(values, ranges)):
                if statistics_mask is not None:
                    value = value[statistics_mask]
                normalized = ((value - limits[0]) / max(limits[1] - limits[0], 1.0e-6)).clamp(0.0, 0.9999)
                bins = torch.bincount((normalized * 10).long(), minlength=10).float()
                self.window_coverage[row] += bins
            boundary = (self._command[:, 2] < low_h + 0.1 * (high_h - low_h)) | (
                self._command[:, 2] > high_h - 0.1 * (high_h - low_h)
            )
            if statistics_mask is not None:
                boundary = boundary[statistics_mask]
            self.window_coverage[4, 0] += boundary.float().sum()

        settled = self.static_gate > 0.5
        if settled.any():
            self.window_settled_vz_square_sum += float(
                torch.square(self._up3_velocity_w()[settled, 2]).sum()
            )
            self.window_static_drift_square_sum += float(
                torch.square(robot.data.root_ang_vel_w[settled]).sum()
            )
            self.window_settled_samples += int(settled.sum())
            self.window_static_samples += int(settled.sum()) * 3
        utilization = robot.data.applied_torque.abs() / robot.data.joint_effort_limits.clamp_min(1.0e-6)
        self.window_torque_high_sum += float((utilization > 0.90).float().mean())

        recovering = self.disturbance_recovery_pending
        if recovering.any():
            self.disturbance_recovery_elapsed[recovering] += self._env.step_dt
            vertical_residual = torch.abs(
                self._up3_velocity_w()[:, 2] - self.commanded_height_rate
            )
            recovery_good = (
                (errors[:, 0] < 0.20)
                & (errors[:, 1] < 0.10)
                & (errors[:, 2] < 0.04)
                & (errors[:, 3] < math.radians(10.0))
                & (vertical_residual < 0.08)
            )
            self.disturbance_recovery_stable[recovering] = torch.where(
                recovery_good[recovering],
                self.disturbance_recovery_stable[recovering] + self._env.step_dt,
                0.0,
            )
            recovered = recovering & (self.disturbance_recovery_stable >= 0.20)
            if recovered.any():
                self.metrics["Disturbance/recovery_seconds"][recovered] += (
                    self.disturbance_recovery_elapsed[recovered] / self._env.max_episode_length
                )
                self.window_recovery_time_sum += float(self.disturbance_recovery_elapsed[recovered].sum())
                self.window_recovery_count += int(recovered.sum())
                self.disturbance_recovery_pending[recovered] = False
                self.disturbance_recovery_elapsed[recovered] = 0.0
                self.disturbance_recovery_stable[recovered] = 0.0
            timed_out = recovering & (self.disturbance_recovery_elapsed >= 2.5)
            if timed_out.any():
                self.metrics["Disturbance/recovery_seconds"][timed_out] += (
                    2.5 / self._env.max_episode_length
                )
                self.metrics["Disturbance/recovery_timeout"][timed_out] += (
                    1.0 / self._env.max_episode_length
                )
                self.window_recovery_time_sum += 2.5 * int(timed_out.sum())
                self.window_recovery_count += int(timed_out.sum())
                self.disturbance_recovery_pending[timed_out] = False
                self.disturbance_recovery_elapsed[timed_out] = 0.0
                self.disturbance_recovery_stable[timed_out] = 0.0

        yaw_relaxation = 1.0 + 1.5 * self._command[:, 0].abs() / self.STAGE_LIMITS[self.stage][0]
        curriculum_residual = self.leg_position_residual.clone()
        curriculum_residual[:, 1:3] /= yaw_relaxation[:, None]
        coordination = torch.stack(
            (
                (self.horizontal_reach > self.REACH_DIAGNOSTIC_LIMITS[0]).float().mean(),
                torch.mean(torch.square(curriculum_residual)),
                ((self.wheel_spacing < 0.42) | (self.wheel_spacing > 0.49)).float().mean(),
                self.cancellation.float().mean(),
                (self.wheel_height_difference > 0.03).float().mean(),
                (self.horizontal_reach > self.REACH_DIAGNOSTIC_LIMITS[1]).float().mean(),
                (self.horizontal_reach > self.REACH_DIAGNOSTIC_LIMITS[2]).float().mean(),
                ((self.knee_spacing < 0.35) | (self.knee_spacing > 0.44)).float().mean(),
                self.com_support_offset.mean(),
            )
        )
        self.window_coordination_sum += coordination
        self.window_reach_samples += self.num_envs
        robust.update_robust_metrics(self)

    @staticmethod
    def _coverage_score(counts: torch.Tensor) -> float:
        if counts.sum() <= 0:
            return 0.0
        return float((counts > 0).float().mean())

    def _normalized_scores(self, completion: float) -> tuple[float, float, float, float, float]:
        stage = min(self.stage, 3)
        rmse = torch.sqrt(self.window_error_square_sum / max(self.window_samples, 1))
        p95 = self.height_error_histogram.quantile()
        self.last_height_p95 = p95

        def lower(name: str, value: float) -> float:
            good, bad = self.SCORE_GOOD_BAD[name][stage]
            return float(nandu1_math.lower_is_better_score(torch.tensor(value), good, bad))

        def higher(name: str, value: float) -> float:
            good, bad = self.SCORE_GOOD_BAD[name][stage]
            return float(nandu1_math.higher_is_better_score(torch.tensor(value), bad, good))

        normalized = {
            "yaw_rmse": lower("yaw", float(rmse[0])),
            "vx_rmse": lower("vx", float(rmse[1])),
            "height_rmse": lower("height_rmse", float(rmse[2])),
            "height_p95": lower("height_p95", p95),
            "completion": higher("completion", completion),
            "tilt": lower("tilt", float(rmse[3])),
        }
        tracking = 0.20 * normalized["yaw_rmse"]
        tracking += 0.30 * normalized["vx_rmse"]
        tracking += 0.35 * normalized["height_rmse"]
        tracking += 0.15 * normalized["height_p95"]

        settled_vz = math.sqrt(self.window_settled_vz_square_sum / max(self.window_settled_samples, 1))
        static_drift = math.sqrt(self.window_static_drift_square_sum / max(self.window_static_samples, 1))
        normalized["settled_vz"] = lower("settled_vz", settled_vz)
        normalized["static_drift"] = lower("static_drift", static_drift)
        torque_rate = self.window_torque_high_sum / max(self.window_samples, 1)
        normalized["torque"] = float(
            nandu1_math.lower_is_better_score(torch.tensor(torque_rate), 0.0, 0.05)
        )
        if self.stage == 0:
            stability = (
                0.25 * normalized["completion"]
                + 0.20 * normalized["tilt"]
                + 0.15 * normalized["settled_vz"]
                + 0.15 * normalized["static_drift"]
                + 0.10 * normalized["torque"]
            ) / 0.85
            normalized["recovery"] = 1.0
        else:
            recovery_time = self.window_recovery_time_sum / max(self.window_recovery_count, 1)
            recovery_good = (0.8, 0.7, 0.6, 0.6)[min(self.stage - 1, 3)]
            recovery_bad = (2.0, 1.6, 1.2, 1.2)[min(self.stage - 1, 3)]
            normalized["recovery"] = float(
                nandu1_math.lower_is_better_score(torch.tensor(recovery_time), recovery_good, recovery_bad)
            ) if self.window_recovery_count > 0 else 0.0
            stability = 0.25 * normalized["completion"]
            stability += 0.20 * normalized["tilt"]
            stability += 0.15 * normalized["settled_vz"]
            stability += 0.15 * normalized["static_drift"]
            stability += 0.15 * normalized["recovery"]
            stability += 0.10 * normalized["torque"]

        coord = self.window_coordination_sum / max(self.window_samples, 1)
        normalized["cancellation"] = float(nandu1_math.lower_is_better_score(coord[3], (0.08, 0.06, 0.04, 0.03)[stage], (0.35, 0.25, 0.18, 0.12)[stage]))
        normalized["mirror"] = float(nandu1_math.lower_is_better_score(torch.sqrt(coord[1]), (0.05, 0.045, 0.04, 0.035)[stage], (0.15, 0.12, 0.10, 0.08)[stage]))
        normalized["wheel_spacing"] = float(nandu1_math.lower_is_better_score(coord[2], (0.10, 0.08, 0.05, 0.03)[stage], (0.50, 0.35, 0.20, 0.10)[stage]))
        normalized["knee_spacing"] = float(nandu1_math.lower_is_better_score(coord[7], (0.03, 0.02, 0.015, 0.01)[stage], (0.15, 0.10, 0.07, 0.04)[stage]))
        normalized["reach"] = float(nandu1_math.lower_is_better_score(coord[5], 0.01, 0.10))
        normalized["com"] = float(nandu1_math.lower_is_better_score(coord[8], 0.03, 0.12))
        coordination = 0.30 * normalized["cancellation"]
        coordination += 0.20 * normalized["mirror"]
        coordination += 0.15 * normalized["wheel_spacing"]
        coordination += 0.15 * normalized["knee_spacing"]
        coordination += 0.10 * normalized["reach"]
        coordination += 0.10 * normalized["com"]

        coverage_values = [self._coverage_score(self.window_coverage[index]) for index in range(3)]
        samples = max(float(self.window_coverage[:3].sum()), 1.0)
        boundary_coverage = min(float(self.window_coverage[4, 0]) / (0.05 * samples), 1.0)
        coverage_good, coverage_bad = ((0.65, 0.35), (0.70, 0.40), (0.75, 0.45), (0.80, 0.50))[stage]
        normalize_coverage = lambda value: float(
            nandu1_math.higher_is_better_score(torch.tensor(value), coverage_bad, coverage_good)
        )
        normalized["height_coverage"] = normalize_coverage(coverage_values[2])
        normalized["vx_coverage"] = normalize_coverage(coverage_values[1])
        normalized["yaw_coverage"] = normalize_coverage(coverage_values[0])
        normalized["boundary_coverage"] = normalize_coverage(boundary_coverage)
        coverage = statistics.sampling_coverage(
            normalized["height_coverage"], normalized["vx_coverage"],
            normalized["yaw_coverage"], normalized["boundary_coverage"],
        )

        self.normalized_metrics = normalized

        weighted = 0.50 * tracking + 0.25 * stability + 0.15 * coordination + 0.10 * coverage
        balance = 0.5 + 0.5 * min(tracking, stability, coordination, coverage)
        return tracking, stability, coordination, coverage, weighted * balance

    def evaluate_curriculum(self, common_step_counter: int) -> dict[str, float]:
        if common_step_counter - self.last_evaluation_step < self.cfg.evaluation_steps or self.window_samples == 0:
            return self.curriculum_state()
        completion = self.window_timeouts / max(self.window_resets, 1)
        safety_failure = 1.0 - completion
        scores = self._normalized_scores(completion)
        self.score_tracking.fill_(scores[0])
        self.score_stability.fill_(scores[1])
        self.score_coordination.fill_(scores[2])
        self.score_coverage.fill_(scores[3])
        self.score_total.fill_(scores[4])
        self.score_tracking_ema.mul_(0.7).add_(0.3 * self.score_tracking)
        self.score_stability_ema.mul_(0.7).add_(0.3 * self.score_stability)
        self.score_coordination_ema.mul_(0.7).add_(0.3 * self.score_coordination)
        self.score_coverage_ema.mul_(0.7).add_(0.3 * self.score_coverage)
        self.score_ema.mul_(0.7).add_(0.3 * self.score_total)
        non_finite = self.window_non_finite > 0
        catastrophic_rate = self.window_catastrophic_tilt / max(self.window_resets, 1)
        forbidden_rate = self.window_forbidden_contact / max(self.window_resets, 1)
        self.safety_veto = bool(
            safety_failure > self.STAGE_SAFETY_THRESHOLDS[min(self.stage, 3)]
            or catastrophic_rate > 0.02
            or forbidden_rate > self.STAGE_SAFETY_THRESHOLDS[min(self.stage, 3)]
            or non_finite
        )
        if self.stage < 4:
            if self.safety_veto:
                if non_finite:
                    self.curriculum_progress.zero_()
                else:
                    self.curriculum_progress.sub_(1.5).clamp_(0.0, 3.0)
            else:
                self.curriculum_progress.copy_(
                    nandu1_math.normalized_curriculum_progress(
                        self.curriculum_progress, self.score_ema, self.STAGE_SCORE_THRESHOLDS[self.stage]
                    )
                )
            if float(self.curriculum_progress) >= 3.0:
                self.stage += 1
                self.curriculum_progress.zero_()
                self.score_ema.zero_()
        self.last_evaluation_step = common_step_counter
        result = self.curriculum_state(completion, safety_failure)
        self.window_error_sum.zero_()
        self.window_error_square_sum.zero_()
        self.window_coordination_sum.zero_()
        self.window_coverage.zero_()
        self.window_samples = self.window_resets = self.window_timeouts = 0
        self.height_error_histogram.clear()
        self.window_settled_vz_square_sum = 0.0
        self.window_settled_samples = 0
        self.window_static_drift_square_sum = 0.0
        self.window_static_samples = 0
        self.window_torque_high_sum = 0.0
        self.window_recovery_time_sum = 0.0
        self.window_recovery_count = 0
        self.window_non_finite = 0
        self.window_forbidden_contact = 0
        self.window_catastrophic_tilt = 0
        return result

    def curriculum_state(self, completion: float | None = None, safety_failure: float | None = None) -> dict[str, float]:
        if completion is not None and safety_failure is not None:
            self.last_completion_rate = float(completion)
            self.last_safety_failure_rate = float(safety_failure)
            self.statistics_audit_valid = True
        result = {
            "stage": float(self.stage),
            "progress": float(self.curriculum_progress),
            "completion_rate": self.last_completion_rate,
            "safety_failure_rate": self.last_safety_failure_rate,
            "statistics_audit_valid": float(self.statistics_audit_valid),
            "height_p95_pooled_approx_m": self.last_height_p95,
            "score_tracking": float(self.score_tracking),
            "score_stability": float(self.score_stability),
            "score_coordination": float(self.score_coordination),
            "score_coverage": float(self.score_coverage),
            "score_tracking_ema": float(self.score_tracking_ema),
            "score_stability_ema": float(self.score_stability_ema),
            "score_coordination_ema": float(self.score_coordination_ema),
            "score_coverage_ema": float(self.score_coverage_ema),
            "score_total": float(self.score_total),
            "score_ema": float(self.score_ema),
            "safety_veto": float(self.safety_veto),
            "command_yaw_limit": float(self.STAGE_LIMITS[self.stage][0]),
            "command_vx_limit": float(self.STAGE_LIMITS[self.stage][1]),
            "command_height_low": float(self.STAGE_HEIGHT_RANGES[self.stage][0]),
            "command_height_high": float(self.STAGE_HEIGHT_RANGES[self.stage][1]),
        }
        result.update({f"metric_{name}_normalized": value for name, value in self.normalized_metrics.items()})
        return result

    def record_safety_resets(self, env_ids: Sequence[int], termination_manager) -> None:
        ids = _ids(env_ids, self.num_envs, self.device)
        self.window_non_finite += int(termination_manager.get_term("non_finite")[ids].sum().item())
        self.window_forbidden_contact += int(
            termination_manager.get_term("forbidden_contact")[ids].sum().item()
        )
        self.window_catastrophic_tilt += int(
            termination_manager.get_term("catastrophic_tilt")[ids].sum().item()
        )

    def update_disturbance_state(self, active: torch.Tensor) -> None:
        ended = self.disturbance_active & ~active
        started = ~self.disturbance_active & active
        self.disturbance_recovery_pending[ended] = True
        self.disturbance_recovery_pending[started] = False
        self.disturbance_recovery_elapsed[ended | started] = 0.0
        self.disturbance_recovery_stable[ended | started] = 0.0
        self.disturbance_active.copy_(active)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, float]:
        ids = _ids(env_ids, self.num_envs, self.device)
        self._update_coordination_state()
        height = self._kinematic_up3_height().squeeze(-1)[ids].clamp(*self.STAGE_HEIGHT_RANGES[self.stage])
        self._command[ids, :2] = 0.0
        self._command[ids, 2] = height
        self.target_command[ids] = self._command[ids]
        self.sampled_height_target[ids] = height
        self.commanded_height_rate[ids] = 0.0
        self.height_low_frequency_velocity[ids] = 0.0
        self.height_high_frequency_velocity[ids] = 0.0
        self.pending_height_step[ids] = False
        self.height_step_segment[ids] = False
        self.alpha_base[ids] = 0.5
        return CommandTerm.reset(self, env_ids)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(
            {
                "checkpoint_schema": self.cfg.checkpoint_schema,
                "curriculum_progress": self.curriculum_progress.detach().cpu(),
                "score_ema": self.score_ema.detach().cpu(),
                "score_tracking": self.score_tracking.detach().cpu(),
                "score_stability": self.score_stability.detach().cpu(),
                "score_coordination": self.score_coordination.detach().cpu(),
                "score_coverage": self.score_coverage.detach().cpu(),
                "score_total": self.score_total.detach().cpu(),
                "score_tracking_ema": self.score_tracking_ema.detach().cpu(),
                "score_stability_ema": self.score_stability_ema.detach().cpu(),
                "score_coordination_ema": self.score_coordination_ema.detach().cpu(),
                "score_coverage_ema": self.score_coverage_ema.detach().cpu(),
                "window_coverage": self.window_coverage.detach().cpu(),
                "statistics_schema": statistics.STATISTICS_SCHEMA,
                "statistics_window": statistics.window_state(self),
                "height_error_histogram": self.height_error_histogram.state_dict(),
                "last_completion_rate": self.last_completion_rate,
                "last_safety_failure_rate": self.last_safety_failure_rate,
                "statistics_audit_valid": self.statistics_audit_valid,
                "last_height_p95": self.last_height_p95,
                "coverage_stride": self._coverage_stride,
                "normalized_metrics": dict(self.normalized_metrics),
                "safety_veto": self.safety_veto,
                "sampled_height_target": self.sampled_height_target.detach().cpu(),
                "commanded_height_rate": self.commanded_height_rate.detach().cpu(),
                "height_step_segment": self.height_step_segment.detach().cpu(),
                "height_low_frequency_velocity": self.height_low_frequency_velocity.detach().cpu(),
                "height_high_frequency_velocity": self.height_high_frequency_velocity.detach().cpu(),
                "disturbance_active": self.disturbance_active.detach().cpu(),
                "disturbance_recovery_pending": self.disturbance_recovery_pending.detach().cpu(),
                "disturbance_recovery_elapsed": self.disturbance_recovery_elapsed.detach().cpu(),
                "disturbance_recovery_stable": self.disturbance_recovery_stable.detach().cpu(),
            }
        )
        return state

    def load_state_dict(self, state):
        if state.get("checkpoint_schema") != self.cfg.checkpoint_schema:
            raise ValueError(
                f"normal resume requires {self.cfg.checkpoint_schema}; use --transfer_checkpoint "
                "for a same-version legacy weight migration"
            )
        super().load_state_dict(state)
        for name in (
            "curriculum_progress",
            "score_ema",
            "score_tracking",
            "score_stability",
            "score_coordination",
            "score_coverage",
            "score_total",
            "score_tracking_ema",
            "score_stability_ema",
            "score_coordination_ema",
            "score_coverage_ema",
            "window_coverage",
            "sampled_height_target",
            "commanded_height_rate",
            "height_step_segment",
            "height_low_frequency_velocity",
            "height_high_frequency_velocity",
            "disturbance_active",
            "disturbance_recovery_pending",
            "disturbance_recovery_elapsed",
            "disturbance_recovery_stable",
        ):
            if name in state:
                value = torch.as_tensor(state[name], device=self.device)
                destination = getattr(self, name)
                if value.shape == destination.shape:
                    destination.copy_(value)
        schema = state.get("statistics_schema")
        if schema is None:
            # Old slice-P95/step-coverage counts cannot be converted to pooled
            # statistics. Preserve stage/weights/progress, start a fresh window.
            statistics.clear_window(self)
            self.last_evaluation_step = self._env.common_step_counter
            self.last_completion_rate = self.last_safety_failure_rate = self.last_height_p95 = 0.0
            self.statistics_audit_valid = False
            self.normalized_metrics = {}
        elif schema == statistics.STATISTICS_SCHEMA:
            statistics.restore_window(self, state["statistics_window"])
            self.height_error_histogram.load_state_dict(state["height_error_histogram"])
            for name in ("last_completion_rate", "last_safety_failure_rate", "last_height_p95"):
                setattr(self, name, float(state[name]))
            self.statistics_audit_valid = bool(state["statistics_audit_valid"])
            self._coverage_stride = int(state["coverage_stride"])
            self.normalized_metrics = dict(state.get("normalized_metrics", {}))
            self.safety_veto = bool(state.get("safety_veto", False))
        else:
            raise ValueError(f"unsupported curriculum statistics schema: {schema}")


def _command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def yaw_huber(env: ManagerBasedRLEnv, command_name: str, scale: float, cap: float) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return nandu1_math.normalized_huber(robot.data.root_ang_vel_w[:, 2] - _command(env, command_name)[:, 0], scale, cap)


def body_vx_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    actual = _body_velocity(term._robot.data.root_quat_w, term._up3_velocity_w())[:, 0]
    return torch.exp(-torch.square((actual - _command(env, command_name)[:, 1]) / std))


def body_vx_huber(env: ManagerBasedRLEnv, command_name: str, scale: float, cap: float) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    actual = _body_velocity(term._robot.data.root_quat_w, term._up3_velocity_w())[:, 0]
    return nandu1_math.normalized_huber(actual - _command(env, command_name)[:, 1], scale, cap)


def height_tracking_exp(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    term: Nandu1HeightCommand = env.command_manager.get_term(command_name)
    height = term._up3_position_w()[:, 2]
    sigma = (0.10, 0.08, 0.065, 0.050, 0.035)[term.stage]
    return torch.exp(-torch.square((height - term.command[:, 2]) / sigma))


def height_huber(env: ManagerBasedRLEnv, command_name: str, scale: float, cap: float) -> torch.Tensor:
    term: Nandu1HeightCommand = env.command_manager.get_term(command_name)
    height = term._up3_position_w()[:, 2]
    return nandu1_math.normalized_huber(height - term.command[:, 2], scale, cap)


def height_velocity_damping(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    term: Nandu1HeightCommand = env.command_manager.get_term(command_name)
    height = term._up3_position_w()[:, 2]
    gate = torch.exp(-torch.square((height - term.command[:, 2]) / 0.08))
    velocity_error = term._up3_velocity_w()[:, 2] - term.commanded_height_rate
    return gate * torch.clamp(torch.square(velocity_error / 0.12), 0.0, 4.0)


def up3_tilt_huber(env: ManagerBasedRLEnv) -> torch.Tensor:
    sensor: Imu = env.scene.sensors["up3_imu"]
    tilt = torch.acos(torch.clamp(-sensor.data.projected_gravity_b[:, 2], -1.0, 1.0))
    return nandu1_math.normalized_huber(tilt, 0.20, 6.0)


def vertical_allocation_tracking(
    env: ManagerBasedRLEnv, command_name: str, base_std: float, arm_std: float
) -> torch.Tensor:
    term: Nandu1HeightCommand = env.command_manager.get_term(command_name)
    error = torch.square((term.actual_base_vz - term.target_base_vz) / base_std)
    error += torch.square((term.actual_arm_vz - term.target_arm_vz) / arm_std)
    active = nandu1_math.vertical_command_gate(term.height_reference_velocity)
    return active * term.tracking_gate * torch.exp(-error)


def opposite_vertical_cancellation(env: ManagerBasedRLEnv, command_name: str, scale: float) -> torch.Tensor:
    term: Nandu1HeightCommand = env.command_manager.get_term(command_name)
    direction = torch.sign(term.height_reference_velocity)
    wrong = torch.relu(-direction * term.actual_base_vz) + torch.relu(-direction * term.actual_arm_vz)
    penalty = torch.clamp(torch.square(wrong / scale), 0.0, 4.0)
    return nandu1_math.vertical_command_gate(term.height_reference_velocity) * penalty


def height_directional_reserve(env: ManagerBasedRLEnv, command_name: str, minimum: float) -> torch.Tensor:
    term: Nandu1HeightCommand = env.command_manager.get_term(command_name)
    penalty = torch.square(torch.relu(minimum - term.base_directional_margin) / minimum)
    penalty += torch.square(torch.relu(minimum - term.arm_directional_margin) / minimum)
    return nandu1_math.vertical_command_gate(term.height_reference_velocity) * penalty


def height_envelope(env: ManagerBasedRLEnv, command_name=None) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    up3_id = robot.find_bodies("up3")[0][0]
    height = robot.data.body_pos_w[:, up3_id, 2] if command_name is None else env.command_manager.get_term(command_name)._up3_position_w()[:, 2]
    return nandu1_math.height_envelope_penalty(height, lower=0.40, upper=1.40, soft_margin=0.05)


def arm_default_deviation_gated(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    term: Nandu1HeightCommand = env.command_manager.get_term(command_name)
    deviation = torch.sum(
        torch.square(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    actual_vx = _body_velocity(robot.data.root_quat_w, term._up3_velocity_w())[:, 0]
    height = term._up3_position_w()[:, 2]
    tracking_error = torch.abs(actual_vx - term.command[:, 1]) + torch.abs(height - term.command[:, 2])
    return deviation * (tracking_error < 0.12)


def static_body_motion(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    term: Nandu1HeightCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene["robot"]
    base_velocity = base_body_linear_velocity(env)
    up3_velocity = _body_velocity(robot.data.root_quat_w, term._up3_velocity_w())
    wheel_ids = robot.find_joints(["wheel_L_Joint", "wheel_R_Joint"], preserve_order=True)[0]
    motion = torch.sum(torch.square(base_velocity / 0.10), dim=1)
    motion += torch.sum(torch.square(robot.data.root_ang_vel_b / 0.15), dim=1)
    motion += torch.sum(torch.square(up3_velocity / 0.10), dim=1)
    motion += torch.sum(torch.square(robot.data.joint_vel[:, wheel_ids] / 1.0), dim=1)
    return term.static_gate * torch.clamp(motion, 0.0, 20.0)


def static_orientation_hold(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    term: Nandu1HeightCommand = env.command_manager.get_term(command_name)
    base_gravity = env.scene.sensors["base_imu"].data.projected_gravity_b
    up3_gravity = env.scene.sensors["up3_imu"].data.projected_gravity_b
    base_roll = torch.atan2(base_gravity[:, 1], -base_gravity[:, 2])
    base_pitch = torch.atan2(-base_gravity[:, 0], -base_gravity[:, 2])
    up3_tilt = torch.acos(torch.clamp(-up3_gravity[:, 2], -1.0, 1.0))
    penalty = torch.square(base_roll / 0.08) + torch.square(base_pitch / 0.12) + torch.square(up3_tilt / 0.08)
    return term.static_gate * torch.clamp(penalty, 0.0, 20.0)


class Up3WrenchCurriculum(ManagerTermBase):
    """Apply short, stage-scaled up3 wrench pulses and explicitly clear them."""

    FORCE_LIMITS = (0.0, 8.0, 15.0, 25.0, 35.0)
    TORQUE_LIMITS = (0.0, 0.8, 1.5, 3.0, 5.0)

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene["robot"]
        self._up3_id = self._robot.find_bodies("up3")[0][0]
        self._time_to_start = torch.empty(env.num_envs, device=env.device).uniform_(8.0, 12.0)
        self._time_left = torch.zeros(env.num_envs, device=env.device)
        self._force_z = torch.zeros(env.num_envs, device=env.device)
        self._torque = torch.zeros(env.num_envs, 3, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str):
        ids = _ids(env_ids, env.num_envs, env.device)
        stage = int(env.command_manager.get_term(command_name).stage)
        self._time_to_start[ids] -= env.step_dt
        self._time_left[ids] -= env.step_dt
        starting = ids[(self._time_to_start[ids] <= 0.0) & (stage > 0)]
        if len(starting) > 0:
            force_limit = self.FORCE_LIMITS[stage]
            torque_limit = self.TORQUE_LIMITS[stage]
            mode = torch.rand(len(starting), device=env.device)
            self._force_z[starting] = torch.empty(len(starting), device=env.device).uniform_(-force_limit, force_limit)
            self._torque[starting] = torch.empty(len(starting), 3, device=env.device).uniform_(
                -torque_limit, torque_limit
            )
            self._force_z[starting[mode >= 0.60]] = 0.0
            self._torque[starting[mode < 0.40]] = 0.0
            self._time_left[starting] = torch.empty(len(starting), device=env.device).uniform_(0.15, 0.35)
            self._time_to_start[starting] = torch.empty(len(starting), device=env.device).uniform_(8.0, 12.0)
        inactive = self._time_left[ids] <= 0.0
        self._force_z[ids[inactive]] = 0.0
        self._torque[ids[inactive]] = 0.0
        force_world = torch.zeros(len(ids), 3, device=env.device)
        force_world[:, 2] = self._force_z[ids]
        up3_quat = self._robot.data.body_quat_w[ids, self._up3_id]
        force_body = quat_apply_inverse(up3_quat, force_world).unsqueeze(1)
        torque_body = self._torque[ids].unsqueeze(1)
        self._robot.set_external_force_and_torque(
            force_body, torque_body, body_ids=[self._up3_id], env_ids=ids
        )
        env._nandu1_v1_wrench_active = self._time_left > 0.0
        env._nandu1_v1_wrench_force_z = self._force_z
        env._nandu1_v1_wrench_torque = self._torque
        env.command_manager.get_term(command_name).update_disturbance_state(self._time_left > 0.0)

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        ids = _ids(env_ids, len(self._time_left), self._time_left.device)
        self._time_left[ids] = 0.0
        self._force_z[ids] = 0.0
        self._torque[ids] = 0.0
        self._time_to_start[ids] = torch.empty(len(ids), device=self._time_left.device).uniform_(8.0, 12.0)

    def state_dict(self) -> dict:
        return {
            "time_to_start": self._time_to_start.detach().cpu(),
            "time_left": self._time_left.detach().cpu(),
            "force_z": self._force_z.detach().cpu(),
            "torque": self._torque.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        for key, destination in (
            ("time_to_start", self._time_to_start),
            ("time_left", self._time_left),
            ("force_z", self._force_z),
            ("torque", self._torque),
        ):
            if key in state:
                value = torch.as_tensor(state[key], device=destination.device, dtype=destination.dtype)
                if value.shape == destination.shape:
                    destination.copy_(value)


def performance_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str) -> dict[str, float]:
    term: Nandu1HeightCommand = env.command_manager.get_term(command_name)
    term.record_resets(env_ids, env.termination_manager.time_outs)
    term.record_safety_resets(env_ids, env.termination_manager)
    return term.evaluate_curriculum(env.common_step_counter)
