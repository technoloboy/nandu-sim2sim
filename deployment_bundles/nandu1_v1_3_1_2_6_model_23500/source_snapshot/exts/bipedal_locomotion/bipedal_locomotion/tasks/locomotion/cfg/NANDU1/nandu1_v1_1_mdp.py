"""NANDU1 v1-1 coordination shaping layered on the immutable v1 task."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from . import nandu1_math
from . import nandu1_v1_1_math as coordination_math
from . import nandu1_v1_mdp as v1


COMMAND_NAME = "nandu1_height_v1_1"


class Nandu1CoordinatedHeightCommand(v1.Nandu1HeightCommand):
    """v1 coordination metrics with direct full-range stage commands."""

    def __init__(self, cfg: v1.Nandu1HeightCommandCfg, env):
        super().__init__(cfg, env)
        self.coordination_task_gate = torch.zeros(self.num_envs, device=self.device)
        self.continuous_cancellation_cost = torch.zeros(self.num_envs, device=self.device)
        self.continuous_cancellation_occupancy = torch.zeros(self.num_envs, device=self.device)
        self.leg_geometry_shaping_cost = torch.zeros(self.num_envs, device=self.device)
        self.leg_geometry_hard_cost = torch.zeros(self.num_envs, device=self.device)
        self.leg_geometry_outward_cost = torch.zeros(self.num_envs, device=self.device)
        self.wheel_geometry_violation = torch.zeros(self.num_envs, device=self.device)
        self.knee_geometry_violation = torch.zeros(self.num_envs, device=self.device)
        self.reach_shaping_cost = torch.zeros(self.num_envs, device=self.device)
        self.reach_hard_cost = torch.zeros(self.num_envs, device=self.device)
        self.reach_outward_cost = torch.zeros(self.num_envs, device=self.device)
        self.reach_curriculum_measure = torch.zeros(self.num_envs, device=self.device)
        self.wheel_spacing_rate = torch.zeros(self.num_envs, device=self.device)
        self.knee_spacing_rate = torch.zeros(self.num_envs, device=self.device)
        self._previous_wheel_spacing = self.wheel_spacing.clone()
        self._previous_knee_spacing = self.knee_spacing.clone()
        self._spacing_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.window_v11_coordination_sum = torch.zeros(6, device=self.device)
        self.metrics.update(
            {
                "Coordination/task_gate": torch.zeros(self.num_envs, device=self.device),
                "Coordination/cancellation_cost": torch.zeros(self.num_envs, device=self.device),
                "Coordination/leg_geometry_cost": torch.zeros(self.num_envs, device=self.device),
                "Coordination/reach_cost": torch.zeros(self.num_envs, device=self.device),
            }
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if getattr(self, "_external_control", False):
            return
        ids = v1._ids(env_ids, self.num_envs, self.device)
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

    def _update_coordination_state(self):
        super()._update_coordination_state()
        dt = self._env.step_dt
        valid = self._spacing_history_valid
        self.wheel_spacing_rate[:] = torch.where(
            valid, (self.wheel_spacing - self._previous_wheel_spacing) / dt, 0.0
        )
        self.knee_spacing_rate[:] = torch.where(
            valid, (self.knee_spacing - self._previous_knee_spacing) / dt, 0.0
        )
        self._previous_wheel_spacing.copy_(self.wheel_spacing)
        self._previous_knee_spacing.copy_(self.knee_spacing)
        self._spacing_history_valid.fill_(True)

        robot: Articulation = self._robot
        up3_velocity_b = v1._body_velocity(
            robot.data.root_quat_w, self._up3_velocity_w()
        )
        height_error = self._command[:, 2] - self._up3_position_w()[:, 2]
        yaw_error = robot.data.root_ang_vel_w[:, 2] - self._command[:, 0]
        vx_error = up3_velocity_b[:, 0] - self._command[:, 1]
        self.coordination_task_gate[:] = coordination_math.task_quality_gate(
            yaw_error, vx_error, height_error
        )
        cancellation = coordination_math.continuous_vertical_cancellation(
            self.actual_base_vz,
            self.actual_arm_vz,
            height_error,
            self.commanded_height_rate,
        )
        self.continuous_cancellation_cost.copy_(cancellation[0])
        self.continuous_cancellation_occupancy.copy_(cancellation[1])

        yaw_fraction = (self._command[:, 0].abs() / self.STAGE_LIMITS[self.stage][0]).clamp(0.0, 1.0)
        mirror_error = torch.sqrt(torch.mean(torch.square(self.leg_position_residual), dim=1))
        leg = coordination_math.leg_geometry_cost(
            self.wheel_spacing,
            self.knee_spacing,
            mirror_error,
            self.wheel_spacing_rate,
            self.knee_spacing_rate,
            yaw_fraction,
        )
        self.leg_geometry_shaping_cost.copy_(leg.shaping)
        self.leg_geometry_hard_cost.copy_(leg.hard)
        self.leg_geometry_outward_cost.copy_(leg.outward)
        self.wheel_geometry_violation.copy_(leg.wheel_violation)
        self.knee_geometry_violation.copy_(leg.knee_violation)

        reach = coordination_math.aligned_reach_cost(
            self.horizontal_reach, self.horizontal_reach_rate,
            soft_limit=self.REACH_SOFT_LIMIT, hard_limit=self.REACH_HARD_LIMIT,
        )
        self.reach_shaping_cost.copy_(reach.shaping)
        self.reach_hard_cost.copy_(reach.hard)
        self.reach_outward_cost.copy_(reach.outward)
        self.reach_curriculum_measure.copy_(reach.curriculum_measure)

    def _update_metrics(self):
        super()._update_metrics()
        scale = 1.0 / float(self._env.max_episode_length)
        self.metrics["Coordination/task_gate"] += self.coordination_task_gate * scale
        self.metrics["Coordination/cancellation_cost"] += self.continuous_cancellation_cost * scale
        self.metrics["Coordination/leg_geometry_cost"] += (
            self.leg_geometry_shaping_cost + 4.0 * self.leg_geometry_hard_cost
        ) * scale
        self.metrics["Coordination/reach_cost"] += (
            self.reach_shaping_cost + 4.0 * self.reach_hard_cost
        ) * scale
        mirror_rms = torch.sqrt(torch.mean(torch.square(self.leg_position_residual), dim=1))
        values = torch.stack(
            (
                self.continuous_cancellation_occupancy.mean(),
                mirror_rms.mean(),
                self.wheel_geometry_violation.mean(),
                self.knee_geometry_violation.mean(),
                self.reach_curriculum_measure.mean(),
                self.com_support_offset.abs().mean(),
            )
        )
        self.window_v11_coordination_sum += values

    def _normalized_scores(self, completion: float) -> tuple[float, float, float, float, float]:
        tracking, stability, _, coverage, _ = super()._normalized_scores(completion)
        stage = min(self.stage, 3)
        raw = self.window_v11_coordination_sum / max(self.window_samples, 1)
        good_bad = {
            "cancellation": ((0.08, 0.35), (0.06, 0.25), (0.04, 0.18), (0.03, 0.12)),
            "mirror": ((0.05, 0.15), (0.045, 0.12), (0.04, 0.10), (0.035, 0.08)),
            "wheel_spacing": ((0.10, 0.50), (0.08, 0.35), (0.05, 0.20), (0.03, 0.10)),
            "knee_spacing": ((0.03, 0.15), (0.02, 0.10), (0.015, 0.07), (0.01, 0.04)),
        }

        def lower(name: str, value: torch.Tensor) -> float:
            good, bad = good_bad[name][stage]
            return float(nandu1_math.lower_is_better_score(value, good, bad))

        normalized = self.normalized_metrics
        normalized["cancellation"] = lower("cancellation", raw[0])
        normalized["mirror"] = lower("mirror", raw[1])
        normalized["wheel_spacing"] = lower("wheel_spacing", raw[2])
        normalized["knee_spacing"] = lower("knee_spacing", raw[3])
        normalized["reach"] = float(nandu1_math.lower_is_better_score(raw[4], 0.01, 0.10))
        normalized["com"] = float(nandu1_math.lower_is_better_score(raw[5], 0.03, 0.12))
        coordination = 0.30 * normalized["cancellation"]
        coordination += 0.20 * normalized["mirror"]
        coordination += 0.15 * normalized["wheel_spacing"]
        coordination += 0.15 * normalized["knee_spacing"]
        coordination += 0.10 * normalized["reach"]
        coordination += 0.10 * normalized["com"]
        total = coordination_math.curriculum_total_score(
            tracking, stability, coordination, coverage
        )
        return tracking, stability, coordination, coverage, total

    def evaluate_curriculum(self, common_step_counter: int) -> dict[str, float]:
        if common_step_counter - self.last_evaluation_step < self.cfg.evaluation_steps or self.window_samples == 0:
            return self.curriculum_state()
        completion = self.window_timeouts / max(self.window_resets, 1)
        safety_failure = 1.0 - completion
        scores = self._normalized_scores(completion)
        for destination, value in zip(
            (self.score_tracking, self.score_stability, self.score_coordination, self.score_coverage, self.score_total),
            scores,
        ):
            destination.fill_(value)
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
            if non_finite:
                self.curriculum_progress.zero_()
            else:
                self.curriculum_progress.copy_(
                    coordination_math.curriculum_progress_step(
                        self.curriculum_progress,
                        self.score_ema,
                        threshold=self.STAGE_SCORE_THRESHOLDS[self.stage],
                        coordination_ema=self.score_coordination_ema,
                        safety_veto=self.safety_veto,
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
        self.window_v11_coordination_sum.zero_()
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
        result = super().curriculum_state(completion, safety_failure)
        result["coordination_progress_floor"] = 0.35
        return result

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, float]:
        result = super().reset(env_ids)
        ids = v1._ids(env_ids, self.num_envs, self.device)
        self._update_coordination_state()
        self._previous_wheel_spacing[ids] = self.wheel_spacing[ids]
        self._previous_knee_spacing[ids] = self.knee_spacing[ids]
        self.wheel_spacing_rate[ids] = 0.0
        self.knee_spacing_rate[ids] = 0.0
        self._spacing_history_valid[ids] = True
        return result

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update(
            {
                "window_v11_coordination_sum": self.window_v11_coordination_sum.detach().cpu(),
                "previous_wheel_spacing": self._previous_wheel_spacing.detach().cpu(),
                "previous_knee_spacing": self._previous_knee_spacing.detach().cpu(),
            }
        )
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for state_name, destination_name in (
            ("previous_wheel_spacing", "_previous_wheel_spacing"),
            ("previous_knee_spacing", "_previous_knee_spacing"),
        ):
            if state_name in state:
                value = torch.as_tensor(state[state_name], device=self.device)
                destination = getattr(self, destination_name)
                if value.shape == destination.shape:
                    destination.copy_(value)


def _term(env, command_name: str) -> Nandu1CoordinatedHeightCommand:
    return env.command_manager.get_term(command_name)


def continuous_vertical_cancellation(env, command_name: str) -> torch.Tensor:
    term = _term(env, command_name)
    return (0.30 + 0.70 * term.coordination_task_gate) * term.continuous_cancellation_cost


def leg_geometry_shaping(env, command_name: str) -> torch.Tensor:
    term = _term(env, command_name)
    return (0.30 + 0.70 * term.coordination_task_gate) * term.leg_geometry_shaping_cost


def leg_geometry_hard_barrier(env, command_name: str) -> torch.Tensor:
    term = _term(env, command_name)
    return 4.0 * term.leg_geometry_hard_cost + term.leg_geometry_outward_cost


def reach_coordination_shaping(env, command_name: str) -> torch.Tensor:
    term = _term(env, command_name)
    return (0.30 + 0.70 * term.coordination_task_gate) * term.reach_shaping_cost


def reach_outward_barrier(env, command_name: str) -> torch.Tensor:
    term = _term(env, command_name)
    return 4.0 * term.reach_hard_cost + term.reach_outward_cost


def performance_curriculum(env, env_ids: Sequence[int], command_name: str) -> dict[str, float]:
    term: Nandu1CoordinatedHeightCommand = env.command_manager.get_term(command_name)
    term.record_resets(env_ids, env.termination_manager.time_outs)
    term.record_safety_resets(env_ids, env.termination_manager)
    return term.evaluate_curriculum(env.common_step_counter)
