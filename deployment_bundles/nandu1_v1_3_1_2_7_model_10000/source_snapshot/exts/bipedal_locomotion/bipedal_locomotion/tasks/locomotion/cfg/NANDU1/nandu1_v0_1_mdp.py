"""Adaptive coordination terms for the isolated NANDU1 v0-1 experiment."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import nandu1_mdp, nandu1_v0_1_math


@configclass
class Nandu1V01CommandCfg(nandu1_mdp.Nandu1CommandCfg):
    """v0 command configuration plus online reward-weight adaptation."""

    adaptive_constraints_enabled: bool = True


class Nandu1V01Command(nandu1_mdp.Nandu1Command):
    """Move coordination objectives into PPO rewards while retaining safety caps."""

    CONSTRAINT_NAMES = (
        "reach",
        "mirror",
        "wheel_spacing",
        "cancellation",
        "wheel_height",
        "knee_spacing",
    )
    MULTIPLIER_METRIC_NAMES = (
        "AdaptiveConstraint/reach_multiplier",
        "AdaptiveConstraint/mirror_multiplier",
        "AdaptiveConstraint/wheel_spacing_multiplier",
        "AdaptiveConstraint/cancellation_multiplier",
        "AdaptiveConstraint/wheel_height_multiplier",
        "AdaptiveConstraint/knee_spacing_multiplier",
    )
    # Order follows nandu1_v0_1_math.coordination_metrics().  These are broad
    # failure ceilings, not desired operating points and not a quality score.
    STAGE_CATASTROPHIC_COORDINATION_LIMITS = (
        (0.35, 0.24, 0.85, 0.65, 0.20, 0.30, 0.20, 0.50),
        (0.32, 0.23, 0.78, 0.60, 0.18, 0.28, 0.18, 0.45),
        (0.30, 0.22, 0.70, 0.55, 0.15, 0.25, 0.15, 0.40),
        (0.25, 0.20, 0.60, 0.50, 0.12, 0.20, 0.12, 0.35),
    )

    def __init__(self, cfg: Nandu1V01CommandCfg, env):
        super().__init__(cfg, env)
        self.adaptive_constraint_multipliers = torch.ones(6, device=self.device)
        self.adaptive_constraint_ema = torch.zeros(6, device=self.device)
        self._last_catastrophic_coordination_safe = True
        for name, metric_name in zip(self.CONSTRAINT_NAMES, self.MULTIPLIER_METRIC_NAMES):
            self.metrics[metric_name] = torch.zeros(
                self.num_envs, device=self.device
            )
            self.metrics[f"AdaptiveConstraint/{name}_ema"] = torch.zeros(
                self.num_envs, device=self.device
            )

    def constraint_multiplier(self, name: str) -> torch.Tensor:
        return self.adaptive_constraint_multipliers[self.CONSTRAINT_NAMES.index(name)]

    def _adaptive_targets_and_hard_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        stage = min(self.stage, 3)
        targets = self.adaptive_constraint_multipliers.new_tensor(
            (
                self.STAGE_REACH_SOFT_THRESHOLDS[stage],
                0.08,
                self.STAGE_WHEEL_SPACING_THRESHOLDS[stage],
                self.STAGE_CANCELLATION_THRESHOLDS[stage],
                0.05,
                self.STAGE_KNEE_SPACING_THRESHOLDS[stage],
            )
        )
        base_hard = self.STAGE_COORDINATION_HARD_LIMITS[stage]
        hard_limits = self.adaptive_constraint_multipliers.new_tensor(
            (base_hard[0], base_hard[1], base_hard[2], base_hard[3], base_hard[4], base_hard[7])
        )
        return targets, hard_limits

    def _adaptive_family_metrics(self, coordination: torch.Tensor) -> torch.Tensor:
        stage = min(self.stage, 3)
        base_hard = coordination.new_tensor(self.STAGE_COORDINATION_HARD_LIMITS[stage])
        reach_targets = coordination.new_tensor(
            (
                self.STAGE_REACH_SOFT_THRESHOLDS[stage],
                self.STAGE_REACH_068_THRESHOLDS[stage],
                self.STAGE_REACH_070_THRESHOLDS[stage],
            )
        )
        reach_values = torch.stack((coordination[0], coordination[5], coordination[6]))
        reach_hard = base_hard[[0, 5, 6]]
        reach_signal = torch.max(
            nandu1_v0_1_math.normalized_constraint_signal(
                reach_values, reach_targets, reach_hard
            )
        )
        return torch.stack(
            (
                reach_signal,
                torch.sqrt(coordination[1].clamp_min(0.0)),
                coordination[2],
                coordination[3],
                coordination[4],
                coordination[7],
            )
        )

    def _update_adaptive_constraints(self, coordination: torch.Tensor) -> None:
        if not self.cfg.adaptive_constraints_enabled:
            return
        metrics = self._adaptive_family_metrics(coordination)
        targets, hard_limits = self._adaptive_targets_and_hard_limits()
        # Reach is already expressed as a normalized worst-case signal.
        targets = targets.clone()
        hard_limits = hard_limits.clone()
        targets[0] = 0.0
        hard_limits[0] = 1.0
        for index in range(len(self.CONSTRAINT_NAMES)):
            multiplier, ema = nandu1_v0_1_math.update_adaptive_multiplier(
                self.adaptive_constraint_multipliers[index],
                self.adaptive_constraint_ema[index],
                metric=metrics[index],
                target=targets[index],
                hard_limit=hard_limits[index],
            )
            self.adaptive_constraint_multipliers[index] = multiplier
            self.adaptive_constraint_ema[index] = ema

    def _update_metrics(self):
        super()._update_metrics()
        metric_scale = 1.0 / float(self._env.max_episode_length)
        for index, name in enumerate(self.CONSTRAINT_NAMES):
            self.metrics[self.MULTIPLIER_METRIC_NAMES[index]] += (
                self.adaptive_constraint_multipliers[index] * metric_scale
            )
            self.metrics[f"AdaptiveConstraint/{name}_ema"] += (
                self.adaptive_constraint_ema[index] * metric_scale
            )

    def evaluate_curriculum(self, common_step_counter: int) -> dict[str, float]:
        if common_step_counter - self.last_evaluation_step < self.cfg.evaluation_steps or self.window_samples == 0:
            return self.curriculum_state()
        error_rmse = torch.sqrt(self.window_error_square_sum / self.window_samples)
        coordination = self.window_coordination_sum / self.window_samples
        completion = self.window_timeouts / max(self.window_resets, 1)
        safety_failure = 1.0 - completion
        self._update_adaptive_constraints(coordination)
        stage = min(self.stage, 3)
        catastrophic_limits = self.STAGE_CATASTROPHIC_COORDINATION_LIMITS[stage]
        self._last_catastrophic_coordination_safe = bool(
            nandu1_v0_1_math.catastrophic_coordination_safe(coordination, catastrophic_limits)
        )
        if self.stage < 4:
            qualified = bool(
                nandu1_v0_1_math.v01_curriculum_window_qualifies(
                    error_rmse=error_rmse,
                    coordination=coordination,
                    tracking_limits=self.STAGE_TRACKING_RMSE_THRESHOLDS[self.stage],
                    completion=completion,
                    completion_minimum=self.STAGE_COMPLETION_THRESHOLDS[self.stage],
                    safety_failure=safety_failure,
                    safety_failure_maximum=self.STAGE_SAFETY_THRESHOLDS[self.stage],
                    catastrophic_limits=catastrophic_limits,
                )
            )
            self.qualifying_windows = self.qualifying_windows + 1 if qualified else 0
            if self.qualifying_windows >= 3:
                self.stage += 1
                self.qualifying_windows = 0
        self.last_evaluation_step = common_step_counter
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

    def curriculum_state(self, completion: float = 0.0, safety_failure: float = 0.0) -> dict[str, float]:
        state = super().curriculum_state(completion, safety_failure)
        coordination = self.window_coordination_sum / max(self.window_samples, 1)
        limits = self.STAGE_CATASTROPHIC_COORDINATION_LIMITS[min(self.stage, 3)]
        catastrophic_safe = bool(
            nandu1_v0_1_math.catastrophic_coordination_safe(coordination, limits)
        )
        state["catastrophic_coordination_safe"] = float(catastrophic_safe)
        for index, name in enumerate(self.CONSTRAINT_NAMES):
            state[f"adaptive_{name}_multiplier"] = float(
                self.adaptive_constraint_multipliers[index]
            )
            state[f"adaptive_{name}_ema"] = float(self.adaptive_constraint_ema[index])
        return state

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["adaptive_constraint_multipliers"] = (
            self.adaptive_constraint_multipliers.detach().cpu()
        )
        state["adaptive_constraint_ema"] = self.adaptive_constraint_ema.detach().cpu()
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        multipliers = state.get("adaptive_constraint_multipliers")
        ema = state.get("adaptive_constraint_ema")
        if multipliers is not None and multipliers.shape == self.adaptive_constraint_multipliers.shape:
            self.adaptive_constraint_multipliers.copy_(multipliers.to(self.device))
        if ema is not None and ema.shape == self.adaptive_constraint_ema.shape:
            self.adaptive_constraint_ema.copy_(ema.to(self.device))


Nandu1V01CommandCfg.class_type = Nandu1V01Command


def _term(env, command_name: str) -> Nandu1V01Command:
    return env.command_manager.get_term(command_name)


def adaptive_horizontal_reach_barrier(env, command_name: str) -> torch.Tensor:
    return nandu1_mdp.horizontal_reach_barrier(env, command_name) * _term(
        env, command_name
    ).constraint_multiplier("reach")


def adaptive_outward_reach_velocity(env, command_name: str) -> torch.Tensor:
    return nandu1_mdp.outward_reach_velocity(env, command_name) * _term(
        env, command_name
    ).constraint_multiplier("reach")


def adaptive_leg_mirror_position(env, command_name: str, beta: float) -> torch.Tensor:
    return nandu1_mdp.leg_mirror_position(env, command_name, beta) * _term(
        env, command_name
    ).constraint_multiplier("mirror")


def adaptive_gross_leg_asymmetry(env, command_name: str) -> torch.Tensor:
    return nandu1_mdp.gross_leg_asymmetry(env, command_name) * _term(
        env, command_name
    ).constraint_multiplier("mirror")


def adaptive_wheel_spacing_envelope(
    env, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    return nandu1_mdp.wheel_spacing_envelope(env, asset_cfg) * _term(
        env, command_name
    ).constraint_multiplier("wheel_spacing")


def adaptive_opposite_vertical_cancellation(
    env, command_name: str, scale: float
) -> torch.Tensor:
    return nandu1_mdp.opposite_vertical_cancellation(env, command_name, scale) * _term(
        env, command_name
    ).constraint_multiplier("cancellation")


def adaptive_wheel_height_difference(env, command_name: str, scale: float) -> torch.Tensor:
    return nandu1_mdp.wheel_height_difference(env, command_name, scale) * _term(
        env, command_name
    ).constraint_multiplier("wheel_height")


def adaptive_knee_spacing_envelope(env, command_name: str) -> torch.Tensor:
    return nandu1_mdp.knee_spacing_envelope(env, command_name) * _term(
        env, command_name
    ).constraint_multiplier("knee_spacing")
