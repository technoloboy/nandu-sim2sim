"""NANDU1 v1.3 tracking-only curriculum layered on the v1.2 task semantics."""

from __future__ import annotations

from collections.abc import Sequence

from . import nandu1_v1_1_mdp as v11
from . import nandu1_v1_3_math as tracking_math


COMMAND_NAME = "nandu1_height_v1_3"


class Nandu1TrackingOnlyHeightCommand(v11.Nandu1CoordinatedHeightCommand):
    """Keep all diagnostics while using only tracking to advance the curriculum."""

    def _normalized_scores(self, completion: float) -> tuple[float, float, float, float, float]:
        tracking, stability, coordination, coverage, _ = super()._normalized_scores(completion)
        promotion = tracking_math.tracking_only_promotion_score(
            tracking=self.score_tracking.new_tensor(tracking),
            stability=self.score_stability.new_tensor(stability),
            coordination=self.score_coordination.new_tensor(coordination),
            coverage=self.score_coverage.new_tensor(coverage),
        )
        return tracking, stability, coordination, coverage, float(promotion)

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
        safety_limit = self.STAGE_SAFETY_THRESHOLDS[min(self.stage, 3)]
        self.safety_veto = tracking_math.minimal_safety_veto(
            non_finite=non_finite,
            catastrophic_rate=catastrophic_rate,
            forbidden_rate=forbidden_rate,
            forbidden_limit=safety_limit,
        )

        if self.stage < 4:
            if non_finite:
                self.curriculum_progress.zero_()
            else:
                self.curriculum_progress.copy_(
                    tracking_math.tracking_progress_step(
                        self.curriculum_progress,
                        tracking_ema=self.score_tracking_ema,
                        threshold=self.STAGE_SCORE_THRESHOLDS[self.stage],
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
        result.pop("coordination_progress_floor", None)
        result["promotion_score"] = float(self.score_tracking)
        result["promotion_score_ema"] = float(self.score_tracking_ema)
        return result


def performance_curriculum(env, env_ids: Sequence[int], command_name: str) -> dict[str, float]:
    term: Nandu1TrackingOnlyHeightCommand = env.command_manager.get_term(command_name)
    term.record_resets(env_ids, env.termination_manager.time_outs)
    term.record_safety_resets(env_ids, env.termination_manager)
    return term.evaluate_curriculum(env.common_step_counter)
