"""Pure adaptive-constraint helpers for the NANDU1 v0-1 task."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def normalized_constraint_signal(
    metric: torch.Tensor,
    target: float | torch.Tensor,
    hard_limit: float | torch.Tensor,
) -> torch.Tensor:
    """Map a coordination metric to [-1, 2] around its target and hard limit."""
    target_tensor = torch.as_tensor(target, dtype=metric.dtype, device=metric.device)
    hard_tensor = torch.as_tensor(hard_limit, dtype=metric.dtype, device=metric.device)
    denominator = (hard_tensor - target_tensor).clamp_min(torch.finfo(metric.dtype).eps)
    return torch.clamp((metric - target_tensor) / denominator, -1.0, 2.0)


def update_adaptive_multiplier(
    multiplier: torch.Tensor,
    ema: torch.Tensor,
    *,
    metric: torch.Tensor,
    target: float | torch.Tensor,
    hard_limit: float | torch.Tensor,
    ema_decay: float = 0.80,
    step_size: float = 0.15,
    minimum: float = 0.50,
    maximum: float = 4.00,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update one reward multiplier from a smoothed normalized violation."""
    signal = normalized_constraint_signal(metric, target, hard_limit)
    new_ema = ema_decay * ema + (1.0 - ema_decay) * signal
    new_multiplier = torch.clamp(multiplier + step_size * new_ema, minimum, maximum)
    return new_multiplier, new_ema


def coordination_metrics(coordination: torch.Tensor) -> torch.Tensor:
    """Convert the v0 window accumulator into its eight human-scale metrics."""
    return torch.stack(
        (
            coordination[0],
            torch.sqrt(coordination[1].clamp_min(0.0)),
            coordination[2],
            coordination[3],
            coordination[4],
            coordination[5],
            coordination[6],
            coordination[7],
        )
    )


def catastrophic_coordination_safe(
    coordination: torch.Tensor,
    limits: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Return true only when every broad non-negotiable geometry cap is safe."""
    metrics = coordination_metrics(coordination)
    limit_tensor = torch.as_tensor(limits, dtype=metrics.dtype, device=metrics.device)
    return torch.all(metrics <= limit_tensor)


def v01_curriculum_window_qualifies(
    *,
    error_rmse: torch.Tensor,
    coordination: torch.Tensor,
    tracking_limits: Sequence[float],
    completion: float,
    completion_minimum: float,
    safety_failure: float,
    safety_failure_maximum: float,
    catastrophic_limits: Sequence[float],
) -> torch.Tensor:
    """v0-1 promotion rule: tracking/survival plus broad catastrophic guards."""
    limits = error_rmse.new_tensor(tracking_limits)
    tracking_safe = torch.all(error_rmse[:3] <= limits)
    tilt_safe = coordination[8] <= 0.05
    geometry_safe = catastrophic_coordination_safe(coordination, catastrophic_limits)
    survival_safe = completion >= completion_minimum and safety_failure <= safety_failure_maximum
    return tracking_safe & tilt_safe & geometry_safe & survival_safe
