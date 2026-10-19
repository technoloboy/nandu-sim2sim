"""Tensor-only curriculum rules for the NANDU1 v1.3 tracking-only ablation."""

from __future__ import annotations

import torch


def tracking_only_promotion_score(
    *,
    tracking: torch.Tensor,
    stability: torch.Tensor,
    coordination: torch.Tensor,
    coverage: torch.Tensor,
) -> torch.Tensor:
    """Return tracking unchanged while accepting all diagnostic scores explicitly."""
    del stability, coordination, coverage
    return tracking


def tracking_progress_step(
    progress: torch.Tensor,
    *,
    tracking_ema: torch.Tensor,
    threshold: float,
    safety_veto: bool,
) -> torch.Tensor:
    """Integrate tracking evidence from the complete current-stage distribution."""
    if safety_veto:
        return torch.clamp(progress - 1.5, 0.0, 3.0)
    delta = torch.clamp((tracking_ema - float(threshold)) / 0.10, -1.0, 1.0)
    return torch.clamp(progress + delta, 0.0, 3.0)


def minimal_safety_veto(
    *,
    non_finite: bool,
    catastrophic_rate: float,
    forbidden_rate: float,
    forbidden_limit: float,
) -> bool:
    """Reject promotion only for numerical failure, severe tilt or forbidden contact."""
    return bool(
        non_finite
        or float(catastrophic_rate) > 0.02
        or float(forbidden_rate) > float(forbidden_limit)
    )
