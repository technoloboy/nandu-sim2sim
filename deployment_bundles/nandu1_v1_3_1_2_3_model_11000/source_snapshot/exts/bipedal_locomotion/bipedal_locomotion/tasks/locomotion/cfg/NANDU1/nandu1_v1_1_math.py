"""Pure tensor helpers for NANDU1 v1-1 coordination shaping.

This module intentionally has no Isaac Lab imports so the reward and curriculum
math can be unit-tested without starting the simulator.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class GeometryCost(NamedTuple):
    shaping: torch.Tensor
    hard: torch.Tensor
    outward: torch.Tensor
    violation: torch.Tensor
    curriculum_measure: torch.Tensor


class LegGeometryCost(NamedTuple):
    shaping: torch.Tensor
    hard: torch.Tensor
    outward: torch.Tensor
    wheel_violation: torch.Tensor
    knee_violation: torch.Tensor
    mirror_error: torch.Tensor


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _rising_gate(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return _smoothstep01((value - low) / max(high - low, 1.0e-6))


def _huber(value: torch.Tensor) -> torch.Tensor:
    absolute = value.abs()
    return torch.where(absolute <= 1.0, 0.5 * absolute.square(), absolute - 0.5)


def continuous_vertical_cancellation(
    base_vz: torch.Tensor,
    arm_vz: torch.Tensor,
    height_error: torch.Tensor,
    commanded_height_rate: torch.Tensor,
    *,
    deadband: float = 0.025,
    scale: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure wasted opposing vertical motion with a demand-dependent gate."""
    cancelled = torch.relu(
        base_vz.abs() + arm_vz.abs() - (base_vz + arm_vz).abs() - deadband
    )
    demand_gate = torch.maximum(
        _rising_gate(height_error.abs(), 0.02, 0.06),
        _rising_gate(commanded_height_rate.abs(), 0.01, 0.04),
    )
    cost = _huber(cancelled / scale).clamp(max=4.0) * demand_gate
    occupancy = (cancelled > 0.01).to(cancelled.dtype) * demand_gate
    return cost, occupancy


def task_quality_gate(
    yaw_error: torch.Tensor, vx_error: torch.Tensor, height_error: torch.Tensor
) -> torch.Tensor:
    return torch.exp(
        -torch.square(yaw_error / 0.25)
        -torch.square(vx_error / 0.12)
        -torch.square(height_error / 0.08)
    )


def _spacing_cost(
    value: torch.Tensor,
    rate: torch.Tensor,
    lower: float,
    upper: float,
    yaw_fraction: torch.Tensor,
    *,
    soft_margin: float = 0.02,
    distance_scale: float = 0.04,
    velocity_scale: float = 0.10,
) -> GeometryCost:
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    relaxed_half_width = half_width * (1.0 + 0.15 * yaw_fraction.clamp(0.0, 1.0))
    effective_lower = center - relaxed_half_width
    effective_upper = center + relaxed_half_width

    distance_from_center = (value - center).abs()
    proximity_start = torch.clamp(relaxed_half_width - soft_margin, min=1.0e-4)
    proximity = torch.relu(distance_from_center - proximity_start) / soft_margin
    shaping = proximity.square().clamp(max=4.0)

    outside = torch.relu(effective_lower - value) + torch.relu(value - effective_upper)
    hard = _huber(outside / distance_scale).clamp(max=4.0)
    direction = torch.sign(value - center)
    outward_rate = torch.relu(direction * rate)
    approach_gate = _smoothstep01((distance_from_center - proximity_start) / soft_margin)
    outward = approach_gate * torch.square(outward_rate / velocity_scale).clamp(max=4.0)
    violation = (shaping + 4.0 * hard).clamp(max=8.0)
    return GeometryCost(shaping, hard, outward, violation, violation.clamp(max=1.0))


def leg_geometry_cost(
    wheel_spacing: torch.Tensor,
    knee_spacing: torch.Tensor,
    mirror_error: torch.Tensor,
    wheel_spacing_rate: torch.Tensor,
    knee_spacing_rate: torch.Tensor,
    yaw_fraction: torch.Tensor,
) -> LegGeometryCost:
    """Combine wheel/knee spacing and mirror shape using one geometry objective."""
    wheel = _spacing_cost(
        wheel_spacing, wheel_spacing_rate, 0.42, 0.49, yaw_fraction,
        soft_margin=0.018, distance_scale=0.04,
    )
    knee = _spacing_cost(
        knee_spacing, knee_spacing_rate, 0.35, 0.44, yaw_fraction,
        soft_margin=0.018, distance_scale=0.04,
    )
    mirror = _huber(mirror_error / 0.08).clamp(max=4.0)
    shaping = 0.55 * wheel.shaping + 0.35 * knee.shaping + 0.10 * mirror
    hard = 0.55 * wheel.hard + 0.35 * knee.hard
    outward = 0.55 * wheel.outward + 0.35 * knee.outward
    return LegGeometryCost(
        shaping,
        hard,
        outward,
        wheel.curriculum_measure,
        knee.curriculum_measure,
        mirror_error,
    )


def aligned_reach_cost(
    distance: torch.Tensor,
    radial_velocity: torch.Tensor,
    *,
    soft_limit: float = 0.60,
    hard_limit: float = 0.70,
) -> GeometryCost:
    """One continuous reach definition shared by reward and curriculum."""
    band = max(hard_limit - soft_limit, 1.0e-6)
    soft_excess = torch.relu(distance - soft_limit)
    shaping = torch.square(soft_excess / band).clamp(max=4.0)
    hard_excess = torch.relu(distance - hard_limit)
    hard = _huber(hard_excess / 0.05).clamp(max=4.0)
    approach_gate = _rising_gate(distance, soft_limit - 0.05, hard_limit)
    outward = approach_gate * torch.square(torch.relu(radial_velocity) / 0.10).clamp(max=4.0)
    violation = (shaping + 4.0 * hard).clamp(max=8.0)
    return GeometryCost(shaping, hard, outward, violation, violation.clamp(max=1.0))


def softmin_score(values: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    """Smooth min that preserves exact endpoints when all entries are equal."""
    return -temperature * torch.logsumexp(-values / temperature, dim=-1) + temperature * torch.log(
        values.new_tensor(values.shape[-1], dtype=values.dtype)
    )


def curriculum_total_score(
    tracking: float, stability: float, coordination: float, coverage: float
) -> float:
    values = torch.tensor((tracking, stability, coordination, coverage), dtype=torch.float32)
    weighted = 0.50 * values[0] + 0.25 * values[1] + 0.15 * values[2] + 0.10 * values[3]
    balance = 0.70 + 0.30 * softmin_score(values)
    return float(weighted * balance)


def curriculum_progress_step(
    progress: torch.Tensor,
    score_ema: torch.Tensor,
    *,
    threshold: float,
    coordination_ema: torch.Tensor,
    safety_veto: bool,
) -> torch.Tensor:
    if safety_veto:
        return torch.clamp(progress - 1.5, 0.0, 3.0)
    delta = torch.clamp((score_ema - threshold) / 0.10, -1.0, 1.0)
    if float(coordination_ema) < 0.35:
        delta = torch.minimum(delta, torch.zeros_like(delta))
    return torch.clamp(progress + delta, 0.0, 3.0)
