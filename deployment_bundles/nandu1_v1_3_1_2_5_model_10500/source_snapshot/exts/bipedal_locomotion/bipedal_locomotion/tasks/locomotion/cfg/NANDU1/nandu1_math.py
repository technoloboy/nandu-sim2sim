"""Small tensor-only calculations shared by NANDU1 curriculum and rewards.

Keeping these independent of Isaac Sim makes coordinate and boundary behavior
testable without launching a simulation application.
"""

from __future__ import annotations

import torch


# NANDU3 native coordinates: pitch axes mirror sign, roll axes mirror sign,
# while yaw and knee coordinates use the same sign for a mirrored pose.
LEG_MIRROR_SIGN = (-1.0, -1.0, 1.0, 1.0)


def slew_limited_height(
    current: torch.Tensor,
    target: torch.Tensor,
    max_rate: float | torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move an absolute height command toward a target with a first-order rate limit."""
    maximum_delta = torch.as_tensor(max_rate, device=current.device, dtype=current.dtype) * float(dt)
    raw_delta = target - current
    delta = torch.clamp(raw_delta, min=-maximum_delta, max=maximum_delta)
    updated = current + delta
    rate = delta / float(dt)
    limited = raw_delta.abs() > maximum_delta
    return updated, rate, limited


def normalized_huber(error: torch.Tensor, scale: float, cap: float) -> torch.Tensor:
    """Return a scale-normalized Huber penalty with a wide numerical cap."""
    normalized = error.abs() / float(scale)
    penalty = torch.where(normalized <= 1.0, 0.5 * torch.square(normalized), normalized - 0.5)
    return torch.clamp(penalty, max=float(cap))


def lower_is_better_score(metric: torch.Tensor, good: float, bad: float) -> torch.Tensor:
    """Normalize a cost metric to one at ``good`` and zero at ``bad``."""
    return torch.clamp((float(bad) - metric) / (float(bad) - float(good)), 0.0, 1.0)


def higher_is_better_score(metric: torch.Tensor, bad: float, good: float) -> torch.Tensor:
    """Normalize a benefit metric to zero at ``bad`` and one at ``good``."""
    return torch.clamp((metric - float(bad)) / (float(good) - float(bad)), 0.0, 1.0)


def sample_uniform_stage_commands(
    count: int,
    *,
    yaw_limit: float,
    vx_limit: float,
    height_low: float,
    height_high: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample yaw, forward velocity, and height independently over a stage box."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if yaw_limit <= 0.0 or vx_limit <= 0.0 or height_high <= height_low:
        raise ValueError("stage command ranges must be ordered and non-degenerate")
    samples = torch.rand((count, 3), device=device, dtype=dtype)
    lower = torch.tensor((-yaw_limit, -vx_limit, height_low), device=device, dtype=dtype)
    upper = torch.tensor((yaw_limit, vx_limit, height_high), device=device, dtype=dtype)
    return lower + samples * (upper - lower)


def coordination_curriculum_score(
    metrics: torch.Tensor,
    targets: torch.Tensor,
    hard_limits: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine coordination costs while preserving wide per-metric safety ceilings.

    A metric earns full credit at or below its target and decreases linearly to
    zero at its hard limit.  Good geometry terms can therefore compensate for
    a moderately lagging auxiliary term, but no term may cross its hard limit.
    """
    if metrics.shape[-1] != targets.shape[-1] or metrics.shape[-1] != hard_limits.shape[-1]:
        raise ValueError("coordination metric, target and hard-limit dimensions must match")
    if weights.shape[-1] != metrics.shape[-1]:
        raise ValueError("coordination weights must match the metric dimension")
    span = (hard_limits - targets).clamp_min(torch.finfo(metrics.dtype).eps)
    component_scores = torch.clamp((hard_limits - metrics) / span, 0.0, 1.0)
    score = torch.sum(component_scores * weights, dim=-1) / weights.sum().clamp_min(
        torch.finfo(metrics.dtype).eps
    )
    hard_safe = torch.all(metrics <= hard_limits, dim=-1)
    return score, hard_safe


def normalized_curriculum_progress(
    progress: torch.Tensor,
    score: torch.Tensor,
    threshold: float,
    width: float = 0.10,
    maximum: float = 3.0,
) -> torch.Tensor:
    """Integrate normalized curriculum evidence without binary window resets."""
    delta = torch.clamp((score - float(threshold)) / float(width), -1.0, 1.0)
    return torch.clamp(progress + delta, 0.0, float(maximum))


def _low_value_gate(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    x = torch.clamp((value.abs() - float(low)) / (float(high) - float(low)), 0.0, 1.0)
    smooth = x * x * (3.0 - 2.0 * x)
    return 1.0 - smooth


def static_hold_gate(
    yaw_command: torch.Tensor,
    vx_command: torch.Tensor,
    height_error: torch.Tensor,
    height_rate: torch.Tensor,
) -> torch.Tensor:
    """Smoothly select settled zero-motion states while retaining an absolute height target."""
    return (
        _low_value_gate(yaw_command, 0.03, 0.10)
        * _low_value_gate(vx_command, 0.02, 0.08)
        * _low_value_gate(height_error, 0.02, 0.06)
        * _low_value_gate(height_rate, 0.01, 0.04)
    )


def build_stage_material_buckets(
    num_buckets: int = 48,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create the three reusable material banks used by the curriculum.

    PhysX has a global limit on unique material objects.  Sampling these
    values once bounds the task to ``3 * num_buckets`` material triplets,
    independent of the number of environments and collision shapes.
    """
    if num_buckets <= 0:
        raise ValueError(f"num_buckets must be positive, got {num_buckets}")
    stage_ranges = (
        (0.8, 1.2, 0.0, 0.0),
        (0.7, 1.2, 0.0, 0.05),
        (0.6, 1.2, 0.0, 0.10),
    )
    banks = []
    for friction_low, friction_high, restitution_low, restitution_high in stage_ranges:
        static = torch.empty(num_buckets, device="cpu").uniform_(
            friction_low, friction_high, generator=generator
        )
        dynamic = torch.empty(num_buckets, device="cpu").uniform_(
            friction_low, friction_high, generator=generator
        )
        dynamic = torch.minimum(static, dynamic)
        restitution = torch.empty(num_buckets, device="cpu").uniform_(
            restitution_low, restitution_high, generator=generator
        )
        banks.append(torch.stack((static, dynamic, restitution), dim=-1))
    return torch.stack(banks, dim=0)


def sample_stage_material_buckets(
    buckets: torch.Tensor,
    stage: int,
    num_envs: int,
    num_shapes: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Assign existing material triplets without creating continuous values."""
    if buckets.ndim != 3 or buckets.shape[0] != 3 or buckets.shape[-1] != 3:
        raise ValueError(f"expected material buckets with shape (3, N, 3), got {tuple(buckets.shape)}")
    stage_bank = 0 if stage < 2 else 1 if stage == 2 else 2
    bucket_ids = torch.randint(
        buckets.shape[1],
        (num_envs, num_shapes),
        device="cpu",
        generator=generator,
    )
    return buckets[stage_bank][bucket_ids]


def resumed_evaluation_anchor(current_step: int, elapsed_steps: int, evaluation_steps: int) -> int:
    """Rebase an unfinished curriculum window onto a new environment clock."""
    elapsed = min(max(int(elapsed_steps), 0), int(evaluation_steps))
    return int(current_step) - elapsed


def limit_vertical_velocity(
    velocity: torch.Tensor,
    nominal_height: torch.Tensor,
    lower: float,
    upper: float,
    buffer: float,
) -> torch.Tensor:
    """Smoothly remove only outward velocity near a nominal height limit."""
    upward_scale = torch.clamp((upper - nominal_height) / buffer, 0.0, 1.0)
    downward_scale = torch.clamp((nominal_height - lower) / buffer, 0.0, 1.0)
    return torch.where(velocity > 0.0, velocity * upward_scale, velocity * downward_scale)


def safe_vertical_pulse_duration(
    velocity: torch.Tensor,
    nominal_height: torch.Tensor,
    lower: float,
    upper: float,
    requested_duration: torch.Tensor,
    safety_fraction: float = 0.80,
) -> torch.Tensor:
    """Clip a vertical pulse duration to its nominal directional height reserve."""
    directional_margin = torch.where(velocity >= 0.0, upper - nominal_height, nominal_height - lower)
    safe_duration = safety_fraction * torch.clamp(directional_margin, min=0.0) / velocity.abs().clamp_min(1.0e-6)
    return torch.where(velocity.abs() > 1.0e-6, torch.minimum(requested_duration, safe_duration), requested_duration)


def sample_axis_counts(
    probabilities: tuple[float, float, float, float],
    count: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample how many of the three command axes are active (zero through three)."""
    if len(probabilities) != 4 or abs(sum(probabilities) - 1.0) > 1.0e-6:
        raise ValueError(f"axis-count probabilities must contain four values summing to one, got {probabilities}")
    probability = torch.tensor(probabilities, dtype=torch.float32, device=device)
    return torch.multinomial(probability, count, replacement=True, generator=generator)


def height_envelope_penalty(
    height: torch.Tensor,
    lower: float = 0.40,
    upper: float = 1.20,
    soft_margin: float = 0.05,
) -> torch.Tensor:
    """Continuous zero-inside penalty, stronger past the requested envelope."""
    soft = torch.relu((lower + soft_margin) - height) + torch.relu(height - (upper - soft_margin))
    hard = torch.relu(lower - height) + torch.relu(height - upper)
    return torch.square(soft / soft_margin) + 4.0 * torch.square(hard / soft_margin)


def wheel_stance_penalties(
    wheel_delta_h: torch.Tensor,
    minimum: float = 0.42,
    maximum: float = 0.49,
    spacing_scale: float = 0.05,
    fore_aft_scale: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independent lateral-spacing and fore-aft wheel penalties."""
    lateral_distance = wheel_delta_h[:, 1].abs()
    lateral_excess = torch.relu(minimum - lateral_distance) + torch.relu(lateral_distance - maximum)
    return torch.square(lateral_excess / spacing_scale), torch.square(wheel_delta_h[:, 0] / fore_aft_scale)


def knee_spacing_penalties(
    knee_delta_h: torch.Tensor,
    minimum: float = 0.35,
    maximum: float = 0.44,
    target: float = 0.395,
    envelope_scale: float = 0.05,
    center_scale: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return soft envelope and nominal-centering penalties for knee spacing."""
    lateral_distance = knee_delta_h[:, 1].abs()
    excess = torch.relu(minimum - lateral_distance) + torch.relu(lateral_distance - maximum)
    envelope = torch.square(excess / envelope_scale)
    centering = torch.square((lateral_distance - target) / center_scale)
    return envelope, centering


def leg_mirror_residual(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the physical left/right mirror residual for pitch, roll, yaw and knee."""
    sign = left.new_tensor(LEG_MIRROR_SIGN)
    return left - sign * right


def horizontal_reach_penalty(
    distance: torch.Tensor,
    soft_limit: float = 0.60,
    scale: float = 0.10,
) -> torch.Tensor:
    """Continuous soft reach barrier; no distance is treated as a hard failure."""
    return torch.square(torch.relu(distance - soft_limit) / scale)


def vertical_command_gate(command_vz: torch.Tensor, deadband: float = 0.02, full_scale: float = 0.05) -> torch.Tensor:
    """Smoothly enable allocation shaping outside the vertical-command deadband."""
    x = torch.clamp((command_vz.abs() - deadband) / (full_scale - deadband), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def tracking_quality_gate(errors: torch.Tensor, scales: tuple[float, ...]) -> torch.Tensor:
    """Smooth gate that is one at accurate tracking and decays across all task axes."""
    normalization = errors.new_tensor(scales)
    return torch.exp(-torch.mean(torch.square(errors / normalization), dim=-1))


def directional_margin(
    command_vz: torch.Tensor, height: torch.Tensor, limits: tuple[float, float]
) -> torch.Tensor:
    lower, upper = limits
    normalized = torch.clamp((height - lower) / (upper - lower), 0.0, 1.0)
    return torch.where(command_vz >= 0.0, 1.0 - normalized, normalized)


def height_allocation_ratio(
    command_vz: torch.Tensor,
    base_height: torch.Tensor,
    arm_height: torch.Tensor,
    base_range: tuple[float, float],
    arm_range: tuple[float, float],
    base_quality: torch.Tensor,
    arm_quality: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Allocate vertical motion from directional reserve and instantaneous quality."""
    base_capacity = directional_margin(command_vz, base_height, base_range) * torch.clamp(base_quality, 0.0, 1.0)
    arm_capacity = directional_margin(command_vz, arm_height, arm_range) * torch.clamp(arm_quality, 0.0, 1.0)
    return base_capacity / (base_capacity + arm_capacity + eps)


def horizontal_allocation_ratio(
    low_frequency_command: torch.Tensor,
    relative_x: torch.Tensor,
    distance: torch.Tensor,
    base_quality: torch.Tensor,
    arm_quality: torch.Tensor,
    soft_limit: float = 0.60,
    hard_limit: float = 0.70,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Allocate low-frequency EE-x motion using reach direction and subsystem quality.

    Fast command residuals remain assigned to the arm.  Near the reach boundary,
    outward motion shifts toward the base while inward motion shifts toward the
    arm so that it actively recovers configuration margin.
    """
    pressure = torch.clamp((distance - soft_limit) / (hard_limit - soft_limit), 0.0, 1.0)
    pressure = pressure * pressure * (3.0 - 2.0 * pressure)
    direction = torch.tanh(relative_x * low_frequency_command / 0.02)
    preferred = 0.65 + 0.30 * pressure * direction
    base = torch.clamp(base_quality, 0.0, 1.0)
    arm = torch.clamp(arm_quality, 0.0, 1.0)
    capability_share = base / (base + arm + eps)
    return torch.clamp(preferred + 0.30 * (capability_share - 0.50), 0.15, 0.95)


def outward_reach_velocity_penalty(
    distance: torch.Tensor,
    radial_velocity: torch.Tensor,
    soft_limit: float = 0.60,
    hard_limit: float = 0.70,
    velocity_scale: float = 0.10,
) -> torch.Tensor:
    """Penalize only outward relative motion as horizontal reach approaches its limit."""
    pressure = torch.clamp((distance - soft_limit) / (hard_limit - soft_limit), 0.0, 1.0)
    pressure = pressure * pressure * (3.0 - 2.0 * pressure)
    outward = torch.relu(radial_velocity) / velocity_scale
    return pressure * torch.square(outward)


def update_sustained_violation(
    elapsed: torch.Tensor, violation: torch.Tensor, dt: float, duration: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate only consecutive violation time and report sustained failures."""
    updated = torch.where(violation, elapsed + dt, torch.zeros_like(elapsed))
    return updated, updated + 1.0e-9 >= duration


def outward_reach_command(
    relative_x: torch.Tensor,
    candidate_vx: torch.Tensor,
    distance: torch.Tensor,
    resample_limit: float = 0.55,
) -> torch.Tensor:
    """Identify sampled forward commands that extend an already long arm reach."""
    return (distance >= resample_limit) & (relative_x * candidate_vx > 0.0)
