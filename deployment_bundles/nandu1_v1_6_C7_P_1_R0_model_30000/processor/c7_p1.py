"""Portable physical contract and tensor math for C7-P-1 only.

No simulator imports or runtime FK/IK. Reset and hard-limit helpers operate on
the supplied articulation without modifying shared assets or control defaults.
"""
from __future__ import annotations

import math

import torch


ARM3_UPPER = 2.24
KNEE_UPPER = 1.2
BASE_HEIGHT_MIN = .363309425
WHEEL_RANGE = (.46, .53)
KNEE_RANGE = (.38, .47)
TCP = (-.066473051764, .161623719479, -.000621484902)
ATTACHMENT_COM = (TCP[0], TCP[1], TCP[2] + .15)
# Offline URDF+wheel collision-mesh calibration: the lowest wheel vertex is
# tangent to the flat ground. Not a runtime kinematics computation.
RESET_ROOT_Z = .6531306790947692
RESET_OVERRIDES = {
    "proximal_roll_L_Joint": .01218427,
    "proximal_roll_R_Joint": -.01218427,
    "proximal_yaw_L_Joint": .05100826,
    "proximal_yaw_R_Joint": .05100826,
}


def physical_profile() -> dict:
    """Return a fresh JSON-compatible record consumed by export/deployment."""
    return {
        "physics_revision": 1,
        "profile_schema": "nandu_c7_p1_physical_v1",
        "checkpoint_schema": "nandu_drl_v1_6_C7_P_1_online_r2",
        "joint_limits": {
            "upper_rad": {"arm3": ARM3_UPPER, "knee_L_Joint": KNEE_UPPER, "knee_R_Joint": KNEE_UPPER},
            "lower_bounds": "unchanged_original_physical_limits",
            "soft_factor": .9,
            "writes": ["simulator_hard", "joint_pos_limits", "default_joint_pos_limits", "soft_joint_pos_limits"],
        },
        "reset": {
            "joint_position_overrides_rad": dict(RESET_OVERRIDES),
            "root_z_m": RESET_ROOT_Z,
            "joint_velocity_rad_s": 0.,
            "default_action_observation_reference": "unchanged_original_default_q",
            "reset_q_storage": "independent_clone_of_default_q_with_reset_overrides",
            "calibration": {
                "method": "offline_URDF_and_wheel_collision_mesh_ground_tangent",
                "reference_root_z_m": .65337,
                "reference_lowest_wheel_mesh_z_m": -.654958507294151,
                "reset_lowest_wheel_mesh_z_m": -.6531306790947692,
                "urdf_revolute_axes": "normalized_before_offline_rotation",
                "calibrated_root_z_m": RESET_ROOT_Z,
                "ground_height_m": 0.,
                "assumption": "STL_surface_ground_tangent; verify_PhysX_convex_contact_separately",
                "runtime_fk_ik": False,
            },
        },
        "posture": {
            "base_floor_m": BASE_HEIGHT_MIN, "base_floor_huber_scale_m": .05,
            "base_floor_cap": 6., "base_floor_weight": -.5,
            "up3_tilt_huber_scale_rad": .12, "up3_tilt_cap": 6., "up3_tilt_weight": -1.,
            "up3_horizontal_reward": "retain_original",
            "static_orientation": {"roll_scale_rad": .08, "pitch_scale_rad": .12,
                                   "up3_tilt_scale_rad": .08, "cap": 20., "weight": -.25,
                                   "gate": "original_static_gate"},
            "dynamic_pitch_envelope": "retain_original",
        },
        "spacing": {
            "wheel_range_m": list(WHEEL_RANGE), "knee_range_m": list(KNEE_RANGE),
            "yaw_half_width_relaxation": .15,
            "geometry_shaping_weight": -.8, "geometry_hard_barrier_weight": -.5,
            "soft_margin_m": .018, "distance_scale_m": .04, "velocity_scale_m_s": .10,
            "diagnostics": "all_raw_and_continuous_metrics_share_ranges_once_per_frame",
        },
        "wheel_axis": {
            "native_local_axis": [-1., 0., 0.],
            "outward_axis_signs_L_R": [1., -1.],
            "rolling_direction_L": "outward_L_cross_world_Z",
            "rolling_direction_R": "world_Z_cross_outward_R",
            "heading_reference": "base_heading_forward_projected_world_XY",
            "deadband_base_deg": 3., "deadband_yaw_extra_deg": 3.,
            "huber_scale_deg": 10., "per_wheel_cap": 4., "aggregation": "mean", "weight": -.15,
            "pair_angle": "unsigned_angle_between_outward_axes_projected_world_XY_rad",
            "common_yaw": "half_sum_signed_heading_errors_rad",
            "toe": "left_minus_right_signed_heading_errors_rad",
            "degenerate_projection": "invalid_flag_and_per_wheel_cap",
        },
        "attachment": {
            "measurement_tcp_local_m": list(TCP), "com_local_m": list(ATTACHMENT_COM),
            "offset_from_tcp_local_m": [0., 0., .15], "frame": "up3_local",
            "mass_distributions": "unchanged_original",
            "reset_inertia": "reconstruct_from_nominal_mass_COM_full_inertia_each_reset",
        },
        "joint_position_bias": {
            "amplitude_rad": .015, "distribution": "Uniform(-amplitude,+amplitude)",
            "sampling": "startup_independent_per_environment_per_11_nonwheel_joint",
            "reset": "preserve", "play_default": False, "fixed_test_override": True,
            "policy_history": "same_cached_measured_positions",
            "critic_label_physical_state": False,
        },
        "disturbance": {
            "bodies": ["base_Link", "up3"], "physics_dt_s": .005,
            "duration_physics_ticks": 2, "duration_s": .01, "onset_interval_s": [2., 4.],
            "force_peaks_N": [0., 2., 4., 7., 10.], "torque_peaks_Nm": [0., .5, 1., 2., 4.],
            "sampling": "independent_per_body_per_axis_uniform_symmetric_peak",
            "force_frame": "world", "torque_frame": "body_local_rotated_to_world_each_physics_tick",
            "application_point": "body_COM", "onset": "shared_between_bodies_per_environment",
            "stage0": False, "play_default": False, "reset_and_end": "clear_wrenches",
            "enable_transition": "fresh_independent_first_delay_2_to_4_seconds",
            "new_physical_world_resume": "clear_temporal_wrench_state",
        },
    }


def reset_joint_positions(default_q: torch.Tensor, joint_names) -> torch.Tensor:
    """Apply reset-only leg offsets; never change action/observation zero points."""
    if default_q.shape[-1] != len(joint_names) or len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names must be unique and match the joint position columns")
    reset_q = default_q.clone()
    for name, angle in RESET_OVERRIDES.items():
        reset_q[..., joint_names.index(name)] = angle
    return reset_q


def write_joint_limits(robot) -> None:
    """Set actual PhysX limits and hard/default/soft caches on this robot only."""
    names = list(robot.joint_names)
    ids = [names.index(name) for name in ("arm3", "knee_L_Joint", "knee_R_Joint")]
    limits = robot.data.joint_pos_limits[:, ids].clone()
    limits[..., 1] = limits.new_tensor([ARM3_UPPER, KNEE_UPPER, KNEE_UPPER])
    robot.write_joint_position_limit_to_sim(limits, joint_ids=ids)
    robot.data.joint_pos_limits[:, ids] = limits
    robot.data.default_joint_pos_limits[:, ids] = limits
    mean = limits.mean(-1)
    half_range = .45 * (limits[..., 1] - limits[..., 0])
    robot.data.soft_joint_pos_limits[:, ids] = torch.stack((mean-half_range, mean+half_range), dim=-1)


def _huber(value: torch.Tensor) -> torch.Tensor:
    absolute = value.abs()
    return torch.where(absolute <= 1., .5 * absolute.square(), absolute - .5)


def base_floor_cost(height: torch.Tensor) -> torch.Tensor:
    return _huber(torch.relu(BASE_HEIGHT_MIN-height)/.05).clamp_max(6.)


def tilt_cost(tilt: torch.Tensor) -> torch.Tensor:
    return _huber(tilt/.12).clamp_max(6.)


def static_orientation_cost(base_gravity: torch.Tensor, up3_tilt: torch.Tensor) -> torch.Tensor:
    roll = torch.atan2(base_gravity[..., 1], -base_gravity[..., 2])
    pitch = torch.atan2(-base_gravity[..., 0], -base_gravity[..., 2])
    return ((roll/.08).square() + (pitch/.12).square() + (up3_tilt/.08).square()).clamp_max(20.)


def wheel_axis_cost(raw_axes_w: torch.Tensor, heading_w: torch.Tensor, yaw_fraction: torch.Tensor) -> dict:
    """Score physical rolling directions relative to heading, with fixed signs.

    Axes are native local -X transformed to world, ordered L/R. All angles are
    radians. ``invalid`` is one boolean per environment; individual invalid
    wheels contribute the bounded per-wheel cap instead of a free zero cost.
    """
    count = raw_axes_w.shape[0]
    if raw_axes_w.shape != (count, 2, 3) or heading_w.shape != (count, 3) or yaw_fraction.shape != (count,):
        raise ValueError("expected axes[N,2,3], heading[N,3], yaw_fraction[N]")
    axes_finite = torch.isfinite(raw_axes_w).all(-1)
    heading_finite = torch.isfinite(heading_w).all(-1)
    axes = torch.nan_to_num(raw_axes_w, nan=0., posinf=0., neginf=0.)[..., :2]
    heading = torch.nan_to_num(heading_w, nan=0., posinf=0., neginf=0.)[..., :2].to(axes)
    # Normalize full vectors before evaluating horizontal degeneracy: a nearly
    # vertical giant vector must not appear valid due to a large raw XY norm.
    full_axis_norm = torch.linalg.vector_norm(torch.nan_to_num(raw_axes_w, nan=0., posinf=0., neginf=0.), dim=-1)
    full_heading_norm = torch.linalg.vector_norm(torch.nan_to_num(heading_w, nan=0., posinf=0., neginf=0.), dim=-1).to(axes)
    axes = axes / full_axis_norm.clamp_min(1e-12)[..., None]
    heading = heading / full_heading_norm.clamp_min(1e-12)[..., None]
    axis_norm = torch.linalg.vector_norm(axes, dim=-1)
    heading_norm = torch.linalg.vector_norm(heading, dim=-1)
    axis_valid = axes_finite & (full_axis_norm > 1e-12) & (axis_norm > 1e-6)
    heading_valid = heading_finite & (full_heading_norm > 1e-12) & (heading_norm > 1e-6)
    valid = axis_valid & heading_valid[:, None] & torch.isfinite(yaw_fraction)[:, None]
    axes = axes / axis_norm.clamp_min(1e-12)[..., None]
    heading = heading / heading_norm.clamp_min(1e-12)[..., None]
    outward = axes * axes.new_tensor([1., -1.])[None, :, None]
    rolling = torch.stack((outward[..., 1], -outward[..., 0]), dim=-1)
    rolling[:, 1] = -rolling[:, 1]
    cross = heading[:, None, 0]*rolling[..., 1] - heading[:, None, 1]*rolling[..., 0]
    dot = (heading[:, None]*rolling).sum(-1)
    psi = torch.atan2(cross, dot)
    psi = torch.where(valid, psi, torch.zeros_like(psi))
    fraction = torch.nan_to_num(yaw_fraction, nan=0., posinf=1., neginf=0.).clamp(0., 1.).to(axes)
    deadband = math.radians(3.) * (1.+fraction)
    per_wheel = _huber(torch.relu(psi.abs()-deadband[:, None])/math.radians(10.)).clamp_max(4.)
    per_wheel = torch.where(valid, per_wheel, torch.full_like(per_wheel, 4.))
    pair_dot = (outward[:, 0]*outward[:, 1]).sum(-1).clamp(-1., 1.)
    pair_angle = torch.acos(pair_dot)
    pair_angle = torch.where(axis_valid.all(-1), pair_angle, torch.full_like(pair_angle, math.pi))
    toe = psi[:, 0]-psi[:, 1]
    common = psi.mean(-1)
    return dict(cost=per_wheel.mean(-1), pair_angle=pair_angle, psi=psi,
                common_yaw=common, toe=toe, invalid=~valid.all(-1))
