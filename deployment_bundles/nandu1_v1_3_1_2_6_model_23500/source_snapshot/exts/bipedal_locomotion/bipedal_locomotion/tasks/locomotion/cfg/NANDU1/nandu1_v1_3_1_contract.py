"""Simulator-free v1.3.1 interface and portable deployment metadata."""
from copy import deepcopy

SCHEMA = "nandu1_height_v1_3_1_tcp_payload_v1"
COMMAND_NAME = "nandu1_height_v1_3_1"
EXPERIMENT_NAME = "nandu1_blind_flat_v1_3_1_tcp_payload"
INTERFACE = dict(policy=49, history=245, history_frames=5, encoder=7,
                 commands=3, actor=59, critic=108, critic_with_commands=111, actions=13)
COMMAND_UNITS = ("rad/s", "m/s", "m")
COMMAND_FRAMES = ("world_z_angular_velocity", "base_body_x", "world_z_absolute")
TCP_LOCAL = (-.066473051764, .161623719479, -.000621484902)
UP3_NOMINAL_MASS = .67148
UP3_MASS_RANGE = (.6, 6.)
UP3_BOX_DIMENSIONS = (.05, .06, .05)
REAR_MASS_RANGE = (0., 5.)
REAR_X_RANGE = (-.35, -.20)
REAR_Z = .07
REAR_CYLINDER_RADIUS = .025
REAR_CYLINDER_LENGTH = .1
PAYLOAD_STAGE_AMPLITUDES = (.1, .25, .5, .75, 1.)
PAYLOAD_MIXTURE_PROBABILITIES = (.2, .2, .2, .4)
YAW_STAGE_LIMITS = (.20, .50, .85, 1.20, 1.57)
BASE_INERTIA_FRAME_ROTATION = ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))


def export_metadata() -> dict:
    """Return an independent JSON-compatible contract; no runtime inertia is assumed."""
    return deepcopy(dict(
        checkpoint_schema=SCHEMA, command_name=COMMAND_NAME,
        experiment_name=EXPERIMENT_NAME, interface=INTERFACE,
        command_units=list(COMMAND_UNITS), command_frames=list(COMMAND_FRAMES),
        quaternion_order="wxyz", tcp_body="up3", tcp_frame="up3_link",
        tcp_local_m=list(TCP_LOCAL), tcp_independent_of_com=True,
        height_range_m=[.4, 1.4],
        yaw=dict(source="body_imu_relative_heading", heading_kp=1.5,
                 deadzone=1e-5, stage_limits_rad_s=list(YAW_STAGE_LIMITS),
                 release_latch=True, tilt_correction=True, saturation_back_calculation=True),
        payload=dict(
            inertia_coordinates="link_axes_about_com", shapes_are_inertia_proxies_only=True,
            nominal_properties_source="immutable_runtime_mass_com_full_inertia",
            up3_nominal_mass_kg=UP3_NOMINAL_MASS, up3_total_mass_kg=list(UP3_MASS_RANGE),
            up3_box_dimensions_m=list(UP3_BOX_DIMENSIONS), up3_box_center="fixed_tcp",
            rear_extra_mass_kg=list(REAR_MASS_RANGE), rear_x_m=list(REAR_X_RANGE),
            rear_y_m=0., rear_z_m=REAR_Z, rear_cylinder_axis="link_y",
            rear_radius_m=REAR_CYLINDER_RADIUS, rear_length_m=REAR_CYLINDER_LENGTH,
            stage_amplitudes=list(PAYLOAD_STAGE_AMPLITUDES),
            mixture_order=["nominal", "up3_only", "rear_only", "both"],
            mixture_probabilities=list(PAYLOAD_MIXTURE_PROBABILITIES),
            base_urdf_inertia_rotation=list(map(list, BASE_INERTIA_FRAME_ROTATION)),
            parameters_constant_within_episode=True),
        resume=dict(old_schema_compatible=False, reset_episodic_heading=True,
                    reset_episodic_payload=True, preserve_course_and_optimizer=True)))
