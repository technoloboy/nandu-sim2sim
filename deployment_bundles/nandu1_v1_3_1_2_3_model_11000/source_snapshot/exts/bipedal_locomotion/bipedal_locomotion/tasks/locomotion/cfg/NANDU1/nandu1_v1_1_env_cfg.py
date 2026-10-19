"""NANDU1 v1-1: v1 interface with improved whole-body coordination shaping."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import nandu1_v1_1_mdp as v11_mdp
from . import nandu1_v1_env_cfg as v1_env


COMMAND_NAME = v11_mdp.COMMAND_NAME


@configclass
class NANDU1V11CommandsCfg:
    nandu1_height_v1_1 = v1_env.v1_mdp.Nandu1HeightCommandCfg(
        class_type=v11_mdp.Nandu1CoordinatedHeightCommand,
        resampling_time_range=(4.0, 8.0),
        debug_vis=False,
        ranges=v1_env.v1_mdp.Nandu1HeightCommandCfg.Ranges(
            yaw_rate=(-1.57, 1.57),
            ee_vx=(-1.0, 1.0),
            up3_height=(0.40, 1.40),
        ),
        acceleration_limits=(2.0, 0.8, 0.5),
        nominal_height_range=(0.40, 1.40),
        height_taper_buffer=0.10,
        vertical_pulse_time_range=(0.0, 0.0),
        vertical_pulse_safety_fraction=0.0,
        evaluation_steps=6000,
        max_reach_resamples=8,
        reach_resample_limit=0.66,
        robustness_enabled=True,
        measurement_command_name=COMMAND_NAME,
        checkpoint_schema="nandu1_height_v1_1_robust_v2",
    )


@configclass
class NANDU1V11ObservationsCfg(v1_env.NANDU1V1ObservationsCfg):
    def __post_init__(self):
        for group in (self.policy, self.obsHistory):
            group.base_ang_vel.params["command_name"] = COMMAND_NAME
            group.base_projected_gravity.params["command_name"] = COMMAND_NAME
            group.up3_ang_vel.params["command_name"] = COMMAND_NAME
            group.up3_projected_gravity.params["command_name"] = COMMAND_NAME
            group.joint_pos.params["command_name"] = COMMAND_NAME
            group.joint_vel.params["command_name"] = COMMAND_NAME
        self.commands.command.params["command_name"] = COMMAND_NAME


@configclass
class NANDU1V11EventsCfg(v1_env.NANDU1V1EventsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.stage_domain_randomization.params["command_name"] = COMMAND_NAME
        self.reset_base.params["command_name"] = COMMAND_NAME
        self.robust_disturbance.params["command_name"] = COMMAND_NAME


@configclass
class NANDU1V11RewardsCfg(v1_env.NANDU1V1RewardsCfg):
    # Remove v1's independently-shaped terms.  v1-1 uses a shared continuous
    # definition for reward and curriculum instead of stacking synonyms.
    opposite_vertical_cancellation = None
    wheel_spacing_envelope = None
    knee_spacing_envelope = None
    leg_mirror_position = None
    horizontal_reach_barrier = None
    outward_reach_velocity = None

    continuous_vertical_cancellation = RewTerm(
        func=v11_mdp.continuous_vertical_cancellation,
        weight=-0.40,
        params={"command_name": COMMAND_NAME},
    )
    leg_geometry_shaping = RewTerm(
        func=v11_mdp.leg_geometry_shaping,
        weight=-0.80,
        params={"command_name": COMMAND_NAME},
    )
    leg_geometry_hard_barrier = RewTerm(
        func=v11_mdp.leg_geometry_hard_barrier,
        weight=-0.50,
        params={"command_name": COMMAND_NAME},
    )
    reach_coordination_shaping = RewTerm(
        func=v11_mdp.reach_coordination_shaping,
        weight=-0.35,
        params={"command_name": COMMAND_NAME},
    )
    reach_outward_barrier = RewTerm(
        func=v11_mdp.reach_outward_barrier,
        weight=-0.30,
        params={"command_name": COMMAND_NAME},
    )

    def __post_init__(self):
        # Every inherited command-dependent reward must query the v1-1 term.
        for name in (
            "yaw_tracking",
            "yaw_huber",
            "ee_vx_tracking",
            "ee_vx_huber",
            "height_tracking_exp",
            "height_huber",
            "height_velocity_damping",
            "vertical_allocation_tracking",
            "height_directional_reserve",
            "com_to_wheel_support",
            "base_pitch_envelope",
            "horizontal_allocation",
            "arm_default",
            "static_body_motion",
            "static_orientation_hold",
            "gross_leg_asymmetry",
        ):
            term = getattr(self, name)
            if term is not None and "command_name" in term.params:
                term.params["command_name"] = COMMAND_NAME


@configclass
class NANDU1V11CurriculumCfg:
    performance = CurrTerm(func=v11_mdp.performance_curriculum, params={"command_name": COMMAND_NAME})


@configclass
class NANDU1BlindFlatV11EnvCfg(v1_env.NANDU1BlindFlatV1EnvCfg):
    observations: NANDU1V11ObservationsCfg = NANDU1V11ObservationsCfg()
    commands: NANDU1V11CommandsCfg = NANDU1V11CommandsCfg()
    rewards: NANDU1V11RewardsCfg = NANDU1V11RewardsCfg()
    events: NANDU1V11EventsCfg = NANDU1V11EventsCfg()
    curriculum: NANDU1V11CurriculumCfg = NANDU1V11CurriculumCfg()


@configclass
class NANDU1BlindFlatV11EnvCfg_PLAY(NANDU1BlindFlatV11EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        self.events.stage_domain_randomization = None
        self.events.up3_wrench = None
        self.events.robust_disturbance = None
        self.commands.nandu1_height_v1_1.robustness_enabled = False
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = 0
            actuator.max_delay = 0
