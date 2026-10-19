"""NANDU1 v1.3: v1.2 dual-head policy with tracking-only curriculum promotion."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from . import nandu1_v1_1_env_cfg as v11_env
from . import nandu1_v1_2_env_cfg as v12_env
from . import nandu1_v1_3_mdp as v13_mdp


COMMAND_NAME = v13_mdp.COMMAND_NAME


@configclass
class NANDU1V13CommandsCfg:
    nandu1_height_v1_3 = v11_env.v1_env.v1_mdp.Nandu1HeightCommandCfg(
        class_type=v13_mdp.Nandu1TrackingOnlyHeightCommand,
        resampling_time_range=(4.0, 8.0),
        debug_vis=False,
        ranges=v11_env.v1_env.v1_mdp.Nandu1HeightCommandCfg.Ranges(
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
        checkpoint_schema="nandu1_height_v1_3_robust_v2",
    )


@configclass
class NANDU1V13ObservationsCfg(v11_env.NANDU1V11ObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        for group in (self.policy, self.obsHistory):
            group.base_ang_vel.params["command_name"] = COMMAND_NAME
            group.base_projected_gravity.params["command_name"] = COMMAND_NAME
            group.up3_ang_vel.params["command_name"] = COMMAND_NAME
            group.up3_projected_gravity.params["command_name"] = COMMAND_NAME
            group.joint_pos.params["command_name"] = COMMAND_NAME
            group.joint_vel.params["command_name"] = COMMAND_NAME
        self.commands.command.params["command_name"] = COMMAND_NAME


@configclass
class NANDU1V13EventsCfg(v11_env.NANDU1V11EventsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.stage_domain_randomization.params["command_name"] = COMMAND_NAME
        self.reset_base.params["command_name"] = COMMAND_NAME
        self.robust_disturbance.params["command_name"] = COMMAND_NAME


@configclass
class NANDU1V13RewardsCfg(v11_env.NANDU1V11RewardsCfg):
    def __post_init__(self):
        super().__post_init__()
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
            "continuous_vertical_cancellation",
            "leg_geometry_shaping",
            "leg_geometry_hard_barrier",
            "reach_coordination_shaping",
            "reach_outward_barrier",
        ):
            term = getattr(self, name)
            if term is not None and "command_name" in term.params:
                term.params["command_name"] = COMMAND_NAME


@configclass
class NANDU1V13CurriculumCfg:
    performance = CurrTerm(func=v13_mdp.performance_curriculum, params={"command_name": COMMAND_NAME})


@configclass
class NANDU1BlindFlatV13EnvCfg(v12_env.NANDU1BlindFlatV12EnvCfg):
    """Isolated v1.3 training environment; only curriculum promotion differs from v1.2."""

    observations: NANDU1V13ObservationsCfg = NANDU1V13ObservationsCfg()
    commands: NANDU1V13CommandsCfg = NANDU1V13CommandsCfg()
    rewards: NANDU1V13RewardsCfg = NANDU1V13RewardsCfg()
    events: NANDU1V13EventsCfg = NANDU1V13EventsCfg()
    curriculum: NANDU1V13CurriculumCfg = NANDU1V13CurriculumCfg()


@configclass
class NANDU1BlindFlatV13EnvCfg_PLAY(NANDU1BlindFlatV13EnvCfg):
    """Deterministic Play configuration for the v1.3 policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        self.events.stage_domain_randomization = None
        self.events.up3_wrench = None
        self.events.robust_disturbance = None
        self.commands.nandu1_height_v1_3.robustness_enabled = False
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = 0
            actuator.max_delay = 0
