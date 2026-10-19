"""v1.4 inherits v1.3 dynamics/robustness/promotion; all overrides are local."""
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from . import nandu1_v1_3_env_cfg as parent
from . import nandu1_v1_3_mdp as curriculum_mdp
from . import nandu1_v1_mdp as height_mdp
from . import nandu1_v1_4_mdp as mdp
from .nandu1_v1_4_contract import COMMAND_NAME, SCHEMA


@configclass
class NANDU1V14CommandsCfg:
    nandu1_height_v1_4 = height_mdp.Nandu1HeightCommandCfg(
        class_type=mdp.Nandu1DecoupledHeightCommand,
        resampling_time_range=(4.,8.), debug_vis=False,
        ranges=height_mdp.Nandu1HeightCommandCfg.Ranges(yaw_rate=(-1.57,1.57), ee_vx=(-1.,1.), up3_height=(.4,1.4)),
        acceleration_limits=(2.,.8,.5), nominal_height_range=(.4,1.4),
        height_taper_buffer=.1, vertical_pulse_time_range=(0.,0.), vertical_pulse_safety_fraction=0.,
        evaluation_steps=6000, robustness_enabled=True,
        measurement_command_name=COMMAND_NAME, checkpoint_schema=SCHEMA,
    )


@configclass
class NANDU1V14ObservationsCfg(parent.NANDU1V13ObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        for group in (self.policy,self.obsHistory):
            for name in ("base_ang_vel","base_projected_gravity","up3_ang_vel","up3_projected_gravity","joint_pos","joint_vel"):
                getattr(group,name).params["command_name"]=COMMAND_NAME
        self.commands.command.func=mdp.command_observation
        self.commands.command.params={"command_name":COMMAND_NAME}
        self.critic.base_lin_vel_body.func=mdp.link_body_velocity
        self.critic.base_lin_vel_body.params={"body_name":"base_Link"}
        self.critic.up3_lin_vel_body.func=mdp.link_body_velocity
        self.critic.up3_lin_vel_body.params={"body_name":"up3"}


@configclass
class NANDU1V14EventsCfg(parent.NANDU1V13EventsCfg):
    def __post_init__(self):
        super().__post_init__()
        for name in ("stage_domain_randomization","reset_base","robust_disturbance"):
            getattr(self,name).params["command_name"]=COMMAND_NAME


@configclass
class NANDU1V14RewardsCfg(parent.NANDU1V13RewardsCfg):
    lateral_hold = RewTerm(func=mdp.reward, weight=-.10, params={"kind":"lateral_hold","command_name":COMMAND_NAME})

    def __post_init__(self):
        super().__post_init__()
        for term in vars(self).values():
            if isinstance(term,RewTerm) and "command_name" in term.params:
                term.params["command_name"]=COMMAND_NAME
        # yaw is the first reward term: its entry point updates the allocation
        # snapshot and statistics before any inherited term reads the buffers.
        replacements={
            "yaw_tracking":"yaw", "yaw_huber":"yaw_huber", "ee_vx_tracking":"vx",
            "ee_vx_huber":"vx_huber", "height_huber":"height_huber", "height_velocity_damping":"damping",
            "vertical_allocation_tracking":"vertical_allocation", "horizontal_allocation":"horizontal_allocation",
            "static_body_motion":"static_motion", "static_orientation_hold":"static_orientation",
            "arm_default":"arm_default", "action_rate":"action_rate", "action_smoothness":"action_smoothness",
        }
        for name,kind in replacements.items():
            term=getattr(self,name)
            term.func=mdp.reward
            term.params={"kind":kind,"command_name":COMMAND_NAME}


@configclass
class NANDU1V14CurriculumCfg:
    performance=CurrTerm(func=curriculum_mdp.performance_curriculum,params={"command_name":COMMAND_NAME})


@configclass
class NANDU1BlindFlatV14EnvCfg(parent.NANDU1BlindFlatV13EnvCfg):
    commands: NANDU1V14CommandsCfg=NANDU1V14CommandsCfg()
    observations: NANDU1V14ObservationsCfg=NANDU1V14ObservationsCfg()
    events: NANDU1V14EventsCfg=NANDU1V14EventsCfg()
    rewards: NANDU1V14RewardsCfg=NANDU1V14RewardsCfg()
    curriculum: NANDU1V14CurriculumCfg=NANDU1V14CurriculumCfg()
    # Deliberately no PhysX external-forces-every-iteration override.


@configclass
class NANDU1BlindFlatV14EnvCfg_PLAY(NANDU1BlindFlatV14EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=32
        self.observations.policy.enable_corruption=False
        self.observations.obsHistory.enable_corruption=False
        self.events.stage_domain_randomization=None
        self.events.up3_wrench=None
        self.events.robust_disturbance=None
        self.commands.nandu1_height_v1_4.robustness_enabled=False
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay=actuator.max_delay=0
