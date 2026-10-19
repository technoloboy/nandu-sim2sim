"""v1.3.1 retains v1.3 policy, rewards, dynamics and promotion defaults."""
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from . import nandu1_v1_3_env_cfg as parent
from . import nandu1_v1_3_mdp as curriculum_mdp
from . import nandu1_v1_mdp as v1
from . import nandu1_v1_3_1_mdp as mdp
from .nandu1_v1_3_1_contract import COMMAND_NAME, SCHEMA


@configclass
class NANDU1V131CommandsCfg:
    nandu1_height_v1_3_1 = v1.Nandu1HeightCommandCfg(
        class_type=mdp.Nandu1TCPHeightCommand, resampling_time_range=(4.,8.), debug_vis=False,
        ranges=v1.Nandu1HeightCommandCfg.Ranges(yaw_rate=(-1.57,1.57),ee_vx=(-1.,1.),up3_height=(.4,1.4)),
        acceleration_limits=(2.,.8,.5),nominal_height_range=(.4,1.4),height_taper_buffer=.1,
        vertical_pulse_time_range=(0.,0.),vertical_pulse_safety_fraction=0.,evaluation_steps=6000,
        max_reach_resamples=8,reach_resample_limit=.83,robustness_enabled=True,
        measurement_command_name=COMMAND_NAME,checkpoint_schema=SCHEMA)


@configclass
class NANDU1V131ObservationsCfg(parent.NANDU1V13ObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        for group in (self.policy,self.obsHistory):
            for name in ("base_ang_vel","base_projected_gravity","up3_ang_vel","up3_projected_gravity","joint_pos","joint_vel"):
                getattr(group,name).params["command_name"] = COMMAND_NAME
        self.commands.command.params["command_name"] = COMMAND_NAME
        self.critic.up3_lin_vel_body.func = mdp.tcp_body_linear_velocity
        self.critic.up3_lin_vel_body.params = {"command_name":COMMAND_NAME}
        self.critic.up3_height.func = mdp.true_tcp_height
        self.critic.up3_height.params = {"command_name":COMMAND_NAME}
        self.critic.rigid_body_mass.func = mdp.actual_body_mass


@configclass
class NANDU1V131EventsCfg(parent.NANDU1V13EventsCfg):
    def __post_init__(self):
        super().__post_init__()
        for name in ("stage_domain_randomization","reset_base","robust_disturbance"):
            getattr(self,name).params["command_name"] = COMMAND_NAME
        self.stage_domain_randomization.func = mdp.TCPPayloadDomainRandomization


@configclass
class NANDU1V131RewardsCfg(parent.NANDU1V13RewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        for term in vars(self).values():
            if term is not None and hasattr(term,"params") and "command_name" in term.params:
                term.params["command_name"] = COMMAND_NAME
        self.height_envelope.params["command_name"] = COMMAND_NAME


@configclass
class NANDU1V131CurriculumCfg:
    performance = CurrTerm(func=curriculum_mdp.performance_curriculum,params={"command_name":COMMAND_NAME})


@configclass
class NANDU1BlindFlatV131EnvCfg(parent.NANDU1BlindFlatV13EnvCfg):
    commands: NANDU1V131CommandsCfg = NANDU1V131CommandsCfg()
    observations: NANDU1V131ObservationsCfg = NANDU1V131ObservationsCfg()
    events: NANDU1V131EventsCfg = NANDU1V131EventsCfg()
    rewards: NANDU1V131RewardsCfg = NANDU1V131RewardsCfg()
    curriculum: NANDU1V131CurriculumCfg = NANDU1V131CurriculumCfg()
    fixed_up3_total_mass: float | None = None
    fixed_rear_mass: float | None = None
    fixed_rear_x: float | None = None


@configclass
class NANDU1BlindFlatV131EnvCfg_PLAY(NANDU1BlindFlatV131EnvCfg):
    fixed_up3_total_mass = .67148
    fixed_rear_mass = 0.
    fixed_rear_x = -.275

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        self.events.up3_wrench = None
        self.events.robust_disturbance = None
        self.commands.nandu1_height_v1_3_1.robustness_enabled = False
        # Keep the payload event; it skips generic DR and applies fixed values.
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = 0
            actuator.max_delay = 0
