"""v0-2 retains the v0-1 task and adds staged robustness."""
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass
from .nandu1_env_cfg import COMMAND_NAME
from .nandu1_v0_1_env_cfg import NANDU1BlindFlatV01EnvCfg, NANDU1V01CommandsCfg
from . import nandu1_v0_2_mdp as robust

@configclass
class NANDU1V02CommandsCfg(NANDU1V01CommandsCfg):
    def __post_init__(self):
        original=self.nandu1_velocity
        self.nandu1_velocity=robust.Nandu1V02CommandCfg(**{
            key:value for key,value in vars(original).items() if key!='class_type'
        })
        self.nandu1_velocity.class_type=robust.Nandu1V02Command

@configclass
class NANDU1BlindFlatV02EnvCfg(NANDU1BlindFlatV01EnvCfg):
    commands: NANDU1V02CommandsCfg=NANDU1V02CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        for group in (self.observations.policy,self.observations.obsHistory):
            for field,kind in [('base_ang_vel','gyro'),('up3_ang_vel','gyro'),
                               ('base_projected_gravity','gravity'),('up3_projected_gravity','gravity'),
                               ('joint_pos','joint_pos'),('joint_vel','joint_vel')]:
                term=getattr(group,field)
                term.func=robust.noisy_observation
                term.params={k:v for k,v in term.params.items() if k in ('sensor_cfg','asset_cfg')}
                term.params.update(kind=kind,command_name=COMMAND_NAME)
        self.events.stage_domain_randomization.func=robust.RobustDomainRandomization
        self.events.reset_base=EventTerm(func=robust.RobustReset,mode='reset',params={'command_name':COMMAND_NAME})
        self.events.reset_joints=None
        self.events.push_robot=None
        self.events.robust_disturbance=EventTerm(func=robust.RobustDisturbance,mode='interval',
                                               interval_range_s=(0.0,0.0),
                                               params={'command_name':COMMAND_NAME})

@configclass
class NANDU1BlindFlatV02EnvCfg_PLAY(NANDU1BlindFlatV02EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=32
        self.observations.policy.enable_corruption=False
        self.observations.obsHistory.enable_corruption=False
        self.events.stage_domain_randomization=None
        self.events.robust_disturbance=None
        self.commands.nandu1_velocity.adaptive_constraints_enabled=False
        self.commands.nandu1_velocity.robustness_enabled=False
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay=0;actuator.max_delay=0
