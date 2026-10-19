"""Local smooth-transition overrides; v1.4 dynamics, sensors and weights inherited."""
import copy
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from . import nandu1_v1_4_env_cfg as parent
from . import nandu1_v1_3_mdp as curriculum_mdp
from . import nandu1_v1_4_1_mdp as mdp
from .nandu1_v1_4_1_contract import COMMAND_NAME, SCHEMA


def _rename_terms(group):
    for term in vars(group).values():
        if hasattr(term, 'params') and 'command_name' in term.params:
            term.params['command_name'] = COMMAND_NAME


@configclass
class NANDU1V141CommandsCfg:
    nandu1_height_v1_4_1 = copy.deepcopy(parent.NANDU1V14CommandsCfg().nandu1_height_v1_4)
    nandu1_height_v1_4_1.class_type = mdp.Nandu1SmoothTransitionHeightCommand
    nandu1_height_v1_4_1.checkpoint_schema = SCHEMA
    nandu1_height_v1_4_1.measurement_command_name = COMMAND_NAME


@configclass
class NANDU1V141ObservationsCfg(parent.NANDU1V14ObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        for group in (self.policy, self.obsHistory, self.commands, self.critic):
            _rename_terms(group)
        self.commands.command.func = mdp.command_observation


@configclass
class NANDU1V141EventsCfg(parent.NANDU1V14EventsCfg):
    def __post_init__(self):
        super().__post_init__()
        _rename_terms(self)


@configclass
class NANDU1V141RewardsCfg(parent.NANDU1V14RewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        _rename_terms(self)
        for term in vars(self).values():
            if hasattr(term, 'params') and 'kind' in term.params:
                term.func = mdp.reward
        self.action_rate.weight = -.01
        self.action_smoothness.weight = -.01


@configclass
class NANDU1V141CurriculumCfg:
    performance = CurrTerm(func=curriculum_mdp.performance_curriculum, params={'command_name':COMMAND_NAME})


@configclass
class NANDU1BlindFlatV141EnvCfg(parent.NANDU1BlindFlatV14EnvCfg):
    commands: NANDU1V141CommandsCfg = NANDU1V141CommandsCfg()
    observations: NANDU1V141ObservationsCfg = NANDU1V141ObservationsCfg()
    events: NANDU1V141EventsCfg = NANDU1V141EventsCfg()
    rewards: NANDU1V141RewardsCfg = NANDU1V141RewardsCfg()
    curriculum: NANDU1V141CurriculumCfg = NANDU1V141CurriculumCfg()


@configclass
class NANDU1BlindFlatV141EnvCfg_PLAY(NANDU1BlindFlatV141EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        self.events.stage_domain_randomization = None
        self.events.up3_wrench = None
        self.events.robust_disturbance = None
        self.commands.nandu1_height_v1_4_1.robustness_enabled = False
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = actuator.max_delay = 0
