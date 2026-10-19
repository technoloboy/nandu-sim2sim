"""Independent E-based adaptive-balance configuration; no H or v13121 inheritance."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from rsl_rl.nandu_adaptive_balance.contract import AdaptiveBalanceExperiment
from . import nandu1_v1_3_1_env_cfg as parent
from . import nandu1_v1_3_1_2_2_mdp as mdp

SCHEMA=AdaptiveBalanceExperiment().schema


def configure(cfg):
    term=cfg.commands.nandu1_height_v1_3_1
    term.class_type=mdp.AdaptiveBalanceCommand
    term.checkpoint_schema=SCHEMA
    term.behavior_revision=2
    cfg.events.stage_domain_randomization.func=mdp.AdaptivePayloadRandomization
    cfg.rewards.yaw_tracking.func=mdp.yaw_tracking
    cfg.rewards.static_body_motion=None
    cfg.rewards.static_orientation_hold.func=mdp.static_orientation_hold
    cfg.rewards.com_to_wheel_support.func=mdp.com_to_wheel_support
    for name,weight in [('capture',-.05),('fine',-.25),('bias',-.15),('residual',-.10),('balance',-.20)]:
        setattr(cfg.rewards,'adaptive_'+name,RewTerm(func=mdp.adaptive_static_cost,weight=weight,
            params={'command_name':mdp.COMMAND_NAME,'kind':name}))


@configclass
class NANDU1BlindFlatV13122EnvCfg(parent.NANDU1BlindFlatV131EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure(self)


@configclass
class NANDU1BlindFlatV13122EnvCfg_PLAY(parent.NANDU1BlindFlatV131EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        configure(self)
