"""Isolated reward-only child of v1.3.1.2 (E); no command/network/physics changes."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from . import nandu1_v1_3_1_env_cfg as parent
from . import nandu1_v1_3_1_2_1_mdp as mdp
from .nandu1_v1_3_1_contract import COMMAND_NAME


@configclass
class NANDU1V13121RewardsCfg(parent.NANDU1V131RewardsCfg):
    static_world_yaw_rate = RewTerm(
        func=mdp.static_world_yaw_rate, weight=-.10,
        params={'command_name': COMMAND_NAME, 'scale': .15, 'cap': 4.},
    )

    def __post_init__(self):
        super().__post_init__()
        self.static_orientation_hold.func = mdp.static_orientation_hold


@configclass
class NANDU1BlindFlatV13121EnvCfg(parent.NANDU1BlindFlatV131EnvCfg):
    rewards: NANDU1V13121RewardsCfg = NANDU1V13121RewardsCfg()


@configclass
class NANDU1BlindFlatV13121EnvCfg_PLAY(parent.NANDU1BlindFlatV131EnvCfg_PLAY):
    rewards: NANDU1V13121RewardsCfg = NANDU1V13121RewardsCfg()
