"""Independent v1.2.5 configuration; same v1.2.3 single-output Encoder."""
from isaaclab.utils import configclass
from . import nandu1_v1_2_3_env_cfg as parent
from .nandu1_v1_2_5_mdp import Nandu1LongTCPHeightCommand
from .nandu1_v1_2_tcp_contract import spec


@configclass
class NANDU1V125CommandsCfg(parent.NANDU1V123CommandsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.nandu1_height_v1_1.class_type = Nandu1LongTCPHeightCommand
        self.nandu1_height_v1_1.checkpoint_schema = spec('v1-2-5')['schema']


@configclass
class NANDU1BlindFlatV125EnvCfg(parent.NANDU1BlindFlatV123EnvCfg):
    commands: NANDU1V125CommandsCfg = NANDU1V125CommandsCfg()


@configclass
class NANDU1BlindFlatV125EnvCfg_PLAY(parent.NANDU1BlindFlatV123EnvCfg_PLAY):
    commands: NANDU1V125CommandsCfg = NANDU1V125CommandsCfg()
