"""v1.2.4 inherits v1.2.3 measurement/physics; its runner changes Encoder heads."""
from isaaclab.utils import configclass
from . import nandu1_v1_2_3_env_cfg as parent
from .nandu1_v1_2_tcp_contract import spec


@configclass
class NANDU1V124CommandsCfg(parent.NANDU1V123CommandsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.nandu1_height_v1_1.checkpoint_schema = spec('v1-2-4')['schema']


@configclass
class NANDU1BlindFlatV124EnvCfg(parent.NANDU1BlindFlatV123EnvCfg):
    commands: NANDU1V124CommandsCfg = NANDU1V124CommandsCfg()


@configclass
class NANDU1BlindFlatV124EnvCfg_PLAY(parent.NANDU1BlindFlatV123EnvCfg_PLAY):
    commands: NANDU1V124CommandsCfg = NANDU1V124CommandsCfg()
