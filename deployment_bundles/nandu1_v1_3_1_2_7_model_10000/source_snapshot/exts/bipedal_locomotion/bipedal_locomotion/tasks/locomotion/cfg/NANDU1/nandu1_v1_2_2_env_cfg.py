"""v1.2.2 changes only Encoder capacity; task semantics remain exactly v1.1."""
from isaaclab.utils import configclass

from .nandu1_v1_1_env_cfg import NANDU1BlindFlatV11EnvCfg, NANDU1BlindFlatV11EnvCfg_PLAY

SCHEMA = "nandu1_height_v1_2_2_hidden_heads_v1"


@configclass
class NANDU1BlindFlatV122EnvCfg(NANDU1BlindFlatV11EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.nandu1_height_v1_1.checkpoint_schema = SCHEMA


@configclass
class NANDU1BlindFlatV122EnvCfg_PLAY(NANDU1BlindFlatV11EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.commands.nandu1_height_v1_1.checkpoint_schema = SCHEMA
