"""B: independent all-random profile, original height rates, zero memory slot."""
from isaaclab.utils import configclass
from . import nandu1_v1_3_1_2_2_env_cfg as parent
from . import nandu1_v1_3_1_2_ablation_mdp as mdp


@configclass
class NANDU1BlindFlatV13123EnvCfg(parent.NANDU1BlindFlatV13122EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        mdp.configure_ablation(self, 3)


@configclass
class NANDU1BlindFlatV13123EnvCfg_PLAY(parent.NANDU1BlindFlatV13122EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        mdp.configure_ablation(self, 3)
