"""NANDU1 v1.2 environment: v1.1 task semantics with dual-head Encoder."""

from isaaclab.utils import configclass

from .nandu1_v1_1_env_cfg import NANDU1BlindFlatV11EnvCfg, NANDU1BlindFlatV11EnvCfg_PLAY


@configclass
class NANDU1BlindFlatV12EnvCfg(NANDU1BlindFlatV11EnvCfg):
    """Training environment shared with v1.1; runner supplies the dual-head Encoder."""

    def __post_init__(self):
        super().__post_init__()
        command = getattr(self.commands, "nandu1_height_v1_1", None)
        if command is not None:
            command.checkpoint_schema = "nandu1_height_v1_2_robust_v2"


@configclass
class NANDU1BlindFlatV12EnvCfg_PLAY(NANDU1BlindFlatV11EnvCfg_PLAY):
    """Deterministic Play variant for the v1.2 dual-head policy."""

    def __post_init__(self):
        super().__post_init__()
        command = getattr(self.commands, "nandu1_height_v1_1", None)
        if command is not None:
            command.checkpoint_schema = "nandu1_height_v1_2_robust_v2"
