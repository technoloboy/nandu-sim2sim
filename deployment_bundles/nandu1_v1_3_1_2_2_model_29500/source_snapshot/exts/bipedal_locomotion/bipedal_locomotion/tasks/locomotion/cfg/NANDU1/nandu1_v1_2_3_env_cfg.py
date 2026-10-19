"""Independent v1.2.3 task: only tracking's measurement point changes from v1.1."""
from isaaclab.utils import configclass
from . import nandu1_v1_1_env_cfg as parent
from . import nandu1_v1_2_3_mdp as mdp
from .nandu1_v1_2_tcp_contract import spec, COMMAND_NAME


@configclass
class NANDU1V123CommandsCfg(parent.NANDU1V11CommandsCfg):
    def __post_init__(self):
        self.nandu1_height_v1_1.class_type = mdp.Nandu1FixedTCPHeightCommand
        self.nandu1_height_v1_1.checkpoint_schema = spec('v1-2-3')['schema']


@configclass
class NANDU1V123ObservationsCfg(parent.NANDU1V11ObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.critic.up3_lin_vel_body.func = mdp.tcp_body_linear_velocity
        self.critic.up3_lin_vel_body.params = {'command_name':COMMAND_NAME}
        self.critic.up3_height.func = mdp.true_tcp_height
        self.critic.up3_height.params = {'command_name':COMMAND_NAME}


@configclass
class NANDU1V123RewardsCfg(parent.NANDU1V11RewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        # v1.1's unbound envelope otherwise reads the physical link origin.
        self.height_envelope.params['command_name'] = COMMAND_NAME


@configclass
class NANDU1BlindFlatV123EnvCfg(parent.NANDU1BlindFlatV11EnvCfg):
    commands: NANDU1V123CommandsCfg = NANDU1V123CommandsCfg()
    observations: NANDU1V123ObservationsCfg = NANDU1V123ObservationsCfg()
    rewards: NANDU1V123RewardsCfg = NANDU1V123RewardsCfg()


@configclass
class NANDU1BlindFlatV123EnvCfg_PLAY(parent.NANDU1BlindFlatV11EnvCfg_PLAY):
    commands: NANDU1V123CommandsCfg = NANDU1V123CommandsCfg()
    observations: NANDU1V123ObservationsCfg = NANDU1V123ObservationsCfg()
    rewards: NANDU1V123RewardsCfg = NANDU1V123RewardsCfg()
