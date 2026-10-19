"""Independent v1.2.6: only episode-fixed dual IMU installation errors added."""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from . import nandu1_v1_2_env_cfg as parent
from . import nandu1_v1_1_env_cfg as base
from . import nandu1_v1_2_6_mdp as mdp
from .nandu1_v1_2_6_contract import COMMAND_NAME, SCHEMA


@configclass
class NANDU1V126CommandsCfg(base.NANDU1V11CommandsCfg):
    def __post_init__(self):
        self.nandu1_height_v1_1.class_type = mdp.Nandu1IMUInstallationHeightCommand
        self.nandu1_height_v1_1.checkpoint_schema = SCHEMA


@configclass
class NANDU1V126EventsCfg(base.NANDU1V11EventsCfg):
    # After the original physical reset. This event writes only observation state.
    imu_installation = EventTerm(func=mdp.IMUInstallationReset, mode='reset', params={'command_name':COMMAND_NAME})


@configclass
class NANDU1BlindFlatV126EnvCfg(parent.NANDU1BlindFlatV12EnvCfg):
    commands: NANDU1V126CommandsCfg = NANDU1V126CommandsCfg()
    events: NANDU1V126EventsCfg = NANDU1V126EventsCfg()
    imu_mount_randomization: bool = True
    base_imu_mount_rpy_deg: tuple[float, float, float] | None = None
    up3_imu_mount_rpy_deg: tuple[float, float, float] | None = None

    def __post_init__(self):
        super().__post_init__()
        self.commands.nandu1_height_v1_1.checkpoint_schema = SCHEMA


@configclass
class NANDU1BlindFlatV126EnvCfg_PLAY(parent.NANDU1BlindFlatV12EnvCfg_PLAY):
    commands: NANDU1V126CommandsCfg = NANDU1V126CommandsCfg()
    events: NANDU1V126EventsCfg = NANDU1V126EventsCfg()
    imu_mount_randomization: bool = False
    base_imu_mount_rpy_deg: tuple[float, float, float] | None = None
    up3_imu_mount_rpy_deg: tuple[float, float, float] | None = None

    def __post_init__(self):
        super().__post_init__()
        self.commands.nandu1_height_v1_1.checkpoint_schema = SCHEMA
