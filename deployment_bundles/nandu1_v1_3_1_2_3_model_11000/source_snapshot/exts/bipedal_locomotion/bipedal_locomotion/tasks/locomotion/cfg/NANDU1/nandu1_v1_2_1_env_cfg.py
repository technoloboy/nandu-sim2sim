"""Independent scheme B configuration; v1.2 remains an unchanged baseline."""
from isaaclab.utils import configclass

from . import nandu1_v1_2_env_cfg as v12
from . import nandu1_v1_1_env_cfg as v11
from . import nandu1_v1_2_1_mdp as mdp
from . import nandu1_v1_2_1_contract as contract


@configclass
class NANDU1V121CommandsCfg:
    nandu1_height_v1_2_1 = mdp.Nandu1AdaptiveHoldCommandCfg(
        resampling_time_range=(4.,8.), debug_vis=False,
        ranges=mdp.Nandu1AdaptiveHoldCommandCfg.Ranges(
            yaw_rate=(-1.57,1.57), ee_vx=(-1.,1.), up3_height=(.4,1.4)),
        acceleration_limits=(2.,.8,.5), nominal_height_range=(.4,1.4),
        height_taper_buffer=.10, vertical_pulse_time_range=(0.,0.),
        vertical_pulse_safety_fraction=0., evaluation_steps=6000,
        max_reach_resamples=8, reach_resample_limit=.66, robustness_enabled=True,
    )


def retarget_commands(config, old_name, new_name):
    """Retarget instance-local manager terms, not shared v1.2 class defaults."""
    for value in vars(config).values():
        if hasattr(value, 'params') and value.params.get('command_name') == old_name:
            value.params['command_name'] = new_name


@configclass
class NANDU1BlindFlatV121EnvCfg(v12.NANDU1BlindFlatV12EnvCfg):
    commands: NANDU1V121CommandsCfg = NANDU1V121CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        for group in (self.observations.policy, self.observations.obsHistory, self.observations.commands,
                      self.events, self.rewards, self.curriculum):
            retarget_commands(group, v11.COMMAND_NAME, contract.COMMAND_NAME)
        self.rewards.height_velocity_damping.func = mdp.height_velocity_damping
        self.rewards.static_body_motion.func = mdp.static_body_motion
        self.rewards.static_orientation_hold.func = mdp.static_orientation_hold


@configclass
class NANDU1BlindFlatV121EnvCfg_PLAY(NANDU1BlindFlatV121EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        self.events.stage_domain_randomization = None
        self.events.up3_wrench = None
        self.events.robust_disturbance = None
        self.commands.nandu1_height_v1_2_1.robustness_enabled = False
        self.commands.nandu1_height_v1_2_1.adaptive_enabled = False
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = 0
            actuator.max_delay = 0
