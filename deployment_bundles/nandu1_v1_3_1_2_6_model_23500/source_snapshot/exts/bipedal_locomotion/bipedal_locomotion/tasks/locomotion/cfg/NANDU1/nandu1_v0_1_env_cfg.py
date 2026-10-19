"""NANDU1 v0-1: v0 control task with adaptive coordination shaping."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import nandu1_v0_1_mdp
from .nandu1_env_cfg import (
    COMMAND_NAME,
    NANDU1BlindFlatEnvCfg,
    NANDU1BlindFlatEnvCfg_PLAY,
    NANDU1CommandsCfg,
    NANDU1RewardsCfg,
)


@configclass
class NANDU1V01CommandsCfg(NANDU1CommandsCfg):
    nandu1_velocity = nandu1_v0_1_mdp.Nandu1V01CommandCfg(
        class_type=nandu1_v0_1_mdp.Nandu1V01Command,
        resampling_time_range=(4.0, 8.0),
        debug_vis=False,
        ranges=nandu1_v0_1_mdp.Nandu1V01CommandCfg.Ranges(
            yaw_rate=(-1.57, 1.57),
            ee_vx=(-1.0, 1.0),
            ee_vz=(-0.5, 0.5),
        ),
        acceleration_limits=(2.0, 0.8, 0.5),
        nominal_height_range=(0.40, 1.20),
        height_taper_buffer=0.10,
        vertical_pulse_time_range=(0.8, 2.0),
        vertical_pulse_safety_fraction=0.8,
        evaluation_steps=6000,
        max_reach_resamples=8,
        reach_resample_limit=0.66,
        adaptive_constraints_enabled=True,
    )


@configclass
class NANDU1V01RewardsCfg(NANDU1RewardsCfg):
    """Only the six coordination families differ from the v0 reward set."""

    horizontal_reach_barrier = RewTerm(
        func=nandu1_v0_1_mdp.adaptive_horizontal_reach_barrier,
        weight=-0.20,
        params={"command_name": COMMAND_NAME},
    )
    outward_reach_velocity = RewTerm(
        func=nandu1_v0_1_mdp.adaptive_outward_reach_velocity,
        weight=-0.20,
        params={"command_name": COMMAND_NAME},
    )
    opposite_vertical_cancellation = RewTerm(
        func=nandu1_v0_1_mdp.adaptive_opposite_vertical_cancellation,
        weight=-0.10,
        params={"command_name": COMMAND_NAME, "scale": 0.05},
    )
    wheel_spacing_envelope = RewTerm(
        func=nandu1_v0_1_mdp.adaptive_wheel_spacing_envelope,
        weight=-0.50,
        params={
            "command_name": COMMAND_NAME,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["wheel_L_Link", "wheel_R_Link"],
                preserve_order=True,
            ),
        },
    )
    wheel_height_difference = RewTerm(
        func=nandu1_v0_1_mdp.adaptive_wheel_height_difference,
        weight=-0.25,
        params={"command_name": COMMAND_NAME, "scale": 0.03},
    )
    knee_spacing_envelope = RewTerm(
        func=nandu1_v0_1_mdp.adaptive_knee_spacing_envelope,
        weight=-0.35,
        params={"command_name": COMMAND_NAME},
    )
    leg_mirror_position = RewTerm(
        func=nandu1_v0_1_mdp.adaptive_leg_mirror_position,
        weight=-0.20,
        params={"command_name": COMMAND_NAME, "beta": 0.05},
    )
    gross_leg_asymmetry = RewTerm(
        func=nandu1_v0_1_mdp.adaptive_gross_leg_asymmetry,
        weight=-0.50,
        params={"command_name": COMMAND_NAME},
    )


@configclass
class NANDU1BlindFlatV01EnvCfg(NANDU1BlindFlatEnvCfg):
    """Training environment; observation/action/network contracts equal v0."""

    commands: NANDU1V01CommandsCfg = NANDU1V01CommandsCfg()
    rewards: NANDU1V01RewardsCfg = NANDU1V01RewardsCfg()


@configclass
class NANDU1BlindFlatV01EnvCfg_PLAY(NANDU1BlindFlatEnvCfg_PLAY):
    """Deterministic v0-1 Play configuration with frozen reward multipliers."""

    commands: NANDU1V01CommandsCfg = NANDU1V01CommandsCfg()
    rewards: NANDU1V01RewardsCfg = NANDU1V01RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.nandu1_velocity.adaptive_constraints_enabled = False
