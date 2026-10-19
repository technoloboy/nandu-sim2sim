"""NANDU1 v1: body-frame velocity and absolute up3-height tracking."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.tasks.locomotion import mdp

from . import nandu1_mdp as v0_mdp
from . import nandu1_v1_mdp as v1_mdp
from . import nandu1_v1_robust_mdp as robust
from .nandu1_env_cfg import (
    ALL_JOINTS,
    ARM_JOINTS,
    POSITION_JOINTS,
    NANDU1ActionsCfg,
    NANDU1BlindFlatEnvCfg,
    NANDU1EventsCfg,
    NANDU1SceneCfg,
    NANDU1TerminationsCfg,
)


COMMAND_NAME = v1_mdp.COMMAND_NAME


@configclass
class NANDU1V1CommandsCfg:
    nandu1_height = v1_mdp.Nandu1HeightCommandCfg(
        class_type=v1_mdp.Nandu1HeightCommand,
        resampling_time_range=(4.0, 8.0),
        debug_vis=False,
        ranges=v1_mdp.Nandu1HeightCommandCfg.Ranges(
            yaw_rate=(-1.57, 1.57),
            ee_vx=(-1.0, 1.0),
            up3_height=(0.40, 1.40),
        ),
        acceleration_limits=(2.0, 0.8, 0.5),
        nominal_height_range=(0.40, 1.40),
        height_taper_buffer=0.10,
        vertical_pulse_time_range=(0.0, 0.0),
        vertical_pulse_safety_fraction=0.0,
        evaluation_steps=6000,
        max_reach_resamples=8,
        reach_resample_limit=0.66,
        robustness_enabled=True,
        measurement_command_name=COMMAND_NAME,
        checkpoint_schema="nandu1_height_v1_robust_v2",
    )


@configclass
class NANDU1V1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=robust.noisy_observation,
            params={
                "sensor_cfg": SceneEntityCfg("base_imu"),
                "command_name": COMMAND_NAME,
                "kind": "gyro",
            },
            scale=0.25,
        )
        base_projected_gravity = ObsTerm(
            func=robust.noisy_observation,
            params={"sensor_cfg": SceneEntityCfg("base_imu"), "command_name": COMMAND_NAME, "kind": "gravity"},
        )
        up3_ang_vel = ObsTerm(
            func=robust.noisy_observation,
            params={
                "sensor_cfg": SceneEntityCfg("up3_imu"),
                "command_name": COMMAND_NAME,
                "kind": "gyro",
            },
            scale=0.25,
        )
        up3_projected_gravity = ObsTerm(
            func=robust.noisy_observation,
            params={"sensor_cfg": SceneEntityCfg("up3_imu"), "command_name": COMMAND_NAME, "kind": "gravity"},
        )
        joint_pos = ObsTerm(
            func=robust.noisy_observation,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=POSITION_JOINTS, preserve_order=True),
                "command_name": COMMAND_NAME,
                "kind": "joint_pos",
            },
        )
        joint_vel = ObsTerm(
            func=robust.noisy_observation,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True),
                "command_name": COMMAND_NAME,
                "kind": "joint_vel",
            },
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HistoryCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel_body = ObsTerm(func=v1_mdp.base_body_linear_velocity)
        up3_lin_vel_body = ObsTerm(
            func=v1_mdp.body_linear_velocity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="up3")},
        )
        # Encoder supervision prefix: base velocity 3D, up3 velocity 3D,
        # then true up3 ground height 1D. This value is never a Policy term.
        up3_height = ObsTerm(func=v1_mdp.true_up3_height)
        base_ang_vel = ObsTerm(
            func=v0_mdp.imu_angular_velocity,
            params={"sensor_cfg": SceneEntityCfg("base_imu")},
            scale=0.25,
        )
        base_projected_gravity = ObsTerm(
            func=v0_mdp.imu_projected_gravity,
            params={"sensor_cfg": SceneEntityCfg("base_imu")},
        )
        up3_ang_vel = ObsTerm(
            func=v0_mdp.imu_angular_velocity,
            params={"sensor_cfg": SceneEntityCfg("up3_imu")},
            scale=0.25,
        )
        up3_projected_gravity = ObsTerm(
            func=v0_mdp.imu_projected_gravity,
            params={"sensor_cfg": SceneEntityCfg("up3_imu")},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=POSITION_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action)
        applied_torque = ObsTerm(func=mdp.robot_joint_torque, scale=0.05)
        joint_acceleration = ObsTerm(func=mdp.robot_joint_acc, scale=0.0025)
        wheel_contact_force = ObsTerm(
            func=mdp.robot_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*")},
        )
        wheel_body_velocity = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*")},
        )
        rigid_body_mass = ObsTerm(func=mdp.robot_mass)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CommandsCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": COMMAND_NAME})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    obsHistory: HistoryCfg = HistoryCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsCfg = CommandsCfg()


@configclass
class NANDU1V1EventsCfg(NANDU1EventsCfg):
    push_robot = None
    up3_wrench = None
    stage_domain_randomization = EventTerm(
        func=robust.HeightRobustDomainRandomization,
        mode="reset",
        params={"command_name": COMMAND_NAME, "asset_cfg": SceneEntityCfg("robot"), "material_buckets_per_stage": 48},
    )
    reset_base = EventTerm(
        func=robust.HeightRobustReset,
        mode="reset",
        params={"command_name": COMMAND_NAME},
    )
    reset_joints = None
    robust_disturbance = EventTerm(
        func=robust.HeightRobustDisturbance,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"command_name": COMMAND_NAME},
    )

    def __post_init__(self):
        self.stage_domain_randomization.params["command_name"] = COMMAND_NAME
        self.reset_base.params["command_name"] = COMMAND_NAME
        self.robust_disturbance.params["command_name"] = COMMAND_NAME


@configclass
class NANDU1V1RewardsCfg:
    yaw_tracking = RewTerm(
        func=v0_mdp.yaw_rate_tracking,
        weight=2.0,
        params={"command_name": COMMAND_NAME, "std": 0.25},
    )
    yaw_huber = RewTerm(
        func=v1_mdp.yaw_huber,
        weight=-0.25,
        params={"command_name": COMMAND_NAME, "scale": 0.35, "cap": 8.0},
    )
    ee_vx_tracking = RewTerm(
        func=v1_mdp.body_vx_tracking_exp,
        weight=3.0,
        params={"command_name": COMMAND_NAME, "std": 0.12},
    )
    ee_vx_huber = RewTerm(
        func=v1_mdp.body_vx_huber,
        weight=-0.40,
        params={"command_name": COMMAND_NAME, "scale": 0.20, "cap": 8.0},
    )
    height_tracking_exp = RewTerm(
        func=v1_mdp.height_tracking_exp,
        weight=4.0,
        params={"command_name": COMMAND_NAME},
    )
    height_huber = RewTerm(
        func=v1_mdp.height_huber,
        weight=-0.60,
        params={"command_name": COMMAND_NAME, "scale": 0.10, "cap": 12.0},
    )
    height_velocity_damping = RewTerm(
        func=v1_mdp.height_velocity_damping,
        weight=-0.25,
        params={"command_name": COMMAND_NAME},
    )
    up3_horizontal = RewTerm(func=v0_mdp.up3_horizontal, weight=3.0, params={"std": 0.12})
    up3_tilt_huber = RewTerm(func=v1_mdp.up3_tilt_huber, weight=-1.0)
    up3_angular_velocity = RewTerm(func=v0_mdp.up3_roll_pitch_angular_velocity, weight=-0.10)
    height_envelope = RewTerm(func=v1_mdp.height_envelope, weight=-0.50)
    horizontal_reach_barrier = RewTerm(
        func=v0_mdp.horizontal_reach_barrier,
        weight=-0.20,
        params={"command_name": COMMAND_NAME},
    )
    vertical_allocation_tracking = RewTerm(
        func=v1_mdp.vertical_allocation_tracking,
        weight=0.60,
        params={"command_name": COMMAND_NAME, "base_std": 0.06, "arm_std": 0.06},
    )
    opposite_vertical_cancellation = RewTerm(
        func=v1_mdp.opposite_vertical_cancellation,
        weight=-0.15,
        params={"command_name": COMMAND_NAME, "scale": 0.05},
    )
    height_directional_reserve = RewTerm(
        func=v1_mdp.height_directional_reserve,
        weight=-0.15,
        params={"command_name": COMMAND_NAME, "minimum": 0.15},
    )
    com_to_wheel_support = RewTerm(
        func=v0_mdp.com_to_wheel_support,
        weight=-0.20,
        params={"command_name": COMMAND_NAME, "scale": 0.08},
    )
    base_roll_stability = RewTerm(func=v0_mdp.base_roll_stability, weight=1.0, params={"std": 0.12})
    base_pitch_envelope = RewTerm(
        func=v0_mdp.base_pitch_soft_envelope,
        weight=-2.0,
        params={"command_name": COMMAND_NAME},
    )
    horizontal_allocation = RewTerm(
        func=v0_mdp.horizontal_allocation_tracking,
        weight=0.50,
        params={"command_name": COMMAND_NAME, "base_std": 0.15, "arm_std": 0.12, "tracking_std": 0.12},
    )
    alive = RewTerm(func=mdp.stay_alive, weight=0.50)
    outward_reach_velocity = RewTerm(
        func=v0_mdp.outward_reach_velocity,
        weight=-0.20,
        params={"command_name": COMMAND_NAME},
    )
    arm_default = RewTerm(
        func=v1_mdp.arm_default_deviation_gated,
        weight=-0.05,
        params={"command_name": COMMAND_NAME, "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    static_body_motion = RewTerm(
        func=v1_mdp.static_body_motion,
        weight=-0.30,
        params={"command_name": COMMAND_NAME},
    )
    static_orientation_hold = RewTerm(
        func=v1_mdp.static_orientation_hold,
        weight=-0.25,
        params={"command_name": COMMAND_NAME},
    )
    wheel_slip = RewTerm(
        func=v0_mdp.wheel_lateral_slip,
        weight=-0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
        },
    )
    wheel_load_imbalance = RewTerm(
        func=v0_mdp.wheel_load_imbalance,
        weight=-0.35,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*")},
    )
    wheel_spacing_envelope = RewTerm(
        func=v0_mdp.wheel_spacing_envelope,
        weight=-0.50,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wheel_L_Link", "wheel_R_Link"], preserve_order=True)},
    )
    wheel_fore_aft_alignment = RewTerm(
        func=v0_mdp.wheel_fore_aft_alignment,
        weight=-0.35,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wheel_L_Link", "wheel_R_Link"], preserve_order=True)},
    )
    knee_spacing_envelope = RewTerm(
        func=v0_mdp.knee_spacing_envelope,
        weight=-0.35,
        params={"command_name": COMMAND_NAME},
    )
    leg_mirror_position = RewTerm(
        func=v0_mdp.leg_mirror_position,
        weight=-0.20,
        params={"command_name": COMMAND_NAME, "beta": 0.05},
    )
    gross_leg_asymmetry = RewTerm(
        func=v0_mdp.gross_leg_asymmetry,
        weight=-0.50,
        params={"command_name": COMMAND_NAME},
    )
    joint_soft_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.50,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POSITION_JOINTS)},
    )
    torque_utilization = RewTerm(func=v0_mdp.normalized_torque_utilization, weight=-0.10)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
    joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-8)
    mechanical_power = RewTerm(func=v0_mdp.mechanical_power, weight=-1.0e-4)


@configclass
class NANDU1V1CurriculumCfg:
    performance = CurrTerm(func=v1_mdp.performance_curriculum, params={"command_name": COMMAND_NAME})


@configclass
class NANDU1BlindFlatV1EnvCfg(NANDU1BlindFlatEnvCfg):
    scene: NANDU1SceneCfg = NANDU1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: NANDU1V1ObservationsCfg = NANDU1V1ObservationsCfg()
    actions: NANDU1ActionsCfg = NANDU1ActionsCfg()
    commands: NANDU1V1CommandsCfg = NANDU1V1CommandsCfg()
    rewards: NANDU1V1RewardsCfg = NANDU1V1RewardsCfg()
    terminations: NANDU1TerminationsCfg = NANDU1TerminationsCfg()
    events: NANDU1V1EventsCfg = NANDU1V1EventsCfg()
    curriculum: NANDU1V1CurriculumCfg = NANDU1V1CurriculumCfg()


@configclass
class NANDU1BlindFlatV1EnvCfg_PLAY(NANDU1BlindFlatV1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        self.events.stage_domain_randomization = None
        self.events.up3_wrench = None
        self.events.robust_disturbance = None
        self.commands.nandu1_height.robustness_enabled = False
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = 0
            actuator.max_delay = 0
