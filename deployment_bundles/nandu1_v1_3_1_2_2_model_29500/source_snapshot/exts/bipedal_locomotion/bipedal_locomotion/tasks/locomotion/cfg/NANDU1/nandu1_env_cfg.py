"""Independent manager-based RL environment for NANDU1 blind-flat control."""

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from bipedal_locomotion.assets.config.nandu1_cfg import NANDU1_CFG
from bipedal_locomotion.tasks.locomotion import mdp

from . import nandu1_mdp


COMMAND_NAME = "nandu1_velocity"

LEG_POSITION_JOINTS = [
    "proximal_pitch_L_Joint",
    "proximal_roll_L_Joint",
    "proximal_yaw_L_Joint",
    "knee_L_Joint",
    "proximal_pitch_R_Joint",
    "proximal_roll_R_Joint",
    "proximal_yaw_R_Joint",
    "knee_R_Joint",
]
ARM_JOINTS = ["arm1", "arm2", "arm3"]
POSITION_JOINTS = LEG_POSITION_JOINTS + ARM_JOINTS
ALL_JOINTS = LEG_POSITION_JOINTS[:4] + ["wheel_L_Joint"] + LEG_POSITION_JOINTS[4:] + ["wheel_R_Joint"] + ARM_JOINTS
WHEEL_JOINTS = ["wheel_L_Joint", "wheel_R_Joint"]
HIP_LATERAL_JOINTS = [
    "proximal_roll_L_Joint",
    "proximal_yaw_L_Joint",
    "proximal_roll_R_Joint",
    "proximal_yaw_R_Joint",
]


@configclass
class NANDU1SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )
    robot: ArticulationCfg = NANDU1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    base_imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        update_period=0.02,
        debug_vis=False,
    )
    up3_imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/up3",
        update_period=0.02,
        debug_vis=False,
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=4,
        track_air_time=True,
    )


@configclass
class NANDU1CommandsCfg:
    nandu1_velocity = nandu1_mdp.Nandu1CommandCfg(
        class_type=nandu1_mdp.Nandu1Command,
        resampling_time_range=(4.0, 8.0),
        debug_vis=False,
        ranges=nandu1_mdp.Nandu1CommandCfg.Ranges(
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
    )


@configclass
class NANDU1ActionsCfg:
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=POSITION_JOINTS,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    wheel_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINTS,
        scale=1.0,
        preserve_order=True,
    )


@configclass
class NANDU1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=nandu1_mdp.stage_noisy_imu_angular_velocity,
            params={
                "sensor_cfg": SceneEntityCfg("base_imu"),
                "command_name": COMMAND_NAME,
                "final_noise": 0.005,
            },
            scale=0.25,
        )
        base_projected_gravity = ObsTerm(
            func=nandu1_mdp.imu_projected_gravity,
            params={"sensor_cfg": SceneEntityCfg("base_imu")},
        )
        up3_ang_vel = ObsTerm(
            func=nandu1_mdp.stage_noisy_imu_angular_velocity,
            params={
                "sensor_cfg": SceneEntityCfg("up3_imu"),
                "command_name": COMMAND_NAME,
                "final_noise": 0.008,
            },
            scale=0.25,
        )
        up3_projected_gravity = ObsTerm(
            func=nandu1_mdp.imu_projected_gravity,
            params={"sensor_cfg": SceneEntityCfg("up3_imu")},
        )
        joint_pos = ObsTerm(
            func=nandu1_mdp.stage_noisy_joint_position,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=POSITION_JOINTS, preserve_order=True),
                "command_name": COMMAND_NAME,
                "noise": 0.002,
            },
        )
        joint_vel = ObsTerm(
            func=nandu1_mdp.stage_noisy_joint_velocity,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True),
                "command_name": COMMAND_NAME,
                "noise": 0.02,
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
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel_h = ObsTerm(func=nandu1_mdp.base_heading_linear_velocity)
        up3_lin_vel_h = ObsTerm(
            func=nandu1_mdp.body_heading_linear_velocity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="up3")},
        )
        base_ang_vel = ObsTerm(
            func=nandu1_mdp.imu_angular_velocity,
            params={"sensor_cfg": SceneEntityCfg("base_imu")},
            scale=0.25,
        )
        base_projected_gravity = ObsTerm(
            func=nandu1_mdp.imu_projected_gravity,
            params={"sensor_cfg": SceneEntityCfg("base_imu")},
        )
        up3_ang_vel = ObsTerm(
            func=nandu1_mdp.imu_angular_velocity,
            params={"sensor_cfg": SceneEntityCfg("up3_imu")},
            scale=0.25,
        )
        up3_projected_gravity = ObsTerm(
            func=nandu1_mdp.imu_projected_gravity,
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
class NANDU1EventsCfg:
    prepare = EventTerm(
        func=mdp.prepare_quantity_for_tron,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "foot_radius": 0.10},
    )
    stage_domain_randomization = EventTerm(
        func=nandu1_mdp.StageDomainRandomization,
        mode="reset",
        params={
            "command_name": COMMAND_NAME,
            "asset_cfg": SceneEntityCfg("robot"),
            "material_buckets_per_stage": 48,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.02, 0.02),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.015, 0.015), "velocity_range": (-0.04, 0.04)},
    )
    push_robot = EventTerm(
        func=nandu1_mdp.stage_gated_push,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "command_name": COMMAND_NAME,
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        },
    )


@configclass
class NANDU1RewardsCfg:
    """Ablation-selected v0 reward set used by variants C/D.

    Keep the five redundant regularizers at their reduced weights here.  The
    A/B ablation environments restore the historical stronger weights only
    through ``_restore_pre_stability_reward_weights`` below.
    """

    yaw_tracking = RewTerm(
        func=nandu1_mdp.yaw_rate_tracking,
        weight=2.0,
        params={"command_name": COMMAND_NAME, "std": 0.25},
    )
    ee_vx_coarse = RewTerm(
        func=nandu1_mdp.ee_velocity_tracking,
        weight=2.5,
        params={"command_name": COMMAND_NAME, "axis": 0, "command_index": 1, "std": 0.12},
    )
    ee_vx_fine = RewTerm(
        func=nandu1_mdp.ee_velocity_tracking,
        weight=0.5,
        params={"command_name": COMMAND_NAME, "axis": 0, "command_index": 1, "std": 0.04},
    )
    ee_vz_coarse = RewTerm(
        func=nandu1_mdp.ee_velocity_tracking,
        weight=2.5,
        params={"command_name": COMMAND_NAME, "axis": 2, "command_index": 2, "std": 0.08},
    )
    ee_vz_fine = RewTerm(
        func=nandu1_mdp.ee_velocity_tracking,
        weight=0.5,
        params={"command_name": COMMAND_NAME, "axis": 2, "command_index": 2, "std": 0.03},
    )
    up3_horizontal = RewTerm(func=nandu1_mdp.up3_horizontal, weight=3.0, params={"std": 0.12})
    arm3_height_envelope = RewTerm(func=nandu1_mdp.arm3_height_envelope, weight=-0.50)
    horizontal_reach_barrier = RewTerm(
        func=nandu1_mdp.horizontal_reach_barrier,
        weight=-0.20,
        params={"command_name": COMMAND_NAME},
    )
    vertical_allocation_tracking = RewTerm(
        func=nandu1_mdp.vertical_allocation_tracking,
        weight=0.75,
        params={"command_name": COMMAND_NAME, "base_std": 0.06, "arm_std": 0.06},
    )
    opposite_vertical_cancellation = RewTerm(
        func=nandu1_mdp.opposite_vertical_cancellation,
        weight=-0.10,
        params={"command_name": COMMAND_NAME, "scale": 0.05},
    )
    height_directional_reserve = RewTerm(
        func=nandu1_mdp.height_directional_reserve,
        weight=-0.20,
        params={"command_name": COMMAND_NAME, "minimum": 0.15},
    )
    com_to_wheel_support = RewTerm(
        func=nandu1_mdp.com_to_wheel_support,
        weight=-0.25,
        params={"command_name": COMMAND_NAME, "scale": 0.08},
    )
    base_roll_stability = RewTerm(func=nandu1_mdp.base_roll_stability, weight=1.0, params={"std": 0.12})
    alive = RewTerm(func=mdp.stay_alive, weight=0.5)
    base_pitch_envelope = RewTerm(
        func=nandu1_mdp.base_pitch_soft_envelope,
        weight=-2.0,
        params={"command_name": COMMAND_NAME},
    )
    up3_angular_velocity = RewTerm(func=nandu1_mdp.up3_roll_pitch_angular_velocity, weight=-0.10)
    horizontal_allocation_tracking = RewTerm(
        func=nandu1_mdp.horizontal_allocation_tracking,
        weight=0.5,
        params={
            "command_name": COMMAND_NAME,
            "base_std": 0.15,
            "arm_std": 0.12,
            "tracking_std": 0.12,
        },
    )
    outward_reach_velocity = RewTerm(
        func=nandu1_mdp.outward_reach_velocity,
        weight=-0.20,
        params={"command_name": COMMAND_NAME},
    )
    arm_default = RewTerm(
        func=nandu1_mdp.arm_default_deviation_gated,
        weight=-0.05,
        params={"command_name": COMMAND_NAME, "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    wheel_slip = RewTerm(
        func=nandu1_mdp.wheel_lateral_slip,
        weight=-0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
        },
    )
    wheel_load_imbalance = RewTerm(
        func=nandu1_mdp.wheel_load_imbalance,
        weight=-0.35,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*")},
    )
    wheel_spacing_envelope = RewTerm(
        func=nandu1_mdp.wheel_spacing_envelope,
        weight=-0.50,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wheel_L_Link", "wheel_R_Link"], preserve_order=True)},
    )
    wheel_fore_aft_alignment = RewTerm(
        func=nandu1_mdp.wheel_fore_aft_alignment,
        weight=-0.35,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wheel_L_Link", "wheel_R_Link"], preserve_order=True)},
    )
    wheel_spacing_centering = RewTerm(
        func=nandu1_mdp.wheel_spacing_centering,
        weight=-0.05,
        params={"command_name": COMMAND_NAME, "target": 0.454, "scale": 0.04},
    )
    wheel_height_difference = RewTerm(
        func=nandu1_mdp.wheel_height_difference,
        weight=-0.25,
        params={"command_name": COMMAND_NAME, "scale": 0.03},
    )
    knee_spacing_envelope = RewTerm(
        func=nandu1_mdp.knee_spacing_envelope,
        weight=-0.35,
        params={"command_name": COMMAND_NAME},
    )
    knee_spacing_centering = RewTerm(
        func=nandu1_mdp.knee_spacing_centering,
        weight=-0.05,
        params={"command_name": COMMAND_NAME},
    )
    hip_lateral_default = RewTerm(
        func=mdp.joint_deviation_from_default_l2,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_LATERAL_JOINTS, preserve_order=True)},
    )
    joint_soft_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.50,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POSITION_JOINTS)},
    )
    torque_utilization = RewTerm(func=nandu1_mdp.normalized_torque_utilization, weight=-0.10)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
    joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-8)
    mechanical_power = RewTerm(func=nandu1_mdp.mechanical_power, weight=-1.0e-4)
    leg_mirror_position = RewTerm(
        func=nandu1_mdp.leg_mirror_position,
        weight=-0.20,
        params={"command_name": COMMAND_NAME, "beta": 0.05},
    )
    leg_mirror_velocity = RewTerm(
        func=nandu1_mdp.leg_mirror_velocity,
        weight=-0.05,
        params={"command_name": COMMAND_NAME, "beta": 0.10},
    )
    gross_leg_asymmetry = RewTerm(
        func=nandu1_mdp.gross_leg_asymmetry,
        weight=-0.50,
        params={"command_name": COMMAND_NAME},
    )


@configclass
class NANDU1TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    non_finite = DoneTerm(func=nandu1_mdp.non_finite_state)
    forbidden_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_Link", "proximal_.*", "knee_.*", "up1", "up2", "up3"],
            ),
            "threshold": 1.0,
        },
    )
    catastrophic_tilt = DoneTerm(
        func=nandu1_mdp.SustainedBadOrientation,
        params={
            "limit_angle": math.radians(65.0),
            "duration_s": 0.15,
            "sensor_cfg": SceneEntityCfg("base_imu"),
        },
    )
@configclass
class NANDU1CurriculumCfg:
    performance = CurrTerm(
        func=nandu1_mdp.performance_curriculum,
        params={"command_name": COMMAND_NAME},
    )


@configclass
class NANDU1BlindFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: NANDU1SceneCfg = NANDU1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: NANDU1ObservationsCfg = NANDU1ObservationsCfg()
    actions: NANDU1ActionsCfg = NANDU1ActionsCfg()
    commands: NANDU1CommandsCfg = NANDU1CommandsCfg()
    rewards: NANDU1RewardsCfg = NANDU1RewardsCfg()
    terminations: NANDU1TerminationsCfg = NANDU1TerminationsCfg()
    events: NANDU1EventsCfg = NANDU1EventsCfg()
    curriculum: NANDU1CurriculumCfg = NANDU1CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.base_imu.update_period = self.decimation * self.sim.dt
        self.scene.up3_imu.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.origin_type = "env"


def _restore_pre_stability_reward_weights(cfg: NANDU1BlindFlatEnvCfg) -> None:
    """Restore only the five reward weights varied by the reward ablation."""
    cfg.rewards.opposite_vertical_cancellation.weight = -0.25
    cfg.rewards.wheel_spacing_centering.weight = -0.10
    cfg.rewards.knee_spacing_centering.weight = -0.10
    cfg.rewards.hip_lateral_default.weight = -0.10
    cfg.rewards.joint_acceleration.weight = -2.5e-7


@configclass
class NANDU1BlindFlatAEnvCfg(NANDU1BlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _restore_pre_stability_reward_weights(self)


@configclass
class NANDU1BlindFlatBEnvCfg(NANDU1BlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _restore_pre_stability_reward_weights(self)


@configclass
class NANDU1BlindFlatCEnvCfg(NANDU1BlindFlatEnvCfg):
    """Old PPO with the stability-tuned reward shaping."""


@configclass
class NANDU1BlindFlatDEnvCfg(NANDU1BlindFlatEnvCfg):
    """Stabilized PPO with the stability-tuned reward shaping."""


@configclass
class NANDU1BlindFlatEnvCfg_PLAY(NANDU1BlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.obsHistory.enable_corruption = False
        self.events.stage_domain_randomization = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
