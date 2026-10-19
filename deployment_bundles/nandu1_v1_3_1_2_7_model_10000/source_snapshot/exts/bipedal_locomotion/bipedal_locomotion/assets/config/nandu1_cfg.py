"""NANDU1 asset configuration.

The complete geometry and kinematic assembly come from NANDU3. Leg and wheel
mass/inertia, physical ranges and actuator parameters are aligned with
WF_TRON2A in the native NANDU3 joint coordinates. The arm uses its own delayed
implicit actuator groups.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from bipedal_locomotion.actuators import DelayedImplicitActuatorCfg


_CURRENT_DIR = os.path.dirname(__file__)
_USD_PATH = os.path.join(_CURRENT_DIR, "../usd/NANDU1/usd/robot.usd")


NANDU1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10000.0,
            max_angular_velocity=10000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # NANDU3's native CAD collision surfaces intentionally overlap at
            # enclosed joints.  Re-enabling all articulation pairs turns
            # those interfaces into multi-kN false contacts at reset.
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Recalibrated on the native NANDU3 geometry.  The forward-biased
        # crouch keeps both wheels grounded and places the full-body COM about
        # 43 mm ahead of the wheel axis, giving the immediate policy more time
        # to arrest the measured reset-time backward fall.
        pos=(0.0, 0.0, 0.65337),
        joint_pos={
            # NANDU3 retains its native mirrored pitch-axis signs.
            "proximal_pitch_L_Joint": -0.475,
            "proximal_pitch_R_Joint": 0.475,
            "proximal_roll_[LR]_Joint": 0.0,
            "proximal_yaw_[LR]_Joint": 0.0,
            "knee_[LR]_Joint": 0.950,
            "wheel_[LR]_Joint": 0.0,
            # The angles sum to zero, so the large up3 plane remains
            # horizontal.  FK gives about 0.886 m height and 0.533 m reach,
            # while doubling the minimum normalized arm-joint margin compared
            # with the previous arm2=-1.99 rad reset.
            "arm1": -0.150,
            "arm2": -1.780,
            "arm3": 1.930,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DelayedImplicitActuatorCfg(
            joint_names_expr=["proximal_pitch_[RL]_Joint", "proximal_roll_[RL]_Joint", "knee_[RL]_Joint"],
            armature=0.161777558,
            effort_limit=140.0,
            effort_limit_sim=140.0,
            velocity_limit=12.57,
            velocity_limit_sim=12.57,
            stiffness=159.67,
            damping=10.16,
            friction=0.0,
            min_delay=0,
            max_delay=2,
        ),
        "base_legs_light": DelayedImplicitActuatorCfg(
            joint_names_expr=["proximal_yaw_[RL]_Joint"],
            armature=0.053923687,
            effort_limit=40.0,
            effort_limit_sim=40.0,
            velocity_limit=14.66,
            velocity_limit_sim=14.66,
            stiffness=53.22,
            damping=3.39,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
        "wheels": DelayedImplicitActuatorCfg(
            joint_names_expr=["wheel_L_Joint", "wheel_R_Joint"],
            effort_limit=20.0,
            effort_limit_sim=20.0,
            velocity_limit=40.0,
            velocity_limit_sim=40.0,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
        "arm1": DelayedImplicitActuatorCfg(
            joint_names_expr=["arm1"],
            armature=0.700,
            effort_limit=181.0,
            effort_limit_sim=181.0,
            velocity_limit=4.14,
            velocity_limit_sim=4.14,
            stiffness=480.0,
            damping=50.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
        "arm2": DelayedImplicitActuatorCfg(
            joint_names_expr=["arm2"],
            armature=0.450,
            effort_limit=136.0,
            effort_limit_sim=136.0,
            velocity_limit=4.14,
            velocity_limit_sim=4.14,
            stiffness=240.0,
            damping=25.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
        "arm3": DelayedImplicitActuatorCfg(
            joint_names_expr=["arm3"],
            armature=0.450,
            effort_limit=136.0,
            effort_limit_sim=136.0,
            velocity_limit=4.14,
            velocity_limit_sim=4.14,
            stiffness=160.0,
            damping=17.0,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
    },
)
