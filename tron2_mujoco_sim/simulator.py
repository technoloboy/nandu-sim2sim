#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

import mujoco
import mujoco.viewer as viewer
try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes


WFYG_JOINT_NAMES = [
    "proximal_pitch_L",
    "proximal_roll_L",
    "proximal_yaw_L",
    "knee_L",
    "wheel_L",
    "proximal_pitch_R",
    "proximal_roll_R",
    "proximal_yaw_R",
    "knee_R",
    "wheel_R",
    "arm1",
    "arm2",
    "arm3",
    "arm4",
    "arm5",
    "arm6",
    "gripper1",
    "gripper2",
]
WFYG_HARDWARE_JOINT_NAMES = [f"{name}_Joint" for name in WFYG_JOINT_NAMES]
BUNDLE_SCHEMA = "limx.wfyg-deployment-bundle/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle_manifest(bundle_dir: Path, asset_root: Path):
    manifest_path = bundle_dir / "manifest.yaml"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read canonical bundle manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"Unsupported bundle schema: {manifest.get('schema')!r}")
    task = manifest.get("task", {})
    if task.get("robot_variant") != "WFYG_TRON2A":
        raise ValueError("The model bundle is not a WFYG_TRON2A policy")
    if task.get("hardware_joint_names") != WFYG_HARDWARE_JOINT_NAMES:
        raise ValueError("Bundle hardware joint order does not match the 18-joint MuJoCo bridge")
    asset = manifest.get("assets", {}).get("files", {}).get("mjcf", {})
    if not asset.get("relative_path") or not asset.get("sha256"):
        raise ValueError("Bundle does not contain an MJCF asset fingerprint")
    model_path = (asset_root / asset["relative_path"]).resolve()
    if not model_path.is_file():
        raise ValueError(f"Bundle MJCF is not installed under asset root: {model_path}")
    actual_hash = sha256_file(model_path)
    if actual_hash != asset["sha256"]:
        raise ValueError(f"Bundle MJCF checksum mismatch: expected {asset['sha256']}, got {actual_hash}")
    default_positions = task.get("default_joint_positions")
    if not isinstance(default_positions, list) or len(default_positions) != 18:
        raise ValueError("Bundle must provide 18 default joint positions")
    if not all(math.isfinite(float(value)) for value in default_positions):
        raise ValueError("Bundle default joint positions contain NaN or Inf")
    return manifest, model_path, [float(value) for value in default_positions]


def load_variant_config(root_dir: str, robot_type: str, model_path: Path | None = None):
    xml_path = os.path.join(root_dir, "robot-description", "tron2", robot_type, "xml", "robot.xml")

    if robot_type == "SF_TRON2A":
        joint_names = [
            "proximal_pitch_L",
            "proximal_roll_L",
            "proximal_yaw_L",
            "knee_L",
            "ankle_pitch_L",
            "proximal_pitch_R",
            "proximal_roll_R",
            "proximal_yaw_R",
            "knee_R",
            "ankle_pitch_R",
        ]
        sensor_joint_names = list(joint_names)
    elif robot_type == "WF_TRON2A":
        joint_names = [
            "proximal_pitch_L",
            "proximal_roll_L",
            "proximal_yaw_L",
            "knee_L",
            "wheel_L",
            "proximal_pitch_R",
            "proximal_roll_R",
            "proximal_yaw_R",
            "knee_R",
            "wheel_R",
        ]
        sensor_joint_names = list(joint_names)
    elif robot_type == "WFYG_TRON2A":
        joint_names = list(WFYG_JOINT_NAMES)
        sensor_joint_names = list(joint_names)
        if model_path is None:
            raise ValueError("WFYG_TRON2A requires --model-bundle; implicit asset selection is forbidden")
        xml_path = str(model_path)
    else:
        raise ValueError(
            f"Unsupported ROBOT_TYPE: {robot_type}. Supported types: SF_TRON2A, WF_TRON2A, WFYG_TRON2A"
        )

    return joint_names, sensor_joint_names, xml_path


class SimulatorMujoco:
    JOY_BTNS = {"R1": (5, 7)}

    def __init__(
        self,
        asset_path,
        joint_names,
        sensor_joint_names,
        robot,
        default_positions=None,
        headless=False,
        duration=0.0,
        command_timeout=0.25,
        runtime_profile="",
    ):
        self.robot = robot
        self.joint_names = joint_names
        self.sensor_joint_names = sensor_joint_names
        self.joint_num = len(joint_names)

        self.mujoco_model = mujoco.MjModel.from_xml_path(asset_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

        self.headless = headless
        self.duration = max(0.0, duration)
        self.command_timeout = max(0.02, command_timeout)
        self.runtime_profile = runtime_profile
        self.viewer = None
        if not self.headless:
            self.viewer = viewer.launch_passive(
                self.mujoco_model,
                self.mujoco_data,
                key_callback=self.key_callback,
                show_left_ui=True,
                show_right_ui=True,
            )
            self.viewer.cam.distance = 10
            self.viewer.cam.elevation = -20

        self.dt = self.mujoco_model.opt.timestep
        self.fps = 1.0 / self.dt
        self.qpos_offset = 7
        self.qvel_offset = 6

        (
            self.sensor_pos_adr,
            self.sensor_vel_adr,
            self.sensor_frc_adr,
        ) = self._resolve_joint_sensor_addrs()

        self.actuator_ids = []
        for name in self.joint_names:
            aid = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                aid = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_ctrl")
            if aid == -1:
                raise RuntimeError(f"Actuator for '{name}' not found in model.")
            self.actuator_ids.append(aid)

        self.default_positions = list(default_positions or [0.0] * self.joint_num)
        if len(self.default_positions) != self.joint_num:
            raise ValueError("Default position count does not match simulator joint count")
        self.joint_qpos_addrs = []
        for sensor_name in self.sensor_joint_names:
            joint_id = mujoco.mj_name2id(
                self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, f"{sensor_name}_Joint"
            )
            if joint_id == -1:
                raise RuntimeError(f"Joint '{sensor_name}_Joint' not found in model")
            self.joint_qpos_addrs.append(int(self.mujoco_model.jnt_qposadr[joint_id]))

        self.imu_quat_adr = self._sensor_adr("base_imu_quat")
        self.imu_gyro_adr = self._sensor_adr("base_imu_gyro")
        self.imu_acc_adr = self._sensor_adr("base_imu_acc")

        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0 for _ in range(self.joint_num)]
        self.robot_cmd.q = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.dq = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.tau = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.Kp = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.Kd = [0.0 for _ in range(self.joint_num)]
        self._set_hold_command(self.default_positions)
        self._last_valid_command_time = None
        self._command_fault = None

        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0.0 for _ in range(self.joint_num)]
        self.robot_state.q = [0.0 for _ in range(self.joint_num)]
        self.robot_state.dq = [0.0 for _ in range(self.joint_num)]

        self.imu_data = datatypes.ImuData()

        self.robot_cmd_callback = partial(self.robot_cmd_cb)
        self.robot.subscribeRobotCmdForSim(self.robot_cmd_callback)
        self.robot.subscribeSensorJoy(self.sensor_joy_cb)
        self._last_r1 = 0
        self._reset_requested = False
        self._pygame_enabled = False
        self._pygame_joystick = None
        self._pygame_last_r1 = 0
        if not self.headless:
            self._setup_pygame_joystick()
        self._reset_sim()

    def _sensor_adr(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid == -1:
            raise RuntimeError(f"Sensor not found: {name}")
        return self.mujoco_model.sensor_adr[sid]

    def _resolve_joint_sensor_addrs(self):
        pos_adr = [self._sensor_adr(f"{n}_pos") for n in self.sensor_joint_names]
        vel_adr = [self._sensor_adr(f"{n}_vel") for n in self.sensor_joint_names]
        frc_adr = [self._sensor_adr(f"{n}_frc") for n in self.sensor_joint_names]
        return pos_adr, vel_adr, frc_adr

    def robot_cmd_cb(self, robot_cmd: datatypes.RobotCmd):
        error = self._validate_robot_cmd(robot_cmd)
        if error is not None:
            if error != self._command_fault:
                print(f"simulator rejected RobotCmd: {error}")
            self._command_fault = error
            self._set_hold_command(self.robot_state.q)
            self._last_valid_command_time = None
            return
        self.robot_cmd = robot_cmd
        self._last_valid_command_time = time.monotonic()
        self._command_fault = None

    def _validate_robot_cmd(self, robot_cmd):
        fields = ("mode", "q", "dq", "tau", "Kp", "Kd")
        for field in fields:
            values = getattr(robot_cmd, field, None)
            if values is None or len(values) != self.joint_num:
                return f"{field} length must be {self.joint_num}"
            if not all(math.isfinite(float(value)) for value in values):
                return f"{field} contains NaN or Inf"
        if any(float(value) < 0.0 or float(value) > 2000.0 for value in robot_cmd.Kp):
            return "Kp must be in [0, 2000]"
        if any(float(value) < 0.0 or float(value) > 200.0 for value in robot_cmd.Kd):
            return "Kd must be in [0, 200]"
        return None

    def _set_hold_command(self, positions):
        positions = list(positions)
        if len(positions) != self.joint_num or not all(math.isfinite(float(value)) for value in positions):
            positions = list(self.default_positions)
        for index, name in enumerate(self.joint_names):
            is_wheel = name.startswith("wheel_")
            is_gripper = name.startswith("gripper")
            is_arm = name.startswith("arm")
            self.robot_cmd.mode[index] = 0
            self.robot_cmd.q[index] = float(positions[index])
            self.robot_cmd.dq[index] = 0.0
            self.robot_cmd.tau[index] = 0.0
            arm_lock = is_arm and self.runtime_profile == "locomotion"
            self.robot_cmd.Kp[index] = 0.0 if is_wheel else (800.0 if is_gripper or arm_lock else (80.0 if is_arm else 159.67))
            self.robot_cmd.Kd[index] = 0.6 if is_wheel else (40.0 if is_gripper or arm_lock else (4.0 if is_arm else 10.16))

    def sensor_joy_cb(self, joy_msg: datatypes.SensorJoy):
        btns = joy_msg.buttons
        r1 = 0
        for btn_idx in self.JOY_BTNS["R1"]:
            if len(btns) > btn_idx and btns[btn_idx]:
                r1 = 1
                break
        if r1 and not self._last_r1:
            self._reset_requested = True
        self._last_r1 = int(r1)

    def _reset_sim(self):
        mujoco.mj_resetData(self.mujoco_model, self.mujoco_data)
        if self.joint_num == 18 and self.mujoco_data.qpos.size >= 3:
            self.mujoco_data.qpos[2] = 0.9
        for address, position in zip(self.joint_qpos_addrs, self.default_positions):
            self.mujoco_data.qpos[address] = position
        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
        self._read_state()
        self._set_hold_command(self.default_positions)
        self._last_valid_command_time = None
        self._command_fault = None

    def _setup_pygame_joystick(self):
        self._pygame_enabled = False
        if not PYGAME_AVAILABLE:
            return
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() <= 0:
                return
            self._pygame_joystick = pygame.joystick.Joystick(0)
            self._pygame_joystick.init()
            self._pygame_enabled = True
            print(f"simulator pygame joystick ready: {self._pygame_joystick.get_name()}")
        except Exception as exc:  # noqa: BLE001
            self._pygame_enabled = False
            self._pygame_joystick = None
            print(f"simulator pygame joystick init failed: {exc}")

    def _poll_pygame_reset(self):
        if not self._pygame_enabled or self._pygame_joystick is None:
            return
        pygame.event.pump()
        joy = self._pygame_joystick
        num_btns = joy.get_numbuttons()
        r1 = 0
        for btn_idx in self.JOY_BTNS["R1"]:
            if num_btns > btn_idx and joy.get_button(btn_idx):
                r1 = 1
                break
        if r1 and not self._pygame_last_r1:
            self._reset_requested = True
        self._pygame_last_r1 = int(r1)

    def key_callback(self, keycode):
        if keycode in (82, 114):
            self._reset_requested = True

    def _read_state(self):
        sensordata = self.mujoco_data.sensordata
        for i in range(self.joint_num):
            self.robot_state.q[i] = float(sensordata[self.sensor_pos_adr[i]])
            self.robot_state.dq[i] = float(sensordata[self.sensor_vel_adr[i]])
            self.robot_state.tau[i] = float(sensordata[self.sensor_frc_adr[i]])

    def _write_ctrl(self):
        for cmd_idx in range(self.joint_num):
            curr_q = self.robot_state.q[cmd_idx]
            curr_dq = self.robot_state.dq[cmd_idx]
            tau = (
                self.robot_cmd.Kp[cmd_idx] * (self.robot_cmd.q[cmd_idx] - curr_q)
                + self.robot_cmd.Kd[cmd_idx] * (self.robot_cmd.dq[cmd_idx] - curr_dq)
                + self.robot_cmd.tau[cmd_idx]
            )
            actuator_id = self.actuator_ids[cmd_idx]
            if not math.isfinite(tau):
                self._command_fault = "computed torque contains NaN or Inf"
                self._set_hold_command(self.robot_state.q)
                tau = 0.0
            if self.mujoco_model.actuator_ctrllimited[actuator_id]:
                lower, upper = self.mujoco_model.actuator_ctrlrange[actuator_id]
                tau = min(float(upper), max(float(lower), tau))
            self.mujoco_data.ctrl[actuator_id] = float(tau)

    def _publish_imu(self):
        sd = self.mujoco_data.sensordata
        self.imu_data.quat[0] = float(sd[self.imu_quat_adr + 0])
        self.imu_data.quat[1] = float(sd[self.imu_quat_adr + 1])
        self.imu_data.quat[2] = float(sd[self.imu_quat_adr + 2])
        self.imu_data.quat[3] = float(sd[self.imu_quat_adr + 3])
        self.imu_data.gyro[0] = float(sd[self.imu_gyro_adr + 0])
        self.imu_data.gyro[1] = float(sd[self.imu_gyro_adr + 1])
        self.imu_data.gyro[2] = float(sd[self.imu_gyro_adr + 2])
        self.imu_data.acc[0] = float(sd[self.imu_acc_adr + 0])
        self.imu_data.acc[1] = float(sd[self.imu_acc_adr + 1])
        self.imu_data.acc[2] = float(sd[self.imu_acc_adr + 2])
        self.imu_data.stamp = time.time_ns()
        self.robot.publishImuDataForSim(self.imu_data)

    def run(self):
        frame_count = 0
        start_time = time.monotonic()
        self.rate = Rate(self.fps)
        while self.viewer is None or self.viewer.is_running():
            now = time.monotonic()
            if self.duration > 0.0 and now - start_time >= self.duration:
                break
            self._poll_pygame_reset()
            if self._reset_requested:
                self._reset_requested = False
                self._reset_sim()
            self._read_state()
            if self._last_valid_command_time is not None and now - self._last_valid_command_time > self.command_timeout:
                if self._command_fault != "command timeout":
                    print("simulator RobotCmd timeout; entering bounded hold")
                self._command_fault = "command timeout"
                self._set_hold_command(self.robot_state.q)
                self._last_valid_command_time = None
            self._write_ctrl()
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

            self.robot_state.stamp = time.time_ns()
            self.robot.publishRobotStateForSim(self.robot_state)
            self._publish_imu()

            if self.viewer is not None and frame_count % 20 == 0:
                self.viewer.sync()
            frame_count += 1
            self.rate.sleep()


def parse_args():
    parser = argparse.ArgumentParser(description="LimX SDK to TRON2 MuJoCo bridge")
    parser.add_argument("legacy_ip", nargs="?", help="Backward-compatible positional robot IP")
    parser.add_argument("--ip", default=None, help="LimX simulator IP (default: 127.0.0.1)")
    parser.add_argument("--model-bundle", type=Path, help="External WFYG or NANDU1 deployment bundle directory")
    parser.add_argument("--task", default=None, help="Canonical NANDU1 task or Play alias")
    parser.add_argument("--aux-port", type=int, default=15012, help="NANDU1 localhost dual-IMU UDP port")
    parser.add_argument("--telemetry-output", type=Path)
    parser.add_argument("--asset-root", type=Path, help="Audited robot-description repository")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="Exit after this many wall-clock seconds; zero is unlimited")
    parser.add_argument("--command-timeout", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model_bundle is not None:
        candidate = json.loads((args.model_bundle / "manifest.yaml").read_text())
        if candidate.get("schema") in ("limx.nandu1-deployment-bundle/v1", "limx.nandu1-adaptive-deployment-bundle/v1"):
            from nandu1_simulator import run_nandu1
            run_nandu1(args)
            sys.exit(0)
    script_dir = Path(__file__).resolve().parent
    default_asset_root = script_dir.parent / "robot-description"
    if not default_asset_root.is_dir():
        default_asset_root = script_dir / "robot-description"
    asset_root = (args.asset_root or default_asset_root).expanduser().resolve()
    manifest = None
    bundle_model_path = None
    default_positions = None
    robot_type = os.getenv("ROBOT_TYPE")
    if args.model_bundle is not None:
        manifest, bundle_model_path, default_positions = load_bundle_manifest(
            args.model_bundle.expanduser().resolve(), asset_root
        )
        if robot_type not in (None, "WFYG_TRON2A"):
            raise ValueError(f"--model-bundle requires ROBOT_TYPE=WFYG_TRON2A, got {robot_type}")
        robot_type = "WFYG_TRON2A"
    if not robot_type:
        print("Error: set ROBOT_TYPE to SF_TRON2A/WF_TRON2A, or pass --model-bundle for WFYG_TRON2A")
        sys.exit(1)
    os.environ["ROBOT_TYPE"] = robot_type

    robot = Robot(RobotType.Tron2, True)
    robot_ip = args.ip or args.legacy_ip or "127.0.0.1"
    if not robot.init(robot_ip):
        sys.exit(1)

    joint_names, sensor_joint_names, model_path = load_variant_config(
        str(script_dir), robot_type, bundle_model_path
    )
    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)

    print(f"*** Model File Loaded: {model_path} ***")
    if manifest is not None:
        print(
            f"*** Bundle task={manifest['task']['task_id']} asset={manifest['assets']['profile']} "
            f"joints={len(joint_names)} ***"
        )
    simulator = SimulatorMujoco(
        model_path,
        joint_names,
        sensor_joint_names,
        robot,
        default_positions=default_positions,
        headless=args.headless,
        duration=args.duration,
        command_timeout=args.command_timeout,
        runtime_profile=manifest["task"]["runtime_profile"] if manifest is not None else "",
    )
    simulator.run()
