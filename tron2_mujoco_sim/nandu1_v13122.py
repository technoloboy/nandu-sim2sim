"""Independent, sensor-only v13122 inference and local MuJoCo closed loop.

The bundle contains the exact training math source and scripted networks.
This module never imports Isaac, ROS, SDK transport, or a hardware driver.
Truth is exposed only by the physics evaluator, separately from policy49.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import nandu1_contract as c

TASK = "Isaac-NANDU1-Blind-Flat-v1-3-1-2-2"
SCHEMA = "nandu_network_v1_3_1_2_2_E_adaptive_balance_r1"
CONTROL_DT = .02
PHYSICS_DT = .005
TCP_LOCAL = (-.066473051764, .161623719479, -.000621484902)


def _source_module(path):
    spec = importlib.util.spec_from_file_location("nandu_v13122_bundle_math_" + hashlib.sha256(str(path).encode()).hexdigest()[:12], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdaptiveRuntime:
    """Explicit hidden64, eta8, memory2, command3 and previous issued13 state.

    ``step`` receives policy49 sensor data and a raw yaw/vx/height intent. The
    final13 observation channels are replaced with the previous issued action,
    so a caller cannot accidentally feed delayed motor actions into Student.
    Initial dynamic state is zero, matching the training runner's first actor
    call. Subsequent sensor frames advance Student/memory exactly once.
    """
    def __init__(self, bundle, *, initial_height, stage=None, device="cpu"):
        self.bundle = Path(bundle).resolve()
        self.metadata = json.loads((self.bundle / "deployment_metadata.json").read_text())
        if self.metadata.get("task") != TASK or self.metadata.get("checkpoint_schema") != SCHEMA:
            raise ValueError("v13122 task/checkpoint schema required")
        expected = dict(policy=49, history=245, encoder=7, commands=3, actor_raw=70, actions=13)
        if any(self.metadata.get("interface", {}).get(key) != value for key, value in expected.items()):
            raise ValueError("v13122 interface dimensions do not match")
        if self.metadata.get("control_dt") != CONTROL_DT:
            raise ValueError("v13122 runtime requires 50Hz control")
        for name, digest in self.metadata.get("file_sha256", {}).items():
            if Path(name).name != name:
                raise ValueError("bundle hash entry must be a local filename")
            path = self.bundle / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"bundle hash mismatch: {name}")
        self.stage = self.metadata.get("curriculum_stage", 4) if stage is None else stage
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or not 0 <= self.stage <= 4:
            raise ValueError("stage must be an integer from0 through4")
        self.device = torch.device(device)
        self.math = _source_module(self.bundle / self.metadata.get("runtime_math", "adaptive_math.py"))
        self.encoder = torch.jit.load(str(self.bundle / "encoder.pt"), map_location=self.device).eval()
        self.policy = torch.jit.load(str(self.bundle / "policy.pt"), map_location=self.device).eval()
        self.context = torch.jit.load(str(self.bundle / "context_step.pt"), map_location=self.device).eval()
        self.processor = self.math.CausalCommandProcessor(1, self.device)
        self.memory = self.math.SlowErrorMemory(1, self.device)
        self.history = c.History()
        self.action_clip = self.metadata.get("clip_actions")
        if self.action_clip is not None and (not np.isfinite(self.action_clip) or self.action_clip <= 0):
            raise ValueError("clip_actions must be positive when configured")
        self.reset(initial_height=initial_height)

    def reset(self, *, initial_height):
        if not np.isfinite(initial_height):
            raise ValueError("initial_height must be finite")
        self.processor.reset(height=float(initial_height))
        self.memory.reset()
        self.history.reset()
        self.hidden = torch.zeros(1, 64, device=self.device)
        self.eta = torch.zeros(1, 8, device=self.device)
        self.previous_issued_action = torch.zeros(1, 13, device=self.device)
        self._known_clipping = False
        self._steps = 0
        self._last_step_id = None
        self._cached = None

    @torch.inference_mode()
    def step(self, observation, raw_intent, *, step_id=None):
        if step_id is not None and self._last_step_id is not None:
            if step_id < self._last_step_id:
                raise ValueError("step_id cannot move backwards")
            if step_id == self._last_step_id:
                return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self._cached.items()}
        obs_array = c.finite(observation, 49).astype(np.float32)
        goal_array = c.finite(raw_intent, 3).astype(np.float32)
        obs = torch.as_tensor(obs_array, device=self.device).reshape(1, 49).clone()
        obs[:, 36:] = self.previous_issued_action
        intent = torch.as_tensor(goal_array, device=self.device).reshape(1, 3)
        # The same sensor-only stage command bounds as training.
        yaw_limit, vx_limit = c.STAGE_LIMITS[self.stage]
        intent[:, 0].clamp_(-yaw_limit, yaw_limit)
        intent[:, 1].clamp_(-vx_limit, vx_limit)
        low, high = c.STAGE_HEIGHT_RANGES[self.stage]
        intent[:, 2].clamp_(low, high)
        command = self.processor.step(intent, CONTROL_DT, step_id=step_id,
            gravity_b=obs[:, 3:6], gyro_b=obs[:, :3] / .25, stage=self.stage,
            yaw_limit=yaw_limit, height_rate_limit=self.math.HEIGHT_RATES[self.stage])
        history = torch.as_tensor(self.history.append(obs[0].cpu().numpy()), device=self.device).reshape(1, 245)
        encoded = self.encoder(history)
        memory = self.memory.normalized()
        if self._steps:
            self.hidden, self.eta, _ = self.context(obs, self.previous_issued_action, self.hidden, self.eta)
            gravity = obs[:, 3:6]
            norm = gravity.norm(dim=-1, keepdim=True)
            up = -gravity / norm.clamp_min(1e-6)
            world_yaw = (obs[:, :3] / .25 * up).sum(-1)
            memory = self.memory.step_errors(command[:, 1] - encoded[:, 3], command[:, 0] - world_yaw,
                intent, self.processor.height_rate, valid=(norm[:, 0] > 1e-6) & encoded.isfinite().all(-1),
                tilt=up[:, 2].clamp(-1., 1.).acos(), clipped=self._known_clipping, dt=CONTROL_DT, step_id=step_id)
        dynamic = torch.cat((memory, self.eta, self.processor.height_rate[:, None] / .5), -1)
        # Training's first actor call has no previous successful physical step.
        # Its entire dynamic11 snapshot is zero, including the hdot feature.
        if not self._steps:
            dynamic.zero_()
        raw = torch.cat((encoded, obs, command, dynamic), -1)
        action = self.policy(raw)
        if action.shape != (1, 13) or not action.isfinite().all():
            raise ValueError("actor returned nonfinite or wrong-shape actions")
        issued = action if self.action_clip is None else action.clamp(-self.action_clip, self.action_clip)
        self._known_clipping = bool((issued != action).any())
        self.previous_issued_action = issued.detach().clone()
        numpy = lambda value: value.detach().cpu().numpy()[0].copy()
        self._cached = dict(action=numpy(issued), raw_action=numpy(action), actor_input=numpy(raw),
                            observation=numpy(obs), history=numpy(history), encoded=numpy(encoded), commands=numpy(command),
                            memory=numpy(memory), hidden=numpy(self.hidden), eta=numpy(self.eta),
                            height_rate=float(self.processor.height_rate[0]), known_clipping=self._known_clipping)
        self._steps += 1
        self._last_step_id = step_id
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in self._cached.items()}

    def state_dict(self):
        return dict(hidden=self.hidden.detach().cpu().numpy().copy(), eta=self.eta.detach().cpu().numpy().copy(),
                    previous_issued_action=self.previous_issued_action.detach().cpu().numpy().copy(),
                    command=self.processor.state_dict(), memory=self.memory.state_dict(), steps=self._steps,
                    last_step_id=self._last_step_id)


class MujocoPhysics:
    """Local explicit XML evaluator with nominal E PD and fixed original TCP."""
    def __init__(self, model_xml, *, up3_total_mass=.67148, rear_mass=0., rear_x=-.275):
        import mujoco
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_xml).resolve()))
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        self.jids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in c.JOINTS])
        self.aids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in c.JOINTS])
        self.bids = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ("base_Link", "up3")}
        if self.model.nu != 13 or min(*self.jids, *self.aids, *self.bids.values()) < 0:
            raise ValueError("XML must contain the13 named NANDU1 joints/actuators and base_Link/up3")
        self.qadr = self.model.jnt_qposadr[self.jids]
        self.vadr = self.model.jnt_dofadr[self.jids]
        self.sensors = {}
        for prefix in ("base_imu", "up3_imu"):
            for suffix, size in (("quat", 4), ("gyro", 3)):
                name = prefix + "_" + suffix
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                if sid < 0 or self.model.sensor_dim[sid] != size:
                    raise ValueError("Missing sensor " + name)
                self.sensors[name] = slice(self.model.sensor_adr[sid], self.model.sensor_adr[sid] + size)
        from nandu1_v131 import FixedPayload
        self.payload = FixedPayload(self.model)
        self.payload.apply(self.data, up3_total_mass, rear_mass, rear_x)
        self.reset()

    def reset(self):
        mj = self.mujoco
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = [0., 0., .65337, 1., 0., 0., 0.]
        self.data.qpos[self.qadr] = c.DEFAULT_Q
        mj.mj_forward(self.model, self.data)
        self.torque_saturations = 0
        self.velocity_projections = 0
        self.target_projections = 0

    def observe(self, previous_issued_action):
        sensor = lambda name: self.data.sensordata[self.sensors[name]].copy()
        return c.observe(sensor("base_imu_quat"), sensor("base_imu_gyro"),
                         sensor("up3_imu_quat"), sensor("up3_imu_gyro"),
                         self.data.qpos[self.qadr], self.data.qvel[self.vadr], previous_issued_action)

    def decode(self, action):
        target_q, target_dq = c.decode(action)
        limits = self.model.jnt_range[self.jids]
        projected = c.clip_position_targets(target_q, limits[:, 0], limits[:, 1])
        self.target_projections += int(np.count_nonzero(target_q != projected))
        return projected, target_dq

    def apply(self, target_q, target_dq):
        q, dq = self.data.qpos[self.qadr], self.data.qvel[self.vadr]
        raw = c.KP * (c.finite(target_q, 13) - q) + c.KD * (c.finite(target_dq, 13) - dq)
        self.data.ctrl[self.aids] = np.clip(raw, -c.EFFORT, c.EFFORT)
        self.torque_saturations += int(np.count_nonzero(np.abs(raw) > c.EFFORT))
        return raw

    def step(self):
        self.mujoco.mj_step(self.model, self.data)
        velocity = self.data.qvel[self.vadr]
        count = int(np.count_nonzero(np.abs(velocity) > c.VELOCITY))
        self.velocity_projections += count
        if count:
            self.data.qvel[self.vadr] = np.clip(velocity, -c.VELOCITY, c.VELOCITY)
            self.mujoco.mj_forward(self.model, self.data)

    def truth(self):
        """Evaluation only: body COM velocities and rigidly transported fixed TCP."""
        mj, base, up3 = self.mujoco, self.bids["base_Link"], self.bids["up3"]
        base_v, up_v = np.empty(6), np.empty(6)
        mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, base, base_v, 0)
        mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, up3, up_v, 0)
        tcp = self.data.xpos[up3] + self.data.xmat[up3].reshape(3, 3) @ np.asarray(TCP_LOCAL)
        tcp_v = up_v[3:] + np.cross(up_v[:3], tcp - self.data.xipos[up3])
        body_rotation = self.data.xmat[base].reshape(3, 3)
        encoder_label = np.r_[body_rotation.T @ base_v[3:], body_rotation.T @ tcp_v, tcp[2]]
        return dict(tcp_position_world=tcp.copy(), tcp_velocity_world=tcp_v.copy(),
                    base_origin_world=self.data.xpos[base].copy(), base_com_world=self.data.xipos[base].copy(),
                    up3_origin_world=self.data.xpos[up3].copy(), up3_com_world=self.data.xipos[up3].copy(),
                    encoder_label=encoder_label,
                    command_truth=np.array([base_v[2], (body_rotation.T @ tcp_v)[0], tcp[2]]))
