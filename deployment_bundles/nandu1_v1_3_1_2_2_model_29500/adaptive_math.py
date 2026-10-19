"""Simulator-independent v13122 causal state, geometry and diagnostic math.

Commands retain E's (yaw, body-vx, world-height) order. Quaternions are wxyz.
All leading dimensions broadcast in pure geometry functions. Stateful objects
operate on one environment batch and advance only in ``step``; ``step_id``
guards repeated observation/reward reads of the same physical control step.
"""
from __future__ import annotations

import math
import torch

TCP_LOCAL = (-.066473051764, .161623719479, -.000621484902)
HEIGHT_RATES = (.05, .10, .20, .35, .50)
YAW_LIMITS = (.20, .50, .85, 1.20, 1.57)


def _as(value, reference=None):
    return torch.as_tensor(value, device=None if reference is None else reference.device,
                           dtype=None if reference is None else reference.dtype)


def _dt(dt):
    dt = float(dt)
    if not math.isfinite(dt) or not 0 < dt <= .1:
        raise ValueError("dt must be finite in (0,.1]")
    return dt


def wrap_angle(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def smooth_ramp(value, low, high):
    """C1 ramp from zero to one; scalar or batched Torch input."""
    if high <= low:
        raise ValueError("ramp high must exceed low")
    x = ((value - low) / (high - low)).clamp(0., 1.)
    return x.square() * (3. - 2. * x)


def quiet_gate(value, low, high):
    return 1. - smooth_ramp(value.abs(), low, high)


def quaternion_normalize(quat, *, validate=True):
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    if quat.shape[-1] != 4 or (validate and (norm <= 1e-12).any()):
        raise ValueError("wxyz quaternion must have nonzero norm")
    return quat / norm


def quaternion_multiply(a, b):
    a, b = torch.broadcast_tensors(a, b)
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv, dim=-1)), -1)


def quaternion_rotate(quat, vector, *, validate=True):
    q = quaternion_normalize(quat, validate=validate)
    qv, vector = torch.broadcast_tensors(q[..., 1:], vector)
    uv = 2 * torch.linalg.cross(qv, vector, dim=-1)
    return vector + q[..., :1] * uv + torch.linalg.cross(qv, uv, dim=-1)


def quaternion_angular_velocity(previous, current, dt, *, validate=True):
    """Shortest quaternion delta rotation vector / dt, in WORLD axes.

    q_current * conjugate(q_previous), rather than Euler yaw differences,
    makes the world-z component correct even while the body tilts.
    """
    dt = _dt(dt)
    previous = quaternion_normalize(previous, validate=validate)
    current = quaternion_normalize(current, validate=validate)
    inverse = torch.cat((previous[..., :1], -previous[..., 1:]), -1)
    delta = quaternion_multiply(current, inverse)
    delta = torch.where(delta[..., :1] < 0, -delta, delta)
    size = torch.linalg.vector_norm(delta[..., 1:], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(size, delta[..., :1].clamp_min(0.))
    scale = torch.where(size > 1e-8, angle / size.clamp_min(1e-12), 2. * torch.ones_like(size))
    return delta[..., 1:] * scale / dt


def heading_frame(gravity_b):
    """Body-frame horizontal forward, left and up from an IMU gravity vector."""
    size = torch.linalg.vector_norm(gravity_b, dim=-1, keepdim=True)
    default = torch.zeros_like(gravity_b)
    default[..., 2] = 1
    up = torch.where(size > 1e-6, -gravity_b / size.clamp_min(1e-6), default)
    ex = torch.zeros_like(up)
    ex[..., 0] = 1
    forward = ex - up[..., :1] * up
    fallback = torch.zeros_like(up)
    fallback[..., 1] = 1
    fallback = fallback - up[..., 1:2] * up
    forward = torch.where(torch.linalg.vector_norm(forward, dim=-1, keepdim=True) > 1e-6,
                          forward, fallback)
    forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(1e-6)
    return forward, torch.linalg.cross(up, forward, dim=-1), up


def imu_heading_rate(gravity_b, gyro_b):
    """v131's projected-forward heading derivative and world-z spin, from IMU."""
    _, left, up = heading_frame(gravity_b)
    projection = (1 - up[..., 0].square()).clamp_min(1e-4).sqrt()
    ex = torch.zeros_like(gyro_b)
    ex[..., 0] = 1
    rate = (torch.linalg.cross(gyro_b, ex, dim=-1) * left).sum(-1) / projection
    world_z = (gyro_b * up).sum(-1)
    return torch.where(projection > .01, rate, world_z), world_z


class _BatchState:
    def __init__(self, num_envs, device="cpu", dtype=torch.float32):
        if isinstance(num_envs, bool) or int(num_envs) != num_envs or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not dtype.is_floating_point:
            raise ValueError("state dtype must be floating")
        self.num_envs, self.device, self.dtype = int(num_envs), torch.device(device), dtype
        self._last_step_id = None

    def _zeros(self, *shape):
        return torch.zeros(self.num_envs, *shape, device=self.device, dtype=self.dtype)

    def _vector(self, value):
        value = _as(value, self._reference)
        if value.shape == (self.num_envs, 1):
            value = value[:, 0]
        return value.expand(self.num_envs)

    def _ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.dtype == torch.bool:
            if ids.shape != (self.num_envs,):
                raise ValueError("boolean reset mask must match num_envs")
            return ids.nonzero().flatten()
        if ids.ndim != 1 or (ids != ids.long()).any():
            raise ValueError("reset ids must be an integer vector")
        ids = ids.long()
        if ((ids < 0) | (ids >= self.num_envs)).any() or ids.unique().numel() != ids.numel():
            raise ValueError("reset ids must be unique and in range")
        return ids

    def _repeat(self, step_id):
        if step_id is None:
            return False
        if isinstance(step_id, bool) or not isinstance(step_id, int):
            raise ValueError("step_id must be an integer")
        if self._last_step_id is not None and step_id < self._last_step_id:
            raise ValueError("step_id must not move backwards")
        return step_id == self._last_step_id

    def _finish(self, step_id):
        if step_id is not None:
            self._last_step_id = step_id


class CausalCommandProcessor(_BatchState):
    """50 Hz causal E commands, sensor heading hold and explicit episode state.

    Heading integrates only supplied body IMU data. Raw yaw intent determines
    release: its anchor latches AFTER that step's gyro integration. Saturation
    back-calculation applies only during intentional turning and cannot move a
    hold anchor. Final yaw command always obeys the 2 rad/s² slew bound.
    """
    SCHEMA = "nandu_v13122_causal_command_r1"
    heading_kp = 1.5

    def __init__(self, num_envs, device="cpu", dtype=torch.float32):
        super().__init__(num_envs, device, dtype)
        self._command = self._zeros(3)
        self._reference = self._command
        self._height_rate = self._zeros()
        self._yaw_intent = self._zeros()
        self.heading = self._zeros()
        self.heading_target = self._zeros()
        self._was_turning = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

    @property
    def command(self):
        return self._command.clone()

    @property
    def height_rate(self):
        return self._height_rate.clone()

    @property
    def hdot(self):
        return self.height_rate

    def reset(self, env_ids=None, *, height=0., heading=0.):
        ids = self._ids(env_ids)
        for state in (self._command, self._height_rate, self._yaw_intent):
            state[ids] = 0
        h = _as(height, self._command)
        psi = _as(heading, self._command)
        self._command[ids, 2] = h[ids] if h.shape == (self.num_envs,) else h
        self.heading[ids] = psi[ids] if psi.shape == (self.num_envs,) else psi
        self.heading_target[ids] = self.heading[ids]
        self._was_turning[ids] = False
        if env_ids is None:
            self._last_step_id = None

    @torch.no_grad()
    def step(self, goal, dt, *, step_id=None, gravity_b=None, gyro_b=None,
             yaw_limit=None, height_rate_limit=None, stage=4):
        if self._repeat(step_id):
            return self.command
        dt = _dt(dt)
        if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage < 5:
            raise ValueError("stage must be an integer in [0,4]")
        goal = _as(goal, self._command)
        if goal.shape != self._command.shape or not torch.isfinite(goal).all():
            raise ValueError("goal must be finite (num_envs,3) in yaw/vx/height order")
        limit = self._vector(YAW_LIMITS[stage] if yaw_limit is None else yaw_limit)
        rate = self._vector(HEIGHT_RATES[stage] if height_rate_limit is None else height_rate_limit)
        if not torch.isfinite(limit).all() or not torch.isfinite(rate).all() or (limit <= 0).any() or (rate <= 0).any():
            raise ValueError("command limits must be finite and positive")
        if gravity_b is None:
            gravity_b = self._zeros(3)
            gravity_b[:, 2] = -1
        if gyro_b is None:
            gyro_b = self._zeros(3)
        gravity, gyro = _as(gravity_b, self._command), _as(gyro_b, self._command)
        if gravity.shape != (self.num_envs, 3) or gyro.shape != gravity.shape:
            raise ValueError("body IMU must have shape (num_envs,3)")
        if not torch.isfinite(gravity).all() or not torch.isfinite(gyro).all():
            raise ValueError("body IMU must be finite")
        heading_rate, world_z = imu_heading_rate(gravity, gyro)
        self.heading.copy_(wrap_angle(self.heading + dt * heading_rate))
        raw_intent = goal[:, 0].clamp(-limit, limit)
        self._yaw_intent.add_((dt / (.10 + dt) * (raw_intent - self._yaw_intent)).clamp(-2 * dt, 2 * dt))
        turning = raw_intent.abs() > 1e-5
        released = self._was_turning & ~turning
        self.heading_target.copy_(torch.where(released, self.heading, self.heading_target))
        feedforward = torch.where(turning, self._yaw_intent, 0.)
        self.heading_target.copy_(wrap_angle(self.heading_target + feedforward * dt))
        tilt_coupling = world_z - heading_rate
        raw_yaw = feedforward + self.heading_kp * wrap_angle(self.heading_target - self.heading) + tilt_coupling
        desired_yaw = raw_yaw.clamp(-limit, limit)
        # Holding can saturate for arbitrarily long; its reference stays fixed.
        backcalc = turning & (raw_yaw != desired_yaw)
        self.heading_target.copy_(wrap_angle(torch.where(backcalc,
            self.heading + (desired_yaw - tilt_coupling - feedforward) / self.heading_kp, self.heading_target)))
        self._command[:, 0].add_((desired_yaw - self._command[:, 0]).clamp(-2 * dt, 2 * dt))
        self._command[:, 1].add_((dt / (.15 + dt) * (goal[:, 1] - self._command[:, 1])).clamp(-.60 * dt, .60 * dt))
        before_height = self._command[:, 2].clone()
        delta = dt / (.15 + dt) * (goal[:, 2] - before_height)
        delta = delta.clamp(-rate * dt, rate * dt)
        # Snap only when both the positional tolerance and rate budget permit it.
        snap = ((goal[:, 2] - before_height).abs() <= 1e-4) & ((goal[:, 2] - before_height).abs() <= rate * dt)
        self._command[:, 2].copy_(torch.where(snap, goal[:, 2], before_height + delta))
        self._height_rate.copy_((self._command[:, 2] - before_height) / dt)
        self._was_turning.copy_(turning)
        self._finish(step_id)
        return self.command

    def state_dict(self):
        return dict(schema=self.SCHEMA, num_envs=self.num_envs,
                    command=self.command, height_rate=self.height_rate,
                    yaw_intent=self._yaw_intent.clone(), heading=self.heading.clone(),
                    heading_target=self.heading_target.clone(), was_turning=self._was_turning.clone(),
                    last_step_id=self._last_step_id)

    def load_state_dict(self, state, *, same_world=False, height=0.):
        if state.get("schema") != self.SCHEMA:
            raise ValueError("incompatible command state schema")
        if not same_world or state.get("num_envs") != self.num_envs:
            self.reset(height=height)
            return
        fields = dict(command=self._command, height_rate=self._height_rate, yaw_intent=self._yaw_intent,
                      heading=self.heading, heading_target=self.heading_target, was_turning=self._was_turning)
        converted = {name: torch.as_tensor(state[name], device=self.device, dtype=target.dtype)
                     for name, target in fields.items()}
        if any(value.shape != fields[name].shape or not torch.isfinite(value).all() for name, value in converted.items()):
            raise ValueError("invalid command temporal state")
        for name, target in fields.items():
            target.copy_(converted[name])
        self._last_step_id = state.get("last_step_id")


class SlowErrorMemory(_BatchState):
    """Actor memory from E estimates/noisy IMU, with no true-state gates.

    Channels are (vx,yaw), unlike external commands (yaw,vx,height). Invalid
    sensor data, tilt >45 degrees or known action clipping freeze both the
    error filter and integral. Raw intent gates feedback-created yaw correction.
    """
    SCHEMA = "nandu_v13122_slow_error_memory_r1"

    def __init__(self, num_envs, device="cpu", dtype=torch.float32):
        super().__init__(num_envs, device, dtype)
        self.error_lpf = self._zeros(2)
        self.z = self._zeros(2)
        self._reference = self.z

    def normalized(self):
        return self.z / self.z.new_tensor((.10, .20))

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        self.error_lpf[ids] = 0
        self.z[ids] = 0
        if env_ids is None:
            self._last_step_id = None

    @torch.no_grad()
    def step_errors(self, vx_error, yaw_error, raw_intent, height_rate, valid=True,
                    tilt=0., clipped=False, dt=.02, *, step_id=None):
        if self._repeat(step_id):
            return self.normalized()
        dt = _dt(dt)
        error = torch.stack((self._vector(vx_error), self._vector(yaw_error)), -1)
        intent = _as(raw_intent, self.z)
        if intent.shape not in ((self.num_envs, 2), (self.num_envs, 3)):
            raise ValueError("raw_intent must be (num_envs,2 or 3), yaw then vx")
        height_rate, tilt = self._vector(height_rate), self._vector(tilt)
        update = self._vector(valid).bool() & ~self._vector(clipped).bool()
        update &= torch.isfinite(error).all(-1) & torch.isfinite(intent).all(-1)
        update &= torch.isfinite(height_rate) & torch.isfinite(tilt) & (tilt.abs() <= math.pi / 4)
        safe_error = torch.nan_to_num(error).clamp(-self.z.new_tensor((.30, .60)), self.z.new_tensor((.30, .60)))
        filtered = self.error_lpf + dt / (.25 + dt) * (safe_error - self.error_lpf)
        self.error_lpf.copy_(torch.where(update[:, None], filtered, self.error_lpf))
        g = torch.stack((quiet_gate(intent[:, 1], .02, .08), quiet_gate(intent[:, 0], .03, .10)), -1)
        g *= quiet_gate(height_rate, .01, .04)[:, None]
        next_z = self.z + dt * (g * self.error_lpf - (1 - g) * self.z)
        caps = self.z.new_tensor((.20, .40))
        self.z.copy_(torch.where(update[:, None], next_z.clamp(-caps, caps), self.z))
        self._finish(step_id)
        return self.normalized()

    def step(self, command, estimate_vx, imu_yaw_rate, dt=.02, *, step_id=None,
             hdot=0., tilt=0., measurement_valid=True, clipped=False, raw_intent=None):
        command = _as(command, self.z)
        return self.step_errors(command[:, 1] - self._vector(estimate_vx),
                                command[:, 0] - self._vector(imu_yaw_rate),
                                command if raw_intent is None else raw_intent, hdot,
                                measurement_valid, tilt, clipped, dt, step_id=step_id)

    def state_dict(self):
        return dict(schema=self.SCHEMA, num_envs=self.num_envs, error_lpf=self.error_lpf.clone(),
                    z=self.z.clone(), last_step_id=self._last_step_id)

    def load_state_dict(self, state, *, same_world=False):
        if state.get("schema") != self.SCHEMA:
            raise ValueError("incompatible memory state schema")
        if not same_world or state.get("num_envs") != self.num_envs:
            self.reset()
            return
        values = [_as(state[key], self.z) for key in ("error_lpf", "z")]
        if any(v.shape != self.z.shape or not torch.isfinite(v).all() for v in values):
            raise ValueError("invalid slow memory temporal state")
        self.error_lpf.copy_(values[0])
        self.z.copy_(values[1])
        self._last_step_id = state.get("last_step_id")


def wheel_contact_points(hub_position, axle_world, radius):
    """Lowest points on actual wheel circles, with no cylinder/zero-tilt shortcut.

    A circle's radial plane is perpendicular to its axle. This returns the
    geometrical low point (not artificially projected onto z=0 when airborne).
    """
    axle = axle_world / torch.linalg.vector_norm(axle_world, dim=-1, keepdim=True).clamp_min(1e-12)
    down = torch.zeros_like(axle)
    down[..., 2] = -1
    radial = down + axle[..., 2:3] * axle
    length = torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
    radial = radial / length.clamp_min(1e-12)
    # Scalar, (wheel,), (...,1) or (...,wheel); ordinary broadcasting keeps
    # a left/right radius vector on the wheel axis even when num_envs == 2.
    radius = _as(radius, hub_position)
    return hub_position + radius[..., None] * radial


def equivalent_balance(nonwheel_com, contact_points, heading, hub_velocity=None):
    """Equivalent sagittal pendulum about the midpoint of real wheel contacts."""
    ground = contact_points.mean(-2)
    heading = heading.clone()
    heading[..., 2] = 0
    heading = heading / torch.linalg.vector_norm(heading, dim=-1, keepdim=True).clamp_min(1e-12)
    offset = nonwheel_com - ground
    h = offset[..., 2]
    theta = torch.atan2((offset * heading).sum(-1), h)
    vG = torch.zeros_like(h) if hub_velocity is None else (hub_velocity.mean(-2) * heading).sum(-1)
    return dict(theta=theta, h=h, vG=vG, ground_center=ground)


def dual_support(wheel_fz, total_mass, gravity=9.81):
    """True-state qualifier for privileged rewards/labels only, never actor input."""
    scale = _as(total_mass, wheel_fz) * gravity
    fraction = wheel_fz / scale[..., None]
    return (fraction > .05).all(-1) & (fraction.sum(-1) > .5)


class GeometryTrajectoryCache(_BatchState):
    """One vectorized trajectory differentiation pass per control step, no FK.

    All positions are world positions supplied from existing geometry caches.
    ``nonwheel_com`` is already the mass-weighted COM excluding wheel bodies.
    Derivatives use tau=.04; initial/partially reset samples have ``valid=False``.
    """
    def __init__(self, num_envs, device="cpu", dtype=torch.float32):
        super().__init__(num_envs, device, dtype)
        self._reference = self._zeros()
        self._previous = {}
        self._cache = {}
        self._initialized = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._derivative_initialized = torch.zeros_like(self._initialized)

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        self._initialized[ids] = False
        self._derivative_initialized[ids] = False
        for value in self._cache.values():
            value[ids] = 0
        if env_ids is None:
            self._last_step_id = None

    def read(self):
        return {key: value.clone() for key, value in self._cache.items()}

    @torch.no_grad()
    def step(self, tcp_pos, base_pos, base_quat, nonwheel_com, wheel_hub_pos,
             wheel_axle_world, wheel_radius, dt, *, step_id=None):
        if self._repeat(step_id):
            return self.read()
        dt = _dt(dt)
        values = dict(tcp_pos=_as(tcp_pos, self._reference), base_pos=_as(base_pos, self._reference),
                      base_quat=_as(base_quat, self._reference), nonwheel_com=_as(nonwheel_com, self._reference),
                      wheel_hub_pos=_as(wheel_hub_pos, self._reference))
        data_valid = torch.ones_like(self._initialized)
        for key, shape in (("tcp_pos", (self.num_envs, 3)), ("base_pos", (self.num_envs, 3)),
                           ("base_quat", (self.num_envs, 4)), ("nonwheel_com", (self.num_envs, 3)),
                           ("wheel_hub_pos", (self.num_envs, 2, 3))):
            if values[key].shape != shape:
                raise ValueError(f"{key} must have shape {shape}")
            data_valid &= torch.isfinite(values[key]).flatten(1).all(-1)
        data_valid &= torch.linalg.vector_norm(values["base_quat"], dim=-1) > 1e-12
        axle = _as(wheel_axle_world, self._reference).expand(self.num_envs, 2, 3)
        radius = _as(wheel_radius, self._reference).expand(self.num_envs, 2)
        data_valid &= torch.isfinite(axle).flatten(1).all(-1) & torch.isfinite(radius).all(-1)
        data_valid &= (radius > 0).all(-1) & (torch.linalg.vector_norm(axle, dim=-1) > 1e-12).all(-1)
        # A failed environment must reach the existing termination machinery;
        # it cannot raise here or contaminate its neighbors' trajectory cache.
        for key, value in values.items():
            default = torch.zeros_like(value)
            if key == "base_quat":
                default[..., 0] = 1
            fallback = self._previous.get(key, default)
            mask = data_valid.reshape(self.num_envs, *([1] * (value.ndim - 1)))
            values[key] = torch.where(mask, value, fallback)
        fallback_axle = torch.zeros_like(axle)
        fallback_axle[..., 1] = 1
        axle = torch.where(data_valid[:, None, None], axle, fallback_axle)
        radius = torch.where(data_valid[:, None], radius, torch.zeros_like(radius))
        values["base_quat"] = quaternion_normalize(values["base_quat"], validate=False)
        heading = quaternion_rotate(values["base_quat"], values["base_pos"].new_tensor((1., 0., 0.)), validate=False)
        contacts = wheel_contact_points(values["wheel_hub_pos"], axle, radius)
        balance = equivalent_balance(values["nonwheel_com"], contacts, heading)
        pose_valid = data_valid & (balance["h"] > 1e-5) & (heading[..., :2].square().sum(-1) > 1e-10)
        valid = self._initialized & pose_valid
        first_derivative = valid & ~self._derivative_initialized
        if not self._previous:
            self._previous = {key: value.clone() for key, value in values.items()}
            self._previous["theta"] = balance["theta"].clone()
        alpha = dt / (.04 + dt)
        result = dict(balance, contact_points=contacts, heading=heading, valid=valid)
        for name, source in (("tcp_vel", "tcp_pos"), ("base_vel", "base_pos"),
                             ("nonwheel_com_vel", "nonwheel_com"), ("hub_vel", "wheel_hub_pos")):
            raw = (values[source] - self._previous[source]) / dt
            old = self._cache.get(name, torch.zeros_like(raw))
            mask = valid.reshape(self.num_envs, *([1] * (raw.ndim - 1)))
            first = first_derivative.reshape(self.num_envs, *([1] * (raw.ndim - 1)))
            result[name] = torch.where(mask, torch.where(first, raw, old + alpha * (raw - old)), 0.)
            result[name + "_raw"] = torch.where(mask, raw, 0.)
        raw_omega = quaternion_angular_velocity(self._previous["base_quat"], values["base_quat"], dt, validate=False)
        result["world_ang_vel_raw"] = torch.where(valid[:, None], raw_omega, 0.)
        result["world_ang_vel"] = torch.where(valid[:, None],
            torch.where(first_derivative[:, None], raw_omega,
                self._cache.get("world_ang_vel", torch.zeros_like(raw_omega)) * (1-alpha) + alpha * raw_omega), 0.)
        raw_tdot = wrap_angle(balance["theta"] - self._previous["theta"]) / dt
        result["theta_dot"] = torch.where(valid,
            torch.where(first_derivative, raw_tdot,
                self._cache.get("theta_dot", torch.zeros_like(raw_tdot)) * (1-alpha) + alpha * raw_tdot), 0.)
        result["vG"] = equivalent_balance(values["nonwheel_com"], contacts, heading, result["hub_vel"])["vG"]
        # Body-frame vx is explicit, while height and dz remain world absolute.
        inv = torch.cat((values["base_quat"][..., :1], -values["base_quat"][..., 1:]), -1)
        result["tcp_v_body"] = quaternion_rotate(inv, result["tcp_vel"], validate=False)
        result["base_v_body"] = quaternion_rotate(inv, result["base_vel"], validate=False)
        self._previous = {key: value.clone() for key, value in values.items()}
        self._previous["theta"] = balance["theta"].clone()
        self._cache = result
        self._initialized.copy_(pose_valid)
        self._derivative_initialized.copy_(valid)
        self._finish(step_id)
        return self.read()


def inertia_about_origin(mass, com, inertia, rotation=None):
    if rotation is not None:
        inertia = rotation @ inertia @ rotation.transpose(-1, -2)
    identity = torch.eye(3, device=inertia.device, dtype=inertia.dtype)
    shift = com.square().sum(-1)[..., None, None] * identity - com[..., :, None] * com[..., None, :]
    return inertia + mass[..., None, None] * shift


def physical_descriptor(mass, com, inertia, nominal_mass, nominal_com, nominal_inertia,
                        *, normalization=None, inertia_rotation=None, nominal_inertia_rotation=None):
    """Two links x [Δmass, Δ(mCOM)xyz, ΔIorigin xx yy zz xy xz yz].

    Inputs end in (2,), (2,3), (2,3,3). Inertias are COM tensors in
    link axes, or provide the fixed inertia-to-link rotations explicitly.
    Normalization is fixed across the curriculum: a DescriptorNormalizer or
    a caller-supplied positive S4 scale20, never per-batch statistics.
    """
    mass = torch.as_tensor(mass)
    com, inertia = _as(com, mass), _as(inertia, mass)
    nominal_mass, nominal_com, nominal_inertia = (_as(x, mass) for x in (nominal_mass, nominal_com, nominal_inertia))
    if mass.shape[-1] != 2 or com.shape[-2:] != (2, 3) or inertia.shape[-3:] != (2, 3, 3):
        raise ValueError("descriptor requires exactly two links with mass/COM/inertia")
    actual = inertia_about_origin(mass, com, inertia, inertia_rotation)
    nominal = inertia_about_origin(nominal_mass, nominal_com, nominal_inertia, nominal_inertia_rotation)
    delta = actual - nominal
    six = torch.stack((delta[..., 0, 0], delta[..., 1, 1], delta[..., 2, 2],
                       delta[..., 0, 1], delta[..., 0, 2], delta[..., 1, 2]), -1)
    dm, first = mass - nominal_mass, mass[..., None] * com - nominal_mass[..., None] * nominal_com
    descriptor = torch.cat((dm[..., None], first, six), -1).flatten(-2)
    if normalization is not None:
        if callable(normalization):
            descriptor = normalization(descriptor)
        else:
            scale = _as(normalization, descriptor)
            if (scale <= 0).any() or not torch.isfinite(scale).all():
                raise ValueError("descriptor normalization scale must be positive and finite")
            descriptor = descriptor / scale
    return descriptor


class DescriptorNormalizer:
    """Immutable full-S4 ranges, floors and constant-dimension mask.

    Scale=max(abs(lo),abs(hi),floor), no centering so nominal remains zero.
    Constant descriptor dimensions are masked out, including decoder losses.
    """
    def __init__(self, scale, constant_mask):
        self.scale = torch.as_tensor(scale).detach().clone()
        self.constant_mask = torch.as_tensor(constant_mask, device=self.scale.device, dtype=torch.bool).detach().clone()
        if self.scale.shape != (20,) or self.constant_mask.shape != (20,):
            raise ValueError("descriptor normalization must have 20 dimensions")
        if not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("descriptor scales must be finite and positive")

    @classmethod
    def from_bounds(cls, low, high, floor=1e-6):
        low = torch.as_tensor(low)
        high, floor = _as(high, low), _as(floor, low)
        if low.shape != (20,) or high.shape != (20,) or (high < low).any():
            raise ValueError("full S4 bounds must be ordered 20-vectors")
        if not torch.isfinite(low).all() or not torch.isfinite(high).all() or (floor <= 0).any():
            raise ValueError("normalization bounds/floors must be finite and positive")
        return cls(torch.maximum(torch.maximum(low.abs(), high.abs()), floor), (high - low).abs() <= 1e-12)

    @property
    def active_mask(self):
        return ~self.constant_mask

    def __call__(self, descriptor):
        scale = self.scale.to(descriptor)
        return (descriptor / scale).masked_fill(self.constant_mask.to(descriptor.device), 0.)

    def state_dict(self):
        return dict(scale=self.scale.clone(), constant_mask=self.constant_mask.clone())


def samepose_counterfactual(link_pos, link_quat, masses, local_com, *, nonwheel_mask=None,
                           contact_points=None, heading=None):
    """Evaluate actual or independently sampled payload at cached link transforms.

    Payload sampling/composition stays with the established v131 payload model.
    This helper only recombines cached transforms and changed local mass/COM;
    it performs no simulator calls, FK, or per-environment loop.
    """
    world_com = link_pos + quaternion_rotate(link_quat, local_com)
    mass = masses if nonwheel_mask is None else masses * torch.as_tensor(nonwheel_mask, device=masses.device)
    total = mass.sum(-1)
    if (mass < 0).any() or (total <= 0).any():
        raise ValueError("nonwheel mass must be nonnegative with positive total")
    center = (mass[..., None] * world_com).sum(-2) / total[..., None]
    result = dict(nonwheel_com=center, nonwheel_mass=total, link_com_world=world_com)
    if contact_points is not None:
        if heading is None:
            raise ValueError("heading required with counterfactual contact points")
        result.update(equivalent_balance(center, contact_points, heading))
    return result


def huber(value):
    magnitude = value.abs()
    return torch.where(magnitude <= 1., .5 * magnitude.square(), magnitude - .5)


def static_costs(tcp_xy, yaw, bias_xy, bias_yaw, *, tcp_vz, base_vz, base_omega_xy,
                 theta, theta_dot, vG):
    """Positive ungated Huber costs capped at 20; caller applies gates/weights.

    TCPxy inputs are velocities from trajectory differentiation (not position
    errors); bias inputs are the two-second EMA velocities. No wheel-zero term.
    """
    def planar(value, scale):
        return huber(value / scale).sum(-1)
    return dict(capture=(planar(tcp_xy, .08) + huber(yaw / .12)).clamp_max(20.),
                fine=(planar(tcp_xy, .02) + huber(yaw / .03)).clamp_max(20.),
                bias=(planar(bias_xy, .005) + huber(bias_yaw / .005)).clamp_max(20.),
                residual=(huber(tcp_vz / .05) + .5 * huber(base_vz / .05)
                          + .5 * planar(base_omega_xy, .15)).clamp_max(20.),
                balance=(huber(theta / .08) + .5 * huber(theta_dot / .15)
                         + .5 * huber(vG / .02)).clamp_max(20.))


def height_reference(height_error, measured_vz, hdot, vmax, *, filtered=None, dt=.02, tau=.15):
    """Causal vertical reference; filtered state cannot oppose reference sign."""
    vmax = _as(vmax, height_error)
    reference = (hdot + 1.5 * height_error - .25 * (measured_vz - hdot)).clamp(-vmax, vmax)
    if filtered is None:
        return reference
    dt = _dt(dt)
    state = filtered + dt / (tau + dt) * (reference - filtered)
    return state.clamp(torch.minimum(torch.zeros_like(reference), reference),
                       torch.maximum(torch.zeros_like(reference), reference))


def drift_scores(net_tcp, path_tcp, wheel_activity, yaw_slope, *, valid_count):
    """Per-environment noncancelling diagnostic scores; absent segments are NaN."""
    metrics = dict(net_tcp=(net_tcp, .002), path_tcp=(path_tcp, .020),
                   wheel_activity=(wheel_activity, .020), yaw_slope=(yaw_slope, .001))
    count = _as(valid_count, net_tcp)
    result = {key: torch.where(count > 0, torch.exp(-(value / scale).square()),
                              torch.full_like(value, float("nan")))
              for key, (value, scale) in metrics.items()}
    result["valid_count"] = count
    return result
