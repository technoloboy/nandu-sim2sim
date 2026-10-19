"""Torch-only v1.3.1 geometry, yaw hold, and composite payload properties.

All vectors use the frame named by the API. Inertias are full tensors about
their COM in link axes. Callers supply immutable nominal mass/COM/inertia;
URDF inertia-frame rotations must be applied once before payload composition.
No simulator import, command sampler, collision shape, or nominal inertia is
embedded here. Tensor operations broadcast and do not loop over environments.
"""
from __future__ import annotations

import math
import torch

TCP_LOCAL = (-.066473051764, .161623719479, -.000621484902)
UP3_NOMINAL_MASS = .67148
UP3_MASS_RANGE = (.6, 6.)
UP3_BOX_DIMENSIONS = (.05, .06, .05)
REAR_X_RANGE = (-.35, -.20)
REAR_Z = .07
REAR_CYLINDER_RADIUS = .025
REAR_CYLINDER_LENGTH = .1
PAYLOAD_STAGE_AMPLITUDES = (.1, .25, .5, .75, 1.)
YAW_STAGE_LIMITS = (.20, .50, .85, 1.20, 1.57)
DT_MIN, DT_MAX = 1e-6, .1


def _tensor(value, name, reference=None, *, validate=True):
    result = torch.as_tensor(value, device=None if reference is None else reference.device,
                             dtype=None if reference is None else reference.dtype)
    if not result.dtype.is_floating_point:
        result = result.to(torch.float32)
    if validate and not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _tail(value, tail, name):
    if value.shape[-len(tail):] != tail:
        raise ValueError(f"{name} must end in shape {tail}")


def _broadcast(*shapes):
    try:
        return torch.broadcast_shapes(*shapes)
    except RuntimeError as error:
        raise ValueError("incompatible batch shapes") from error


def _stage(stage):
    if isinstance(stage, bool) or not isinstance(stage, (int, float)) or not math.isfinite(stage) or int(stage) != stage or not 0 <= stage < 5:
        raise ValueError("stage must be an integer in [0,4]")
    return int(stage)


def transport_velocity(velocity_at_source, angular_velocity, target_minus_source):
    """Rigid transport v_target=v_source+omega×(p_target-p_source), all in one frame."""
    v = _tensor(velocity_at_source, "source velocity")
    w = _tensor(angular_velocity, "angular velocity", v)
    offset = _tensor(target_minus_source, "point offset", v)
    for value in (v, w, offset):
        _tail(value, (3,), "transport vector")
    shape = _broadcast(v.shape, w.shape, offset.shape)
    return v.expand(shape) + torch.linalg.cross(w.expand(shape), offset.expand(shape), dim=-1)


def point_state_world(link_pos_w, link_quat_w, com_pos_w, com_lin_vel_w, ang_vel_w,
                      point_local=TCP_LOCAL):
    """World state of a fixed link-local point using world COM linear velocity.

    ``link_quat_w`` is scalar-first wxyz; ``link_pos_w`` is the link origin,
    not the COM. ``com_pos_w`` must match the supplied COM velocity. The default
    TCP never follows randomized COM. Leading dimensions broadcast.
    """
    pos = _tensor(link_pos_w, "link position")
    quat = _tensor(link_quat_w, "link quaternion", pos)
    local = _tensor(point_local, "local point", pos)
    com = _tensor(com_pos_w, "COM position", pos)
    velocity = _tensor(com_lin_vel_w, "COM velocity", pos)
    omega = _tensor(ang_vel_w, "angular velocity", pos)
    _tail(pos, (3,), "link position")
    _tail(quat, (4,), "link quaternion")
    _tail(local, (3,), "local point")
    _tail(com, (3,), "COM position")
    _tail(velocity, (3,), "COM velocity")
    _tail(omega, (3,), "angular velocity")
    shape = _broadcast(pos.shape[:-1], quat.shape[:-1], local.shape[:-1], com.shape[:-1],
                       velocity.shape[:-1], omega.shape[:-1])
    quat = quat.expand(*shape, 4)
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    if (norm < 1e-8).any():
        raise ValueError("quaternion must have nonzero norm")
    quat = quat / norm
    local = local.expand(*shape, 3)
    twice_cross = 2 * torch.linalg.cross(quat[..., 1:], local, dim=-1)
    rotated = local + quat[..., :1] * twice_cross + torch.linalg.cross(quat[..., 1:], twice_cross, dim=-1)
    point = pos.expand(*shape, 3) + rotated
    return point, transport_velocity(velocity, omega, point - com)


def _wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _heading_frame(gravity_b):
    # Identical frame and pole fallback to the v1.4 yaw implementation.
    norm = torch.linalg.vector_norm(gravity_b, dim=-1, keepdim=True)
    up_default = torch.zeros_like(gravity_b)
    up_default[..., 2] = 1
    up = torch.where(norm > 1e-6, -gravity_b / norm.clamp_min(1e-6), up_default)
    ex = torch.zeros_like(up)
    ex[..., 0] = 1
    forward = ex - up[..., :1] * up
    fallback = torch.zeros_like(up)
    fallback[..., 1] = 1
    fallback = fallback - up[..., 1:2] * up
    forward_norm = torch.linalg.vector_norm(forward, dim=-1, keepdim=True)
    forward = torch.where(forward_norm > 1e-6, forward, fallback)
    forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(1e-6)
    return forward, torch.linalg.cross(up, forward, dim=-1), up


class HeadingController:
    """Yaw-only sensor controller; callers keep v1.3 vx/height sampling unchanged.

    ``step`` takes scalar or (N,) yaw intent and (N,3) body IMU vectors. It
    returns (N,) world-Z yaw rate. Stage affects yaw bounds only. ``reset``
    clears selected episodic state. Checkpoint restore defaults to clearing
    heading for a new physical world, even for the same environment count.
    ``validate_inputs=False`` skips step input finite reductions for trusted
    runtime sensor caches; the caller then owns nonfinite handling. Shapes,
    scalar configuration, reset IDs and checkpoint state still validate.
    """
    SCHEMA = "nandu1_v1_3_1_heading_controller_v1"
    _STATE_FIELDS = ("heading", "heading_target", "yaw_rate", "_was_turning")
    heading_kp = 1.5
    yaw_deadzone = 1e-5

    def __init__(self, num_envs, device="cpu", dtype=torch.float32, *, validate_inputs=True):
        if isinstance(num_envs, bool) or int(num_envs) != num_envs or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not dtype.is_floating_point:
            raise ValueError("heading requires a floating dtype")
        self.num_envs, self.device, self.dtype = int(num_envs), torch.device(device), dtype
        self.validate_inputs = bool(validate_inputs)
        self.heading = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self.heading_target = torch.zeros_like(self.heading)
        self.yaw_rate = torch.zeros_like(self.heading)
        self._was_turning = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _ids(self, ids):
        all_ids = torch.arange(self.num_envs, device=self.device)
        if ids is None:
            return all_ids
        if isinstance(ids, slice):
            return all_ids[ids]
        selected = torch.as_tensor(ids, device=self.device)
        if selected.dtype == torch.bool:
            if selected.shape != (self.num_envs,):
                raise ValueError("boolean ids must match num_envs")
            return selected.nonzero(as_tuple=False).flatten()
        if selected.ndim != 1 or not torch.isfinite(selected).all() or (selected != selected.to(torch.long)).any():
            raise ValueError("ids must be a finite integer vector")
        selected = selected.to(torch.long)
        if ((selected < 0) | (selected >= self.num_envs)).any() or selected.unique().numel() != selected.numel():
            raise ValueError("ids must be unique and in range")
        return selected

    def reset(self, ids=None):
        selected = self._ids(ids)
        for name in self._STATE_FIELDS:
            getattr(self, name)[selected] = 0

    def step(self, dt, gravity_b, gyro_b, yaw_intent, stage):
        if not math.isfinite(dt) or not DT_MIN <= dt <= DT_MAX:
            raise ValueError(f"dt must be finite in [{DT_MIN},{DT_MAX}]")
        stage = _stage(stage)
        gravity = _tensor(gravity_b, "gravity", self.heading, validate=self.validate_inputs)
        gyro = _tensor(gyro_b, "gyro", self.heading, validate=self.validate_inputs)
        intent = _tensor(yaw_intent, "yaw intent", self.heading, validate=self.validate_inputs)
        if gravity.shape != (self.num_envs, 3) or gyro.shape != (self.num_envs, 3):
            raise ValueError("IMU inputs must have shape (num_envs,3)")
        if intent.shape not in ((), (self.num_envs,)):
            raise ValueError("yaw intent must be scalar or shape (num_envs,)")
        limit = YAW_STAGE_LIMITS[stage]
        joystick = intent.expand(self.num_envs).clamp(-limit, limit)
        _, left, up = _heading_frame(gravity)
        projection = (1 - up[:, 0].square()).clamp_min(1e-4).sqrt()
        ex = torch.zeros_like(gyro)
        ex[:, 0] = 1
        heading_rate = (torch.linalg.cross(gyro, ex, dim=-1) * left).sum(-1) / projection
        heading_rate = torch.where(projection > .01, heading_rate, (gyro * up).sum(-1))
        self.heading[:] = _wrap_angle(self.heading + heading_rate * dt)
        turning = joystick.abs() > self.yaw_deadzone
        released = self._was_turning & ~turning
        self.heading_target[:] = torch.where(released, self.heading, self.heading_target)
        self.heading_target[:] = _wrap_angle(self.heading_target + torch.where(turning, joystick * dt, 0.))
        heading_error = _wrap_angle(self.heading_target - self.heading)
        tilt_coupling = (gyro * up).sum(-1) - heading_rate
        raw_yaw = joystick + self.heading_kp * heading_error + tilt_coupling
        applied_yaw = raw_yaw.clamp(-limit, limit)
        saturated = raw_yaw != applied_yaw
        self.heading_target[:] = _wrap_angle(torch.where(saturated,
            self.heading + (applied_yaw - tilt_coupling - joystick) / self.heading_kp, self.heading_target))
        self._was_turning.copy_(turning)
        self.yaw_rate.copy_(applied_yaw)
        return self.yaw_rate

    def state_dict(self):
        state = {name: getattr(self, name).detach().clone() for name in self._STATE_FIELDS}
        state.update(schema=self.SCHEMA, num_envs=self.num_envs)
        return state

    def load_state_dict(self, state, *, reset_heading=True):
        """Validate atomically; explicit ``False`` restores only same-world batch state.

        A different environment count always resets state. New-world resume must
        use the default; neither absolute yaw nor old episode targets are valid.
        """
        if state.get("schema") != self.SCHEMA:
            raise ValueError("incompatible heading controller schema")
        count = state.get("num_envs")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("invalid saved environment count")
        converted = {}
        for name in self._STATE_FIELDS:
            if name not in state:
                raise ValueError(f"missing heading state {name}")
            source = torch.as_tensor(state[name], device=self.device)
            if source.shape != (count,) or not torch.isfinite(source).all():
                raise ValueError(f"invalid heading state {name}")
            if name == "_was_turning" and ((source != 0) & (source != 1)).any():
                raise ValueError("invalid turning latch")
            converted[name] = source.to(dtype=getattr(self, name).dtype)
            if not torch.isfinite(converted[name]).all():
                raise ValueError(f"heading state {name} overflows receiving dtype")
        if reset_heading or count != self.num_envs:
            self.reset()
        else:
            for name, source in converted.items():
                getattr(self, name).copy_(source)


def _inertia(value, name="inertia", reference=None):
    inertia = _tensor(value, name, reference)
    _tail(inertia, (3, 3), name)
    eps = torch.finfo(inertia.dtype).eps * 64
    scale = inertia.abs().amax(dim=(-2, -1)).clamp_min(torch.finfo(inertia.dtype).tiny)
    if ((inertia - inertia.transpose(-1, -2)).abs().amax(dim=(-2, -1)) > eps * scale).any():
        raise ValueError(f"{name} must be symmetric")
    eig = torch.linalg.eigvalsh(inertia)
    if (eig[..., 0] < -eps * scale).any() or (eig[..., 2] > eig[..., :2].sum(-1) + eps * scale).any():
        raise ValueError(f"{name} must be a physically valid inertia")
    return inertia


def rotate_inertia_to_link(inertia, inertia_to_link_rotation):
    """Convert full COM inertia I to R I Rᵀ, including URDF base z90 rotation."""
    inertia = _inertia(inertia)
    rotation = _tensor(inertia_to_link_rotation, "inertia rotation", inertia)
    _tail(rotation, (3, 3), "rotation")
    _broadcast(inertia.shape[:-2], rotation.shape[:-2])
    identity = torch.eye(3, dtype=inertia.dtype, device=inertia.device)
    eps = torch.finfo(inertia.dtype).eps * 128
    if ((rotation @ rotation.transpose(-1, -2) - identity).abs() > eps).any() or (torch.linalg.det(rotation) - 1).abs().gt(eps).any():
        raise ValueError("inertia rotation must be a proper orthonormal rotation")
    return rotation @ inertia @ rotation.transpose(-1, -2)


def composite_mass_properties(masses, centers, inertias):
    """Combine (...,K), (...,K,3), (...,K,3,3) link-frame components.

    Each component inertia is about its own COM. Zero-mass components have
    zero inertia. Return total mass (...), COM (...,3), full inertia (...,3,3).
    """
    tensors = _inertia(inertias, "component inertia")
    mass = _tensor(masses, "component masses", tensors)
    com = _tensor(centers, "component COMs", tensors)
    _tail(com, (3,), "component COMs")
    if mass.ndim < 1 or com.ndim < 2 or tensors.ndim < 3 or mass.shape[-1] != com.shape[-2] or mass.shape[-1] != tensors.shape[-3]:
        raise ValueError("components must share a nonempty K axis")
    if mass.shape[-1] == 0 or (mass < 0).any():
        raise ValueError("component masses must be nonnegative with nonempty K")
    shape = _broadcast(mass.shape[:-1], com.shape[:-2], tensors.shape[:-3])
    k = mass.shape[-1]
    mass, com, tensors = mass.expand(*shape, k), com.expand(*shape, k, 3), tensors.expand(*shape, k, 3, 3)
    if ((mass == 0) & (tensors.abs().amax(dim=(-2, -1)) != 0)).any():
        raise ValueError("zero mass components must have zero inertia")
    total = mass.sum(-1)
    if (total <= 0).any():
        raise ValueError("total mass must be positive")
    center = (mass[..., None] * com).sum(-2) / total[..., None]
    delta = com - center[..., None, :]
    identity = torch.eye(3, dtype=tensors.dtype, device=tensors.device)
    shift = delta.square().sum(-1)[..., None, None] * identity - delta[..., :, None] * delta[..., None, :]
    result = (tensors + mass[..., None, None] * shift).sum(-3)
    return total, center, result


def _nominal(nominal_mass, nominal_com, nominal_inertia):
    inertia = _inertia(nominal_inertia, "nominal inertia")
    mass = _tensor(nominal_mass, "nominal mass", inertia)
    com = _tensor(nominal_com, "nominal COM", inertia)
    _tail(com, (3,), "nominal COM")
    shape = _broadcast(mass.shape, com.shape[:-1], inertia.shape[:-2])
    if (mass <= 0).any() or (torch.linalg.eigvalsh(inertia)[..., 0] <= 0).any():
        raise ValueError("nominal mass and inertia must be positive")
    return mass.expand(shape), com.expand(*shape, 3), inertia.expand(*shape, 3, 3)


def up3_mass_properties(nominal_mass, nominal_com, nominal_inertia, total_mass):
    """Scale bare link below m0; above m0 add a .05×.06×.05 box at fixed TCP.

    The supplied bare mass must match m0=.67148kg. Inertia/COM remain runtime
    inputs. Total requested mass is bounded [.6,6]kg. Repeated calls rebuild
    from the same nominal tensors and cannot accumulate payload.
    """
    mass, com, inertia = _nominal(nominal_mass, nominal_com, nominal_inertia)
    total = _tensor(total_mass, "up3 total mass", inertia)
    if ((total < .6) | (total > 6.)).any():
        raise ValueError("up3 total mass must be in [.6,6]kg")
    if ((mass - UP3_NOMINAL_MASS).abs() > 1e-6).any():
        raise ValueError("up3 bare nominal mass must be .67148kg")
    shape = _broadcast(mass.shape, total.shape)
    mass, com, inertia, total = mass.expand(shape), com.expand(*shape, 3), inertia.expand(*shape, 3, 3), total.expand(shape)
    extra = (total - mass).clamp_min(0.)
    dimensions = inertia.new_tensor(UP3_BOX_DIMENSIONS)
    diag = extra[..., None] * (dimensions.square().sum() - dimensions.square()) / 12
    tcp = inertia.new_tensor(TCP_LOCAL).expand(*shape, 3)
    _, combined_com, combined_inertia = composite_mass_properties(
        torch.stack((mass, extra), -1), torch.stack((com, tcp), -2),
        torch.stack((inertia, torch.diag_embed(diag)), -3))
    small = total <= mass
    return total, torch.where(small[..., None], com, combined_com), torch.where(
        small[..., None, None], inertia * (total / mass)[..., None, None], combined_inertia)


def rear_mass_properties(nominal_mass, nominal_com, nominal_inertia, extra_mass, rear_x):
    """Attach cylinderY proxy (radius .025,length .1) at [rear_x,0,.07]."""
    mass, com, inertia = _nominal(nominal_mass, nominal_com, nominal_inertia)
    extra = _tensor(extra_mass, "rear extra mass", inertia)
    x = _tensor(rear_x, "rear x", inertia)
    if ((extra < 0.) | (extra > 5.)).any() or ((x < -.35) | (x > -.20)).any():
        raise ValueError("rear mass/x must be in [0,5]kg/[-.35,-.20]m")
    shape = _broadcast(mass.shape, extra.shape, x.shape)
    mass, com, inertia = mass.expand(shape), com.expand(*shape, 3), inertia.expand(*shape, 3, 3)
    extra, x = extra.expand(shape), x.expand(shape)
    center = torch.stack((x, torch.zeros_like(x), torch.full_like(x, REAR_Z)), -1)
    radial = extra * (3 * REAR_CYLINDER_RADIUS ** 2 + REAR_CYLINDER_LENGTH ** 2) / 12
    axial = extra * REAR_CYLINDER_RADIUS ** 2 / 2
    result_mass, result_com, result_inertia = composite_mass_properties(
        torch.stack((mass, extra), -1), torch.stack((com, center), -2),
        torch.stack((inertia, torch.diag_embed(torch.stack((radial, axial, radial), -1))), -3))
    # Preserve exact nominal identity, avoiding multiply/divide round-off.
    nominal = extra == 0
    return result_mass, torch.where(nominal[..., None], com, result_com), torch.where(nominal[..., None, None], inertia, result_inertia)


def payload_stage_bounds(stage):
    """Return scalar (up3_min,up3_max,rear_extra_max) for current stage 0..4."""
    amplitude = PAYLOAD_STAGE_AMPLITUDES[_stage(stage)]
    return (UP3_NOMINAL_MASS + amplitude * (.6 - UP3_NOMINAL_MASS),
            UP3_NOMINAL_MASS + amplitude * (6. - UP3_NOMINAL_MASS), 5. * amplitude)


def sample_payload(num_envs, stage, *, device="cpu", dtype=torch.float32, generator=None):
    """Sample reset parameters only; callers hold them fixed for the episode.

    Mixture IDs: 0 nominal, 1 up3-only, 2 rear-only, 3 both, with probabilities
    .2/.2/.2/.4. Active dimensions and rear x are independently uniform.
    An optional caller-owned torch Generator makes reset sampling reproducible.
    """
    if isinstance(num_envs, bool) or int(num_envs) != num_envs or num_envs < 0:
        raise ValueError("num_envs must be a nonnegative integer")
    if not dtype.is_floating_point:
        raise ValueError("payload requires a floating dtype")
    lo, hi, rear_hi = payload_stage_bounds(stage)
    draws = torch.rand(int(num_envs), 4, device=device, dtype=dtype, generator=generator)
    mixture = torch.bucketize(draws[:, 0].contiguous(), draws.new_tensor((.2, .4, .6)))
    up3_active = (mixture == 1) | (mixture == 3)
    rear_active = (mixture == 2) | (mixture == 3)
    return dict(up3_mass=torch.where(up3_active, lo + (hi - lo) * draws[:, 1], UP3_NOMINAL_MASS),
                rear_mass=torch.where(rear_active, rear_hi * draws[:, 2], 0.),
                rear_x=-.35 + .15 * draws[:, 3], mixture=mixture)
