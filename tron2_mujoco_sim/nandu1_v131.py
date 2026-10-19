"""Independent NumPy deployment of the v1.3.1 yaw/TCP/payload contract.

Shapes are inertia proxies only. MuJoCo stores principal diagonal inertia;
composition operates on full COM tensors expressed in the link axes.
"""
import math
import numpy as np
from nandu1_contract import finite, rotation

TCP = np.array([-.066473051764, .161623719479, -.000621484902])
YAW_LIMITS = (.2, .5, .85, 1.2, 1.57)
PROCESSOR_CONTRACT = {'implementation': 'nandu1_v131.CommandProcessor', 'heading_kp': 1.5,
    'yaw_deadzone': 1e-5, 'stage_yaw_limits': list(YAW_LIMITS), 'release_latch': True,
    'tilt_correction': True, 'saturation_back_calculation': True,
    'vx_frame': 'base_body', 'height_frame': 'world_z_absolute', 'command_dimension': 3}
PAYLOAD_CONTRACT = dict(
    inertia_coordinates='link_axes_about_com', shapes_are_inertia_proxies_only=True,
    nominal_properties_source='immutable_runtime_mass_com_full_inertia',
    up3_nominal_mass_kg=.67148, up3_total_mass_kg=[.6,6.], up3_box_dimensions_m=[.05,.06,.05],
    up3_box_center='fixed_tcp', rear_extra_mass_kg=[0.,5.], rear_x_m=[-.35,-.20],
    rear_y_m=0., rear_z_m=.07, rear_cylinder_axis='link_y', rear_radius_m=.025, rear_length_m=.1,
    stage_amplitudes=[.1,.25,.5,.75,1.], mixture_order=['nominal','up3_only','rear_only','both'],
    mixture_probabilities=[.2,.2,.2,.4], base_urdf_inertia_rotation=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],
    parameters_constant_within_episode=True)


def validate_payload(up3_mass=.67148, rear_mass=0., rear_x=-.275):
    values = finite([up3_mass, rear_mass, rear_x], 3)
    if not (.6 <= values[0] <= 6 and 0 <= values[1] <= 5 and -.35 <= values[2] <= -.2):
        raise ValueError('Payload requires up3 total [.6,6], rear extra [0,5], rear x [-.35,-.2]')
    return values


def tcp_state(link_position, link_rotation, com_position, com_velocity, angular_velocity):
    point = np.asarray(link_position) + np.asarray(link_rotation).reshape(3, 3) @ TCP
    return point, np.asarray(com_velocity) + np.cross(angular_velocity, point - com_position)


def full_inertia(model, body_id):
    axes = rotation(model.body_iquat[body_id])
    return axes @ np.diag(model.body_inertia[body_id]) @ axes.T


def _combine(mass, com, inertia, extra, extra_com, extra_inertia):
    total = mass + extra
    center = (mass * com + extra * extra_com) / total
    result = inertia + extra_inertia
    for m, p in ((mass, com), (extra, extra_com)):
        d = p - center
        result = result + m * (d @ d * np.eye(3) - np.outer(d, d))
    return total, center, result


class FixedPayload:
    """Capture immutable nominal body properties once, rebuild for every apply."""
    def __init__(self, model):
        import mujoco  # Physics-only dependency; ROS validation needs only the contract.
        self.model = model
        self.ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    for name in ('base_Link', 'up3')}
        if min(self.ids.values()) < 0:
            raise ValueError('Payload requires base_Link and up3')
        self.nominal = {name: (float(model.body_mass[i]), model.body_ipos[i].copy(), full_inertia(model, i))
                        for name, i in self.ids.items()}
        self.raw = {name: (model.body_inertia[i].copy(), model.body_iquat[i].copy()) for name, i in self.ids.items()}
        if abs(self.nominal['up3'][0] - .67148) > 1e-6:
            raise ValueError('up3 nominal mass differs from .67148kg')
        self.parameters = dict(up3_total_mass=.67148, rear_mass=0., rear_x=-.275)

    def apply(self, data, up3_mass=.67148, rear_mass=0., rear_x=-.275):
        import mujoco
        up3_mass, rear_mass, rear_x = validate_payload(up3_mass, rear_mass, rear_x)
        bm, bc, bi = self.nominal['base_Link']
        um, uc, ui = self.nominal['up3']
        nominal_up3 = up3_mass == .67148
        if nominal_up3:
            # USD authored float32 mass becomes .671480000019... in MuJoCo;
            # nominal CLI identity preserves that immutable runtime baseline.
            up = um, uc, ui
        elif up3_mass <= um:
            up = up3_mass, uc, ui * (up3_mass / um)
        else:
            extra = up3_mass - um
            box_diag = extra * np.array([.06**2+.05**2, .05**2+.05**2, .05**2+.06**2]) / 12
            up = _combine(um, uc, ui, extra, TCP, np.diag(box_diag))
        cyl_diag = rear_mass * np.array([(3*.025**2+.1**2)/12, .025**2/2, (3*.025**2+.1**2)/12])
        base = _combine(bm, bc, bi, rear_mass, np.array([rear_x, 0., .07]), np.diag(cyl_diag))
        for name, properties, unchanged in (('base_Link', base, rear_mass == 0), ('up3', up, nominal_up3)):
            bid = self.ids[name]
            mass, center, tensor = properties
            self.model.body_mass[bid], self.model.body_ipos[bid] = mass, center
            if unchanged:
                self.model.body_ipos[bid] = self.nominal[name][1]
                self.model.body_inertia[bid], self.model.body_iquat[bid] = self.raw[name]
            else:
                diagonal, axes = np.linalg.eigh(tensor)
                if np.linalg.det(axes) < 0: axes[:, 0] *= -1
                quat = np.empty(4); mujoco.mju_mat2Quat(quat, axes.reshape(9))
                self.model.body_inertia[bid], self.model.body_iquat[bid] = diagonal, quat
        # Recompute model constants on scratch data: preserve the live qpos/qvel.
        mujoco.mj_setConst(self.model, mujoco.MjData(self.model))
        mujoco.mj_forward(self.model, data)
        self.parameters = dict(up3_total_mass=float(up3_mass), rear_mass=float(rear_mass), rear_x=float(rear_x))


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class CommandProcessor:
    """Yaw hold from body IMUs, raw base-body vx/absolute TCP height command."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.heading = self.heading_target = 0.
        self.was_turning = False

    def step(self, dt, gravity, gyro, external, stage):
        if not math.isfinite(dt) or not 1e-6 <= dt <= .1:
            raise ValueError('dt outside [1e-6,.1]')
        if type(stage) is not int or stage not in range(5):
            raise ValueError('stage must be 0..4')
        gravity, gyro, command = finite(gravity, 3), finite(gyro, 3), finite(external, 3).copy()
        norm = np.linalg.norm(gravity)
        up = -gravity / norm if norm > 1e-6 else np.array([0., 0., 1.])
        forward = np.array([1., 0., 0.]) - up[0] * up
        if np.linalg.norm(forward) <= 1e-6: forward = np.array([0., 1., 0.]) - up[1] * up
        forward /= max(np.linalg.norm(forward), 1e-6)
        left = np.cross(up, forward)
        projection = math.sqrt(max(1 - up[0]**2, 1e-4))
        rate = np.dot(np.cross(gyro, [1., 0., 0.]), left) / projection if projection > .01 else np.dot(gyro, up)
        self.heading = _wrap(self.heading + rate * dt)
        intent = np.clip(command[0], -YAW_LIMITS[stage], YAW_LIMITS[stage])
        turning = abs(intent) > 1e-5
        if self.was_turning and not turning: self.heading_target = self.heading
        self.heading_target = _wrap(self.heading_target + (intent * dt if turning else 0))
        coupling = np.dot(gyro, up) - rate
        raw = intent + 1.5 * _wrap(self.heading_target - self.heading) + coupling
        command[0] = np.clip(raw, -YAW_LIMITS[stage], YAW_LIMITS[stage])
        if raw != command[0]: self.heading_target = _wrap(self.heading + (command[0] - coupling - intent) / 1.5)
        self.was_turning = turning
        return command
