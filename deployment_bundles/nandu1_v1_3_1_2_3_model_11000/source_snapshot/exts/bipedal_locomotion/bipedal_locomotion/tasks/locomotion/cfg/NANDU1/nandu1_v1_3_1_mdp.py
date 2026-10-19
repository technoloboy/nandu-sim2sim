"""Isolated fixed-TCP semantics and episodic payloads for NANDU1 v1.3.1."""
from __future__ import annotations

import math
import torch
from isaaclab.managers import CommandTerm, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from . import nandu1_v1_mdp as v1
from . import nandu1_v1_3_mdp as v13
from . import nandu1_v1_robust_mdp as robust
from . import nandu1_v1_3_1_math as cm
from . import nandu1_v1_3_1_contract as contract

COMMAND_NAME = contract.COMMAND_NAME
SCHEMA = contract.SCHEMA
TCP_LOCAL = contract.TCP_LOCAL


def validate_fixed_payload(up3_mass, rear_mass, rear_x):
    values = (up3_mass, rear_mass, rear_x)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("fixed payload requires up3 total mass, rear mass and rear x together")
    limits = ((.6, 6.), (0., 5.), (-.35, -.20))
    for value, (low, high) in zip(values, limits):
        if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("fixed payload must be finite and within up3 [.6,6], rear [0,5], x [-.35,-.20]")
    return tuple(float(value) for value in values)


class Nandu1TCPHeightCommand(v13.Nandu1TrackingOnlyHeightCommand):
    REACH_SOFT_LIMIT = .77
    REACH_HARD_LIMIT = .87
    REACH_DIAGNOSTIC_LIMITS = (.84, .85, .87)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.heading_controller = cm.HeadingController(self.num_envs, self.device, self._command.dtype,
                                                       validate_inputs=False)
        self._tcp_cache_step = None
        self._tcp_local = self._command.new_tensor(TCP_LOCAL).expand(self.num_envs, 3)
        self.evaluation_snapshot = {}
        for name in ("Yaw/intent", "Yaw/applied", "Yaw/heading_error", "Geometry/TCP_height",
                     "Geometry/up3_origin_height", "Geometry/up3_com_height"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _invalidate_point_cache(self):
        self._tcp_cache_step = None

    def tcp_state_world(self):
        step = self._env.common_step_counter
        if getattr(self, "_tcp_cache_step", None) != step:
            d = self._robot.data
            local = getattr(self, "_tcp_local", None)
            if local is None:
                local = d.body_pos_w.new_tensor(TCP_LOCAL).expand(self.num_envs, 3)
                self._tcp_local = local
            position = d.body_pos_w[:, self._up3_id] + quat_apply(d.body_quat_w[:, self._up3_id], local)
            velocity = d.body_lin_vel_w[:, self._up3_id] + torch.linalg.cross(
                d.body_ang_vel_w[:, self._up3_id], position - d.body_com_pos_w[:, self._up3_id], dim=-1)
            self._tcp_cache = (position, velocity)
            self._tcp_cache_step = step
        return self._tcp_cache

    def _up3_position_w(self):
        return self.tcp_state_world()[0]

    def _up3_velocity_w(self):
        return self.tcp_state_world()[1]

    def _base_point_velocity_w(self, up3_position):
        d = self._robot.data
        return d.body_lin_vel_w[:, self._base_id] + torch.linalg.cross(
            d.body_ang_vel_w[:, self._base_id], up3_position - d.body_com_pos_w[:, self._base_id], dim=-1)

    def _vertical_velocity_components(self, up3_velocity, base_point_velocity):
        return base_point_velocity[:, 2], up3_velocity[:, 2] - base_point_velocity[:, 2]

    def _kinematic_up3_height(self):
        return v1.kinematic_up3_height(self._env, terminal_origin=TCP_LOCAL)

    def _update_command(self):
        previous_yaw = self._command[:, 0].clone()
        super()._update_command()
        gravity = robust.noisy_observation(self._env, "gravity", COMMAND_NAME, sensor_cfg=SceneEntityCfg("base_imu"))
        gyro = robust.noisy_observation(self._env, "gyro", COMMAND_NAME, sensor_cfg=SceneEntityCfg("base_imu"))
        self._command[:, 0] = self.heading_controller.step(
            self._env.step_dt, gravity, gyro, self.target_command[:, 0], self.active_stage)
        self.command_acceleration[:, 0] = (self._command[:, 0] - previous_yaw) / self._env.step_dt

    def set_external_command(self, command):
        applied_yaw = self._command[:, 0].clone()
        super().set_external_command(command)
        # External yaw is intent; retain the sensor controller's applied value
        # until the next physical command step. vx/height keep v1.3 semantics.
        self._command[:, 0] = applied_yaw

    def reset(self, env_ids=None):
        ids = v1._ids(env_ids, self.num_envs, self.device)
        self._invalidate_point_cache()
        height = self._kinematic_up3_height().squeeze(-1)[ids].clamp(*self.STAGE_HEIGHT_RANGES[self.active_stage])
        self.heading_controller.reset(ids)
        self._command[ids, :2] = 0
        self._command[ids, 2] = height
        self.target_command[ids] = self._command[ids]
        self.sampled_height_target[ids] = height
        for name in ("commanded_height_rate", "height_reference_velocity", "height_low_frequency_velocity",
                     "height_high_frequency_velocity", "filtered_base_vx_target", "static_gate",
                     "wheel_spacing_rate", "knee_spacing_rate", "disturbance_recovery_elapsed", "disturbance_recovery_stable"):
            getattr(self, name)[ids] = 0
        self.alpha_base[ids] = self.horizontal_alpha_base[ids] = .5
        self._spacing_history_valid[ids] = False
        self.command_acceleration[ids] = 0
        self.height_step_segment[ids] = self.pending_height_step[ids] = False
        self.disturbance_active[ids] = self.disturbance_recovery_pending[ids] = False
        return CommandTerm.reset(self, ids)

    def _update_metrics(self):
        super()._update_metrics()
        d = self._robot.data
        position, velocity = self.tcp_state_world()
        scale = 1. / self._env.max_episode_length
        h = self.heading_controller
        heading_error = torch.atan2(torch.sin(h.heading_target-h.heading),torch.cos(h.heading_target-h.heading))
        for name,value in (("Yaw/intent",self.target_command[:,0]),("Yaw/applied",self.command[:,0]),
                           ("Yaw/heading_error",heading_error), ("Geometry/TCP_height",position[:,2]),
                           ("Geometry/up3_origin_height",d.body_pos_w[:,self._up3_id,2]),
                           ("Geometry/up3_com_height",d.body_com_pos_w[:,self._up3_id,2])):
            self.metrics[name] += value * scale
        # Copies are diagnostics only; no simulator truth enters the Actor.
        self.evaluation_snapshot = dict(step=self._env.common_step_counter, command=self.command.clone(),
            intent=self.target_command.clone(), position=position.clone(), velocity=velocity.clone(),
            actual=torch.stack((d.root_ang_vel_w[:,2],quat_apply_inverse(d.root_quat_w,velocity)[:,0],position[:,2]),-1))

    def state_dict(self):
        state = super().state_dict()
        state["heading_controller"] = self.heading_controller.state_dict()
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if "heading_controller" in state:
            self.heading_controller.load_state_dict(state["heading_controller"])
        else:
            self.heading_controller.reset()
        self._invalidate_point_cache()

    @torch.inference_mode()
    def restart_after_resume(self):
        """New physics world at restored course; no curriculum update or fake reset record."""
        env = self._env
        ids = torch.arange(self.num_envs, device=self.device)
        env.scene.reset(ids)
        env.event_manager.apply(mode="reset", env_ids=ids, global_env_step_count=env.common_step_counter)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.)
        env._nandu_v1_measurement_cache = {}
        env.action_manager.reset(ids)
        env.observation_manager.reset(ids)
        self.reset(ids)
        env.event_manager.reset(ids)
        env.episode_length_buf[ids] = 0
        env.obs_buf = env.observation_manager.compute(update_history=True)


class TCPPayloadDomainRandomization(robust.HeightRobustDomainRandomization):
    """Replace only base/up3 mass properties; PhysX API tensors are already link axes.

    omni.physics.tensors ArticulationView.get/set_inertias documents full COM
    inertia in rigid-body-prim axes (column-major 9). get_coms quaternion is the
    principal-axis rotation, NOT an extra rotation to apply to these tensors.
    """
    def __init__(self, cfg, env):
        robot = env.scene["robot"]
        self._payload_base_id = robot.find_bodies("base_Link")[0][0]
        self._payload_up3_id = robot.find_bodies("up3")[0][0]
        cfg.params["excluded_body_ids"] = (self._payload_base_id, self._payload_up3_id)
        super().__init__(cfg, env)
        view = robot.root_physx_view
        self.nominal_masses = view.get_masses().clone().cpu()
        self.nominal_coms = view.get_coms().clone().cpu()
        self.nominal_inertias = view.get_inertias().clone().cpu().reshape(env.num_envs, robot.num_bodies,3,3).transpose(-1,-2).contiguous()
        self.fixed_payload = validate_fixed_payload(env.cfg.fixed_up3_total_mass, env.cfg.fixed_rear_mass, env.cfg.fixed_rear_x)
        env._nandu1_payload_event = self

    def __call__(self, env, env_ids, command_name, asset_cfg, material_buckets_per_stage=48, excluded_body_ids=()):
        ids = env_ids.to(device="cpu", dtype=torch.long)
        if not len(ids):
            return
        term = env.command_manager.get_term(command_name)
        if bool(getattr(getattr(term,"cfg",None), "robustness_enabled", True)):
            super().__call__(env, env_ids, command_name, asset_cfg, material_buckets_per_stage)
        if self.fixed_payload is None:
            params = cm.sample_payload(len(ids), int(term.stage))
        else:
            u,r,x = self.fixed_payload
            params = dict(up3_mass=torch.full((len(ids),),u),rear_mass=torch.full((len(ids),),r),
                          rear_x=torch.full((len(ids),),x),mixture=torch.full((len(ids),),-1,dtype=torch.long))
        view = self._robot.root_physx_view
        masses, coms, inertias = view.get_masses(),view.get_coms(),view.get_inertias()
        for body, function, arguments in ((self._payload_up3_id,cm.up3_mass_properties,(params["up3_mass"],)),
                                          (self._payload_base_id,cm.rear_mass_properties,(params["rear_mass"],params["rear_x"]))):
            mass,com,inertia = function(self.nominal_masses[ids,body],self.nominal_coms[ids,body,:3],
                                       self.nominal_inertias[ids,body],*arguments)
            masses[ids,body] = mass
            coms[ids,body] = self.nominal_coms[ids,body]
            coms[ids,body,:3] = com
            inertias[ids,body] = inertia.transpose(-1,-2).reshape(-1,9)
        # set_coms sets mass-frame pose; set_inertias then recomputes its principal
        # axes from the full link-frame tensor, so do not overwrite that rotation.
        view.set_masses(masses,ids)
        view.set_coms(coms,ids)
        view.set_inertias(inertias,ids)
        device_ids = ids.to(env.device)
        if not hasattr(env,"_nandu1_randomized_body_masses"):
            env._nandu1_randomized_body_masses = self.nominal_masses.to(env.device).clone()
        env._nandu1_randomized_body_masses[device_ids] = masses[ids].to(env.device)
        if not hasattr(env,"_nandu1_payload_parameters"):
            env._nandu1_payload_parameters = {k:torch.zeros(env.num_envs,device=env.device,dtype=v.dtype) for k,v in params.items()}
        for key,value in params.items():
            env._nandu1_payload_parameters[key][device_ids] = value.to(env.device)
        if hasattr(term,"_invalidate_point_cache"):
            term._invalidate_point_cache()


def tcp_body_linear_velocity(env, command_name=COMMAND_NAME):
    term = env.command_manager.get_term(command_name)
    return quat_apply_inverse(term._robot.data.root_quat_w,term._up3_velocity_w())


def true_tcp_height(env, command_name=COMMAND_NAME):
    return env.command_manager.get_term(command_name)._up3_position_w()[:,2:3]


def actual_body_mass(env):
    return getattr(env,"_nandu1_randomized_body_masses",env.scene["robot"].data.default_mass).to(env.device)
