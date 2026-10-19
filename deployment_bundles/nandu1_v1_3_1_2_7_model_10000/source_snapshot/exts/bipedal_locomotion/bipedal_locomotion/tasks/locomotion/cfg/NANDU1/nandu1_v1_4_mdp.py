"""v1.4 command/reward bridge. No changes to the parent v1.3 implementation.

Stateful allocation and statistics run once after physics, before command
resampling. Rewards and diagnostics therefore use the command that produced the
action, never the next reference. Truth is used only for these training targets.
"""
from __future__ import annotations

import math
import torch
from isaaclab.managers import CommandTerm, SceneEntityCfg

from . import nandu1_math as common
from . import nandu1_mdp as v0
from . import nandu1_v1_mdp as v1
from . import nandu1_v1_1_math as coordination
from . import nandu1_v1_3_mdp as v13
from . import nandu1_v1_robust_mdp as robust
from . import nandu1_v1_4_math as command_math
from . import nandu1_v1_4_coordination as cm
from .nandu1_v1_4_contract import COMMAND_NAME, SCHEMA


def _link_velocity(robot, body_id):
    return robot.data.body_link_lin_vel_w[:, body_id]


class Nandu1DecoupledHeightCommand(v13.Nandu1TrackingOnlyHeightCommand):
    PROCESSOR_CLASS = command_math.CommandProcessor
    CHECKPOINT_SCHEMA = SCHEMA
    COMMAND_NAME = COMMAND_NAME

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._wheel_joint_ids = self._robot.find_joints(["wheel_L_Joint", "wheel_R_Joint"], preserve_order=True)[0]
        self.processor = self.PROCESSOR_CLASS(self.num_envs, self.device, self._command.dtype)
        self.applied_intent = cm.applied_intent(self.processor.goal,self.processor.mode,self.processor.active_axes)
        self._state_latch = cm.StepLatch()
        self._statistics_latch = cm.StepLatch()
        self.static_intent = torch.zeros_like(self.static_gate)
        self.arm_high_frequency_availability = torch.zeros_like(self.static_gate)
        self.errors = torch.zeros(self.num_envs, 4, device=self.device)
        self.base_velocity_h = torch.zeros(self.num_envs, 3, device=self.device)
        self.up3_velocity_h = torch.zeros_like(self.base_velocity_h)
        self._target_history = torch.zeros(4, self.num_envs, 13, device=self.device)
        self._target_history_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.action_rate_cost = torch.zeros_like(self.static_gate)
        self.action_smoothness_cost = torch.zeros_like(self.static_gate)
        self.jerk = torch.zeros(self.num_envs, 13, device=self.device)
        self.evaluation_snapshot = {}
        self._metric_names = (
            "V14/static_intent", "V14/arm_high_frequency_availability", "V14/position_target_jerk",
            "V14/wheel_target_jerk", "V14/heading_error", "V14/lateral_velocity",
            "V14/mode_static", "V14/mode_single", "V14/mode_double", "V14/mode_triple",
            "V14/height_transition_error", "V14/height_hold_error",
        )
        self.metrics.update({name: torch.zeros(self.num_envs, device=self.device) for name in self._metric_names})
        # The parent had FK diagnostic slots; v1.4 does not log zero placeholders
        # that could be mistaken for an actual height estimate.
        for name in ("Height/kinematic_estimate", "Height/estimate_error"):
            self.metrics.pop(name, None)
        self.metrics.update({name:torch.zeros(self.num_envs,device=self.device) for name in (
            "Geometry/base_height", "Geometry/arm_relative_height", "Geometry/up3_base_horizontal_distance",
            "Geometry/COM_support_offset", "Allocation/actual_base_vz", "Allocation/actual_arm_vz",
            "Allocation/target_base_vz", "Allocation/target_arm_vz", "Symmetry/wheel_spacing", "Symmetry/knee_spacing",
            "V14/height_transition_fraction", "V14/height_hold_fraction")})

    def _resample_command(self, env_ids):
        if self._external_control or not hasattr(self, "processor"):
            return
        ids = v1._ids(env_ids, self.num_envs, self.device)
        self.processor.resample(ids, self.active_stage)
        # The processor owns mode dwell and asynchronous axis timers.
        self.time_left[ids] = self.processor.mode_time_left[ids]

    def _update_command(self):
        gravity = robust.noisy_observation(self._env, "gravity", self.COMMAND_NAME,
                                           sensor_cfg=SceneEntityCfg("base_imu"))
        gyro = robust.noisy_observation(self._env, "gyro", self.COMMAND_NAME,
                                        sensor_cfg=SceneEntityCfg("base_imu"))
        previous = self._command.clone()
        self._command.copy_(self._advance_processor(gravity, gyro))
        self.applied_intent = cm.applied_intent(self.processor.goal,self.processor.mode,self.processor.active_axes)
        self.target_command.copy_(self.processor.goal)
        self.sampled_height_target.copy_(self.processor.goal[:, 2])
        self.commanded_height_rate.copy_(self.processor.height_rate)
        self.height_step_segment.copy_(self.processor.is_step)
        self.height_rate_limited.copy_((self.processor.goal[:, 2] - self.command[:, 2]).abs() > 1e-6)
        self.command_acceleration.copy_(((self._command - previous) / self._env.step_dt).clamp(-5., 5.))
        self.command_acceleration[:, 2] = self.commanded_height_rate
        gain = self._env.step_dt / (.8 + self._env.step_dt)
        self.filtered_base_vx_target.add_(gain * (self.command[:, 1] - self.filtered_base_vx_target))
        # Disable CommandTerm's timer: processor resamples its own mode/axis timers.
        self.time_left.fill_(1e9)

    def _advance_processor(self, gravity, gyro):
        return self.processor.step(self._env.step_dt, gravity, gyro, self.active_stage)

    def set_external_command(self, command):
        if self._evaluation_stage is None:
            raise RuntimeError("call set_evaluation_stage before external control")
        self.processor.set_external(command, self.active_stage)
        self._external_control = True
        self.time_left.fill_(1e9)

    def reset(self, env_ids=None):
        ids = v1._ids(env_ids, self.num_envs, self.device)
        # Initial reference may use proprioceptive FK; no continuous position loop.
        height = v1.kinematic_up3_height(self._env).squeeze(-1)[ids].clamp(*self.STAGE_HEIGHT_RANGES[self.active_stage])
        self.processor.reset(ids, height)
        for name in self.applied_intent:
            self.applied_intent[name][ids] = getattr(self.processor,name)[ids]
        self._command[ids, :2] = 0
        self._command[ids, 2] = height
        self.target_command[ids] = self._command[ids]
        self.sampled_height_target[ids] = height
        for name in ("commanded_height_rate", "height_reference_velocity", "height_low_frequency_velocity",
                     "height_high_frequency_velocity", "filtered_base_vx_target", "static_gate", "static_intent"):
            getattr(self, name)[ids] = 0
        self.alpha_base[ids] = self.horizontal_alpha_base[ids] = .5
        self._spacing_history_valid[ids] = False
        self._target_history_count[ids] = 0
        self._target_history[:, ids] = 0
        self.command_acceleration[ids] = 0
        self.height_step_segment[ids] = self.pending_height_step[ids] = False
        # CommandTerm.reset emits/clears metrics and invokes our sampler. Do not
        # call parent resets that advance filters using pre-reset/stale geometry.
        return CommandTerm.reset(self, ids)

    def _update_coordination_state(self):
        if not hasattr(self, "_state_latch") or not self._state_latch.begin(self._env.common_step_counter):
            return
        r = self._robot
        dt = self._env.step_dt
        bp, ep = r.data.body_pos_w[:, self._base_id], r.data.body_pos_w[:, self._up3_id]
        wheels = r.data.body_pos_w[:, self._wheel_ids]
        rel = ep - bp
        vb, ve = _link_velocity(r, self._base_id), _link_velocity(r, self._up3_id)
        omega = r.data.body_ang_vel_w[:, self._base_id]
        vb_at_ee = vb + torch.linalg.cross(omega, rel, dim=-1)
        va = ve - vb_at_ee
        heading = lambda x: v0._heading_velocity(r.data.root_quat_w, x)
        self.base_velocity_h.copy_(heading(vb))
        self.up3_velocity_h.copy_(heading(ve))
        self.relative_up3_h.copy_(heading(rel))
        self.horizontal_reach.copy_(rel[:, :2].norm(dim=-1))
        self.horizontal_reach_rate.copy_((rel[:, :2] * (ve-vb)[:, :2]).sum(-1) / self.horizontal_reach.clamp_min(1e-6))
        self.base_height.copy_(bp[:, 2] - (wheels[:, :, 2].mean(-1) - .1))
        self.arm_relative_height.copy_(rel[:, 2])
        self.actual_base_vx.copy_(heading(vb_at_ee)[:, 0])
        self.actual_arm_vx.copy_(heading(va)[:, 0])
        self.actual_base_vz.copy_(vb_at_ee[:, 2])
        self.actual_arm_vz.copy_(va[:, 2])
        masses = getattr(self._env, "_nandu1_randomized_body_masses", r.data.default_mass).to(self.device)
        com = (r.data.body_com_pos_w * masses[..., None]).sum(1) / masses.sum(1, keepdim=True).clamp_min(1e-6)
        self.com_support_offset.copy_((com[:, :2]-wheels.mean(1)[:, :2]).norm(dim=-1))
        loads = self._env.scene.sensors["contact_forces"].data.net_forces_w[:, self._wheel_contact_ids, 2].clamp_min(0.)
        self.wheel_load_difference.copy_((loads[:, 0]-loads[:, 1]).abs() / loads.sum(-1).clamp_min(1.))
        util = r.data.applied_torque.abs() / r.data.joint_effort_limits.clamp_min(1e-6)
        bq = (1-util[:, self._leg_ids].amax(-1)).clamp(0, 1)
        aq = (1-util[:, self._arm_ids].amax(-1)).clamp(0, 1)
        cq = torch.exp(-(self.com_support_offset/.08).square())
        lq = (1-self.wheel_load_difference).clamp(0, 1)
        rq = ((.70-self.horizontal_reach)/.10).clamp(0, 1)
        sq = (1-r.data.joint_vel[:, self._arm_ids].abs().div(r.data.joint_vel_limits[:, self._arm_ids].clamp_min(.1)).amax(-1)).clamp(0, 1)
        self.arm_high_frequency_availability.copy_((aq * sq * rq).sqrt())
        eh = self.command[:, 2] - ep[:, 2]
        self._update_vertical_reference(eh, ve[:, 2], dt)
        br, ar = self.STAGE_BASE_HEIGHT_RANGES[self.active_stage], self.STAGE_ARM_HEIGHT_RANGES[self.active_stage]
        self.base_directional_margin.copy_(common.directional_margin(self.height_reference_velocity, self.base_height, br))
        self.arm_directional_margin.copy_(common.directional_margin(self.height_reference_velocity, self.arm_relative_height, ar))
        raw = common.height_allocation_ratio(self.height_reference_velocity, self.base_height, self.arm_relative_height,
                                             br, ar, bq*cq*lq, aq*rq)
        self.alpha_base.add_(dt/(.3+dt)*(raw-self.alpha_base))
        bv, av = cm.allocate(self.height_reference_velocity, self.height_low_frequency_velocity,
                             self.alpha_base, self.arm_high_frequency_availability)
        self.target_base_vz.copy_(bv)
        self.target_arm_vz.copy_(av)
        ha = common.horizontal_allocation_ratio(self.filtered_base_vx_target, self.relative_up3_h[:, 0],
                                                self.horizontal_reach, bq*cq*lq, aq*rq)
        self.horizontal_alpha_base.add_(dt/(.3+dt)*(ha-self.horizontal_alpha_base))
        hx, ax = cm.allocate(self.command[:,1],self.filtered_base_vx_target,
                              self.horizontal_alpha_base,self.arm_high_frequency_availability)
        self.target_base_vx.copy_(hx)
        self.target_arm_vx.copy_(ax)
        self.leg_position_residual.copy_(common.leg_mirror_residual(r.data.joint_pos[:, self._left_leg_ids], r.data.joint_pos[:, self._right_leg_ids]))
        self.leg_velocity_residual.copy_(common.leg_mirror_residual(r.data.joint_vel[:, self._left_leg_ids], r.data.joint_vel[:, self._right_leg_ids]))
        wd = heading(wheels[:, 0]-wheels[:, 1])
        self.wheel_height_difference.copy_(wd[:, 2].abs())
        self.wheel_spacing.copy_(wd[:, 1].abs())
        knees = r.data.body_pos_w[:, self._knee_body_ids]
        self.knee_delta_h.copy_(heading(knees[:, 0]-knees[:, 1]))
        self.knee_spacing.copy_(self.knee_delta_h[:, 1].abs())
        self.wheel_spacing_rate.copy_(torch.where(self._spacing_history_valid, (self.wheel_spacing-self._previous_wheel_spacing)/dt, 0.))
        self.knee_spacing_rate.copy_(torch.where(self._spacing_history_valid, (self.knee_spacing-self._previous_knee_spacing)/dt, 0.))
        self._previous_wheel_spacing.copy_(self.wheel_spacing)
        self._previous_knee_spacing.copy_(self.knee_spacing)
        self._spacing_history_valid.fill_(True)
        tilt = torch.acos((-self._env.scene.sensors["up3_imu"].data.projected_gravity_b[:, 2]).clamp(-1, 1))
        self.errors.copy_(torch.stack((r.data.root_ang_vel_w[:, 2]-self.command[:, 0], self.up3_velocity_h[:, 0]-self.command[:, 1], -eh, tilt), -1))
        self.tracking_gate.copy_(torch.exp(-(eh/.1).square()-(tilt/.15).square()))
        self.symmetry_gate.copy_(common.tracking_quality_gate(self.errors.abs(), (.25,.14,.1,math.radians(15))))
        self.coordination_task_gate.copy_(coordination.task_quality_gate(self.errors[:, 0], self.errors[:, 1], eh))
        cost, occupancy = coordination.continuous_vertical_cancellation(self.actual_base_vz, self.actual_arm_vz, eh, self.commanded_height_rate)
        self.continuous_cancellation_cost.copy_(cost)
        self.continuous_cancellation_occupancy.copy_(occupancy)
        self.cancellation.copy_(occupancy > .5)
        leg = coordination.leg_geometry_cost(self.wheel_spacing, self.knee_spacing, self.leg_position_residual.square().mean(-1).sqrt(),
                                             self.wheel_spacing_rate, self.knee_spacing_rate,
                                             (self.command[:, 0].abs()/self.STAGE_LIMITS[self.active_stage][0]).clamp(0,1))
        for name, val in (("leg_geometry_shaping_cost",leg.shaping),("leg_geometry_hard_cost",leg.hard),
                          ("leg_geometry_outward_cost",leg.outward),("wheel_geometry_violation",leg.wheel_violation),
                          ("knee_geometry_violation",leg.knee_violation)):
            getattr(self,name).copy_(val)
        reach = coordination.aligned_reach_cost(self.horizontal_reach, self.horizontal_reach_rate)
        for name,val in (("reach_shaping_cost",reach.shaping),("reach_hard_cost",reach.hard),
                         ("reach_outward_cost",reach.outward),("reach_curriculum_measure",reach.curriculum_measure)):
            getattr(self,name).copy_(val)
        intent, settled = cm.static_gates(self.applied_intent['goal'], eh, self.commanded_height_rate)
        self.static_intent.copy_(intent)
        self.static_gate.copy_(settled)
        action = self._env.action_manager.action
        self._update_action_history(action, dt)

    def _update_vertical_reference(self, eh, actual_vz, dt):
        vmax = self.STAGE_VREF_LIMITS[self.active_stage]
        self.height_reference_velocity.copy_((1.5*eh-.25*(actual_vz-self.commanded_height_rate)).clamp(-vmax, vmax))
        self.height_low_frequency_velocity.add_(dt/(.8+dt)*(self.height_reference_velocity-self.height_low_frequency_velocity))
        self.height_high_frequency_velocity.copy_(self.height_reference_velocity-self.height_low_frequency_velocity)

    def _update_action_history(self, action, dt):
        # Group-normalized physical targets: rad for positions, rad/s for wheels.
        physical = action * action.new_tensor([.25]*11+[1.]*2)
        hist = self._target_history
        hist[:-1] = hist[1:].clone()
        hist[-1] = physical
        self._target_history_count.add_(1)
        scales = physical.new_tensor([.25]*11+[1.]*2)
        d1, d2 = (hist[-1]-hist[-2])/scales, (hist[-1]-2*hist[-2]+hist[-3])/scales
        group_mean = lambda x: (x[:, :8].square().mean(-1)+x[:, 8:11].square().mean(-1)+x[:, 11:].square().mean(-1))/3
        self.action_rate_cost.copy_(group_mean(d1)*(self._target_history_count >= 2))
        self.action_smoothness_cost.copy_(group_mean(d2)*(self._target_history_count >= 3))
        self.jerk.copy_(cm.target_jerk(hist,dt)*(self._target_history_count >= 4)[:,None])

    def ensure_reward_state(self):
        self._update_coordination_state()
        self._record_metrics()

    def _update_metrics(self):
        # CommandTerm calls this after reset: the reward's pre-reset record wins.
        self.ensure_reward_state()

    def _record_metrics(self):
        if not self._statistics_latch.begin(self._env.common_step_counter):
            return
        e = self.errors.abs()
        scale = 1. / self._env.max_episode_length
        self.evaluation_snapshot = {
            "actual":torch.stack((self._robot.data.root_ang_vel_w[:,2],self.up3_velocity_h[:,0],
                                  self._robot.data.body_pos_w[:,self._up3_id,2]),-1).clone(),
            "position":self._robot.data.body_pos_w[:,self._up3_id].clone(),
            "command":self.command.clone(), "mode":self.applied_intent['mode'].clone(),
            "is_step":self.height_step_segment.clone(), "active_axes":self.applied_intent['active_axes'].clone(),
            "height_rate":self.commanded_height_rate.clone(),
        }
        self.window_error_sum.add_(e.mean(0))
        self.window_error_square_sum.add_(e.square().mean(0))
        self.window_samples += 1
        self.height_error_histogram.update(e[:, 2])
        for name, val in (("yaw_abs_error",e[:,0]),("ee_vx_abs_error",e[:,1]),("height_abs_error",e[:,2]),("up3_tilt",e[:,3]),
                          ("Command/applied_height",self.command[:,2]),("Command/sample_height_target",self.sampled_height_target),
                          ("Command/height_rate",self.commanded_height_rate),("Command/is_step",self.height_step_segment.float()),
                          ("Command/rate_limit_active",self.height_rate_limited.float()),("Height/sim_true",self._robot.data.body_pos_w[:,self._up3_id,2]),
                          ("Allocation/vref",self.height_reference_velocity),("Allocation/vlow",self.height_low_frequency_velocity),
                          ("Allocation/vhigh",self.height_high_frequency_velocity),("Allocation/alpha_base",self.alpha_base),
                          ("Static/gate",self.static_gate),("V14/static_intent",self.static_intent),
                          ("V14/arm_high_frequency_availability",self.arm_high_frequency_availability),
                          ("V14/position_target_jerk",self.jerk[:,:11].square().mean(-1).sqrt()),
                          ("V14/wheel_target_jerk",self.jerk[:,11:].square().mean(-1).sqrt()),
                          ("V14/lateral_velocity",self.up3_velocity_h[:,1].abs()),
                          ("Coordination/task_gate",self.coordination_task_gate),
                          ("Coordination/cancellation_cost",self.continuous_cancellation_cost),
                          ("Coordination/leg_geometry_cost",self.leg_geometry_shaping_cost+4*self.leg_geometry_hard_cost),
                          ("Coordination/reach_cost",self.reach_shaping_cost+4*self.reach_hard_cost),
                          ("Geometry/base_height",self.base_height),("Geometry/arm_relative_height",self.arm_relative_height),
                          ("Geometry/up3_base_horizontal_distance",self.horizontal_reach),("Geometry/COM_support_offset",self.com_support_offset),
                          ("Allocation/actual_base_vz",self.actual_base_vz),("Allocation/actual_arm_vz",self.actual_arm_vz),
                          ("Allocation/target_base_vz",self.target_base_vz),("Allocation/target_arm_vz",self.target_arm_vz),
                          ("Symmetry/wheel_spacing",self.wheel_spacing),("Symmetry/knee_spacing",self.knee_spacing)):
            self.metrics[name].add_(val*scale)
        self.metrics["stage"].fill_(self.stage)
        for n, name in enumerate(("static","single","double","triple")):
            self.metrics[f"V14/mode_{name}"].add_((self.applied_intent['mode']==n)*scale)
        self.metrics["V14/heading_error"].add_(torch.atan2(torch.sin(self.processor.heading_target-self.processor.heading),torch.cos(self.processor.heading_target-self.processor.heading)).abs()*scale)
        transition = self.height_rate_limited | self.height_step_segment
        self.metrics["V14/height_transition_error"].add_(e[:,2]*transition*scale)
        self.metrics["V14/height_hold_error"].add_(e[:,2]*(~transition)*scale)
        self.metrics["V14/height_transition_fraction"].add_(transition*scale)
        self.metrics["V14/height_hold_fraction"].add_((~transition)*scale)
        self._coverage_stride += 1
        if self._coverage_stride % 10 == 0:
            lo,hi = self.STAGE_HEIGHT_RANGES[self.active_stage]
            yl,xl = self.STAGE_LIMITS[self.active_stage]
            for row,(low,high) in enumerate(((-yl,yl),(-xl,xl),(lo,hi))):
                ids = (((self.command[:,row]-low)/(high-low)).clamp(0,.9999)*10).long()
                self.window_coverage[row].add_(torch.bincount(ids,minlength=10))
            self.window_coverage[4,0] += ((self.command[:,2]<lo+.1*(hi-lo))|(self.command[:,2]>hi-.1*(hi-lo))).sum()
        settled = self.static_gate>.5
        self.window_settled_vz_square_sum += float((self.up3_velocity_h[:,2].square()*settled).sum())
        self.window_settled_samples += int(settled.sum())
        self.window_static_drift_square_sum += float((self._robot.data.root_ang_vel_w.square().sum(-1)*settled).sum())
        self.window_static_samples += int(settled.sum())*3
        util = self._robot.data.applied_torque.abs()/self._robot.data.joint_effort_limits.clamp_min(1e-6)
        self.window_torque_high_sum += float((util>.9).float().mean())
        self.window_v11_coordination_sum.add_(torch.stack((self.continuous_cancellation_occupancy.mean(),self.leg_position_residual.square().mean(-1).sqrt().mean(),
                                                       self.wheel_geometry_violation.mean(),self.knee_geometry_violation.mean(),self.reach_curriculum_measure.mean(),self.com_support_offset.mean())))
        self.window_coordination_sum.add_(torch.stack(((self.horizontal_reach>.67).float().mean(),self.leg_position_residual.square().mean(),self.wheel_geometry_violation.mean(),
                                                      self.cancellation.float().mean(),(self.wheel_height_difference>.03).float().mean(),(self.horizontal_reach>.68).float().mean(),
                                                      (self.horizontal_reach>.70).float().mean(),self.knee_geometry_violation.mean(),self.com_support_offset.mean())))
        self.window_reach_samples += self.num_envs
        self._recovery_metrics(e, scale)

    def _recovery_metrics(self, errors, scale):
        # Retain robust diagnostics but evaluate vx in Heading and link-origin vz.
        scheduler = getattr(self._env,"_nandu_v1_robust_disturbance",None)
        for key,val in robust.robust_math.STAGES[self.stage].items():
            self.metrics[f"Robustness/{key}_limit"].add_(val*scale)
        active = self.disturbance_active
        pending = self.disturbance_recovery_pending
        recovery = pending & ~active
        normalized = (errors/errors.new_tensor([.25,.14,.1,math.radians(10)])).mean(-1)
        for label,mask in (("clean",~(active|recovery)),("disturbed",active),("recovery",recovery)):
            self.metrics[f"Tracking/{label}_error"].add_(normalized*mask*scale)
            self.metrics[f"Tracking/{label}_fraction"].add_(mask*scale)
        if scheduler is not None:
            self.metrics["Disturbance/active"].add_(active*scale)
            self.metrics["Disturbance/force_norm"].add_(scheduler.applied_force.norm(dim=-1)*scale)
            self.metrics["Disturbance/torque_norm"].add_(scheduler.applied_torque.norm(dim=-1)*scale)
        good = (errors[:,:3] < errors.new_tensor([.2,.1,.04])).all(-1) & (errors[:,3]<math.radians(10)) & ((self.up3_velocity_h[:,2]-self.commanded_height_rate).abs()<.08)
        self.disturbance_recovery_elapsed.add_(recovery*self._env.step_dt)
        self.disturbance_recovery_stable.copy_(torch.where(recovery & good,self.disturbance_recovery_stable+self._env.step_dt,0.))
        done = recovery & ((self.disturbance_recovery_stable>=.2)|(self.disturbance_recovery_elapsed>=2.5))
        self.window_recovery_time_sum += float((self.disturbance_recovery_elapsed*done).sum())
        self.window_recovery_count += int(done.sum())
        self.metrics["Disturbance/recovery_seconds"].add_(self.disturbance_recovery_elapsed*done*scale)
        self.metrics["Disturbance/recovery_timeout"].add_((done & (self.disturbance_recovery_elapsed>=2.5))*scale)
        self.disturbance_recovery_pending[done]=False
        self.disturbance_recovery_elapsed[done]=0
        self.disturbance_recovery_stable[done]=0

    def state_dict(self):
        state = super().state_dict()
        state["v14_processor"] = self.processor.state_dict()
        state["v14_applied_intent"] = {k:v.detach().cpu() for k,v in self.applied_intent.items()}
        state["v14_buffers"] = {name:getattr(self,name).detach().cpu() for name in ("_target_history","_target_history_count","horizontal_alpha_base","filtered_base_vx_target")}
        return state

    def load_state_dict(self,state):
        if state.get("checkpoint_schema") != self.CHECKPOINT_SCHEMA or "v14_processor" not in state:
            raise ValueError("v1.4 requires its own checkpoint; v1.3 migration is not supported")
        super().load_state_dict(state)
        self.processor.load_state_dict(state["v14_processor"])
        self.applied_intent=cm.applied_intent(self.processor.goal,self.processor.mode,self.processor.active_axes)
        for name,val in state.get('v14_applied_intent',{}).items():
            if val.shape==self.applied_intent[name].shape:
                self.applied_intent[name].copy_(val.to(self.device))
        self._command.copy_(self.processor.commands)
        self.target_command.copy_(self.processor.goal)
        self.sampled_height_target.copy_(self.processor.goal[:,2])
        self.commanded_height_rate.copy_(self.processor.height_rate)
        self.height_step_segment.copy_(self.processor.is_step)
        self.time_left.fill_(1e9)
        for name,val in state.get("v14_buffers",{}).items():
            dest=getattr(self,name)
            if dest.shape==val.shape:
                dest.copy_(val.to(self.device))
        self._state_latch.step=self._statistics_latch.step=-1


def command_observation(env,command_name=COMMAND_NAME):
    term=env.command_manager.get_term(command_name)
    return torch.cat((term.command,term.processor.trend),-1)


def link_body_velocity(env, body_name):
    """Clean Encoder/Critic label at the same link origin used by v1.4 rewards."""
    robot=env.scene["robot"]
    index=robot.find_bodies(body_name)[0][0]
    return v1._body_velocity(robot.data.root_quat_w,_link_velocity(robot,index))


def reward(env, kind, command_name=COMMAND_NAME):
    """All changed terms enter through this latch before accessing shared state."""
    t=env.command_manager.get_term(command_name)
    t.ensure_reward_state()
    r=t._robot
    if kind=="yaw": return torch.exp(-(t.errors[:,0]/.25).square())
    if kind=="vx": return torch.exp(-(t.errors[:,1]/.12).square())
    if kind=="yaw_huber": return cm.hold_huber(t.errors[:,0],.35,8.,cm.low_gate(t.applied_intent['goal'][:,0],.03,.1))
    if kind=="vx_huber": return cm.hold_huber(t.errors[:,1],.2,8.,cm.low_gate(t.command[:,1],.02,.08))
    if kind=="height_huber": return cm.hold_huber(t.errors[:,2],.1,12.,cm.low_gate(t.commanded_height_rate,.01,.04)*(~t.height_step_segment))
    if kind=="damping":
        near=torch.exp(-(t.errors[:,2]/.08).square())
        residual=t.up3_velocity_h[:,2]-t.commanded_height_rate
        # Far-target excess speed only; do not oppose necessary rise/fall.
        excess=(residual.abs()-t.STAGE_VREF_LIMITS[t.active_stage]).clamp_min(0)/.5
        return near*(residual/.12).square().clamp_max(4)+(1-near)*excess.square().clamp_max(4)*.25
    if kind=="vertical_allocation":
        attitude=torch.exp(-(t.errors[:,3]/.15).square())
        gate=attitude*(.2+.8*torch.exp(-(t.errors[:,2]/.1).square()))
        return gate*torch.exp(-.5*(((t.actual_base_vz-t.target_base_vz)/.06).square()+((t.actual_arm_vz-t.target_arm_vz)/.06).square()))
    if kind=="horizontal_allocation":
        return torch.exp(-(t.errors[:,1]/.12).square())*torch.exp(-.5*(((t.actual_base_vx-t.target_base_vx)/.15).square()+((t.actual_arm_vx-t.target_arm_vx)/.12).square()))
    if kind=="static_motion":
        # Preserve weak intent cost outside the settled band; avoid freezing the
        # vertical recovery channels when still far from target.
        horizontal=(t.base_velocity_h[:,:2]/.1).square().sum(-1)+(t.up3_velocity_h[:,:2]/.1).square().sum(-1)
        residual_omega=r.data.root_ang_vel_w.clone()
        residual_omega[:,2]-=t.command[:,0]
        omega=(residual_omega/.15).square().sum(-1)
        wheels=(r.data.joint_vel[:,t._wheel_joint_ids]/1.).square().sum(-1)
        return (.25*t.static_intent+.75*t.static_gate)*(horizontal+omega+wheels).clamp_max(20)
    if kind=="static_orientation":
        g=env.scene.sensors["base_imu"].data.projected_gravity_b
        roll=torch.atan2(g[:,1],-g[:,2]); pitch=torch.atan2(-g[:,0],-g[:,2])
        return (.25*t.static_intent+.75*t.static_gate)*((roll/.08).square()+(pitch/.12).square()+(t.errors[:,3]/.08).square()).clamp_max(20)
    if kind=="lateral_hold": return cm.lateral_hold_gate(t.command)*(t.up3_velocity_h[:,1]/.12).square().clamp_max(4)
    if kind=="arm_default":
        deviation=(r.data.joint_pos[:,t._arm_ids]-r.data.default_joint_pos[:,t._arm_ids]).square().sum(-1)
        return deviation*torch.exp(-(t.errors[:,1]/.12).square()-(t.errors[:,2]/.1).square())
    if kind=="action_rate": return t.action_rate_cost
    if kind=="action_smoothness": return t.action_smoothness_cost
    raise ValueError(f"unknown v1.4 reward {kind}")
