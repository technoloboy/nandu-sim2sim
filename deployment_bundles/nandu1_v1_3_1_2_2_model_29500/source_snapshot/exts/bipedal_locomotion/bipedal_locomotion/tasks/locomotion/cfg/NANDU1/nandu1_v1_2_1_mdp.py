"""v1.2.1 scheme B. Original course metrics/thresholds remain inherited."""
from __future__ import annotations

import math
import torch
from isaaclab.utils import configclass

from . import nandu1_math
from . import nandu1_v1_mdp as v1
from . import nandu1_v1_1_mdp as v11
from . import nandu1_v1_2_1_math as cm
from . import nandu1_v1_2_1_contract as contract


class Nandu1AdaptiveHoldCommand(v11.Nandu1CoordinatedHeightCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._init_scheme_b_state()
        self._hold_wheel_ids = self._robot.find_joints(
            ['wheel_L_Joint', 'wheel_R_Joint'], preserve_order=True
        )[0]

    def _init_scheme_b_state(self):
        self.command_age = torch.zeros(self.num_envs, device=self.device)
        self.hold_segment = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.adapter = cm.SlowRewardAdapter()
        self.learning_iteration = 0
        # fixed-size global statistics also participate in parent's window checkpoint.
        self.window_b_sums = torch.zeros(18, dtype=torch.float64, device=self.device)
        self.window_b_dynamic_bins = torch.zeros(3, 10, device=self.device)
        self._b_env_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.b_diagnostics = {name: 0. for name in (
            'quiet_samples','quiet_environments','quiet_vz_rms','quiet_base_angular_rms',
            'quiet_height_rmse','response_height_rmse','dynamic_yaw_rmse','dynamic_vx_rmse',
            'dynamic_height_rmse','dynamic_bin_coverage','intent_occupancy','sampled_hold_fraction',
            'damping_raw','motion_raw','damping_scaled','motion_scaled')}

    def _resample_command(self, env_ids):
        if self._external_control:
            return
        ids = v1._ids(env_ids, self.num_envs, self.device)
        super()._resample_command(ids)
        mixed, hold = cm.mix_hold_commands(
            self.target_command[ids], self._command[ids, 2],
            torch.rand(len(ids), device=self.device), self.cfg.hold_probability,
        )
        self.target_command[ids] = mixed
        self.sampled_height_target[ids] = mixed[:, 2]
        self.hold_segment[ids] = hold
        self.window_b_sums[12] += hold.sum()
        self.window_b_sums[13] += len(ids)

    def _update_command(self):
        before = self._command.clone()
        super()._update_command()
        changed = (self._command != before).any(dim=-1)
        self.command_age.copy_(torch.where(changed, 0., self.command_age + self._env.step_dt))

    def reset(self, env_ids=None):
        ids = v1._ids(env_ids, self.num_envs, self.device)
        result = super().reset(env_ids)
        self.command_age[ids] = 0.
        return result

    def begin_rollout(self, iteration):
        self.learning_iteration = int(iteration)
        if self.cfg.adaptive_enabled:
            self.adapter.begin_rollout()

    def reward_hold_gate(self):
        height_error = self._command[:, 2] - self._up3_position_w()[:, 2]
        # Keep the original curriculum static_gate unchanged. This gate is reward-only.
        return nandu1_math.static_hold_gate(
            self._command[:, 0], self._command[:, 1], height_error, self.commanded_height_rate
        ) * cm.hold_phase(self.command_age)

    def damping_cost(self):
        return v1.height_velocity_damping(self._env, contract.COMMAND_NAME) * cm.hold_phase(self.command_age)

    def motion_cost(self):
        data = self._robot.data
        return self.reward_hold_gate() * cm.channel_motion_cost(
            v1._body_velocity(data.root_quat_w, data.root_lin_vel_w), data.root_ang_vel_b,
            v1._body_velocity(data.root_quat_w, self._up3_velocity_w()),
            data.joint_vel[:, self._hold_wheel_ids],
        )

    def _update_metrics(self):
        super()._update_metrics()
        data = self._robot.data
        intent = cm.static_intent_gate(self._command) > .5
        # A fast/drifting robot remains in this population. Actual error or
        # velocity is never an eligibility filter for adaptation statistics.
        quiet = intent & (self.command_age >= 1.) & ~self.disturbance_active & ~self.disturbance_recovery_pending
        response = self.command_age < 1.
        dynamic = ~self.hold_segment
        height_error = self._up3_position_w()[:, 2] - self._command[:, 2]
        up3_velocity = self._up3_velocity_w()
        errors = torch.stack((data.root_ang_vel_w[:, 2] - self._command[:, 0],
                              v1._body_velocity(data.root_quat_w, up3_velocity)[:, 0] - self._command[:, 1],
                              height_error), dim=-1).square()
        s = self.window_b_sums
        s[0] += quiet.sum()
        s[1] += torch.where(quiet, up3_velocity[:, 2].square(), 0.).sum()
        s[2] += torch.where(quiet, data.root_ang_vel_w.square().mean(dim=-1), 0.).sum()
        s[3] += response.sum()
        s[4] += torch.where(response, height_error.square(), 0.).sum()
        s[5] += torch.where(quiet, height_error.square(), 0.).sum()
        s[6] += dynamic.sum()
        s[7:10] += torch.where(dynamic[:, None], errors, 0.).sum(dim=0)
        s[10] += intent.sum()
        s[11] += self.num_envs
        damping, motion = self.damping_cost().sum(), self.motion_cost().sum()
        s[14] += damping; s[15] += motion
        s[16] += self.adapter.active[0] * damping
        s[17] += self.adapter.active[1] * motion
        self._b_env_seen |= quiet
        if self._coverage_stride % 10 == 0:
            yaw, vx = self.STAGE_LIMITS[self.stage]
            low, high = self.STAGE_HEIGHT_RANGES[self.stage]
            for row, bounds in enumerate(((-yaw,yaw),(-vx,vx),(low,high))):
                bins = (((self._command[:,row] - bounds[0]) / (bounds[1] - bounds[0])).clamp(0.,.9999) * 10).long()
                self.window_b_dynamic_bins[row] += torch.bincount(bins[dynamic], minlength=10)

    def evaluate_curriculum(self, common_step_counter):
        due = common_step_counter - self.last_evaluation_step >= self.cfg.evaluation_steps and self.window_samples > 0
        if not due:
            return super().evaluate_curriculum(common_step_counter)
        old_stage = self.stage
        s = self.window_b_sums.tolist()
        seen = int(self._b_env_seen.sum())
        rms = lambda numerator, denominator: math.sqrt(numerator / max(denominator, 1.))
        raw = (rms(s[1],s[0]), rms(s[2],s[0]))
        # Dynamic-only tracking prevents the extra easy hold samples hiding regression.
        tracking = tuple(rms(s[i],s[6]) for i in (7,8,9))
        self.b_diagnostics.update(
            quiet_samples=s[0], quiet_environments=seen, quiet_vz_rms=raw[0], quiet_base_angular_rms=raw[1],
            quiet_height_rmse=rms(s[5],s[0]), response_height_rmse=rms(s[4],s[3]),
            dynamic_yaw_rmse=tracking[0], dynamic_vx_rmse=tracking[1], dynamic_height_rmse=tracking[2],
            dynamic_bin_coverage=float((self.window_b_dynamic_bins >= 64).float().mean()),
            intent_occupancy=s[10]/max(s[11],1.), sampled_hold_fraction=s[12]/max(s[13],1.),
            damping_raw=s[14]/max(s[11],1.), motion_raw=s[15]/max(s[11],1.),
            damping_scaled=s[16]/max(s[11],1.), motion_scaled=s[17]/max(s[11],1.),
        )
        super().evaluate_curriculum(common_step_counter)
        if self.cfg.adaptive_enabled:
            self.adapter.set_stage(old_stage)
            pairs = [self.SCORE_GOOD_BAD[name][min(old_stage,3)] for name in ('settled_vz','static_drift')]
            self.adapter.audit(iteration=self.learning_iteration, raw=raw,
                               good=tuple(p[0] for p in pairs), bad=tuple(p[1] for p in pairs),
                               samples=int(s[0]) if s[6] >= 256 else 0, environments=seen,
                               tracking=tracking, safety_veto=self.safety_veto)
            if self.stage != old_stage:
                self.adapter.set_stage(self.stage)
        self.window_b_sums.zero_()
        self.window_b_dynamic_bins.zero_()
        self._b_env_seen.zero_()
        return self.curriculum_state()

    def curriculum_state(self, completion=None, safety_failure=None):
        result = super().curriculum_state(completion, safety_failure)
        result.update({f'SchemeB/{key}': value for key,value in self.b_diagnostics.items()})
        result.update({'SchemeB/lambda_z':self.adapter.active[0], 'SchemeB/lambda_static':self.adapter.active[1],
                       'SchemeB/pending_z':self.adapter.pending[0], 'SchemeB/pending_static':self.adapter.pending[1],
                       'SchemeB/adaptation_reason':float(self.adapter.reason)})
        return result

    def state_dict(self):
        state = super().state_dict()
        state.update(scheme_b_version=contract.VERSION, adaptive_reward=self.adapter.state_dict(),
                     learning_iteration=self.learning_iteration, scheme_b_diagnostics=dict(self.b_diagnostics),
                     scheme_b_sums=self.window_b_sums.detach().cpu().clone(),
                     scheme_b_bins=self.window_b_dynamic_bins.detach().cpu().clone(),
                     scheme_b_seen=self._b_env_seen.detach().cpu().clone())
        return state

    def load_state_dict(self, state):
        if state.get('scheme_b_version') != contract.VERSION:
            raise ValueError('v1.2.1 requires matching scheme B configuration')
        self.adapter.load_state_dict(state['adaptive_reward'])
        super().load_state_dict(state)
        self.learning_iteration = int(state['learning_iteration'])
        self.b_diagnostics.update(state['scheme_b_diagnostics'])
        self.window_b_sums.copy_(torch.as_tensor(state['scheme_b_sums'],device=self.device))
        self.window_b_dynamic_bins.copy_(torch.as_tensor(state['scheme_b_bins'],device=self.device))
        seen = torch.as_tensor(state['scheme_b_seen'],device=self.device)
        self._b_env_seen.zero_()
        if seen.shape == self._b_env_seen.shape:
            self._b_env_seen.copy_(seen)
        # Physics is not checkpointed: never reuse old per-episode phase/timers.
        self.command_age.zero_()
        self.hold_segment.zero_()

    @torch.inference_mode()
    def restart_after_resume(self):
        env = self._env
        ids = torch.arange(self.num_envs,device=self.device)
        env.scene.reset(ids)
        env.event_manager.apply(mode='reset',env_ids=ids,global_env_step_count=env.common_step_counter)
        env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(0.)
        env._nandu_v1_measurement_cache = {}
        env.action_manager.reset(ids); env.observation_manager.reset(ids)
        self.reset(ids)
        env.event_manager.reset(ids)
        self.disturbance_active.zero_(); self.disturbance_recovery_pending.zero_()
        self.disturbance_recovery_elapsed.zero_(); self.disturbance_recovery_stable.zero_()
        env.episode_length_buf[ids] = 0
        env.obs_buf = env.observation_manager.compute(update_history=True)


@configclass
class Nandu1AdaptiveHoldCommandCfg(v1.Nandu1HeightCommandCfg):
    class_type: type = Nandu1AdaptiveHoldCommand
    hold_probability: float = .15
    adaptive_enabled: bool = True
    checkpoint_schema: str = contract.SCHEMA
    measurement_command_name: str = contract.COMMAND_NAME


def height_velocity_damping(env, command_name):
    term = env.command_manager.get_term(command_name)
    return term.adapter.active[0] * term.damping_cost()


def static_body_motion(env, command_name):
    term = env.command_manager.get_term(command_name)
    return term.adapter.active[1] * term.motion_cost()


def static_orientation_hold(env, command_name):
    term = env.command_manager.get_term(command_name)
    # Unchanged orientation targets/base weight; do not adapt pitch-to-zero.
    return v1.static_orientation_hold(env, command_name) * cm.hold_phase(term.command_age)
