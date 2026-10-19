"""v1.4.1 overrides share v1.4's single owner for reward-state orchestration."""
import torch
from . import nandu1_v1_4_mdp as parent
from . import nandu1_v1_4_1_math as command_math
from . import nandu1_v1_4_1_coordination as cm
from .nandu1_v1_4_1_contract import COMMAND_NAME, SCHEMA


class Nandu1SmoothTransitionHeightCommand(parent.Nandu1DecoupledHeightCommand):
    PROCESSOR_CLASS = command_math.CommandProcessor
    CHECKPOINT_SCHEMA = SCHEMA
    COMMAND_NAME = COMMAND_NAME

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.action_history = cm.ActionHistory(self.num_envs, self.device, self._command.dtype)
        self.static_diagnostics = cm.StaticDiagnostics(self.num_envs, self.device, self._command.dtype)
        self.reference_saturation = torch.zeros_like(self.static_gate)
        self.reference_lag = torch.zeros_like(self.static_gate)
        self.applied_static_bin = self.processor.static_bin.clone()
        self.applied_static_arrived = self.processor.static_arrived.clone()
        self._command_latch = cm.StepLatch()
        self._static_metrics_latch = cm.StepLatch()
        self.metrics.update({name:torch.zeros(self.num_envs, device=self.device) for name in (
            'V141/reference_saturation', 'V141/reference_lag',
            *(f'V141/action_{kind}_{group}' for kind in ('rate','smoothness','physical_delta') for group in ('legs','arm','wheels')),
        )})

    def _advance_processor(self, gravity, gyro):
        return self.processor.step(self._env.step_dt, gravity, gyro, self.active_stage,
                                   tick=self._env.common_step_counter)

    def _update_command(self):
        if not hasattr(self, '_command_latch') or not self._command_latch.begin(self._env.common_step_counter):
            return
        super()._update_command()
        self.applied_static_bin.copy_(torch.where(self.processor.static_bin >= 0,
            self.processor.static_bin + self.processor.static_stage*10, self.processor.static_bin))
        self.applied_static_arrived.copy_(self.processor.static_arrived)

    def _update_vertical_reference(self, eh, actual_vz, dt):
        reference, low, saturation, lag = cm.vertical_reference(
            self.commanded_height_rate, eh, actual_vz, self.height_low_frequency_velocity,
            dt, self.STAGE_VREF_LIMITS[self.active_stage])
        self.height_reference_velocity.copy_(reference)
        self.height_low_frequency_velocity.copy_(low)
        self.height_high_frequency_velocity.copy_(lag)
        self.reference_saturation.copy_(saturation)
        self.reference_lag.copy_(lag)

    def _update_action_history(self, action, dt):
        rate, smooth = self.action_history.update(action, dt)
        self.action_rate_cost.copy_(rate)
        self.action_smoothness_cost.copy_(smooth)
        self.jerk.copy_(self.action_history.jerk)

    def _record_metrics(self):
        super()._record_metrics()
        if not hasattr(self, '_static_metrics_latch') or not self._static_metrics_latch.begin(self._env.common_step_counter):
            return
        dt = self._env.step_dt
        scale = 1. / self._env.max_episode_length
        self.metrics['V141/reference_saturation'].add_(self.reference_saturation*scale)
        self.metrics['V141/reference_lag'].add_(self.reference_lag.abs()*scale)
        for kind, values in (('rate',self.action_history.rate_groups), ('smoothness',self.action_history.smoothness_groups),
                             ('physical_delta',self.action_history.physical_delta_groups)):
            for i, group in enumerate(('legs','arm','wheels')):
                self.metrics[f'V141/action_{kind}_{group}'].add_(values[:,i]*scale)
        position = self._robot.data.body_pos_w[:, self._up3_id]
        self.static_diagnostics.update(self.applied_static_bin, self.applied_static_arrived,
            position[:,2]-self.command[:,2], self.up3_velocity_h[:,2], self.errors[:,3], position[:,:2], dt,
            action_rate=self.action_rate_cost, action_smoothness=self.action_smoothness_cost)
        self.evaluation_snapshot.update(static_bin=self.applied_static_bin.clone(),
                                        static_hold=self.applied_static_arrived.clone())

    def reset(self, env_ids=None):
        ids = parent.v1._ids(env_ids, self.num_envs, self.device)
        if hasattr(self, 'action_history'):
            terminated = getattr(self._env.termination_manager, 'terminated', torch.zeros(self.num_envs,device=self.device,dtype=torch.bool))
            self.static_diagnostics.record_reset(ids, terminated[ids])
            self.action_history.reset(ids)
            self.applied_static_bin[ids] = -1
            self.applied_static_arrived[ids] = False
            self.reference_saturation[ids] = self.reference_lag[ids] = 0
            self.action_rate_cost[ids] = self.action_smoothness_cost[ids] = 0
            self.jerk[ids] = 0
        result = super().reset(env_ids)
        if hasattr(self, 'static_diagnostics'):
            result.update(self.static_diagnostics.summary())
        return result

    def state_dict(self):
        state = super().state_dict()
        state['v141_action_history'] = self.action_history.state_dict()
        state['v141_static_diagnostics'] = self.static_diagnostics.state_dict()
        state['v141_buffers'] = {name:getattr(self,name).detach().clone() for name in (
            'applied_static_bin','applied_static_arrived','alpha_base','reference_saturation','reference_lag')}
        return state

    def load_state_dict(self, state):
        if state.get('checkpoint_schema') != SCHEMA:
            raise ValueError('v1.4.1 requires same-version checkpoint schema')
        if 'v141_action_history' not in state or 'v141_static_diagnostics' not in state:
            raise ValueError('v1.4.1 requires complete transition state')
        super().load_state_dict(state)
        self.action_history.load_state_dict(state['v141_action_history'])
        # A checkpoint reconstructs a new physical scene, so old actions are
        # not valid same-episode predecessors in this scene.
        self.action_history.reset(slice(None))
        self.static_diagnostics.load_state_dict(state['v141_static_diagnostics'])
        self.static_diagnostics.clear_episode_memory()
        for name, value in state['v141_buffers'].items():
            dest = getattr(self,name)
            if dest.shape == value.shape:
                dest.copy_(value.to(self.device))
        self.applied_static_bin.fill_(-1)
        self.applied_static_arrived.zero_()
        # Simulator ticks restart in a newly constructed world; replay state
        # survives, but the latch's old world counter must not suppress its first tick.
        self.processor._last_tick = None
        self._command_latch.step = -1
        self._static_metrics_latch.step = -1


def command_observation(env, command_name=COMMAND_NAME):
    return parent.command_observation(env, command_name)


def reward(env, kind, command_name=COMMAND_NAME):
    return parent.reward(env, kind, command_name)
