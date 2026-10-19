"""Isolated B–F sampling profiles and height diagnostics; A remains unchanged."""
from __future__ import annotations

import torch

from rsl_rl.nandu_adaptive_balance.ablations import AdaptiveBalanceAblationExperiment
from rsl_rl.nandu_adaptive_balance.modes import TrainingModes
from . import nandu1_v1_3_1_2_2_mdp as parent


class AllRandomTrainingModes(TrainingModes):
    """Use A's NORMAL path for every row, preserving E's stage-box sampler."""

    def reset(self, ids, low, high):
        ids = (torch.arange(self.mode.numel(), device=self.mode.device) if ids is None
               else torch.as_tensor(ids, device=self.mode.device, dtype=torch.long))
        self.mode[ids] = self.NORMAL
        self.age[ids] = 0.
        self.stop_time[ids] = 0.
        self.height_a[ids] = low
        self.height_b[ids] = high

    def validate_state_dict(self, state):
        """Reject an incompatible recipe before any checkpoint state is applied."""
        if not isinstance(state, dict):
            raise ValueError('All-random checkpoint requires NORMAL mode state')
        for name in ('mode', 'age', 'stop_time', 'height_a', 'height_b'):
            if name not in state:
                raise ValueError('All-random checkpoint requires complete NORMAL mode state')
            value = torch.as_tensor(state[name], device=self.mode.device)
            if value.shape != getattr(self, name).shape or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid all-random NORMAL mode state dimensions or values')
            if name == 'mode' and bool((value != self.NORMAL).any()):
                raise ValueError('All-random checkpoint modes must all be NORMAL')

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        super().load_state_dict(state)


class AblationAdaptiveBalanceCommand(parent.AdaptiveBalanceCommand):
    """Change episode sampling only; inherited A processing uses profile rates."""

    HEIGHT_DIAGNOSTICS = ('raw_target_m', 'applied_command_m', 'actual_tcp_m',
                          'applied_error_m', 'raw_applied_lag_m', 'raw_actual_error_m',
                          'ramp_fraction', 'hold_fraction', 'reference_saturated_fraction')

    def __init__(self, cfg, env):
        if cfg.ablation_mode_sampling not in ('random', 'four_modes'):
            raise ValueError('Unknown adaptive ablation mode sampling profile')
        super().__init__(cfg, env)
        if cfg.ablation_mode_sampling == 'random':
            self.modes = AllRandomTrainingModes(self.num_envs, self.device)
        self.tracking_statistics_mask = self.modes.normal_mask()
        self._height_diagnostic_step = None
        self._height_diagnostic_sum = torch.zeros(self.num_envs, len(self.HEIGHT_DIAGNOSTICS), device=self.device)
        self._height_diagnostic_count = torch.zeros(self.num_envs, device=self.device)
        self._height_diagnostic_reset_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        for name in self.HEIGHT_DIAGNOSTICS:
            self.metrics['Height/' + name] = torch.zeros(self.num_envs, device=self.device)
        self.metrics['Height/valid_sample_count'] = torch.zeros(self.num_envs, device=self.device)

    def _accumulate_metrics(self):
        super()._accumulate_metrics()
        step = self._env.common_step_counter
        if self._height_diagnostic_step == step:
            return
        raw = self.target_command[:, 2]
        applied = self.command[:, 2]
        actual = self._up3_position_w()[:, 2]
        # Exposure describes the applied reference: a ramp has nonzero hdot;
        # saturation means the existing vertical reference reaches its cap.
        ramp = (self.commanded_height_rate.abs() > 1e-4).float()
        vmax = self.STAGE_VREF_LIMITS[self.active_stage]
        saturated = (self.height_reference_velocity.abs() >= vmax - 1e-6).float()
        values = torch.stack((raw, applied, actual, applied - actual, raw - applied, raw - actual,
                              ramp, 1. - ramp, saturated), dim=-1)
        valid = torch.isfinite(values).all(-1) & torch.isfinite(self.commanded_height_rate)
        valid &= torch.isfinite(self.height_reference_velocity) & (self._height_diagnostic_reset_step != step)
        self._height_diagnostic_sum.add_(torch.where(valid[:, None], values, 0.))
        self._height_diagnostic_count.add_(valid)
        means = self._height_diagnostic_sum / self._height_diagnostic_count.clamp_min(1.)[:, None]
        for index, name in enumerate(self.HEIGHT_DIAGNOSTICS):
            self.metrics['Height/' + name].copy_(means[:, index])
        self.metrics['Height/valid_sample_count'].copy_(self._height_diagnostic_count)
        self._height_diagnostic_step = step

    def reset(self, env_ids=None):
        # Parent CommandTerm.reset exports this episode's means before clearing
        # metrics. Clear only those rows' accumulators after that export.
        result = super().reset(env_ids)
        if hasattr(self, '_height_diagnostic_count'):
            ids = slice(None) if env_ids is None else env_ids
            self._height_diagnostic_sum[ids] = 0.
            self._height_diagnostic_count[ids] = 0.
            self._height_diagnostic_reset_step[ids] = self._env.common_step_counter
            # Preserve the shared step guard: surviving rows must not be
            # sampled twice, and reset rows have no interval to measure yet.
        return result

    def load_state_dict(self, state):
        if isinstance(self.modes, AllRandomTrainingModes):
            self.modes.validate_state_dict(state.get('adaptive_modes'))
        super().load_state_dict(state)
        # F keeps all four restored modes; B–E validated their all-NORMAL state.
        self.tracking_statistics_mask = self.modes.normal_mask()
        self._height_diagnostic_step = None


def configure_ablation(cfg, suffix):
    """Assign this instance's immutable profile without modifying A defaults."""
    experiment = AdaptiveBalanceAblationExperiment(suffix=suffix)
    term = cfg.commands.nandu1_height_v1_3_1
    term.class_type = AblationAdaptiveBalanceCommand
    term.checkpoint_schema = experiment.schema
    term.behavior_revision = experiment.behavior_revision
    term.ablation_mode_sampling = experiment.mode_sampling
    term.height_rate_limits = experiment.height_rates
