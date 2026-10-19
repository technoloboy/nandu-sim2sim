"""Independent adaptive-balance identity and deployable Actor feature order."""
from dataclasses import dataclass, replace

import torch
from torch import nn

from ..nandu_experiments.contract import Experiment
from ..nandu_experiments.features import tracking_errors


ADAPTIVE_BALANCE_TASK = 'Isaac-NANDU1-Blind-Flat-v1-3-1-2-2'


@dataclass(frozen=True)
class AdaptiveBalanceExperiment(Experiment):
    family: str = 'v1-3-1'
    variant: int = 2
    behavior_revision: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.family != 'v1-3-1' or self.variant != 2:
            raise ValueError('Adaptive balance must inherit v1-3-1 experiment 2 (E)')
        if type(self.behavior_revision) is not int or self.behavior_revision not in (1, 2):
            raise ValueError('Unsupported adaptive behavior revision')

    @property
    def task(self): return ADAPTIVE_BALANCE_TASK
    @property
    def schema(self): return f'nandu_network_v1_3_1_2_2_E_adaptive_balance_r{self.behavior_revision}'
    @property
    def experiment_name(self): return 'nandu1_network_v1_3_1_2_2_E_adaptive_balance'
    @property
    def actor_dim(self): return 73
    @property
    def interface(self):
        return dict(policy=49, history=245, encoder=7, commands=3, actor_raw=70,
                    actor_features=73, critic=123, critic_with_commands=126, actions=13)

    def manifest(self):
        result = super().manifest()
        # Reproduce existing r1 manifests exactly, including those written
        # before this dataclass acquired a behavior_revision field.
        result.pop('behavior_revision', None)
        if self.behavior_revision == 2:
            result.update(revision=2, behavior_revision=2,
                          statistics_timing='post_physics_pre_autoreset_once_per_step',
                          allocation_tracking_gate='current_post_physics_state')
        result.update(parent_task=Experiment('v1-3-1', 2).task,
                      dynamic_features=['memory_vx', 'memory_yaw'] + [f'student_eta_{i}' for i in range(8)] + ['height_rate_over_0.5'],
                      dynamic_features_stored_in_ppo=True,
                      actor_feature_order='original_E_62_then_dynamic_11',
                      critic_feature_order='original_108_dynamic_11_balance_3_valid_1_commands_3',
                      student_input='noisy_policy_first36_previous_software_clipped_issued13',
                      adaptation_actor_gradient=False,
                      reward_changes=['static_capture', 'static_fine_hold', 'static_bias',
                                      'static_residual', 'joint_balance', 'gated_com_x'])
        return result

    def validate_checkpoint(self, state, require_optimizer=False):
        super().validate_checkpoint(state, require_optimizer)
        adaptation = state.get('adaptation_state_dict')
        if not isinstance(adaptation, dict) or not adaptation:
            raise ValueError('Missing checkpoint state: adaptation_state_dict')
        def check_finite(value):
            if isinstance(value, torch.Tensor) and not bool(value.isfinite().all()):
                raise ValueError('Non-finite checkpoint tensor in adaptation_state_dict')
            if isinstance(value, dict):
                for item in value.values(): check_finite(item)
            elif isinstance(value, (list, tuple)):
                for item in value: check_finite(item)
        # Failed/invalid frames deliberately remain in masked replay. Learning
        # tensors and live context must be finite; old-scene replay is discarded
        # on resume and the learner validates its shapes and validity masks.
        for key in ('modules', 'optimizer', 'runtime'):
            check_finite(adaptation.get(key))


def checkpoint_experiment(state):
    """Read an explicit saved identity; never reinterpret r1 as r2 on load."""
    exp = Experiment.from_task(state.get('task_name', ''))
    if isinstance(exp, AdaptiveBalanceExperiment):
        for revision in (1, 2):
            candidate = replace(exp, behavior_revision=revision)
            if state.get('checkpoint_schema') == candidate.schema:
                return candidate
        raise ValueError('Unsupported adaptive checkpoint schema/behavior revision')
    return exp


def actor_features(raw: torch.Tensor) -> torch.Tensor:
    """E's original 62 features, then detached memory2/Student8/height-rate1."""
    if raw.shape[-1] != 70:
        raise ValueError('Adaptive Actor requires raw70')
    return torch.cat((raw[..., :56], tracking_errors(raw), raw[..., 56:59],
                      raw[..., 59:70].detach()), dim=-1)


class AdaptiveBalanceActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(73, 512), nn.ELU(), nn.Linear(512, 256),
                                     nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 13))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.network(actor_features(raw))
