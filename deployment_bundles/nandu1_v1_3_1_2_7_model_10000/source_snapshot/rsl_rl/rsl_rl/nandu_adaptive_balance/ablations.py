"""Immutable B–F identities: target sampling, height slew, and height memory only."""
from dataclasses import dataclass

import torch
from torch import nn

from .contract import AdaptiveBalanceExperiment
from ..nandu_experiments.features import tracking_errors


ORIGINAL_HEIGHT_RATES = (.05, .10, .20, .35, .50)
FASTER_HEIGHT_RATES = (.08, .15, .25, .38, .50)
ABLATION_SUFFIXES = (3, 4, 5, 6, 7)


@dataclass(frozen=True)
class AdaptiveBalanceAblationExperiment(AdaptiveBalanceExperiment):
    suffix: int = 3

    def __post_init__(self):
        super().__post_init__()
        if type(self.suffix) is not int or self.suffix not in ABLATION_SUFFIXES:
            raise ValueError('Unknown adaptive-balance ablation suffix')

    @property
    def group(self): return {3: 'B', 4: 'C', 5: 'D', 6: 'E', 7: 'F'}[self.suffix]
    @property
    def task(self): return f'Isaac-NANDU1-Blind-Flat-v1-3-1-2-{self.suffix}'
    @property
    def schema(self): return f'nandu_network_v1_3_1_2_{self.suffix}_ablation_{self.group}_r{self.behavior_revision}'
    @property
    def experiment_name(self): return f'nandu1_network_v1_3_1_2_{self.suffix}_ablation_{self.group}'
    @property
    def mode_sampling(self): return 'four_modes' if self.suffix == 7 else 'random'
    @property
    def height_rates(self): return FASTER_HEIGHT_RATES if self.suffix in (4, 6) else ORIGINAL_HEIGHT_RATES
    @property
    def height_memory_enabled(self): return self.suffix in (5, 6, 7)
    @property
    def actor_dim(self): return 74
    @property
    def interface(self):
        return dict(policy=49, history=245, encoder=7, commands=3, actor_raw=71,
                    actor_features=74, critic=124, critic_with_commands=127, actions=13)

    def manifest(self):
        from .height_memory import HeightSlowErrorMemory
        result = super().manifest()
        result.update(parent_task=AdaptiveBalanceExperiment().task, ablation_group=self.group,
                      mode_sampling=self.mode_sampling, height_rate_limits_m_s=list(self.height_rates),
                      height_memory_enabled=self.height_memory_enabled,
                      height_memory_profile=dict(HeightSlowErrorMemory.PROFILE),
                      dynamic_features=result['dynamic_features'] + ['memory_height'],
                      actor_feature_order='original_A_73_then_memory_height',
                      critic_feature_order='original_108_dynamic_12_balance_3_valid_1_commands_3',
                      initialization='A_shared_weights_zero_new_actor_column73_critic_column119',
                      tracking_population='NORMAL_only' if self.mode_sampling == 'four_modes' else 'all_environments',
                      command_sampling='E_independent_uniform_stage_targets_every_4_to_8_s',
                      command_processing='A_causal_processor_with_profile_height_rates')
        return result

    def validate_checkpoint(self, state, require_optimizer=False):
        super().validate_checkpoint(state, require_optimizer)
        from .height_memory import HeightSlowErrorMemory
        memory = state.get('height_memory_state')
        # Validate without restoring the saved scene's temporal state.
        probe = HeightSlowErrorMemory(1, enabled=self.height_memory_enabled)
        if not isinstance(memory, dict):
            raise ValueError('Missing height memory state for ablation checkpoint')
        probe.load_state_dict(memory, same_world=False)


def ablation_actor_features(raw: torch.Tensor) -> torch.Tensor:
    if raw.shape[-1] != 71:
        raise ValueError('Adaptive ablation Actor requires raw71')
    return torch.cat((raw[..., :56], tracking_errors(raw), raw[..., 56:59], raw[..., 59:71].detach()), -1)


class AdaptiveBalanceAblationActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(74, 512), nn.ELU(), nn.Linear(512, 256),
                                     nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 13))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.network(ablation_actor_features(raw))
