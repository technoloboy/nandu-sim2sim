"""Simulator-independent identities and dimensions for paired ablations."""
from dataclasses import dataclass, asdict
import re

VARIANTS = {
    0: '', 1: 'H', 2: 'E', 3: 'G', 4: 'GF', 5: 'P', 6: 'HE',
    7: 'EG', 8: 'HGF', 9: 'EGF', 10: 'HEG', 11: 'HEGF',
    12: 'HEGFP', 13: 'HEP',
}


@dataclass(frozen=True)
class Experiment:
    family: str
    variant: int

    def __post_init__(self):
        if self.family not in ('v1-3-1', 'v1-4-1') or type(self.variant) is not int or self.variant not in VARIANTS:
            raise ValueError('Unknown NANDU network experiment')

    @classmethod
    def from_task(cls, task):
        if re.fullmatch(r'Isaac-NANDU1-Blind-Flat-(?:Play-)?v1-3-1-2-[3-7]', task):
            from ..nandu_adaptive_balance.ablations import AdaptiveBalanceAblationExperiment
            return AdaptiveBalanceAblationExperiment(suffix=int(task.rsplit('-', 1)[1]))
        if task in ('Isaac-NANDU1-Blind-Flat-v1-3-1-2-2', 'Isaac-NANDU1-Blind-Flat-Play-v1-3-1-2-2'):
            from ..nandu_adaptive_balance.contract import AdaptiveBalanceExperiment
            return AdaptiveBalanceExperiment()
        if task in (STATIC_PITCH_YAW_TASK, STATIC_PITCH_YAW_TASK.replace('Flat-', 'Flat-Play-')):
            return StaticPitchYawExperiment('v1-3-1', 2)
        match = re.fullmatch(r'Isaac-NANDU1-Blind-Flat-(?:Play-)?(v1-[34]-1)-(\d+)', task)
        if not match:
            raise ValueError(f'Not a network experiment task: {task}')
        return cls(match[1], int(match[2]))

    @property
    def factors(self): return VARIANTS[self.variant]
    @property
    def task(self): return f'Isaac-NANDU1-Blind-Flat-{self.family}-{self.variant}'
    @property
    def play_task(self): return self.task.replace('Flat-', 'Flat-Play-')
    @property
    def schema(self): return f'nandu_network_{self.family}_{self.variant}_{self.factors or "control"}_r1'
    @property
    def command_name(self): return 'nandu1_height_' + self.family.replace('-', '_')
    @property
    def experiment_name(self): return f'nandu1_network_{self.family.replace("-", "_")}_{self.variant}_{self.factors or "control"}'
    @property
    def commands(self): return 6 if self.family == 'v1-4-1' else 3
    @property
    def velocity_frame(self): return 'heading' if self.family == 'v1-4-1' else 'body'
    @property
    def history_dim(self): return 260 if 'H' in self.factors else 245
    @property
    def actor_dim(self): return 56 + self.commands + 3*('E' in self.factors) + 3*('G' in self.factors)
    @property
    def interface(self):
        return dict(policy=49, history=self.history_dim, encoder=7, commands=self.commands,
                    actor_raw=56+self.commands, actor_features=self.actor_dim,
                    critic=108, critic_with_commands=108+self.commands, actions=13)

    def manifest(self):
        return {**asdict(self), 'factors': self.factors, 'revision': 1,
                'velocity_frame': self.velocity_frame, 'interface': self.interface,
                'history_command_timing': 'previous_applied_command_producing_current_state',
                'reward_changes': [], 'command_gp': .0002 if 'P' in self.factors else 0.,
                'sensory_gp': .002 if 'P' in self.factors or self.family == 'v1-4-1' else 0.,
                'gp_warmup_iterations': 1000, 'encoder_actor_gradient': False}

    def validate_checkpoint(self, state, require_optimizer=False):
        if state.get('checkpoint_schema') != self.schema or state.get('task_name') != self.task:
            raise ValueError('Network experiment checkpoint schema/task mismatch; start fresh for another suffix')
        if state.get('network_experiment') != self.manifest() or state.get('interface_contract') != self.interface:
            raise ValueError('Network experiment manifest/interface mismatch')
        for key in ('model_state_dict', 'encoder_state_dict', 'environment_state_dict'):
            if not isinstance(state.get(key), dict) or not state[key]:
                raise ValueError(f'Missing checkpoint state: {key}')
        stage = state['environment_state_dict'].get(self.command_name, {}).get('stage')
        if hasattr(stage, 'item'): stage = stage.item()
        if type(stage) is not int or stage not in range(5):
            raise ValueError('Missing/invalid baseline curriculum stage')
        if require_optimizer:
            for key in ('optimizer_state_dict', 'extra_optimizer_state_dict'):
                if not state.get(key): raise ValueError(f'Resume requires {key}')
        for group in ('model_state_dict', 'encoder_state_dict'):
            if any(not hasattr(t, 'isfinite') or not bool(t.isfinite().all()) for t in state[group].values()):
                raise ValueError(f'Non-finite checkpoint tensor in {group}')


STATIC_PITCH_YAW_TASK = 'Isaac-NANDU1-Blind-Flat-v1-3-1-2-1'


class StaticPitchYawExperiment(Experiment):
    """Reward-only child of E; never change the original 28 experiment manifests."""

    def __post_init__(self):
        super().__post_init__()
        if self.family != 'v1-3-1' or self.variant != 2:
            raise ValueError('Static pitch/yaw ablation must inherit v1-3-1 experiment 2 (E)')

    @property
    def task(self): return STATIC_PITCH_YAW_TASK
    @property
    def schema(self): return 'nandu_network_v1_3_1_2_1_E_static_pitch_yaw_r1'
    @property
    def experiment_name(self): return 'nandu1_network_v1_3_1_2_1_E_static_pitch_yaw'

    def manifest(self):
        result = super().manifest()
        result.update(reward_parent_task=Experiment('v1-3-1', 2).task, reward_changes=[
            dict(term='static_orientation_hold', change='remove_base_pitch_only',
                 weight=-.25, base_roll_scale=.08, up3_tilt_scale=.08, cap=20., gate='static_gate'),
            dict(term='static_world_yaw_rate', change='add', weight=-.10,
                 frame='world_z', body='base', scale=.15, cap=4., gate='static_gate'),
        ])
        return result
