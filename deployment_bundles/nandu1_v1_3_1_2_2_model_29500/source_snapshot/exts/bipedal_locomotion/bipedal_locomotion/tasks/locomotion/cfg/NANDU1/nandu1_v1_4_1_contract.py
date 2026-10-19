"""Strict independent smooth-transition checkpoint and command contract."""
SCHEMA = 'nandu1_height_v1_4_1_smooth_transition_v1'
COMMAND_NAME = 'nandu1_height_v1_4_1'
TASK = 'Isaac-NANDU1-Blind-Flat-v1-4-1'
PLAY_TASK = 'Isaac-NANDU1-Blind-Flat-Play-v1-4-1'
INTERFACE = dict(policy=49, history=245, encoder=7, commands=6,
                 critic=108, critic_with_commands=114, actions=13)
COMMAND_UNITS = ('rad/s', 'm/s', 'm')
COMMAND_FRAMES = ('world_z_angular_velocity', 'base_heading_x', 'ground_absolute_z')


def validate_training_request(transfer_checkpoint, resume, checkpoint_path):
    if transfer_checkpoint:
        raise ValueError('v1.4.1 transfer is disabled; initialize fresh Actor/Critic/Encoder')
    if checkpoint_path and not resume:
        raise ValueError('v1.4.1 checkpoint_path requires --resume')


def validate_checkpoint(checkpoint):
    if checkpoint.get('checkpoint_schema') != SCHEMA:
        raise ValueError(f'v1.4.1 requires checkpoint schema {SCHEMA}')
    if checkpoint.get('task_name') not in (TASK, PLAY_TASK):
        raise ValueError('v1.4.1 requires matching checkpoint task')
    if checkpoint.get('interface_contract') != INTERFACE:
        raise ValueError('v1.4.1 checkpoint interface must match 49/245/7/6/108/114/13')
    state = checkpoint.get('environment_state_dict', {}).get(COMMAND_NAME)
    if not isinstance(state, dict) or state.get('checkpoint_schema') != SCHEMA:
        raise ValueError('v1.4.1 checkpoint requires matching command state schema')
    stage = state.get('stage')
    if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= 4:
        raise ValueError('v1.4.1 command stage must be integer 0..4')
    if state.get('v14_processor', {}).get('schema') != 'nandu1_v1_4_1_command_processor_v1':
        raise ValueError('v1.4.1 checkpoint requires matching processor schema')
    if state.get('v141_action_history', {}).get('schema') != 'nandu1_v1_4_1_action_history_v1':
        raise ValueError('v1.4.1 checkpoint requires matching action history schema')
    if not isinstance(state.get('v141_static_diagnostics'), dict) or not isinstance(state.get('v141_buffers'), dict):
        raise ValueError('v1.4.1 checkpoint requires complete transition diagnostics and buffers')
