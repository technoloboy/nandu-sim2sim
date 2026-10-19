"""Independent E/r2 deployment identity; no training, torch or MuJoCo imports."""
import math

TASK = 'Isaac-NANDU1-Blind-Flat-v1-3-1-2-6'
PLAY = 'Isaac-NANDU1-Blind-Flat-Play-v1-3-1-2-6'
SCHEMA = 'nandu_network_v1_3_1_2_6_ablation_E_r2'
BUNDLE_SCHEMA = 'limx.nandu1-adaptive-deployment-bundle/v1'
RUNTIME = 'nandu1_adaptive_ablation_e_r2'
HEIGHT_MEMORY_ENABLED = True
IS_ABLATION=True
ABLATION_GROUP='E'
MODE_SAMPLING='random'
EVALUATION_STAGE=4
INTERFACE = dict(policy=49, history=245, encoder=7, commands=3, actor_raw=71,
                 actor_features=74, critic=124, critic_with_commands=127, actions=13)
HEIGHT_RATES = [.08, .15, .25, .38, .50]
HEIGHT_MEMORY = dict(error_lpf_tau_s=.5, error_clip_m=.1, integral_clip_m_s=.1,
                     actor_scale_m_s=.05, quiet_rate_low_m_s=.005,
                     quiet_rate_high_m_s=.02, quiet_leak_tau_s=10.,
                     moving_leak_tau_s=1., freeze_tilt_rad=math.pi / 4)
TCP = [-.066473051764, .161623719479, -.000621484902]
CHECKPOINT_SHA256 = '1db8e6e7f58e4613dd2a0144d80ec50e7a2e20573e7d05239408057bf4909115'
OBSERVATION_ORDER = ['base_gyro_scaled_0.25:3', 'base_gravity:3',
    'up3_gyro_scaled_0.25:3', 'up3_gravity:3', 'relative_joint_position:11',
    'joint_velocity_scaled_0.05:13', 'previous_raw_action:13']
DYNAMIC_ORDER = ['memory_vx_over_0.10', 'memory_yaw_over_0.20'] + [
    'filtered_eta_' + str(i) for i in range(8)] + ['hdot_over_0.5', 'memory_height_over_0.05']
HISTORY_ORDER = 'oldest_to_newest_5_policy49_frames; clear_on_new_physics_or_episode_reset'
RAW_ACTOR_ORDER = 'Encoder7,policy49,applied_commands3,normalized_memory2,filtered_student8,hdot_over_0.5,normalized_memory_height1'
ACTOR_FEATURES_ORDER = 'original_E62_in_original_order_then_normalized_memory2_filtered_student8_hdot_over_0.5_then_normalized_memory_height1'


def contract_for(task):
    if task in (TASK,PLAY):
        import sys
        return sys.modules[__name__]
    from nandu1_adaptive_balance_contract import TASK as balance,PLAY as alias
    if task in (balance,alias):
        import nandu1_adaptive_balance_contract
        return nandu1_adaptive_balance_contract
    from nandu1_ablation_contract import PROFILES
    for profile in PROFILES.values():
        if task in (profile.TASK,profile.PLAY):return profile
    raise ValueError('Unsupported adaptive task: '+str(task))


def canonical_task(task):
    return contract_for(task).TASK


def adaptive_tasks(include_play=False):
    from nandu1_adaptive_balance_contract import TASK as balance
    from nandu1_ablation_contract import PROFILES
    profiles=[contract_for(t) for t in (TASK,balance,*PROFILES)]
    return tuple(c.TASK for c in profiles)+(tuple(c.PLAY for c in profiles) if include_play else ())


def deployment_profile(task):
    """Explicit new identity; never widen or reinterpret legacy TASKS/profiles."""
    if task in adaptive_tasks(include_play=True):
        c=contract_for(task)
        return dict(task_id=c.TASK,play_alias=c.PLAY,checkpoint_schema=c.SCHEMA,runtime_profile=c.RUNTIME,
                    encoder_architecture='dual',command=3,policy_input=c.INTERFACE['actor_raw'],
                    velocity_reference='fixed_tcp',actor_velocity_frame='body')
    from nandu1_contract import task_profile
    return task_profile(task)
