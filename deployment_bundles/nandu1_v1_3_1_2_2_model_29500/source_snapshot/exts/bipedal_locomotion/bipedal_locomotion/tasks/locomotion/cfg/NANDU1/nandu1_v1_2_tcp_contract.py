"""Simulator-free identities and fixed measurement contracts for TCP ablations."""
from copy import deepcopy

TCP_LOCAL = (-.066473051764, .161623719479, -.000621484902)
LONG_TCP_LOCAL = (TCP_LOCAL[0], .5, TCP_LOCAL[2])
COMMAND_NAME = 'nandu1_height_v1_1'
_SPECS = {
    'v1-2-3': dict(schema='nandu1_height_v1_2_3_fixed_tcp_v1',
                   experiment='nandu1_blind_flat_v1_2_3_fixed_tcp', tcp=TCP_LOCAL,
                   encoder='MLP_Encoder', parent='v1-1'),
    'v1-2-4': dict(schema='nandu1_height_v1_2_4_tcp_hidden_heads_v1',
                   experiment='nandu1_blind_flat_v1_2_4_tcp_hidden_heads', tcp=TCP_LOCAL,
                   encoder='DualHeadHiddenMLPEncoder', parent='v1-2-3'),
    'v1-2-5': dict(schema='nandu1_height_v1_2_5_long_tcp_v1',
                   experiment='nandu1_blind_flat_v1_2_5_long_tcp', tcp=LONG_TCP_LOCAL,
                   encoder='MLP_Encoder', parent='v1-2-3'),
}


def spec(version):
    if version not in _SPECS:
        raise ValueError(f'Unsupported TCP task: {version}')
    result = deepcopy(_SPECS[version])
    result.update(version=version, task='Isaac-NANDU1-Blind-Flat-' + version,
                  play_task='Isaac-NANDU1-Blind-Flat-Play-' + version, command_name=COMMAND_NAME)
    return result


def export_metadata(version):
    s = spec(version)
    return dict(task=s['task'], checkpoint_schema=s['schema'], parent=s['parent'],
        command_name=COMMAND_NAME, tcp_body='up3', tcp_frame='up3_link', tcp_local_m=list(s['tcp']),
        tcp_independent_of_com=True, velocity_reference='fixed_up3_tcp',
        velocity_frame='full_base_body', base_velocity_reference='base_com',
        height_reference='fixed_up3_tcp_world_z', reach_reference='up3_link_origin_legacy',
        reach_velocity_reference='legacy_up3_com_transport', imu_reference='up3_link_origin',
        reach_soft_limit_m=.60, reach_enhanced_limit_m=.70, reach_hard_termination=False,
        payload_model='unchanged_v1_1', heading_outer_loop=False, height_range_m=[.4,1.4],
        command_order=['base_yaw_rate_world_z','up3_tcp_vx_base_body','up3_tcp_height_world_z'],
        units=['rad/s','m/s','m'], encoder_class=s['encoder'],
        encoder_architecture=('245->256_ELU->128_ELU; velocity:128_ELU->6; height:128_ELU->1'
                              if s['encoder']=='DualHeadHiddenMLPEncoder' else '245->256_ELU->128_ELU->7'),
        dimensions=dict(observation=49,history=245,encoder=7,command=3,actor=59,critic=111,action=13),
        encoder_order=['base_velocity_base_body_3','up3_tcp_velocity_base_body_3','up3_tcp_height_world'],
        observation_order=['base_gyro*.25','base_gravity','up3_gyro*.25','up3_gravity',
                           'position_relative_11','velocity_13*.05','last_action_13'],
        actor_input_order=['encoder_7','observation_49','commands_3'],
        history_order='oldest_to_newest',history_length=5,policy_dt=.02,
        mujoco_bundle_included=False,
        scope='Neural models and measurement contract only; independent MuJoCo task integration is required.')
