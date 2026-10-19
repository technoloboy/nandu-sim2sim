"""Strict NANDU1 bundle verification, shared by simulator and ROS launcher."""
import hashlib
import json
from pathlib import Path
import nandu1_contract as c
import numpy as np

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def require_controller_runtime(manifest):
    if manifest.get('task',{}).get('task_id')==c.V141_TASK and manifest['task'].get('ros_runtime_supported') is not True:
        raise ValueError('v1.4.1 ROS controller runtime is unsupported; use the local direct physics probe')

def load_bundle(directory, task=None):
    directory=Path(directory).resolve()
    m=json.loads((directory/'manifest.yaml').read_text())
    if m.get('schema')=='limx.nandu1-adaptive-deployment-bundle/v1':
        from nandu1_adaptive_bundle import load_adaptive_bundle
        return load_adaptive_bundle(directory,task)
    t=m.get('task',{})
    profile=c.task_profile(task or t.get('task_id'))
    if m.get('schema')!=c.BUNDLE_SCHEMA or t.get('task_id')!=profile['task_id']:
        raise ValueError('NANDU1 task/schema mismatch')
    if t.get('checkpoint_schema')!=profile['checkpoint_schema'] or t.get('hardware_joint_names')!=c.JOINTS:
        raise ValueError('NANDU1 checkpoint/joint contract mismatch')
    expected={'observation':49,'history':245,'encoder':7,'command':profile['command'],'policy_input':profile['policy_input'],'action':13}
    if m.get('dimensions')!=expected or t.get('history_order')!='oldest_to_newest':
        raise ValueError('NANDU1 dimensions/history mismatch')
    for key,wanted in [('default_joint_positions',c.DEFAULT_Q),('kp',c.KP),('kd',c.KD),('effort',c.EFFORT),
                       ('velocity_limits',c.VELOCITY),('armature',c.ARMATURE)]:
        if not np.array_equal(np.asarray(t.get(key,[])),wanted): raise ValueError('NANDU1 motor contract mismatch: '+key)
    if (t.get('history_length')!=5 or t.get('history_first_frame')!='repeat' or t.get('policy_dt')!=.02
        or t.get('position_scale')!=.25 or t.get('wheel_scale')!=1 or t.get('action_clipping','missing') is not None
        or t.get('root_position')!=[0,0,.65337]
        or t.get('action_joint_names')!=[c.JOINTS[i] for i in c.ACTION_TO_HARDWARE]
        or t.get('observation_order')!=['base_gyro*.25','base_gravity','up3_gyro*.25','up3_gravity','position_relative_11','velocity_13*.05','last_action_13']):
        raise ValueError('NANDU1 runtime contract mismatch')
    if profile['velocity_reference']=='body_com':
        legacy=profile['task_id']==c.TASK
        if not legacy:
            stage=t.get('evaluation_stage')
            if type(stage) is not int or stage not in range(5):raise ValueError('Invalid body-task evaluation stage')
            yl,vl=c.STAGE_LIMITS[stage];lo,hi=c.STAGE_HEIGHT_RANGES[stage]
            if (t.get('runtime_profile')!=profile['runtime_profile']
                or t.get('encoder_architecture')!=profile['encoder_architecture']
                or t.get('velocity_reference')!='body_com' or t.get('actor_velocity_frame')!='body'):
                raise ValueError('NANDU1 body-task runtime contract mismatch')
        if (t.get('command_order')!=['base_yaw_rate_world_z','up3_vx_base_body','up3_height_world']
            or t.get('command_min')!=([-1.57,-1,.4] if legacy else [-yl,-vl,lo])
            or t.get('command_max')!=([1.57,1,1.4] if legacy else [yl,vl,hi])
            or t.get('command_rate')!=([2,.8,.5] if legacy else [2,.8,[.05,.1,.2,.35,.5][stage]])
            or t.get('encoder_order')!=['base_velocity_base_body_3','up3_velocity_base_body_3','up3_height_world']):
            raise ValueError('NANDU1 runtime contract mismatch')
    elif profile['task_id'] in (c.V131_TASK,c.V1312_TASK):
        from nandu1_v131 import PROCESSOR_CONTRACT,PAYLOAD_CONTRACT,TCP,validate_payload
        stage=t.get('evaluation_stage')
        payload=t.get('fixed_payload',{})
        if type(stage) is not int or stage not in range(5):raise ValueError('Invalid NANDU1 v1.3.1 stage')
        if (t.get('runtime_profile')!=profile['runtime_profile'] or t.get('play_alias')!=profile['play_alias']
            or t.get('velocity_reference')!='fixed_tcp' or t.get('actor_velocity_frame')!='body'
            or t.get('base_velocity_reference')!='body_com' or t.get('tcp_frame')!='up3_link'
            or t.get('tcp_local_m')!=TCP.tolist()
            or t.get('external_command_order')!=['relative_heading_rate','tcp_vx_base_body','tcp_height_world_z']
            or t.get('command_order')!=['base_yaw_rate_world_z','tcp_vx_base_body','tcp_height_world_z']
            or t.get('encoder_order')!=['base_com_velocity_base_body_3','tcp_velocity_base_body_3','tcp_height_world']
            or t.get('command_min')!=[-1.57,-1,.4] or t.get('command_max')!=[1.57,1,1.4]
            or t.get('command_processor')!=PROCESSOR_CONTRACT
            or t.get('payload_contract')!=PAYLOAD_CONTRACT
            or set(payload)!= {'up3_total_mass','rear_mass','rear_x'}):
            raise ValueError('NANDU1 v1.3.1 TCP/payload/runtime contract mismatch')
        validate_payload(payload['up3_total_mass'],payload['rear_mass'],payload['rear_x'])
        required={'command_processor.py','nandu1_v131.py','nandu1_contract.py','yaw_conformance.json','deployment_metadata.json',
                  'actor_pipeline.onnx','encoder.pt','policy.pt','actor_pipeline.pt','conformance.npz'}
        if not required<=m.get('files',{}).keys():raise ValueError('NANDU1 v1.3.1 missing runtime fingerprints')
        if profile['task_id']==c.V1312_TASK:
            from nandu1_network_e import CHECKPOINT_SHA256,INFERENCE_SHA256,validate_metadata
            metadata=json.loads((directory/'deployment_metadata.json').read_text())
            validate_metadata(metadata)
            if (m.get('checkpoint',{}).get('sha256')!=CHECKPOINT_SHA256
                or any(m['files'].get(k)!=v for k,v in INFERENCE_SHA256.items())
                or t.get('evaluation_stage')!=4 or t.get('actor_feature_width')!=62
                or t.get('feature_transform_included') is not True
                or t.get('command_rate')!=[2.,.8,.5] or t.get('ros_runtime_supported') is not True
                or any(m['files'].get(k)!=v for k,v in metadata['file_sha256'].items())):
                raise ValueError('Original network-E artifact/checkpoint/feature mismatch')
    elif profile['task_id'] in (c.V14_TASK,c.V141_TASK):
        if profile['task_id']==c.V141_TASK:
            from nandu1_v141 import PROCESSOR_CONTRACT,STAGE_LIMITS,STAGE_HEIGHT_RANGES,HEIGHT_RATE_LIMITS,COMMAND_ORDER
            required={'command_processor.py','nandu1_v1_4_math.py','nandu1_v141.py','nandu1_v14.py','nandu1_contract.py',
                      'deployment_metadata.json','actor_pipeline.onnx','encoder.pt','policy.pt','actor_pipeline.pt',
                      'conformance.npz','runtime_conformance.txt','deploy_reference/NANDU1V141.h',
                      'deploy_reference/nandu1_v141_conformance.cpp'}
            if not required<=m.get('files',{}).keys():raise ValueError('v1.4.1 missing runtime fingerprints')
            metadata=json.loads((directory/'deployment_metadata.json').read_text())
            if (metadata.get('checkpoint_schema')!=c.V141_SCHEMA
                or metadata.get('processor_contract')!=PROCESSOR_CONTRACT
                or metadata.get('ros_runtime_supported') is not t.get('ros_runtime_supported',False)
                or metadata.get('dimensions')!={'policy_observation':49,'history':245,'encoder':7,
                    'external_command':3,'command_observation':6,'actor_input':62,'critic_input':114,'action':13}):
                raise ValueError('v1.4.1 deployment metadata/runtime contract mismatch')
        else:
            from nandu1_v14 import PROCESSOR_CONTRACT,STAGE_LIMITS,STAGE_HEIGHT_RANGES,HEIGHT_RATE_LIMITS
            COMMAND_ORDER=['base_yaw_rate_world_z','up3_vx_base_heading','up3_height_world_z','yaw_trend','vx_trend','height_trend']
        stage=t.get('evaluation_stage')
        if type(stage) is not int or stage not in range(5):raise ValueError('Invalid NANDU1 v1.4 evaluation stage')
        yl,vl=STAGE_LIMITS[stage];lo,hi=STAGE_HEIGHT_RANGES[stage]
        if (t.get('runtime_profile')!=profile['runtime_profile'] or t.get('play_alias')!=profile['play_alias']
            or t.get('velocity_reference')!='body_link_origin' or t.get('actor_velocity_frame')!='heading'
            or t.get('heading_conversion_gravity')!='policy_observation[3:6]'
            or t.get('external_command_order')!=['relative_heading_rate','up3_vx_base_heading','up3_height_world_z']
            or t.get('command_order')!=COMMAND_ORDER
            or t.get('command_min')!=[-yl,-vl,lo] or t.get('command_max')!=[yl,vl,hi]
            or t.get('height_rate_limit')!=HEIGHT_RATE_LIMITS[stage]
            or t.get('command_processor')!=PROCESSOR_CONTRACT
            or t.get('encoder_order')!=['base_link_velocity_base_body_3','up3_link_velocity_base_body_3','up3_height_world']):
            raise ValueError('NANDU1 v1.4 runtime contract mismatch')
    else:
        raise ValueError('Unsupported runtime profile')
    lower=np.asarray(t.get('joint_position_lower',[]));upper=np.asarray(t.get('joint_position_upper',[]))
    if lower.shape!=(13,) or upper.shape!=(13,) or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>=upper):
        raise ValueError('NANDU1 physical joint limit contract mismatch')
    for relative,digest in m.get('files',{}).items():
        p=(directory/relative).resolve()
        if directory not in p.parents or sha256(p)!=digest: raise ValueError('Bundle hash mismatch: '+relative)
    if not {'encoder.onnx','policy.onnx','assets/robot.xml','assets/robot.urdf'}<=m.get('files',{}).keys():
        raise ValueError('Bundle missing required file fingerprints')
    sums={}
    for line in (directory/'SHA256SUMS').read_text().splitlines():
        digest,relative=line.split('  ',1);sums[relative]=digest
    if sums.get('manifest.yaml')!=sha256(directory/'manifest.yaml'):
        raise ValueError('Manifest checksum mismatch')
    if any(sums.get(relative)!=digest for relative,digest in m['files'].items()):
        raise ValueError('SHA256SUMS differs from manifest')
    return m,directory/'assets/robot.xml'
