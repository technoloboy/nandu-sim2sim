"""Strict typed bundle loader. Safe to import inside the lightweight ROS image."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import nandu1_contract as base
from nandu1_adaptive_contract import contract_for

REQUIRED = {f'{name}.{suffix}' for name in ('encoder', 'policy', 'context_step', 'actor_pipeline')
            for suffix in ('onnx', 'pt')} | {'conformance.npz', 'deployment_metadata.json',
            'adaptive_math.py', 'adaptive_height_memory.py', 'processing_reference.py',
            'assets/robot.xml', 'assets/robot.urdf', 'provenance.json'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe bundle path: ' + str(relative))
    resolved = (root / p).resolve()
    if root not in resolved.parents: raise ValueError('Bundle path escapes root')
    return resolved


def validate_metadata(m,task=None):
    c=contract_for(task or m.get('task'))
    if (m.get('task') != c.TASK or m.get('checkpoint_schema') != c.SCHEMA
        or m.get('checkpoint_sha256')!=c.CHECKPOINT_SHA256
        or m.get('interface') != c.INTERFACE or m.get('control_dt') != .02
        or m.get('encoder_output_frame') != 'base_body' or m.get('actor_velocity_frame') != 'body'
        or m.get('tcp_local_m') != c.TCP
        or m.get('physical_command_units') != ['rad/s', 'm/s', 'm']
        or m.get('observation_order') != c.OBSERVATION_ORDER
        or m.get('raw_actor_input_order') != c.RAW_ACTOR_ORDER
        or m.get('actor_features_order') != c.ACTOR_FEATURES_ORDER
        or m.get('feature_transform_included') is not True
        or m.get('stateful_processing_reference') != 'processing_reference.py'
        or m.get('runtime_math') != 'adaptive_math.py'
        or m.get('curriculum_stage') != c.EVALUATION_STAGE):
        raise ValueError('Adaptive metadata contract mismatch')
    if c.IS_ABLATION:
        if (m.get('height_rate_limits_m_s')!=c.HEIGHT_RATES or m.get('height_memory',{}).get('enabled') is not c.HEIGHT_MEMORY_ENABLED
            or m.get('height_memory',{}).get('profile')!=c.HEIGHT_MEMORY):
            raise ValueError('Adaptive height memory contract mismatch')
    elif 'height_memory' in m:
        raise ValueError('Original adaptive balance must not include height memory')
    ctx = m.get('context_step', {})
    if (ctx.get('hidden_width') != 64 or ctx.get('latent_width') != 8 or ctx.get('lpf_tau_s') != .3
        or ctx.get('input_order') != ['observation', 'previous_issued_action', 'hidden', 'filtered_eta']
        or ctx.get('output_order') != ['next_hidden', 'next_filtered_eta', 'raw_eta']
        or ctx.get('student_input') != 'noisy_policy_first36_previous_software_clipped_issued13'):
        raise ValueError('Adaptive context contract mismatch')
    if (m.get('experiment', {}).get('behavior_revision') != 2
        or (c.IS_ABLATION and (m.get('experiment', {}).get('ablation_group') != c.ABLATION_GROUP
            or m.get('experiment',{}).get('mode_sampling')!=c.MODE_SAMPLING))
        or (not c.IS_ABLATION and (m.get('experiment',{}).get('factors')!='E'
            or m.get('experiment',{}).get('revision')!=2 or 'ablation_group' in m.get('experiment',{})))
        or m.get('dynamic_input_order') != c.DYNAMIC_ORDER
        or m.get('history_order') != c.HISTORY_ORDER):
        raise ValueError('Adaptive temporal contract mismatch')
    for name, digest in m.get('file_sha256', {}).items():
        if Path(name).name != name or len(digest) != 64:
            raise ValueError('Invalid exported fingerprint path')


def load_adaptive_bundle(directory, task=None):
    root = Path(directory).resolve()
    m = json.loads((root / 'manifest.yaml').read_text())
    t = m.get('task', {})
    c=contract_for(task or t.get('task_id'))
    yaw,vx=base.STAGE_LIMITS[c.EVALUATION_STAGE];low,high=base.STAGE_HEIGHT_RANGES[c.EVALUATION_STAGE]
    if m.get('schema') != c.BUNDLE_SCHEMA or t.get('task_id') != c.TASK or t.get('checkpoint_schema') != c.SCHEMA:
        raise ValueError('Adaptive bundle task/schema mismatch')
    if m.get('checkpoint',{}).get('sha256')!=c.CHECKPOINT_SHA256:
        raise ValueError('Checkpoint differs from selected adaptive profile')
    expected = dict(observation=49, history=245, encoder=7, command=3,
        dynamic=c.INTERFACE['actor_raw']-59, policy_input=c.INTERFACE['actor_raw'], action=13)
    if m.get('dimensions') != expected or t.get('runtime_profile') != c.RUNTIME:
        raise ValueError('Adaptive bundle dimensions/runtime mismatch')
    if (t.get('hardware_joint_names') != base.JOINTS
        or t.get('action_joint_names') != [base.JOINTS[i] for i in base.ACTION_TO_HARDWARE]
        or t.get('policy_dt') != .02 or t.get('history_length') != 5
        or t.get('history_order') != 'oldest_to_newest' or t.get('history_first_frame') != 'repeat'
        or t.get('action_clipping', 'missing') is not None
        or t.get('position_scale') != .25 or t.get('wheel_scale') != 1
        or t.get('velocity_reference') != 'fixed_tcp' or t.get('tcp_local_m') != c.TCP
        or t.get('actor_velocity_frame') != 'body' or t.get('root_position') != [0,0,.65337]
        or t.get('ros_runtime_supported') is not True
        or t.get('fixed_payload') != dict(up3_total_mass=.67148,rear_mass=0.,rear_x=-.275)
        or t.get('observation_order') != ['base_gyro*.25','base_gravity','up3_gyro*.25','up3_gravity','position_relative_11','velocity_13*.05','last_action_13']
        or t.get('command_min') != [-yaw,-vx,low] or t.get('command_max') != [yaw,vx,high]
        or t.get('evaluation_stage') != c.EVALUATION_STAGE):
        raise ValueError('Adaptive motor/command contract mismatch')
    if (t.get('height_rate_limits_m_s')!=c.HEIGHT_RATES
        or t.get('command_processor')!={'implementation':c.RUNTIME[len('nandu1_'):],'control_dt':.02}
        or (not c.IS_ABLATION and 'height_memory_profile' in t)):
        raise ValueError('Adaptive command processor/height schedule mismatch')
    for key, expected_values in [('default_joint_positions', base.DEFAULT_Q), ('kp', base.KP), ('kd', base.KD),
                                 ('effort', base.EFFORT), ('velocity_limits', base.VELOCITY), ('armature', base.ARMATURE)]:
        if not np.array_equal(np.asarray(t.get(key, [])), expected_values):
            raise ValueError('Adaptive motor contract mismatch: ' + key)
    lo, hi = [np.asarray(t.get(key, [])) for key in ('joint_position_lower', 'joint_position_upper')]
    if lo.shape != (13,) or hi.shape != (13,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
        raise ValueError('Invalid physical limits')
    if not low<=t.get('initial_up3_height', float('nan'))<=high: raise ValueError('Invalid initial height')
    files = m.get('files', {})
    required=REQUIRED if c.IS_ABLATION else REQUIRED-{'adaptive_height_memory.py'}
    if not required <= files.keys(): raise ValueError('Adaptive bundle missing required files')
    if not c.IS_ABLATION and 'adaptive_height_memory.py' in files:raise ValueError('Original adaptive balance has no height memory feature')
    pins=getattr(c,'INFERENCE_SHA256',None)
    if pins is not None and (len(pins)!=4 or any(files.get(k)!=v for k,v in pins.items())):
        raise ValueError('Adaptive inference identity mismatch')
    for name, digest in files.items():
        path = safe_path(root, name)
        if not path.is_file() or sha256(path) != digest: raise ValueError('Bundle hash mismatch: ' + name)
    # MJCF/URDF file references are part of the sealed asset, not arbitrary
    # absolute host paths. Resolve compiler meshdir explicitly before loading.
    model=ET.parse(root/'assets/robot.xml').getroot();compiler=model.find('compiler')
    meshdir=compiler.get('meshdir','') if compiler is not None else ''
    texturedir=compiler.get('texturedir','') if compiler is not None else ''
    for element in model.iter():
        relative=element.get('file')
        if relative:
            folder=meshdir if element.tag=='mesh' else texturedir if element.tag=='texture' else ''
            path=safe_path(root, str(Path('assets')/folder/relative))
            if str(path.relative_to(root)) not in files:raise ValueError('Unsealed asset reference')
    for element in ET.parse(root/'assets/robot.urdf').getroot().iter('mesh'):
        path=safe_path(root,str(Path('assets')/element.get('filename','')))
        if str(path.relative_to(root)) not in files:raise ValueError('Unsealed URDF reference')
    sums = dict((line.split('  ', 1)[1], line.split('  ', 1)[0])
                for line in (root / 'SHA256SUMS').read_text().splitlines())
    if sums.get('manifest.yaml') != sha256(root / 'manifest.yaml') or any(sums.get(k) != v for k, v in files.items()):
        raise ValueError('Bundle checksums disagree')
    meta = json.loads((root / 'deployment_metadata.json').read_text())
    validate_metadata(meta,c.TASK)
    if meta['checkpoint_sha256'] != m.get('checkpoint', {}).get('sha256'):
        raise ValueError('Checkpoint identity mismatch')
    for name, digest in meta['file_sha256'].items():
        if files.get(name) != digest: raise ValueError('Export fingerprints disagree: ' + name)
    return m, root / 'assets/robot.xml'
