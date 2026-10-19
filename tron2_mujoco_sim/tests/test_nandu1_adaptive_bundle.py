"""Catch wrong model identity, missing inference artifacts and unsafe bundle paths."""
import importlib
import json
import hashlib
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'deployment_bundles/nandu1_v1_3_1_2_6_model_23500'
TASK = 'Isaac-NANDU1-Blind-Flat-v1-3-1-2-6'


def test_new_task_alias_cannot_be_misrouted_to_legacy_actor():
    spec = importlib.util.find_spec('nandu1_adaptive_contract')
    assert spec is not None, 'new E/r2 contract is not implemented'
    contract = importlib.import_module('nandu1_adaptive_contract')
    assert contract.canonical_task(TASK.replace('Flat-v', 'Flat-Play-v')) == TASK
    with pytest.raises(ValueError):
        contract.canonical_task('Isaac-NANDU1-Blind-Flat-v1-3-1')


def test_actual_bundle_matches_explicit_checkpoint():
    from nandu1_adaptive_bundle import load_adaptive_bundle
    m, model = load_adaptive_bundle(BUNDLE, TASK)
    assert m['checkpoint']['sha256'] == '1db8e6e7f58e4613dd2a0144d80ec50e7a2e20573e7d05239408057bf4909115'
    assert m['dimensions']['policy_input'] == 71
    assert model.is_file()
    from nandu1_bundle import load_bundle
    assert load_bundle(BUNDLE, TASK)[0] == m


def test_graphs_are_compatible_with_existing_ros_ort():
    import onnx
    for name in ('encoder', 'policy', 'context_step', 'actor_pipeline'):
        m = onnx.load(BUNDLE / (name + '.onnx'))
        assert all(o.version <= 15 for o in m.opset_import if o.domain in ('', 'ai.onnx'))


@pytest.mark.parametrize('change', ['r1', 'dimensions', 'memory', 'context', 'escape', 'nan',
    'dynamic_order', 'units', 'observation_order', 'actor_order', 'feature_transform', 'history_suffix'])
def test_metadata_rejects_incompatible_contract(change):
    from nandu1_adaptive_bundle import validate_metadata
    m = json.loads((BUNDLE / 'deployment_metadata.json').read_text())
    if change == 'r1': m['checkpoint_schema'] = 'nandu_network_v1_3_1_2_6_ablation_E_r1'
    if change == 'dimensions': m['interface']['actor_raw'] = 59
    if change == 'memory': m['height_memory']['enabled'] = False
    if change == 'context': m['context_step']['hidden_width'] = 63
    if change == 'escape': m['file_sha256']['../escape'] = '0' * 64
    if change == 'nan': m['control_dt'] = float('nan')
    if change == 'dynamic_order': m['dynamic_input_order'].reverse()
    if change == 'units': m['physical_command_units'] = ['degrees/s', 'cm/s', 'cm']
    if change == 'observation_order': m['observation_order'].reverse()
    if change == 'actor_order': m['raw_actor_input_order'] = 'policy49,Encoder7,commands3,dynamic12'
    if change == 'feature_transform': m['feature_transform_included'] = False
    if change == 'history_suffix': m['history_order'] += '; reverse_before_encoder'
    with pytest.raises(ValueError): validate_metadata(m)


@pytest.mark.parametrize('change',['height','root','payload','observation','actor_frame','mesh_escape'])
def test_resealed_bundle_still_rejects_untrained_physical_semantics(tmp_path,change):
    from nandu1_adaptive_bundle import load_adaptive_bundle
    b=tmp_path/'bundle';shutil.copytree(BUNDLE,b)
    m=json.loads((b/'manifest.yaml').read_text())
    if change=='height':m['task']['initial_up3_height']=3.
    if change=='root':m['task']['root_position']=[0,0,2.]
    if change=='payload':m['task']['fixed_payload']['up3_total_mass']=float('nan')
    if change=='observation':m['task']['observation_order']=[]
    if change=='actor_frame':m['task']['actor_velocity_frame']='world'
    if change=='mesh_escape':
        f=b/'assets/robot.xml';f.write_text(f.read_text().replace('meshdir="meshes"','meshdir="/tmp"'))
        m['files']['assets/robot.xml']=hashlib.sha256(f.read_bytes()).hexdigest()
    f=b/'manifest.yaml';f.write_text(json.dumps(m))
    sums=dict(m['files'],**{'manifest.yaml':hashlib.sha256(f.read_bytes()).hexdigest()})
    (b/'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k,v in sums.items()))
    with pytest.raises(ValueError):load_adaptive_bundle(b)
