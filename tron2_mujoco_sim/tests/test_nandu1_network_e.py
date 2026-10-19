"""Original E is a distinct raw59/feature62 task, never adaptive E/r2."""
import copy
import importlib.util
import json
import hashlib
import shutil
from pathlib import Path

import pytest
import nandu1_contract as c

ROOT=Path(__file__).resolve().parents[2]
TASK='Isaac-NANDU1-Blind-Flat-v1-3-1-2'
SCHEMA='nandu_network_v1-3-1_2_E_r1'
SOURCE=ROOT/'tron2_rl_lab/logs/rsl_rl/autodl/v1-3-1-2-261018-27500/exported'
BUNDLE=ROOT/'deployment_bundles/nandu1_v1_3_1_2_model_27500'


def test_original_e_task_and_play_are_distinct_from_adaptive_and_baseline():
    p=c.task_profile(TASK)
    assert p['checkpoint_schema']==SCHEMA and p['policy_input']==59
    assert p['actor_features']==62 and p['velocity_reference']=='fixed_tcp'
    assert c.task_profile(TASK.replace('Flat-v','Flat-Play-v'))==p
    assert len(c.TASKS)==7  # existing collection registries keep their semantics
    from nandu1_adaptive_contract import deployment_profile
    assert deployment_profile('Isaac-NANDU1-Blind-Flat-v1-3-1-2-6')['policy_input']==71


@pytest.mark.parametrize('change',['valid','task','schema','features','raw','transform','units','obs','frame','history','order','factors'])
def test_metadata_rejects_same_width_wrong_feature_semantics(change):
    assert importlib.util.find_spec('nandu1_network_e'), 'Original-E contract missing'
    from nandu1_network_e import validate_metadata
    m=json.loads((BUNDLE/'deployment_metadata.json').read_text())
    if change=='task':m['task']='Isaac-NANDU1-Blind-Flat-v1-3-1'
    if change=='schema':m['checkpoint_schema']='nandu_network_v1_3_1_2_6_ablation_E_r2'
    if change=='features':m['experiment']['interface']['actor_features']=59
    if change=='raw':m['experiment']['interface']['actor_raw']=62
    if change=='transform':m['feature_transform_included']=False
    if change=='units':m['physical_command_units']=['deg/s','m/s','cm']
    if change=='obs':m['observation_order'].reverse()
    if change=='frame':m['actor_velocity_frame']='heading'
    if change=='history':m['history_order']='newest_to_oldest'
    if change=='order':m['raw_actor_input_order']='policy49,Encoder7,commands3'
    if change=='factors':m['experiment']['factors']='H'
    if change=='valid':validate_metadata(m)
    else:
        with pytest.raises(ValueError):validate_metadata(m)


def test_sealed_e_bundle_and_checkpoint_identity_are_real():
    assert BUNDLE.is_dir(), 'Original-E deployment bundle not yet exported'
    from nandu1_bundle import load_bundle
    m,model=load_bundle(BUNDLE,TASK.replace('Flat-v','Flat-Play-v'))
    assert m['checkpoint']['sha256']=='96fbe2f9ab0c347c5877aee7b20ee1db141c27f42b57d84210504e613c3ee2ef'
    assert m['task']['checkpoint_schema']==SCHEMA and model.is_file()
    assert m['dimensions']==dict(observation=49,history=245,encoder=7,command=3,policy_input=59,action=13)
    assert 'context_step.onnx' not in m['files']
    with pytest.raises(ValueError):load_bundle(BUNDLE,c.V131_TASK)


def test_plain_tcp_bundle_cannot_pass_as_e_even_with_same_raw_width():
    from nandu1_bundle import load_bundle
    with pytest.raises(ValueError):load_bundle(ROOT/'deployment_bundles/nandu1_v1_3_1_model_30000',TASK)


@pytest.mark.parametrize('change',['policy','encoder','actor_pipeline','checkpoint'])
def test_resealed_same_width_wrong_graph_or_checkpoint_is_rejected(tmp_path,change):
    """Self-consistent bundle hashes are not an independent graph identity."""
    from nandu1_bundle import load_bundle
    root=tmp_path/'resealed';shutil.copytree(BUNDLE,root)
    m=json.loads((root/'manifest.yaml').read_text());meta=json.loads((root/'deployment_metadata.json').read_text())
    if change=='checkpoint':
        m['checkpoint']['sha256']=meta['checkpoint_sha256']='0'*64
    else:
        name=change+'.onnx'
        shutil.copyfile(ROOT/'deployment_bundles/nandu1_v1_3_1_model_30000'/name,root/name)
        digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
        m['files'][name]=meta['file_sha256'][name]=digest
    (root/'deployment_metadata.json').write_text(json.dumps(meta))
    m['files']['deployment_metadata.json']=hashlib.sha256((root/'deployment_metadata.json').read_bytes()).hexdigest()
    (root/'manifest.yaml').write_text(json.dumps(m))
    sums=dict(m['files'],**{'manifest.yaml':hashlib.sha256((root/'manifest.yaml').read_bytes()).hexdigest()})
    (root/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in sums.items()))
    with pytest.raises(ValueError):load_bundle(root,TASK)
