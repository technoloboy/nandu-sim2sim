"""Reject r1 / E-ablation substitution for the specified local A/r2 run."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]
TASK='Isaac-NANDU1-Blind-Flat-v1-3-1-2-2'
SOURCE=ROOT/'tron2_rl_lab/logs/rsl_rl/nandu1_network_v1_3_1_2_2_E_adaptive_balance/2026-10-16_23-34-49_124751_local_A_v13122_r2_seed42/exported'
BUNDLE=ROOT/'deployment_bundles/nandu1_v1_3_1_2_2_model_29500'


def test_new_r2_routes_independently_from_r1_and_height_memory_ablation():
    from nandu1_adaptive_contract import deployment_profile
    p=deployment_profile(TASK)
    assert p['task_id']==TASK and p['policy_input']==70
    assert p['checkpoint_schema']=='nandu_network_v1_3_1_2_2_E_adaptive_balance_r2'
    assert deployment_profile(TASK.replace('Flat-v','Flat-Play-v'))==p
    assert deployment_profile('Isaac-NANDU1-Blind-Flat-v1-3-1-2-6')['policy_input']==71
    import nandu1_contract as base
    assert len(base.TASKS)==7


@pytest.mark.parametrize('change',['valid','r1','task','raw','features','height_memory',
    'dynamic','history','frame','context','checkpoint','units'])
def test_balance_metadata_rejects_incompatible_temporal_semantics(change):
    from nandu1_adaptive_bundle import validate_metadata
    m=json.loads((BUNDLE/'deployment_metadata.json').read_text())
    if change=='r1':m['checkpoint_schema']=m['checkpoint_schema'].replace('_r2','_r1')
    if change=='task':m['task']='Isaac-NANDU1-Blind-Flat-v1-3-1-2-6'
    if change=='raw':m['interface']['actor_raw']=71
    if change=='features':m['interface']['actor_features']=74
    if change=='height_memory':m['height_memory']={'enabled':True}
    if change=='dynamic':m['dynamic_input_order'].reverse()
    if change=='history':m['history_order']='newest_to_oldest'
    if change=='frame':m['actor_velocity_frame']='heading'
    if change=='context':m['context_step']['hidden_width']=63
    if change=='checkpoint':m['checkpoint_sha256']='0'*64
    if change=='units':m['physical_command_units']=['rad/s','cm/s','m']
    if change=='valid':validate_metadata(m)
    else:
        with pytest.raises(ValueError):validate_metadata(m)


def test_local_checkpoint_bundle_is_sealed_and_not_an_ablation_alias():
    assert BUNDLE.is_dir(), 'Specified local29500 deployment bundle missing'
    from nandu1_bundle import load_bundle
    m,asset=load_bundle(BUNDLE,TASK)
    assert m['checkpoint']['sha256']=='ff430c7709b9e9a5786c93fb33ebc1e4f4383fad83506d62de4ef6de6eb0d112'
    assert m['dimensions']==dict(observation=49,history=245,encoder=7,command=3,dynamic=11,policy_input=70,action=13)
    assert asset.is_file() and 'adaptive_height_memory.py' not in m['files']
    with pytest.raises(ValueError):load_bundle(BUNDLE,'Isaac-NANDU1-Blind-Flat-v1-3-1-2-6')
