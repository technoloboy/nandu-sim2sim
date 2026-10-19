"""Catch same-width profile substitution and silently promoted curriculum limits."""
import hashlib
import json
from pathlib import Path
import shutil
import pytest

ROOT=Path(__file__).resolve().parents[2]
CASES=[(3,'B',11000,1,False,[-.5,-.2,.68],[.5,.2,1.12]),
       (5,'D',10500,4,True,[-1.57,-1,.4],[1.57,1,1.4]),
       (7,'F',10000,2,True,[-.85,-.4,.58],[.85,.4,1.22])]


@pytest.mark.parametrize('suffix,group,iteration,stage,enabled,lower,upper',CASES)
def test_downloaded_ablation_routes_to_own_stage_and_preserves_width(suffix,group,iteration,stage,enabled,lower,upper):
    from nandu1_adaptive_contract import contract_for,deployment_profile
    task=f'Isaac-NANDU1-Blind-Flat-v1-3-1-2-{suffix}'
    c=contract_for(task)
    assert c.EVALUATION_STAGE==stage and c.HEIGHT_MEMORY_ENABLED is enabled
    assert c.IS_ABLATION and c.INTERFACE['actor_raw']==71
    assert deployment_profile(task)['checkpoint_schema']==f'nandu_network_v1_3_1_2_{suffix}_ablation_{group}_r2'
    assert deployment_profile(task.replace('Flat-v','Flat-Play-v'))==deployment_profile(task)
    import nandu1_contract as base
    assert len(base.TASKS)==7


@pytest.mark.parametrize('suffix,group,iteration,stage,enabled,lower,upper',CASES)
def test_actual_bundle_identity_and_curriculum_limits(suffix,group,iteration,stage,enabled,lower,upper):
    from nandu1_bundle import load_bundle
    path=ROOT/f'deployment_bundles/nandu1_v1_3_1_2_{suffix}_model_{iteration}'
    assert path.is_dir(), 'New ablation deployment bundle missing'
    m,_=load_bundle(path,f'Isaac-NANDU1-Blind-Flat-v1-3-1-2-{suffix}')
    assert m['dimensions']['policy_input']==71 and m['dimensions']['dynamic']==12
    assert m['task']['evaluation_stage']==stage
    assert m['task']['command_min']==lower and m['task']['command_max']==upper
    assert m['task']['height_rate_limits_m_s']==[.05,.1,.2,.35,.5]
    meta=json.loads((path/'deployment_metadata.json').read_text())
    assert meta['height_memory']['enabled'] is enabled and meta['curriculum_stage']==stage
    assert 'adaptive_height_memory.py' in m['files']  # even disabled B keeps its channel


@pytest.mark.parametrize('suffix,group,iteration,stage,enabled,lower,upper',CASES)
@pytest.mark.parametrize('change',['stage','bounds','memory','graph','metadata_profile','manifest_checkpoint'])
def test_resealed_bundle_rejects_wrong_semantics(tmp_path,suffix,group,iteration,stage,enabled,lower,upper,change):
    from nandu1_bundle import load_bundle
    source=ROOT/f'deployment_bundles/nandu1_v1_3_1_2_{suffix}_model_{iteration}'
    assert source.is_dir(), 'New ablation deployment bundle missing'
    b=tmp_path/'bundle';shutil.copytree(source,b)
    m=json.loads((b/'manifest.yaml').read_text());meta=json.loads((b/'deployment_metadata.json').read_text())
    if change=='stage':m['task']['evaluation_stage']=(stage+1)%5
    if change=='bounds':m['task']['command_min'][1]-=.01
    if change=='memory':meta['height_memory']['enabled']=not enabled
    if change=='metadata_profile':
        # Both metadata and manifest digest consistently name ANOTHER valid
        # profile while the selected task and independently pinned graphs stay.
        other=ROOT/'deployment_bundles'/('nandu1_v1_3_1_2_5_model_10500' if suffix==3 else 'nandu1_v1_3_1_2_3_model_11000')
        meta=json.loads((other/'deployment_metadata.json').read_text())
        meta['file_sha256']={k:m['files'][k] for k in meta['file_sha256']}
        m['checkpoint']['sha256']=meta['checkpoint_sha256']
    if change=='manifest_checkpoint':m['checkpoint']['sha256']='0'*64
    if change=='graph':
        shutil.copyfile(ROOT/'deployment_bundles/nandu1_v1_3_1_2_6_model_23500/policy.onnx',b/'policy.onnx')
        meta['file_sha256']['policy.onnx']=m['files']['policy.onnx']=hashlib.sha256((b/'policy.onnx').read_bytes()).hexdigest()
    (b/'deployment_metadata.json').write_text(json.dumps(meta))
    m['files']['deployment_metadata.json']=hashlib.sha256((b/'deployment_metadata.json').read_bytes()).hexdigest()
    (b/'manifest.yaml').write_text(json.dumps(m))
    sums=dict(m['files'],**{'manifest.yaml':hashlib.sha256((b/'manifest.yaml').read_bytes()).hexdigest()})
    (b/'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k,v in sums.items()))
    with pytest.raises(ValueError,match='inference identity' if change=='graph' else None):
        load_bundle(b)
