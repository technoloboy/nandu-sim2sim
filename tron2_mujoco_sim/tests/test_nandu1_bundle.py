import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_asset_matches_source_and_fk(tmp_path):
    source = Path(__file__).resolve().parents[2]/'tron2_rl_lab/exts/bipedal_locomotion/bipedal_locomotion/assets/usd/NANDU1'
    if not source.is_dir():pytest.skip('Optional training USD/URDF source is not distributed; sealed assets are tested separately')
    from nandu1_asset import build_asset, audit_asset
    path = build_asset(source, tmp_path/'assets')
    report = audit_asset(source, path)
    assert report['max_position_error_m'] < .001
    assert report['max_rotation_error_deg'] < .1
    assert report['actuators'] == 13

def test_bad_task_or_checksum_is_rejected(tmp_path):
    from nandu1_bundle import load_bundle
    import json
    (tmp_path/'manifest.yaml').write_text(json.dumps({'schema':'limx.nandu1-deployment-bundle/v1','task':{'task_id':'wrong'}}))
    with pytest.raises(ValueError):
        load_bundle(tmp_path)

@pytest.mark.parametrize('field,value',[('command_rate',[20,.8,.5]),('observation_order',[]),('encoder_order',[])])
def test_wrong_runtime_contract_rejected(tmp_path,field,value):
    import json
    from nandu1_bundle import load_bundle
    bundle=Path(__file__).resolve().parents[2]/'deployment_bundles/nandu1_v1_2_model_40000'
    manifest=json.loads((bundle/'manifest.yaml').read_text())
    manifest['task'][field]=value
    (tmp_path/'manifest.yaml').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='runtime contract'):
        load_bundle(tmp_path)

def test_manifest_tamper_rejected(tmp_path):
    import shutil
    from nandu1_bundle import load_bundle
    bundle=Path(__file__).resolve().parents[2]/'deployment_bundles/nandu1_v1_2_model_40000'
    shutil.copytree(bundle,tmp_path/'bundle')
    manifest=tmp_path/'bundle/manifest.yaml'
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError,match='Manifest checksum'):
        load_bundle(tmp_path/'bundle')
