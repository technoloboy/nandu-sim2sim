"""The new tasks must route to their own checkpoint/actor contracts."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nandu1_contract as c

@pytest.mark.parametrize('version,schema,encoder,actor',[
    ('1','nandu1_height_v1_1_robust_v2','single',59),
    ('3','nandu1_height_v1_3_robust_v2','dual',59),
    ('2-1','nandu1_height_v1_2_1_adaptive_hold_v1','dual',59),
    ('3-1','nandu1_height_v1_3_1_tcp_payload_v1','dual',59),
    ('4-1','nandu1_height_v1_4_1_smooth_transition_v1','dual',62),
])
def test_train_and_play_select_checkpoint_and_encoder(version,schema,encoder,actor):
    task='Isaac-NANDU1-Blind-Flat-v1-'+version
    profile=c.task_profile(task)
    assert c.task_profile('Isaac-NANDU1-Blind-Flat-Play-v1-'+version)==profile
    assert profile['task_id']==task
    assert profile['checkpoint_schema']==schema
    assert profile['encoder_architecture']==encoder
    assert profile['policy_input']==actor

def test_unsupported_suffix_is_not_accepted_as_legacy_body():
    with pytest.raises(ValueError):c.task_profile('Isaac-NANDU1-Blind-Flat-v1-3-2')

def test_export_rejects_wrong_schema_and_missing_stage():
    if not (Path(__file__).resolve().parents[2]/'tron2_rl_lab/scripts/rsl_rl/export_nandu1_deployment.py').is_file():
        pytest.skip('Optional training exporter is not distributed; deployed bundle identity is tested separately')
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tron2_rl_lab/scripts/rsl_rl'))
    from export_nandu1_deployment import validate_body_checkpoint
    profile=c.task_profile(c.V121_TASK)
    checkpoint={'task_name':c.V121_TASK,'checkpoint_schema':profile['checkpoint_schema'],
                'environment_state_dict':{'nandu1_height_v1_2_1':{'stage':3}}}
    assert validate_body_checkpoint(checkpoint,profile)==3
    checkpoint['checkpoint_schema']='nandu1_height_v1_2_robust_v2'
    with pytest.raises(ValueError,match='schema'):validate_body_checkpoint(checkpoint,profile)
    checkpoint['checkpoint_schema']=profile['checkpoint_schema'];checkpoint['environment_state_dict']={}
    with pytest.raises(ValueError,match='stage'):validate_body_checkpoint(checkpoint,profile)

def test_v141_ros_activation_requires_explicit_supported_bundle():
    from nandu1_bundle import require_controller_runtime
    manifest={'task':{'task_id':c.V141_TASK,'ros_runtime_supported':True}}
    require_controller_runtime(manifest)
    manifest['task']['ros_runtime_supported']=False
    with pytest.raises(ValueError,match='unsupported'):require_controller_runtime(manifest)

def test_local_probe_tcp_pipeline_holds_heading_instead_of_raw_zero_yaw():
    from nandu1_inference import CommandPipeline
    obs=[0.]*49;obs[5]=-1
    pipeline=CommandPipeline({'task_id':c.V131_TASK,'evaluation_stage':4,'initial_up3_height':.9,
                             'command_min':[-1.57,-1,.4],'command_max':[1.57,1,1.4]})
    values=pipeline.step([0,0,.9],obs,[0,0,.2])
    assert values[0]==pytest.approx(-.006,abs=1e-7)
    assert len(pipeline.actor_input([0.]*7,obs,values))==59
    pipeline.reset()
    assert pipeline.step([0,0,.9],obs,[0,0,0])[0]==0

@pytest.mark.parametrize('name,task',[
    ('nandu1_v1_1_model_30000',c.V11_TASK),('nandu1_v1_3_model_30000',c.V13_TASK),
    ('nandu1_v1_2_1_model_29000',c.V121_TASK),('nandu1_v1_3_1_model_30000',c.V131_TASK),
    ('nandu1_v1_4_1_model_14000',c.V141_TASK),
])
def test_sealed_new_bundle_cannot_be_relabelled_as_another_task(name,task):
    from nandu1_bundle import load_bundle,require_controller_runtime
    root=Path(__file__).resolve().parents[2]/'deployment_bundles'/name
    manifest,_=load_bundle(root,task.replace('Flat-v','Flat-Play-v'))
    require_controller_runtime(manifest)
    assert manifest['task']['task_id']==task
    with pytest.raises(ValueError,match='mismatch'):load_bundle(root,c.TASK)

def test_ros_validator_has_no_mujoco_dependency_for_tcp_bundle():
    import subprocess
    root=Path(__file__).resolve().parents[2]
    code=f'import sys;sys.modules["mujoco"]=None;sys.path.insert(0,{str(root/"tron2_mujoco_sim")!r});from nandu1_bundle import load_bundle;load_bundle({str(root/"deployment_bundles/nandu1_v1_3_1_model_30000")!r})'
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
