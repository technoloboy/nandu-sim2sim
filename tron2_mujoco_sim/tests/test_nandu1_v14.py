"""Catch task alias, frame, state ordering and command-processor divergence."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

SIM=Path(__file__).resolve().parents[1]
ROOT=SIM.parent
sys.path.insert(0,str(SIM))

def training_math():
    path=ROOT/'tron2_rl_lab/exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/NANDU1/nandu1_v1_4_math.py'
    if not path.is_file():path=ROOT/'deployment_bundles/nandu1_v1_4_model_24000/command_processor.py'
    spec=importlib.util.spec_from_file_location('training_v14_math',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def runtime():
    assert (SIM/'nandu1_v14.py').exists(),'v1.4 runtime is not implemented'
    import nandu1_v14
    return nandu1_v14

def test_v14_task_alias_selects_heading_62_input():
    import nandu1_contract as c
    assert hasattr(c,'task_profile'),'v1.4 task registration is missing'
    p=c.task_profile('Isaac-NANDU1-Blind-Flat-Play-v1-4')
    assert p['task_id']=='Isaac-NANDU1-Blind-Flat-v1-4'
    assert p['policy_input']==62 and p['command']==6
    assert c.task_profile(c.PLAY)['policy_input']==59
    with pytest.raises(ValueError):c.task_profile('Isaac-NANDU1-Blind-Flat-v1-3-2')

def test_heading_adapter_tilt_pole_and_zero_gravity():
    m=runtime();t=training_math();import torch
    gravity=np.array([[0,0,-1],[.5,0,-.8660254],[-1,0,0],[0,0,0]],dtype=np.float32)
    velocity=np.array([[1,2,3]]*4,dtype=np.float32)
    result=np.array([m.body_to_heading(v,g) for v,g in zip(velocity,gravity)])
    expected=t.body_to_heading(torch.tensor(velocity),torch.tensor(gravity)).numpy()
    np.testing.assert_allclose(result,expected,atol=1e-6)
    np.testing.assert_allclose(result[0],[1,2,3],atol=1e-6)
    obs=np.zeros(49,np.float32);obs[3:6]=gravity[1];enc=np.array([1,2,3,4,5,6,.9],np.float32);commands=np.array([0,.1,.9,0,1,0],np.float32)
    inp=m.actor_input(enc,obs,commands)
    assert inp.shape==(62,);np.testing.assert_array_equal(inp[-6:],commands)
    np.testing.assert_allclose(inp[:3],expected[1],atol=1e-6)
    with pytest.raises(ValueError):m.actor_input(enc,obs,np.zeros(3))

def test_processor_matches_training_turn_release_tilt_saturation_reset():
    m=runtime();t=training_math();import torch
    ref=t.CommandProcessor(1,'cpu');ref.reset(None,.8856453)
    actual=m.CommandProcessor(4,.8856453)
    for step in range(160):
        if step==90:ref.reset(None,.8);actual.reset(.8)
        gravity=np.array([.45,.10,-.88741195],np.float32)
        gyro=np.array([.12,-.09,.15 if step<60 else .4],np.float32)
        external=np.array([.3 if step<40 else (2. if step>=120 else 0),.1 if step%30<15 else -.1,1.32 if step<90 else .7],np.float32)
        ref.set_external(torch.tensor(external),4);ref.step(.02,torch.tensor(gravity[None]),torch.tensor(gyro[None]),4)
        expected=np.r_[ref.commands[0].numpy(),ref.trend[0].numpy()]
        np.testing.assert_allclose(actual.step(external,gravity,gyro,.02),expected,atol=3e-5,rtol=1e-5)
        assert abs(actual.heading-float(ref.heading[0]))<3e-5
    with pytest.raises(ValueError):actual.step([np.nan,0,.9],gravity,gyro,.02)
    with pytest.raises(ValueError):actual.step([0,0,.9],gravity,gyro,0)

def test_zero_intent_uses_imu_heading_hold_not_true_yaw():
    m=runtime();p=m.CommandProcessor(4,.9)
    out=p.step([0,0,.9],[0,0,-1],[0,0,.2],.02)
    assert out[0]==pytest.approx(-.006,abs=1e-6)
    assert out[2]==pytest.approx(.9)
    assert out[3]==pytest.approx(-.006/(.02*1.57),abs=1e-6)
    p.reset(.9)
    np.testing.assert_allclose(p.step([0,0,.9],[0,0,-1],[0,0,0],.02),[0,0,.9,0,0,0],atol=1e-6)

def test_v14_bundle_and_strict_rejections(tmp_path):
    import json
    import nandu1_contract as c
    from nandu1_bundle import load_bundle
    bundle=ROOT/'deployment_bundles/nandu1_v1_4_model_24000'
    manifest,_=load_bundle(bundle,c.V14_PLAY)
    assert manifest['dimensions']['policy_input']==62
    assert manifest['task']['evaluation_stage']==4
    with pytest.raises(ValueError):load_bundle(bundle,c.TASK)
    for key,value in [('evaluation_stage',4.0),('velocity_reference','body_com'),('actor_velocity_frame','body'),('command_processor',{})]:
        changed=json.loads(json.dumps(manifest));changed['task'][key]=value
        (tmp_path/'manifest.yaml').write_text(json.dumps(changed))
        with pytest.raises(ValueError):load_bundle(tmp_path,c.V14_TASK)

def test_link_origin_truth_matches_mujoco_jacobian():
    import mujoco
    from nandu1_simulator import Nandu1Physics
    from nandu1_v14 import body_to_heading
    physics=Nandu1Physics(ROOT/'deployment_bundles/nandu1_v1_4_model_24000')
    physics.data.qvel[:]=np.linspace(-.2,.3,physics.model.nv)
    mujoco.mj_forward(physics.model,physics.data)
    packet=physics.packet(0);rotation=physics.data.xmat[physics.bids['base_Link']].reshape(3,3)
    velocities=[]
    for name in ['base_Link','up3']:
        jp=np.zeros((3,physics.model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacBody(physics.model,physics.data,jp,jr,physics.bids[name])
        velocities.extend(rotation.T@jp@physics.data.qvel)
    np.testing.assert_allclose(packet['truth'][:6],velocities,atol=1e-10)
    np.testing.assert_allclose(packet['command_truth'][1],body_to_heading(velocities[3:],rotation.T@[0,0,-1])[0],atol=1e-7)
