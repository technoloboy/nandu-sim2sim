import sys
from pathlib import Path
import numpy as np
import pytest
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nandu1_simulator import Nandu1Physics
import nandu1_contract as c

BUNDLE=Path(__file__).resolve().parents[2]/'deployment_bundles/nandu1_v1_2_model_40000'

def test_pd_torque_and_reset():
    p=Nandu1Physics(BUNDLE)
    q,dq,_=p.state()
    p.apply(q+100,dq,np.full(13,1000),np.zeros(13),np.zeros(13))
    np.testing.assert_array_equal(p.data.ctrl[p.aids],c.EFFORT)
    with pytest.raises(ValueError):p.apply(np.full(13,np.inf),dq,c.KP,c.KD,np.zeros(13))
    epoch=p.epoch;p.step();p.reset()
    assert p.epoch==epoch+1 and p.data.time==0
    np.testing.assert_allclose(p.state()[0],c.DEFAULT_Q)
    np.testing.assert_allclose(p.state()[1],0)

def test_estimator_truth_uses_base_frame_and_body_com():
    p=Nandu1Physics(BUNDLE)
    p.data.qvel[p.vadr[10]]=1
    mujoco.mj_forward(p.model,p.data)
    packet=p.packet(1)
    up=np.empty(6)
    mujoco.mj_objectVelocity(p.model,p.data,mujoco.mjtObj.mjOBJ_BODY,p.bids['up3'],up,0)
    base=p.data.xmat[p.bids['base_Link']].reshape(3,3)
    np.testing.assert_allclose(packet['truth'][3:6],base.T@up[3:])
    assert packet['command_truth'][1]==pytest.approx((base.T@up[3:])[0])
    assert np.linalg.norm(up[3:])>.1

def test_velocity_projection_is_visible_and_resettable():
    p=Nandu1Physics(BUNDLE)
    p.data.qvel[p.vadr[4]]=100.
    count=p.step()
    assert count>0
    assert p.packet(1)['velocity_projection_events']==count
    p.reset()
    assert p.packet(2)['velocity_projection_events']==0
