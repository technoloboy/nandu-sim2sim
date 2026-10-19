"""Protect the B71/disabled-memory case and each checkpoint's command envelope."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
CASES=[(3,'b',11000,1,False,.2,.5,.68,1.12),
       (5,'d',10500,4,True,1.,1.57,.4,1.4),
       (7,'f',10000,2,True,.4,.85,.58,1.22)]


@pytest.mark.parametrize('suffix,group,it,stage,memory,vx,yaw,lo,hi',CASES)
def test_actual_runtime_default_stage_and_temporal_state(suffix,group,it,stage,memory,vx,yaw,lo,hi):
    from nandu1_adaptive_runtime import AdaptiveRuntime
    path=ROOT/f'deployment_bundles/nandu1_v1_3_1_2_{suffix}_model_{it}'
    assert path.is_dir(), 'New ablation deployment bundle missing'
    r=AdaptiveRuntime(path,initial_height=.9)
    assert r.stage==stage
    obs=np.zeros(49,np.float32);obs[5]=obs[11]=-1
    first=r.step(obs,[0,0,.9],step_id=0)
    assert first['actor_input'].shape==(71,)
    np.testing.assert_array_equal(first['dynamic'],np.zeros(12))
    assert r.processor.height_memory.enabled is memory
    for i in range(1,41):
        out=r.step(obs,[99,99,99],step_id=i)
        assert abs(out['commands'][0])<=yaw+1e-6 and abs(out['commands'][1])<=vx+1e-6
        assert lo-1e-6<=out['commands'][2]<=hi+1e-6
        if not memory:assert out['dynamic'][11]==0 and out['height_filter'][0]==0
    repeated=r.step(obs,[-99,-99,-99],step_id=40)
    np.testing.assert_array_equal(repeated['hidden'],out['hidden'])
    np.testing.assert_array_equal(repeated['commands'],out['commands'])
    r.reset(initial_height=.9)
    np.testing.assert_array_equal(r.step(obs,[0,0,.9],step_id=0)['action'],first['action'])


@pytest.mark.parametrize('suffix,group,it,stage,memory,vx,yaw,lo,hi',CASES)
def test_ablation_command_launch_and_layout_are_independent(suffix,group,it,stage,memory,vx,yaw,lo,hi):
    from nandu1_process import launch_file
    from nandu1_mouse_protocol import command_interface
    task=f'Isaac-NANDU1-Blind-Flat-v1-3-1-2-{suffix}'
    assert launch_file(task)==f'nandu1_ablation_{group}_mujoco_policy.launch'
    assert launch_file(task.replace('Flat-v','Flat-Play-v'))==launch_file(task)
    assert command_interface(task)==('tcp',f'/nandu1/v1_3_1_2_{suffix}/command')
    spec=importlib.util.spec_from_file_location('bdf_logger',ROOT/'tron2_rl_deploy_ros/tron2_controllers/scripts/nandu1_adaptive_telemetry.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    fields=m.tracking_fields('ablation_'+group)
    assert len(fields)==284 and fields[140]=='dynamic_11' and fields[-1]=='actor_input_70'
    assert m.Pairer('ablation_'+group).tracking([0.]*282)==[]
