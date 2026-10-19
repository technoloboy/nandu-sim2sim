"""Catch dynamic12 padding, first-frame GRU advancement and logger mislabeling."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'deployment_bundles/nandu1_v1_3_1_2_2_model_29500'
TASK='Isaac-NANDU1-Blind-Flat-v1-3-1-2-2'


def test_balance_priming_prev_action_duplicate_reset_and_no_height_memory():
    from nandu1_adaptive_runtime import AdaptiveRuntime
    r=AdaptiveRuntime(BUNDLE,initial_height=.9)
    obs=np.zeros(49,np.float32);obs[5]=obs[11]=-1
    a=r.step(obs,[0,0,.9],step_id=0)
    np.testing.assert_array_equal(a['dynamic'],np.zeros(11))
    np.testing.assert_array_equal(a['history'],np.tile(obs,5))
    assert a['actor_input'].shape==(70,)
    assert not hasattr(r.processor,'height_memory')
    b=r.step(obs,[0,.2,1.],step_id=1)
    assert b['commands'][1]==pytest.approx(.012,abs=1e-7)
    assert b['dynamic'][10]==pytest.approx(1.,abs=2e-6)
    np.testing.assert_array_equal(b['observation'][36:],a['action'])
    repeat=r.step(obs,[1,1,1.4],step_id=1)
    np.testing.assert_array_equal(repeat['hidden'],b['hidden'])
    np.testing.assert_array_equal(repeat['commands'],b['commands'])
    bad=obs.copy();bad[0]=np.nan
    with pytest.raises(ValueError):r.step(bad,[0,0,.9],step_id=2)
    np.testing.assert_array_equal(r.step(obs,[0,0,.9],step_id=1)['action'],b['action'])
    r.reset(initial_height=.9)
    np.testing.assert_array_equal(r.step(obs,[0,0,.9],step_id=0)['action'],a['action'])


def test_balance_task_routes_tcp_command_and_independent_ros_launch():
    from nandu1_process import launch_file
    from nandu1_mouse_protocol import command_interface
    assert launch_file(TASK)=='nandu1_adaptive_balance_mujoco_policy.launch'
    assert launch_file(TASK.replace('Flat-v','Flat-Play-v'))=='nandu1_adaptive_balance_mujoco_policy.launch'
    assert command_interface(TASK)==('tcp','/nandu1/v1_3_1_2_2/command')


@pytest.mark.parametrize('stage,rate',[(0,.05),(1,.10),(2,.20),(3,.35),(4,.50)])
def test_balance_uses_original_five_stage_height_rates(stage,rate):
    from nandu1_adaptive_runtime import AdaptiveRuntime
    r=AdaptiveRuntime(BUNDLE,initial_height=.9,stage=stage)
    obs=np.zeros(49,np.float32);obs[5]=obs[11]=-1
    a=r.step(obs,[0,0,1.],step_id=0)
    assert a['commands'][2]-.9==pytest.approx(rate*.02,abs=1e-7)


def test_balance_telemetry_layout_preserves_exact_old_and_new_widths():
    f=ROOT/'tron2_rl_deploy_ros/tron2_controllers/scripts/nandu1_adaptive_telemetry.py'
    spec=importlib.util.spec_from_file_location('balance_logger_test',f)
    logger=importlib.util.module_from_spec(spec);spec.loader.exec_module(logger)
    fields=logger.tracking_fields('adaptive_balance')
    assert len(fields)==282 and len(set(fields))==282
    assert fields[129:140]==['dynamic_'+str(i) for i in range(11)]
    assert fields[212:]==['actor_input_'+str(i) for i in range(70)]
    assert len(logger.tracking_fields())==284
    assert len(logger.tracking_fields('network_e_tcp'))==121
    assert logger.Pairer('adaptive_balance').tracking([0.]*284)==[]
