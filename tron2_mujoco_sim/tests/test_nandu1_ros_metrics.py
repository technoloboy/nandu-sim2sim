import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ros_metrics', ROOT/'scripts/analyze_nandu1_ros_tracking.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

def rows():
    out=[]
    for t in [1.,1.02,1.04]:
        r=dict(epoch=1.,tracking_epoch=1.,sample_age=.001,tracking_age=.002,tracking_stamp=t,sim_time=t)
        for prefix,n in [('command',3),('command_truth',3),('estimate',7),('truth',7),('q',13),('dq',13),('q_target',13),('torque_actual',13),('torque_saturated',13)]:
            r.update({f'{prefix}_{i}':0. for i in range(n)})
        r['command_truth_1']=.02
        out.append(r)
    return out

def manifest():
    return {'task':{'task_id':'Isaac-NANDU1-Blind-Flat-v1-2','joint_position_lower':[-5]*13,'joint_position_upper':[5]*13}}

def test_metrics_and_deduplication():
    r=rows();duplicate=dict(r[0],tracking_age=.01)
    result=metrics.summarize(r+[duplicate],manifest())
    assert result['synchronized_policy_samples']==3
    assert result['policy_hz_average']==pytest.approx(50)
    assert result['tracking_rmse_yaw_vx_height']==pytest.approx([0,.02,0])

def test_invalid_epochs_and_nonfinite_rejected():
    r=rows();r[0]['epoch']=2
    with pytest.raises(ValueError,match='reset epochs'):metrics.summarize(r,manifest())
    r=rows();r[0]['truth_0']=float('nan')
    with pytest.raises(ValueError,match='Non-finite'):metrics.summarize(r,manifest())

def test_wrong_tracking_epoch_not_used():
    r=rows();r[0]['tracking_epoch']=2
    assert metrics.summarize(r,manifest())['synchronized_policy_samples']==2

def test_step_response_requires_five_consecutive_samples():
    r=[]
    for i in range(100):
        sample=dict(rows()[0],tracking_stamp=i*.02,sim_time=i*.02)
        sample['command_1']=.1 if i>=50 else 0.
        sample['command_truth_1']=.1 if i>=55 else 0.
        r.append(sample)
    response=metrics.summarize(r,manifest())['command_plateaus'][1]['response_from_applied_command_including_slew']
    assert [x['delay_s'] for x in response]==pytest.approx([.1,.1])

def test_v14_heading_hold_does_not_fragment_external_step_segments():
    r=[]
    for i in range(100):
        sample=dict(rows()[0],tracking_stamp=i*.02,sim_time=i*.02)
        sample.update(external_intent_0=0.,external_intent_1=.1 if i>=50 else 0.,external_intent_2=.9)
        sample['command_0']=i*.0001;sample['command_1']=sample['external_intent_1'];sample['command_2']=.9
        sample['command_truth_1']=.1 if i>=55 else 0.
        r.append(sample)
    m=manifest();m['task']['task_id']='Isaac-NANDU1-Blind-Flat-v1-4'
    report=metrics.summarize(r,m)
    assert report['segment_reference']=='external_intent'
    assert len(report['command_plateaus'])==2
    response=report['command_plateaus'][1]['response_from_applied_command_including_slew']
    assert [x['delay_s'] for x in response]==pytest.approx([.1,.1])
