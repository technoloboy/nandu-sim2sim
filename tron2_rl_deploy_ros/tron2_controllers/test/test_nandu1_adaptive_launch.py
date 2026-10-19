import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tron2_mujoco_sim'))
sys.path.insert(0,str(ROOT/'tron2_rl_deploy_ros/tron2_controllers/scripts'))


def test_adaptive_launch_selects_independent_controller_and_logger():
    path=ROOT/'tron2_rl_deploy_ros/tron2_hw/launch/nandu1_adaptive_mujoco_policy.launch'
    assert path.is_file(), 'Independent adaptive launch missing'
    xml=ET.parse(path).getroot()
    values={p.get('name'):p.get('value') for p in xml.findall('.//param')}
    assert values['tron2_controller/type']=='tron2_controller/NANDU1AdaptiveController'
    nodes={n.get('type') for n in xml.findall('.//node')}
    assert 'nandu1_aux_receiver.py' in nodes and 'nandu1_adaptive_telemetry.py' in nodes


def test_adaptive_logger_retains_imu_and_rejects_cross_epoch():
    assert importlib.util.find_spec('nandu1_adaptive_telemetry'), 'Versioned adaptive logger missing'
    from nandu1_adaptive_telemetry import align_packet, TRACKING_FIELDS
    track=[0.]*len(TRACKING_FIELDS);track[0]=100.;track[1]=7
    packet=dict(stamp_ns=100_010_000_000,sim_time=1.,epoch=7,seq=4,
        base=dict(quat=[1,0,0,0],gyro=[.1,.2,.3],acc=[0,0,9.81]),
        up3=dict(quat=[1,0,0,0],gyro=[.4,.5,.6],acc=[0,0,9.81]),truth=[0.]*7,command_truth=[0.]*3,
        q=[.1]*13,dq=[.2]*13,tau=[.3]*13,
        point_diagnostics=dict(tcp_world=[0,0,.9],up3_origin_world=[0,0,1],up3_com_world=[0,0,1]))
    row=align_packet(packet,track)
    assert row['base_gyro_2']==.3 and row['up3_gyro_0']==.4
    assert row['raw_q_0']==.1 and row['tcp_world_2']==.9
    assert row['tracking_age']==pytest.approx(.01)
    packet['epoch']=8
    assert align_packet(packet,track) is None
    packet['epoch']=7;packet['stamp_ns']=101_000_000_000
    assert align_packet(packet,track) is None


def test_delayed_tracking_pairs_with_retained_sensor_time_not_latest_packet():
    import nandu1_adaptive_telemetry as log
    assert hasattr(log,'Pairer'), 'Delayed tracking needs timestamp pairing buffer'
    pair=log.Pairer()
    packet=dict(stamp_ns=100_000_000_000,sim_time=1.,epoch=7,seq=1,
        base=dict(quat=[1,0,0,0],gyro=[0,0,0],acc=[0,0,9.81]),
        up3=dict(quat=[1,0,0,0],gyro=[0,0,0],acc=[0,0,9.81]),truth=[0.]*7,command_truth=[0.]*3,
        q=[0.]*13,dq=[0.]*13,tau=[0.]*13,
        point_diagnostics=dict(tcp_world=[0,0,.9],up3_origin_world=[0,0,1],up3_com_world=[0,0,1]))
    pair.packet(packet)
    pair.packet(dict(packet,stamp_ns=100_050_000_000,seq=2))
    track=[0.]*len(log.TRACKING_FIELDS);track[0]=100.;track[1]=7
    result=pair.tracking(track)
    assert len(result)==1 and result[0]['stamp']==100.
    assert pair.tracking(track)==[]


def test_state_events_persist_fault_reason_passive_and_session_identity():
    import nandu1_adaptive_telemetry as log
    assert hasattr(log, 'StateEvents'), 'FAULT/PASSIVE must survive offline without tracking'
    events=log.StateEvents()
    values={'epoch':'12','generation':'3','mode':'POLICY'}
    start=events.update(100.,0,'POLICY',values)
    assert start['mode']=='POLICY' and start['generation']==3
    assert events.update(100.01,0,'POLICY',values) is None
    fault=events.update(100.1,2,'Sensor state expired',dict(values,mode='FAULT'))
    assert fault['reason']=='Sensor state expired' and fault['epoch']==12
    passive=events.update(100.2,0,'PASSIVE',dict(values,mode='PASSIVE',generation='4'))
    assert passive['mode']=='PASSIVE' and passive['reason']==''
    resume=events.update(100.3,0,'POLICY',dict(values,generation='5'))
    assert resume['generation']==5 and resume['stamp']==100.3
