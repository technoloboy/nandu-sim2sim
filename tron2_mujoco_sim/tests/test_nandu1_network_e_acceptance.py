import csv
import importlib.util
from pathlib import Path
import socket
import pytest

ROOT=Path(__file__).resolve().parents[2]


def load():
    path=ROOT/'tron2_mujoco_sim/scripts/check_nandu1_network_e_ros.py'
    assert path.is_file(), 'Original-E SDK acceptance entry missing'
    spec=importlib.util.spec_from_file_location('network_e_acceptance',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_network_e_acceptance_is_local_and_fresh_only(tmp_path):
    module=load()
    with pytest.raises(ValueError):module.preflight(tmp_path)
    with pytest.raises(ValueError):module.preflight(tmp_path/'new','192.168.1.1')
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        sock.bind(('127.0.0.1',0))
        with pytest.raises(RuntimeError):module.preflight(tmp_path/'new',ports=[(socket.SOCK_DGRAM,sock.getsockname()[1])])


def test_e_report_separates_estimation_from_tracking_error(tmp_path):
    module=load()
    from nandu1_bundle import load_bundle
    import nandu1_contract as c
    m,_=load_bundle(ROOT/'deployment_bundles/nandu1_v1_3_1_2_model_27500')
    row=dict(tracking_stamp=100.,tracking_age=.001,velocity_projection_events=0)
    for prefix,n in [('command',3),('command_truth',3),('estimate',7),('truth',7),
        ('q',13),('q_target',13),('dq',13),('action',13),('torque_actual',13),('raw_tau',13),('external_intent',3)]:
        row.update({f'{prefix}_{i}':0. for i in range(n)})
    for prefix,values in [('command',[.1,.2,.9]),('external_intent',[.1,.2,.9]),
        ('command_truth',[.08,.17,.86]),('estimate',[0,0,0,0,0,0,.865]),('truth',[0,0,0,0,0,0,.86]),
        ('q',c.DEFAULT_Q),('q_target',c.DEFAULT_Q)]:
        row.update({f'{prefix}_{i}':float(v) for i,v in enumerate(values)})
    path=tmp_path/'tracking.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
        for i in range(200):writer.writerow(dict(row,tracking_stamp=100+i*.02))
    report=module.summarize(path,m,104.)
    assert report['tracking_rmse_yaw_vx_height']==pytest.approx([.02,.03,.04])
    assert report['estimator_rmse_base3_tcp3_height'][-1]==pytest.approx(.005)
    assert report['policy_telemetry_hz']==pytest.approx(50.)
    assert report['position_targets_outside_limits']==0
