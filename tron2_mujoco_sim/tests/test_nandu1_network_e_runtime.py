import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tron2_rl_deploy_ros/tron2_controllers/scripts'))
TASK='Isaac-NANDU1-Blind-Flat-v1-3-1-2'


def test_network_e_uses_tcp_heading_processor_without_double_features():
    from nandu1_inference import CommandPipeline
    from nandu1_bundle import load_bundle
    m,_=load_bundle(ROOT/'deployment_bundles/nandu1_v1_3_1_2_model_27500')
    pipeline=CommandPipeline(m['task']);obs=np.zeros(49);obs[5]=obs[11]=-1
    applied=pipeline.step([0,0,.8850238269508085],obs,[0,0,.2])
    assert abs(applied[0]+.006)<1e-7
    raw=pipeline.actor_input(np.arange(7),obs,applied)
    assert len(raw)==59 and raw[3]==3 and raw[56]==applied[0]
    pipeline.reset()
    assert pipeline.step([0,0,.8850238269508085],obs,[0,0,0])[0]==0


def test_task_uses_separate_launch_and_tcp_command_topic():
    from nandu1_process import launch_file
    from nandu1_mouse_protocol import command_interface
    assert launch_file(TASK)=='nandu1_network_e_mujoco_policy.launch'
    assert command_interface(TASK)==('tcp','/nandu1/v1_3_1_2/command')
    assert launch_file('Isaac-NANDU1-Blind-Flat-v1-3-1')=='nandu1_mujoco_policy.launch'
    assert launch_file(TASK+'-6')=='nandu1_adaptive_mujoco_policy.launch'


def test_network_e_launch_uses_121_layout_not_adaptive_state():
    file=ROOT/'tron2_rl_deploy_ros/tron2_hw/launch/nandu1_network_e_mujoco_policy.launch'
    assert file.is_file(), 'Explicit original-E launch missing'
    xml=ET.parse(file).getroot();values={p.get('name'):p.get('value') for p in xml.findall('.//param')}
    assert values['tron2_controller/type']=='tron2_controller/NANDU1Controller'
    assert values['layout']=='network_e_tcp'


def test_tcp_logger_pairs_full_raw_state_and_rejects_adaptive_width():
    import nandu1_adaptive_telemetry as log
    assert hasattr(log,'tracking_fields'), 'Explicit tcp121 logger layout missing'
    fields=log.tracking_fields('network_e_tcp');assert len(fields)==121
    assert len(log.tracking_fields('adaptive'))==284
    packet=dict(stamp_ns=100_000_000_000,sim_time=1.,epoch=7,seq=1,
        base=dict(quat=[1,0,0,0],gyro=[.1,.2,.3],acc=[0,0,9.81]),
        up3=dict(quat=[1,0,0,0],gyro=[.4,.5,.6],acc=[0,0,9.81]),truth=[0.]*7,command_truth=[0.]*3,
        q=[.1]*13,dq=[.2]*13,tau=[.3]*13,
        point_diagnostics=dict(tcp_world=[0,0,.9],up3_origin_world=[0,0,1],up3_com_world=[0,0,1]))
    pair=log.Pairer('network_e_tcp');pair.packet(packet);pair.packet(dict(packet,seq=2,stamp_ns=100_010_000_000))
    track=[0.]*121;track[:2]=[100.,7];track[118:]=[.1,.2,.9]
    rows=pair.tracking(track)
    assert len(rows)==1 and rows[0]['raw_q_0']==.1 and rows[0]['external_intent_1']==.2
    assert rows[0]['base_gyro_2']==.3 and 'hidden_0' not in rows[0]
    assert pair.tracking(track)==[] and pair.tracking(track+[0.]*163)==[]
    track[0]=100.02;track[1]=8;assert pair.tracking(track)==[]
