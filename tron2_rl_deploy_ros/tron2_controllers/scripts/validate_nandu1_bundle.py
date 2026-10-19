#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tron2_mujoco_sim'))
from nandu1_bundle import load_bundle,require_controller_runtime

def ros_parameters(manifest):
    # ROS1 XML-RPC cannot serialize JSON null. Only these validated physical
    # parameters are consumed by the controller; keep the bundle unchanged.
    keys=('initial_up3_height','joint_position_lower','joint_position_upper','task_id','runtime_profile','evaluation_stage',
          'command_min','command_max','command_rate')
    if manifest['task'].get('runtime_profile','').startswith('nandu1_adaptive_') or manifest['task'].get('runtime_profile')=='nandu1_network_v1_3_1_2_e_r1':
        keys=keys+('checkpoint_schema',)
    return {'task': {key:manifest['task'][key] for key in keys if key in manifest['task']}}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--task',default=None)
    p.add_argument('--robot-ip',default='127.0.0.1')
    p.add_argument('--emit-urdf',action='store_true')
    p.add_argument('--emit-config',action='store_true')
    p.add_argument('--emit-ros-params',action='store_true')
    a=p.parse_args()
    if a.robot_ip!='127.0.0.1': raise ValueError('Only localhost sim2sim is supported')
    m,_=load_bundle(a.bundle,a.task)
    require_controller_runtime(m)
    if a.emit_urdf:
        import xml.etree.ElementTree as ET
        tree=ET.parse(a.bundle/'assets/robot.urdf')
        for mesh in tree.findall('.//mesh'):
            mesh.set('filename',str((a.bundle/'assets'/mesh.get('filename')).resolve()))
        print(ET.tostring(tree.getroot(),encoding='unicode'))
    elif a.emit_ros_params:
        print(json.dumps(ros_parameters(m)))
    elif a.emit_config:
        print(json.dumps({'Tron2Cfg':{'init_state':{'joint_names':m['task']['hardware_joint_names']}}}))
    else: print('validated')
