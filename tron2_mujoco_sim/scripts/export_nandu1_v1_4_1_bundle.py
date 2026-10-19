#!/usr/bin/env python3
"""Seal a strict v1.4.1 export and physical assets for localhost SDK+ROS.

Requires a Python environment with Torch, ONNX and MuJoCo. No baseline bundle
or trained v1.4 weights are used.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch

SIM=Path(__file__).resolve().parents[1]
LAB=SIM.parent/'tron2_rl_lab'
sys.path[:0]=[str(SIM),str(LAB/'scripts/rsl_rl')]
import nandu1_contract as c
from nandu1_bundle import load_bundle,sha256
from nandu1_v141 import PROCESSOR_CONTRACT,COMMAND_ORDER,STAGE_LIMITS,STAGE_HEIGHT_RANGES,HEIGHT_RATE_LIMITS

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def build_bundle(checkpoint,output,task=c.V141_TASK):
    import mujoco
    import onnx
    if task not in (c.V141_TASK,c.V141_PLAY):raise ValueError('v1.4.1 task required')
    checkpoint,output=Path(checkpoint),Path(output)
    from export_nandu1_deployment import export_v141
    export_v141(checkpoint,output)
    for name in ('encoder','policy','actor_pipeline'):
        path=output/(name+'.onnx');graph=onnx.load(str(path))
        for tensor in list(graph.graph.input)+list(graph.graph.output):
            dimension=tensor.type.tensor_type.shape.dim[0];dimension.ClearField('dim_param');dimension.dim_value=1
        onnx.checker.check_model(graph);onnx.save(graph,str(path))
    from nandu1_asset import build_asset,audit_asset
    source=LAB/'exts/bipedal_locomotion/bipedal_locomotion/assets/usd/NANDU1'
    xml=build_asset(source,output/'assets');audit=audit_asset(source,xml)
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in c.JOINTS]
    data.qpos[:7]=[0,0,.65337,1,0,0,0];data.qpos[model.jnt_qposadr[ids]]=c.DEFAULT_Q
    mujoco.mj_forward(model,data);height=float(data.xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'up3'),2])
    metadata=json.loads((output/'deployment_metadata.json').read_text());stage=metadata['curriculum_stage']
    yl,vl=STAGE_LIMITS[stage];lo,hi=STAGE_HEIGHT_RANGES[stage]
    t=dict(task_id=c.V141_TASK,play_alias=c.V141_PLAY,checkpoint_schema=c.V141_SCHEMA,robot_variant='NANDU1',
        runtime_profile='nandu1_height_v1_4_1',evaluation_stage=stage,initial_up3_height=height,
        ros_runtime_supported=True,
        velocity_reference='body_link_origin',actor_velocity_frame='heading',heading_conversion_gravity='policy_observation[3:6]',
        external_command_order=['relative_heading_rate','up3_vx_base_heading','up3_height_world_z'],command_order=COMMAND_ORDER,
        encoder_order=['base_link_velocity_base_body_3','up3_link_velocity_base_body_3','up3_height_world'],
        command_min=[-yl,-vl,lo],command_max=[yl,vl,hi],height_rate_limit=HEIGHT_RATE_LIMITS[stage],command_processor=PROCESSOR_CONTRACT,
        hardware_joint_names=c.JOINTS,action_joint_names=[c.JOINTS[i] for i in c.ACTION_TO_HARDWARE],
        default_joint_positions=c.DEFAULT_Q.tolist(),root_position=[0,0,.65337],
        history_order='oldest_to_newest',history_length=5,history_first_frame='repeat',policy_dt=.02,
        observation_order=['base_gyro*.25','base_gravity','up3_gyro*.25','up3_gravity','position_relative_11','velocity_13*.05','last_action_13'],
        position_scale=.25,wheel_scale=1,action_clipping=None,
        joint_position_lower=model.jnt_range[ids,0].tolist(),joint_position_upper=model.jnt_range[ids,1].tolist())
    for key,value in [('kp',c.KP),('kd',c.KD),('effort',c.EFFORT),('velocity_limits',c.VELOCITY),('armature',c.ARMATURE)]:t[key]=value.tolist()
    for name in ('nandu1_v141.py','nandu1_v14.py','nandu1_contract.py'):
        shutil.copyfile(SIM/name,output/name)
    shutil.copytree(SIM/'deploy_reference',output/'deploy_reference')
    sys.path.insert(0,str(output))
    training=module('sealed_v141_processor',output/'command_processor.py')
    rows=[]
    for current in range(5):
        ref=training.CommandProcessor(1,'cpu');ref.reset(None,height)
        for i in range(90):
            reset=i in (0,45);reset_height=height if i<45 else .8
            if reset:ref.reset(None,reset_height)
            dt=[1e-6,.018,.02,.1][i%4]
            g=np.array([[0,0,-1],[.45,.1,-.88],[-1,0,0],[0,0,0]][i%4],np.float32)
            gyro=np.array([.12,-.09,.4],np.float32)
            ext=np.array([2 if i<30 else (-.3 if i>60 else 0),.1 if i%3 else -.1,1.4 if i<60 else .4],np.float32)
            ref.set_external(torch.tensor(ext),current);ref.step(dt,torch.tensor(g[None]),torch.tensor(gyro[None]),current)
            rows.append(np.r_[current,reset,reset_height,dt,g,gyro,ext,ref.commands[0].numpy(),ref.trend[0].numpy(),ref.heading.item(),ref.heading_target.item()])
    np.savetxt(output/'runtime_conformance.txt',np.array(rows),fmt='%.9g',header=str(len(rows)),comments='')
    vectors=np.load(output/'conformance.npz')
    with (output/'conformance.txt').open('w') as stream:
        for inputs,expected in ((vectors['history'],vectors['encoder_output_body']),
                                (vectors['actor_input'],vectors['actions'])):
            stream.write(f'{len(inputs)} {inputs.shape[1]} {expected.shape[1]}\n')
            for x,y in zip(inputs,expected):stream.write(' '.join(format(float(v),'.9g') for v in np.r_[x,y])+'\n')
    with (output/'runtime_conformance.txt').open('a') as stream:
        stream.write(str(len(vectors['actions']))+'\n')
        for enc,obs,cmd,expected in zip(vectors['encoder_output_body'],vectors['observation'],vectors['commands'],vectors['actor_input']):
            stream.write(' '.join(format(float(x),'.9g') for x in np.r_[enc,obs,cmd,expected])+'\n')
    metadata.update(mujoco_runtime_supported=True,mujoco_runtime_scope='localhost_sdk_ros_or_direct_probe',ros_runtime_supported=True,
                    initial_height_default=height)
    (output/'deployment_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    files={str(path.relative_to(output)):sha256(path) for path in sorted(output.rglob('*')) if path.is_file() and '__pycache__' not in path.parts}
    manifest=dict(schema=c.BUNDLE_SCHEMA,task=t,dimensions=dict(observation=49,history=245,encoder=7,command=6,policy_input=62,action=13),
        checkpoint=dict(path=str(checkpoint.resolve()),sha256=sha256(checkpoint),iteration=saved['iter']),
        source=dict(git_commit=subprocess.check_output(['git','-C',str(LAB),'rev-parse','HEAD'],text=True).strip(),
                    exporter_sha256=sha256(__file__),training_exporter_sha256=sha256(LAB/'scripts/rsl_rl/export_nandu1_deployment.py'),
                    deployment_sources={name:sha256(SIM/name) for name in ('nandu1_v141.py','nandu1_v14.py','nandu1_contract.py','nandu1_bundle.py','nandu1_asset.py')},
                    urdf_sha256=sha256(source/'urdf/robot.urdf')),
        physics=dict(timestep=.001,internal_collisions=False,joint_velocity_constraint='post_step_qvel_projection_if_exceeded',
                     equivalence_note='MuJoCo velocity projection approximates PhysX maxJointVelocity'),asset_audit=audit,
        onnx_max_abs_error=metadata['onnx_max_abs_error'],files=files)
    (output/'manifest.yaml').write_text(json.dumps(manifest,indent=2)+'\n')
    sums={**files,'manifest.yaml':sha256(output/'manifest.yaml')}
    (output/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in sums.items()))
    load_bundle(output,task)
    return dict(bundle=str(output),stage=stage,ros_runtime_supported=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=(c.V141_TASK,c.V141_PLAY),required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build_bundle(a.checkpoint,a.output,a.task),indent=2))
if __name__=='__main__':main()
