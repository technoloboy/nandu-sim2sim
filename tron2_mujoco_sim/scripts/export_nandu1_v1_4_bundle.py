#!/usr/bin/env python3
"""Seal an explicit v1.4 checkpoint, training export and USD-derived assets."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import onnx
import onnxruntime as ort
import mujoco

SIM=Path(__file__).resolve().parents[1]
ROOT=SIM.parent
sys.path.insert(0,str(SIM))
import nandu1_contract as c
from nandu1_bundle import load_bundle,sha256
from nandu1_asset import build_asset,audit_asset
from nandu1_v14 import PROCESSOR_CONTRACT,STAGE_LIMITS,STAGE_HEIGHT_RANGES,HEIGHT_RATE_LIMITS

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',required=True,choices=[c.V14_TASK,c.V14_PLAY])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--provided-export',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();lab=ROOT/'tron2_rl_lab'
    exporter=lab/'scripts/rsl_rl/export_nandu1_deployment.py'
    module('deployment_export',exporter).export_v14(a.checkpoint,a.output)
    # ROS Network requires a fixed batch of one; do not alter source exports.
    for name in ['encoder','policy','actor_pipeline']:
        path=a.output/(name+'.onnx');graph=onnx.load(str(path))
        for tensor in list(graph.graph.input)+list(graph.graph.output):
            dim=tensor.type.tensor_type.shape.dim[0];dim.ClearField('dim_param');dim.dim_value=1
        onnx.checker.check_model(graph);onnx.save(graph,str(path))
    vectors=np.load(a.output/'conformance.npz')
    checks={}
    for name,inputs,expected in [('encoder',[vectors['history']],vectors['encoder_output_body']),
                                ('policy',[vectors['actor_input']],vectors['actions']),
                                ('actor_pipeline',[vectors['history'],vectors['observation'],vectors['commands']],vectors['actions'])]:
        for source in [a.output,a.provided_export]:
            path=source/(name+'.onnx');session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
            result=[]
            for row in range(len(expected)):
                feed={}
                for info,data in zip(session.get_inputs(),inputs):
                    feed[info.name]=data[row] if len(info.shape)==1 else data[row:row+1]
                result.append(session.run(None,feed)[0].reshape(-1))
            result=np.array(result);np.testing.assert_allclose(result,expected,atol=1e-5,rtol=1e-4)
            checks[str(path)]=float(np.max(np.abs(result-expected)))
    with (a.output/'conformance.txt').open('w') as stream:
        for inp,out in [(vectors['history'],vectors['encoder_output_body']),(vectors['actor_input'],vectors['actions'])]:
            stream.write(f'{len(inp)} {inp.shape[1]} {out.shape[1]}\n')
            for x,y in zip(inp,out):stream.write(' '.join(format(float(v),'.9g') for v in np.r_[x,y])+'\n')
    source=lab/'exts/bipedal_locomotion/bipedal_locomotion/assets/usd/NANDU1'
    xml=build_asset(source,a.output/'assets');audit=audit_asset(source,xml)
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in c.JOINTS]
    data.qpos[:7]=[0,0,.65337,1,0,0,0];data.qpos[model.jnt_qposadr[ids]]=c.DEFAULT_Q
    mujoco.mj_forward(model,data);height=float(data.xpos[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'up3'),2])
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    stage=int(checkpoint['environment_state_dict']['nandu1_height_v1_4']['stage'])
    yl,vl=STAGE_LIMITS[stage];lo,hi=STAGE_HEIGHT_RANGES[stage]
    # Reuse the shared physical contract, then replace every policy-specific field.
    baseline,_=load_bundle(ROOT/'deployment_bundles/nandu1_v1_2_model_40000')
    task=baseline['task'].copy();task.pop('command_rate')
    task.update(task_id=c.V14_TASK,play_alias=c.V14_PLAY,checkpoint_schema=c.V14_SCHEMA,
        runtime_profile='nandu1_height_v1_4',evaluation_stage=stage,initial_up3_height=height,
        velocity_reference='body_link_origin',actor_velocity_frame='heading',heading_conversion_gravity='policy_observation[3:6]',
        external_command_order=['relative_heading_rate','up3_vx_base_heading','up3_height_world_z'],
        command_order=['base_yaw_rate_world_z','up3_vx_base_heading','up3_height_world_z','yaw_trend','vx_trend','height_trend'],
        encoder_order=['base_link_velocity_base_body_3','up3_link_velocity_base_body_3','up3_height_world'],
        command_min=[-yl,-vl,lo],command_max=[yl,vl,hi],height_rate_limit=HEIGHT_RATE_LIMITS[stage],
        command_processor=PROCESSOR_CONTRACT,joint_position_lower=model.jnt_range[ids,0].tolist(),
        joint_position_upper=model.jnt_range[ids,1].tolist())
    training_math_path=lab/'exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/NANDU1/nandu1_v1_4_math.py'
    training=module('training_processor',training_math_path);ref=training.CommandProcessor(1,'cpu');ref.reset(None,height)
    with (a.output/'runtime_conformance.txt').open('w') as stream:
        stream.write(f'256 {stage} {height:.9g}\n')
        for i in range(256):
            reset=int(i==128);reset_height=.8
            if reset:ref.reset(None,reset_height)
            dt=.02 if i%3 else .018
            g=np.array([.45,.10,-.88741195],np.float32);gyro=np.array([.12,-.09,.15 if i<90 else .4],np.float32)
            ext=np.array([.3 if i<40 else (2 if i>=190 else 0),.1 if i%30<15 else -.1,1.32 if i<128 else .7],np.float32)
            ref.set_external(torch.tensor(ext),stage);ref.step(dt,torch.tensor(g[None]),torch.tensor(gyro[None]),stage)
            values=np.r_[reset,reset_height,dt,g,gyro,ext,ref.commands[0].numpy(),ref.trend[0].numpy(),float(ref.heading[0]),float(ref.heading_target[0])]
            stream.write(' '.join(format(float(v),'.9g') for v in values)+'\n')
        stream.write(f'{len(vectors["history"])}\n')
        for enc,obs,cmd,expected in zip(vectors['encoder_output_body'],vectors['observation'],vectors['commands'],vectors['actor_input']):
            stream.write(' '.join(format(float(v),'.9g') for v in np.r_[enc,obs,cmd,expected])+'\n')
    metadata=json.loads((a.output/'deployment_metadata.json').read_text());metadata['mujoco_runtime_supported']=True
    metadata['initial_height_default']=height;metadata['provided_export_errors']=checks
    (a.output/'deployment_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    files={str(x.relative_to(a.output)):sha256(x) for x in sorted(a.output.rglob('*')) if x.is_file()}
    manifest={'schema':c.BUNDLE_SCHEMA,'task':task,'dimensions':{'observation':49,'history':245,'encoder':7,'command':6,'policy_input':62,'action':13},
        'checkpoint':{'path':str(a.checkpoint.resolve()),'sha256':sha256(a.checkpoint),'iteration':checkpoint['iter']},
        'source':{'git_commit':subprocess.check_output(['git','-C',str(lab),'rev-parse','HEAD'],text=True).strip(),
            'exporter_sha256':sha256(__file__),'training_exporter_sha256':sha256(exporter),'training_processor_sha256':sha256(training_math_path),
            'urdf_sha256':sha256(source/'urdf/robot.urdf'),
            'usd_files':{str(x.relative_to(source)):sha256(x) for x in sorted((source/'usd').rglob('*.usd'))},
            'provided_exports':{x.name:sha256(x) for x in sorted(a.provided_export.iterdir()) if x.is_file()},
            'deployment_sources':{name:sha256(SIM/name) for name in ['nandu1_contract.py','nandu1_v14.py','nandu1_asset.py','nandu1_bundle.py']}},
        'physics':baseline['physics'],'asset_audit':audit,'onnx_max_abs_error':checks,'files':files}
    (a.output/'manifest.yaml').write_text(json.dumps(manifest,indent=2)+'\n')
    files['manifest.yaml']=sha256(a.output/'manifest.yaml')
    (a.output/'SHA256SUMS').write_text(''.join(f'{digest}  {path}\n' for path,digest in files.items()))
    load_bundle(a.output,a.task)
    print(json.dumps({'bundle':str(a.output),'stage':stage,'initial_height':height,'validation':checks},indent=2))

if __name__=='__main__':main()
