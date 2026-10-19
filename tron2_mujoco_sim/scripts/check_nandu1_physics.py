#!/usr/bin/env python3
"""Local physics/network conformance probe, not a replacement ROS controller."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import mujoco
import onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nandu1_contract as c
from nandu1_simulator import Nandu1Physics

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--duration',type=float,default=100)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    physics=Nandu1Physics(a.bundle);history=c.History();last=np.zeros(13)
    v14=physics.manifest['task']['task_id'] in (c.V14_TASK,c.V141_TASK)
    from nandu1_inference import CommandPipeline
    sessions={n:ort.InferenceSession(str(a.bundle/(n+'.onnx')),providers=['CPUExecutionProvider']) for n in ['encoder','policy']}
    run=lambda name,x:sessions[name].run(None,{sessions[name].get_inputs()[0].name:x.astype(np.float32)[None]})[0][0]
    height=physics.manifest['task']['initial_up3_height'];cmd=np.array([0.,0.,height]);qtarget=c.DEFAULT_Q.copy();dtarget=np.zeros(13)
    processor=CommandPipeline(physics.manifest['task'])
    segments=[(60,[0,0,height]),(5,[0,.1,height]),(5,[0,-.1,height]),(5,[.2,0,height]),(5,[-.2,0,height]),
              (5,[0,0,height+.05]),(5,[0,0,height-.05]),(5,[.1,.05,height]),(5,[0,0,height])]
    records=[];fault='';saturations=0;speed_projections=0
    with (a.output/'physics_tracking.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['sim_time']+[f'{name}_{i}' for name,count in [('command',3),('actual',3),('estimate',7),('truth',7)] for i in range(count)])
        for step in range(round(a.duration/.001)):
            if step%20==0:
                elapsed=physics.data.time;desired=np.array([0,0,height],float)
                for duration,value in segments:
                    if elapsed<duration:desired=np.array(value);break
                    elapsed-=duration
                b,u=physics.imu('base_imu'),physics.imu('up3_imu');q,dq,_=physics.state()
                obs=c.observe(b['quat'],b['gyro'],u['quat'],u['gyro'],q,dq,last)
                estimated=run('encoder',history.append(obs))
                features=processor.step(desired,obs,b['gyro'],.02);cmd=features[:3]
                policy_input=processor.actor_input(estimated,obs,features)
                last=run('policy',policy_input)
                qtarget,dtarget=c.decode(last)
                lo=physics.model.jnt_range[physics.jids,0];hi=physics.model.jnt_range[physics.jids,1]
                qtarget=c.clip_position_targets(qtarget,lo,hi)
                packet=physics.packet(step)
                row=np.r_[physics.data.time,cmd,packet['command_truth'],estimated,packet['truth']]
                if not np.isfinite(row).all():fault='non_finite';break
                records.append(row);writer.writerow(row)
            if not v14 or step%2==0:
                raw=physics.apply(qtarget,dtarget,c.KP,c.KD,np.zeros(13))
                saturations+=int(np.count_nonzero(np.abs(raw)>c.EFFORT))
            speed_projections+=physics.step()
    data=np.asarray(records)
    results={'kind':'local physics and ONNX probe; ROS transport not tested','samples':len(data),'sim_duration':float(physics.data.time),
             'position_target_mode':'clip_to_inclusive_physical_limits',
             'task_id':physics.manifest['task']['task_id'],'velocity_reference':physics.manifest['task'].get('velocity_reference','body_com'),
             'fault':fault,'torque_saturation_events':saturations,'velocity_projection_events':speed_projections,
             'collision_approximation':physics.manifest['asset_audit']['collision_note']}
    if len(data):
        results['tracking_rmse']=np.sqrt(np.mean((data[:,1:4]-data[:,4:7])**2,axis=0)).tolist()
        results['estimator_rmse']=np.sqrt(np.mean((data[:,7:14]-data[:,14:21])**2,axis=0)).tolist()
        results['height_range_m']=[float(data[:,6].min()),float(data[:,6].max())]
        results['segments']=[]
        start=0;previous=None
        for duration,value in segments:
            window=data[(data[:,0]>=start+min(1,duration*.2))&(data[:,0]<start+duration)]
            if len(window):
                error=window[:,4:7]-window[:,1:4]
                response=[]
                full=data[(data[:,0]>=start)&(data[:,0]<start+duration)]
                baseline=data[(data[:,0]>=start-1)&(data[:,0]<start)]
                if previous is not None and len(baseline) and len(full):
                    for axis,delta in enumerate(np.asarray(value)-previous):
                        if abs(delta)<1e-8:continue
                        change=(full[:,4+axis]-np.mean(baseline[:,4+axis]))*np.sign(delta)
                        delays={}
                        for fraction in [.5,.9]:
                            passed=change>=abs(delta)*fraction
                            # Require 5 consecutive policy samples (100 ms).
                            hits=np.flatnonzero(np.convolve(passed.astype(int),np.ones(5,dtype=int),'valid')==5) if len(passed)>=5 else []
                            delays[str(fraction)]=float(full[hits[0],0]-start) if len(hits) else None
                        response.append({'axis':axis,'step':float(delta),'threshold_delays_s':delays})
                results['segments'].append({'start':start,'duration':duration,'command':value,
                    'tracking_rmse':np.sqrt(np.mean(error**2,axis=0)).tolist(),
                    'steady_bias':np.mean(error[-min(50,len(window)):],axis=0).tolist(),
                    'estimator_rmse':np.sqrt(np.mean((window[:,7:14]-window[:,14:21])**2,axis=0)).tolist(),
                    'response_from_step_including_command_slew':response})
            start+=duration;previous=np.asarray(value)
    (a.output/'report.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
    if fault:raise SystemExit(1)

if __name__=='__main__':main()
