#!/usr/bin/env python3
"""Owned original-E SDK+ROS acceptance, never a hardware control entry."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2];SIM=ROOT/'tron2_mujoco_sim'
sys.path.insert(0,str(SIM));sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_nandu1_adaptive_ros import preflight
from nandu1_bundle import load_bundle,sha256
from nandu1_network_e import TASK
from nandu1_process import _stop


def summarize(path,manifest,before):
    import numpy as np
    import nandu1_contract as c
    with path.open() as f:rows=[r for r in csv.DictReader(f) if float(r['tracking_stamp'])<before]
    if len(rows)<100:raise RuntimeError('Insufficient synchronized original-E samples')
    t=np.array([float(r['tracking_stamp']) for r in rows]);dt=np.diff(t)
    def array(prefix,n):return np.array([[float(r[f'{prefix}_{i}']) for i in range(n)] for r in rows])
    command,truth,encoded,label=array('command',3),array('command_truth',3),array('estimate',7),array('truth',7)
    q,qt,dq,tau,rawtau,action=(array(k,13) for k in ('q','q_target','dq','torque_actual','raw_tau','action'))
    if not np.isfinite(np.c_[t,command,truth,encoded,label,q,qt,dq,tau,rawtau,action]).all():raise RuntimeError('Nonfinite original-E telemetry')
    task=manifest['task'];ids=c.POSITION_IDS;lo=np.array(task['joint_position_lower']);hi=np.array(task['joint_position_upper'])
    report=dict(rows=len(rows),duration_s=float(t[-1]-t[0]),policy_telemetry_hz=float((len(t)-1)/(t[-1]-t[0])),
        period_quantiles_s=np.quantile(dt,[.5,.9,.99,1]).tolist(),
        max_alignment_age_s=max(abs(float(r['tracking_age'])) for r in rows),
        tracking_rmse_yaw_vx_height=np.sqrt(np.mean((command-truth)**2,0)).tolist(),
        estimator_rmse_base3_tcp3_height=np.sqrt(np.mean((encoded-label)**2,0)).tolist(),
        mean_tracking_error=(truth-command).mean(0).tolist(),
        position_targets_outside_limits=int(np.count_nonzero((qt[:,ids]<lo[ids]-1e-9)|(qt[:,ids]>hi[ids]+1e-9))),
        position_states_outside_005_tolerance=int(np.count_nonzero((q[:,ids]<lo[ids]-.05)|(q[:,ids]>hi[ids]+.05))),
        velocity_states_outside_limits=int(np.count_nonzero(abs(dq)>c.VELOCITY+1e-6)),
        measured_torque_outside_limits=int(np.count_nonzero(abs(tau)>c.EFFORT+1e-6)),
        raw_torque_outside_limits=int(np.count_nonzero(abs(rawtau)>c.EFFORT+1e-6)),
        position_target_projection_samples_inferred=int(np.count_nonzero(abs(qt[:,ids]-(c.DEFAULT_Q[ids]+.25*action[:,:11]))>1e-6)),
        velocity_projection_events=int(max(float(r['velocity_projection_events']) for r in rows)))
    requested=array('external_intent',3)
    edges=np.r_[0,np.nonzero(np.max(abs(np.diff(requested,axis=0)),axis=1)>1e-6)[0]+1,len(t)]
    report['plateaus']=[]
    for begin,end in zip(edges[:-1],edges[1:]):
        if t[end-1]-t[begin]<1:continue
        ix=np.arange(begin,end);steady=ix[t[ix]>t[end-1]-2]
        report['plateaus'].append(dict(start_s=float(t[begin]-t[0]),duration_s=float(t[end-1]-t[begin]),
            requested=requested[begin].tolist(),steady_bias=(truth[steady]-command[steady]).mean(0).tolist(),
            rmse=np.sqrt(np.mean((truth[ix]-command[ix])**2,0)).tolist()))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--robot-ip',default='127.0.0.1')
    a=p.parse_args();a.bundle=a.bundle.resolve();a.output=a.output.resolve();preflight(a.output,a.robot_ip)
    manifest,_=load_bundle(a.bundle,TASK)
    lock=(SIM/'.nandu1_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.output.mkdir(parents=True);container='nandu1-network-e-test-'+uuid.uuid4().hex;processes=[];streams=[]
    def start(name,command,env=None):
        f=(a.output/(name+'.log')).open('w');streams.append(f)
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env);processes.append(proc);return proc
    def inside(command):
        return ['docker','exec',container,'bash','-lc',
            'source /opt/ros/noetic/setup.bash && source /ws/devel/setup.bash && export PYTHONPATH='+shlex.quote(str(SIM))+':"${PYTHONPATH:-}" && '+command]
    def wait(proc,timeout):
        deadline=time.monotonic()+timeout
        while proc.poll() is None:
            if ros.poll() is not None or sim.poll() is not None:raise RuntimeError('Owned ROS/sim stopped during probe')
            if time.monotonic()>deadline:raise RuntimeError('Original-E probe timeout')
            time.sleep(.05)
        if proc.returncode:raise RuntimeError('Probe failed; inspect '+str(a.output))
    try:
        relay=start('relay',[sys.executable,str(SIM/'scripts/nandu1_test_imu_relay.py'),'--duration','170'])
        ros=start('ros',[str(ROOT/'tron2_rl_deploy_ros/scripts/nandu1_noetic.sh'),'run','roslaunch','tron2_hw','nandu1_network_e_mujoco_policy.launch',
            'task:='+TASK,'model_bundle:='+str(a.bundle),'telemetry_output:='+str(a.output/'tracking.csv')],dict(os.environ,NANDU1_CONTAINER_NAME=container))
        sim=start('simulator',[sys.executable,str(SIM/'simulator.py'),'--task',TASK,'--model-bundle',str(a.bundle),'--ip','127.0.0.1','--headless','--aux-port','15013','--duration','170'])
        deadline=time.monotonic()+25
        while not (a.output/'tracking.csv').exists() or sum(1 for _ in (a.output/'tracking.csv').open())<3:
            if time.monotonic()>deadline or sim.poll() is not None or ros.poll() is not None:raise RuntimeError('No original-E synchronized control telemetry')
            time.sleep(.1)
        print('Original-E SDK+ROS ready;60s zero intent then positive/negative vx/yaw/height',flush=True)
        command=start('commands',inside('rosrun tron2_controllers nandu1_command.py --task '+TASK+' --mode steps --height '+str(manifest['task']['initial_up3_height'])))
        wait(command,110);finished=time.time();report=summarize(a.output/'tracking.csv',manifest,finished)
        events=[json.loads(line) for line in (a.output/'tracking.events.jsonl').read_text().splitlines()]
        if any(e['mode']=='FAULT' and e['stamp']<finished for e in events):raise RuntimeError('Nominal original-E run faulted')
        print('Nominal complete;testing command timeout, dual IMU failures, reset and short POLICY reentry',flush=True)
        tests=ROOT/'tron2_rl_deploy_ros/tron2_controllers/test'
        for name,script in [('safety','nandu1_adaptive_ros_safety.py'),('lifecycle','nandu1_adaptive_ros_lifecycle.py')]:
            proc=start(name,inside('python3 '+shlex.quote(str(tests/script))+' --task '+TASK+' --output '+shlex.quote(str(a.output/(name+'.json')))))
            wait(proc,30)
        subprocess.run(inside('rostopic pub -1 /nandu1/set_mode std_msgs/String "data: PASSIVE"'),capture_output=True,check=True,timeout=10)
        events=[json.loads(line) for line in (a.output/'tracking.events.jsonl').read_text().splitlines()]
        sources=[SIM/name for name in ('nandu1_contract.py','nandu1_network_e.py','nandu1_bundle.py','nandu1_simulator.py')]
        controller=ROOT/'tron2_rl_deploy_ros/tron2_controllers'
        sources += [controller/name for name in ('src/NANDU1Controller.cpp','include/tron2_controllers/NANDU1Profiles.h',
            'include/tron2_controllers/NANDU1Onnx.h','scripts/nandu1_adaptive_telemetry.py')]
        hard=('position_targets_outside_limits','position_states_outside_005_tolerance','velocity_states_outside_limits',
            'measured_torque_outside_limits','raw_torque_outside_limits','velocity_projection_events')
        report.update(task=TASK,checkpoint=manifest['checkpoint'],nominal_finished_wall=finished,
            scope='localhost actual SDK+ROS+MuJoCo; not physical hardware',
            numerical_conformance=manifest['numerical_conformance'],
            deployment_source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sources},
            safety=json.loads((a.output/'safety.json').read_text()),lifecycle=json.loads((a.output/'lifecycle.json').read_text()),
            state_events=dict(count=len(events),fault_reasons=[e['reason'] for e in events if e['mode']=='FAULT'],
                passive_recorded=any(e['mode']=='PASSIVE' for e in events),
                fault_context_valid=all(e['epoch']>0 and e['generation']>0 for e in events if e['mode']=='FAULT')))
        report['correctness_acceptance']=(all(report[k]==0 for k in hard) and 45<report['policy_telemetry_hz']<55
            and report['period_quantiles_s'][-1]<.1 and len(report['plateaus'])>=9
            and all(c['passed'] for c in report['safety']['checks']) and report['lifecycle']['passed']
            and report['state_events']['passive_recorded'] and report['state_events']['fault_context_valid'])
        report['tracking_performance']='Measured separately; passing implementation gates does not certify precision or drift'
        write= a.output/'report.json';write.write_text(json.dumps(report,indent=2)+'\n')
        if not report['correctness_acceptance']:raise RuntimeError('Original-E acceptance gate failed')
        print(json.dumps(dict(output=str(a.output),passed=True,policy_hz=report['policy_telemetry_hz'],rmse=report['tracking_rmse_yaw_vx_height'])),flush=True)
    finally:
        for proc in processes:_stop(proc)
        subprocess.run(['docker','stop','--time','3',container],capture_output=True,timeout=8,check=False)
        for stream in streams:stream.close()
        lock.close()


if __name__=='__main__':main()
