#!/usr/bin/env python3
"""Owned, finite SDK+ROS E/r2 acceptance. Physical robot IPs are forbidden."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2];SIM=ROOT/'tron2_mujoco_sim'
sys.path.insert(0,str(SIM))
from nandu1_adaptive_contract import TASK,adaptive_tasks
from nandu1_adaptive_balance_contract import TASK as BALANCE_TASK
from nandu1_adaptive_bundle import load_adaptive_bundle
from nandu1_adaptive_bundle import sha256
from nandu1_process import _stop


def preflight(output, robot_ip='127.0.0.1', ports=None):
    if robot_ip!='127.0.0.1':raise ValueError('Only localhost simulation permitted')
    if Path(output).exists():raise ValueError('Choose a fresh output directory')
    ports=ports if ports is not None else [(socket.SOCK_STREAM,11311)]+[(socket.SOCK_DGRAM,p) for p in (15012,15013,15015)]
    for kind,port in ports:
        with socket.socket(socket.AF_INET,kind) as s:
            if kind==socket.SOCK_STREAM:
                s.settimeout(.2)
                if s.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Port occupied: '+str(port))
            else:
                try:s.bind(('127.0.0.1',port))
                except OSError:raise RuntimeError('Port occupied: '+str(port)) from None


def summarize(path,manifest,before):
    import numpy as np
    with path.open() as f:rows=[r for r in csv.DictReader(f) if float(r['tracking_stamp'])<before]
    if len(rows)<100:raise RuntimeError('Insufficient synchronized policy samples')
    t=np.array([float(r['tracking_stamp']) for r in rows]);period=np.diff(t)
    def matrix(prefix,n):return np.array([[float(r[f'{prefix}_{i}']) for i in range(n)] for r in rows])
    command,truth,estimate,labels=matrix('command',3),matrix('command_truth',3),matrix('estimate',7),matrix('truth',7)
    q,qt,tau=matrix('q',13),matrix('q_target',13),matrix('torque_limited',13)
    from nandu1_contract import VELOCITY
    if not np.isfinite(np.c_[t,command,truth,estimate,labels,q,qt,tau]).all():raise RuntimeError('Nonfinite telemetry')
    task=manifest['task'];ids=[0,1,2,3,5,6,7,8,10,11,12]
    lower,upper=np.asarray(task['joint_position_lower']),np.asarray(task['joint_position_upper'])
    report=dict(rows=len(rows),duration_s=float(t[-1]-t[0]),policy_telemetry_hz=float((len(rows)-1)/(t[-1]-t[0])),
        period_quantiles_s=np.quantile(period,[.5,.9,.99,1]).tolist(),
        reported_policy_hz=float(1/np.mean([float(r['policy_period']) for r in rows])),
        max_alignment_age_s=max(abs(float(r['tracking_age'])) for r in rows),
        tracking_rmse_yaw_vx_height=np.sqrt(np.mean((command-truth)**2,axis=0)).tolist(),
        estimator_rmse_base3_tcp3_height=np.sqrt(np.mean((estimate-labels)**2,axis=0)).tolist(),
        mean_tracking_error=(truth-command).mean(axis=0).tolist(),
        position_targets_outside_limits=int(np.count_nonzero((qt[:,ids]<lower[ids]-1e-9)|(qt[:,ids]>upper[ids]+1e-9))),
        position_states_outside_005_tolerance=int(np.count_nonzero((q[:,ids]<lower[ids]-.05)|(q[:,ids]>upper[ids]+.05))),
        torque_outside_limits=int(np.count_nonzero(np.abs(tau)>np.asarray(task['effort'])+1e-6)),
        velocity_states_outside_limits=int(np.count_nonzero(np.abs(matrix('dq',13))>VELOCITY+1e-6)),
        target_projection_events=int(sum(float(r['target_projection_count']) for r in rows)),
        torque_saturated_samples=int(np.count_nonzero(matrix('torque_saturated',13))),
        velocity_projection_events=int(max(float(r['velocity_projection_events']) for r in rows)))
    requested=matrix('requested',3);changes=np.r_[0,np.nonzero(np.max(np.abs(np.diff(requested,axis=0)),axis=1)>1e-6)[0]+1,len(rows)]
    report['plateaus']=[]
    for start,end in zip(changes[:-1],changes[1:]):
        if t[end-1]-t[start]<1:continue
        subset=np.arange(start,end);steady=subset[t[subset]>t[end-1]-2]
        report['plateaus'].append(dict(start_s=float(t[start]-t[0]),duration_s=float(t[end-1]-t[start]),
            requested=requested[start].tolist(),steady_bias=(truth[steady]-command[steady]).mean(0).tolist(),
            rmse=np.sqrt(np.mean((truth[subset]-command[subset])**2,0)).tolist()))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',choices=adaptive_tasks(),default=TASK)
    p.add_argument('--duration',type=float,default=130);p.add_argument('--robot-ip',default='127.0.0.1')
    a=p.parse_args();output=a.output.resolve();preflight(output,a.robot_ip)
    if not math.isfinite(a.duration) or a.duration<120:raise ValueError('Full acceptance requires >=120 seconds')
    manifest,_=load_adaptive_bundle(a.bundle,a.task)
    lock=(SIM/'.nandu1_launch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output.mkdir(parents=True);processes=[];streams=[];container='nandu1-adaptive-test-'+uuid.uuid4().hex
    helper=ROOT/'tron2_rl_deploy_ros/scripts/nandu1_noetic.sh'
    def start(name,cmd,env=None):
        f=(output/(name+'.log')).open('w');streams.append(f)
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env);processes.append(proc);return proc
    def inside(command):
        prefix='source /opt/ros/noetic/setup.bash && source /ws/devel/setup.bash && export ROS_MASTER_URI=http://127.0.0.1:11311 ROS_IP=127.0.0.1 PYTHONPATH='+shlex.quote(str(SIM))+':"${PYTHONPATH:-}" && '
        return ['docker','exec',container,'bash','-lc',prefix+command]
    def wait_proc(proc,timeout):
        deadline=time.monotonic()+timeout
        while proc.poll() is None:
            if ros.poll() is not None or sim.poll() is not None:raise RuntimeError('ROS/simulation ended during probe')
            if time.monotonic()>deadline:raise RuntimeError('Probe timeout')
            time.sleep(.1)
        if proc.returncode:raise RuntimeError('Probe failed; inspect '+str(output))
    result=None
    try:
        relay=start('relay',[sys.executable,str(SIM/'scripts/nandu1_test_imu_relay.py'),'--duration',str(a.duration+40)])
        from nandu1_process import launch_file
        ros=start('ros',[str(helper),'run','roslaunch','tron2_hw',launch_file(a.task),
            'task:='+a.task,'model_bundle:='+str(a.bundle.resolve()),'telemetry_output:='+str(output/'tracking.csv')],
            env=dict(os.environ,NANDU1_CONTAINER_NAME=container))
        sim=start('simulator',[sys.executable,str(SIM/'simulator.py'),'--task',a.task,'--model-bundle',str(a.bundle.resolve()),
            '--ip','127.0.0.1','--headless','--aux-port','15013','--duration',str(a.duration+40)])
        deadline=time.monotonic()+25
        while True:
            ready=False
            if (output/'tracking.csv').exists():
                with (output/'tracking.csv').open() as f:ready=all(f.readline().strip() for _ in range(3))
            if ready:break
            if time.monotonic()>deadline or sim.poll() is not None or ros.poll() is not None:raise RuntimeError('No synchronized adaptive policy telemetry')
            time.sleep(.1)
        print('SDK+ROS ready; 60s zero + +/-vx/yaw/height + combination steps',flush=True)
        height=manifest['task']['initial_up3_height']
        commands=start('commands',inside('rosrun tron2_controllers nandu1_command.py --task '+a.task+' --mode steps --height '+str(height)))
        wait_proc(commands,110)
        main_finished=time.time()
        if 'NANDU1 adaptive FAULT:' in (output/'ros.log').read_text():raise RuntimeError('Unexpected FAULT during nominal step sequence')
        report=summarize(output/'tracking.csv',manifest,main_finished)
        print('Nominal steps complete; testing expiry, fault latch, reset and dual-IMU transport',flush=True)
        safety_script=ROOT/'tron2_rl_deploy_ros/tron2_controllers/test/nandu1_adaptive_ros_safety.py'
        safety=start('safety',inside('python3 '+shlex.quote(str(safety_script))+' --task '+a.task+' --output '+shlex.quote(str(output/'safety.json'))))
        wait_proc(safety,25)
        lifecycle_script=ROOT/'tron2_rl_deploy_ros/tron2_controllers/test/nandu1_adaptive_ros_lifecycle.py'
        lifecycle=start('lifecycle',inside('python3 '+shlex.quote(str(lifecycle_script))+' --task '+a.task+' --output '+shlex.quote(str(output/'lifecycle.json'))))
        wait_proc(lifecycle,25)
        # Disarm before stopping physics, so planned shutdown is not counted
        # as an unexpected loss of the simulator's sensor stream.
        subprocess.run(inside('rostopic pub -1 /nandu1/set_mode std_msgs/String "data: PASSIVE"'),check=True,capture_output=True,timeout=10)
        report.update(task=a.task,checkpoint=manifest['checkpoint'],scope='localhost MuJoCo + actual SDK + ROS, not physical robot',
            safety=json.loads((output/'safety.json').read_text()),nominal_finished_wall=main_finished,
            lifecycle=json.loads((output/'lifecycle.json').read_text()),
            numerical_conformance=json.loads((a.bundle/'runtime_conformance.json').read_text()))
        source_files=[SIM/name for name in ('nandu1_adaptive_contract.py','nandu1_adaptive_bundle.py','nandu1_simulator.py')]
        if a.task==BALANCE_TASK:source_files.append(SIM/'nandu1_adaptive_balance_contract.py')
        from nandu1_ablation_contract import PROFILES
        if a.task in PROFILES:source_files.append(SIM/'nandu1_ablation_contract.py')
        controller=ROOT/'tron2_rl_deploy_ros/tron2_controllers'
        source_files += [controller/'include/tron2_controllers'/('NANDU1Adaptive'+name+'.h')
            for name in ('Onnx','Command','Memory','Runtime','Controller')]
        source_files += [controller/'src/NANDU1AdaptiveController.cpp',controller/'scripts/nandu1_adaptive_telemetry.py',
                         controller/'include/tron2_controllers/NANDU1AdaptiveProfiles.h']
        report['deployment_source_sha256']={str(path.relative_to(ROOT)):sha256(path) for path in source_files}
        events=[json.loads(line) for line in (output/'tracking.events.jsonl').read_text().splitlines()]
        report['state_events']=dict(count=len(events),fault_reasons=[e['reason'] for e in events if e['mode']=='FAULT'],
            passive_recorded=any(e['mode']=='PASSIVE' for e in events),
            fault_context_valid=all(e['epoch']>0 and e['generation']>0 for e in events if e['mode']=='FAULT'))
        hard=('position_targets_outside_limits','position_states_outside_005_tolerance','torque_outside_limits','velocity_states_outside_limits','velocity_projection_events')
        report['correctness_acceptance']=(all(report[k]==0 for k in hard) and 45<=report['policy_telemetry_hz']<=55 and report['period_quantiles_s'][-1]<.1 and len(report['plateaus'])>=9
            and all(c['passed'] for c in report['safety']['checks']) and report['lifecycle']['passed']
            and report['state_events']['passive_recorded'] and report['state_events']['fault_context_valid'])
        report['tracking_performance']='Measured separately; correctness pass does not imply drift/accuracy target met'
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if not report['correctness_acceptance']:raise RuntimeError('Acceptance gate failed; inspect report')
        result=report
    finally:
        for proc in processes:
            try:_stop(proc)
            except Exception as exc:print('Cleanup:',exc,file=sys.stderr)
        subprocess.run(['docker','stop','--time','3',container],capture_output=True,timeout=8,check=False)
        for stream in streams:stream.close()
        lock.close()
    print(json.dumps(dict(output=str(output),pass_correctness=result['correctness_acceptance'],policy_hz=result['policy_telemetry_hz'])),flush=True)


if __name__=='__main__':main()
