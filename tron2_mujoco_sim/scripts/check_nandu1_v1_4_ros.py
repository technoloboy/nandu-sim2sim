#!/usr/bin/env python3
"""Finite localhost SDK+ROS acceptance; never contacts physical hardware."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import socket
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HELPER=ROOT/'tron2_rl_deploy_ros/scripts/nandu1_noetic.sh'
TASK='Isaac-NANDU1-Blind-Flat-v1-4'

def docker(command):return ['sg','docker','-c',str(HELPER)+' run '+command]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=120)
    p.add_argument('--bundle',type=Path,default=ROOT/'deployment_bundles/nandu1_v1_4_model_24000')
    a=p.parse_args();a.output=a.output.resolve()
    if a.duration<120 or not __import__('math').isfinite(a.duration):raise ValueError('Full step/lifecycle acceptance requires at least 120 seconds')
    with socket.socket() as sock:
        sock.settimeout(.2)
        if sock.connect_ex(('127.0.0.1',11311))==0:raise RuntimeError('ROS master already running; stop that simulation first')
    if a.output.exists():raise ValueError('Choose a fresh acceptance output directory')
    a.output.mkdir(parents=True);processes=[];streams=[]
    def start(name,command):
        stream=(a.output/(name+'.log')).open('w');streams.append(stream)
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);processes.append(process);return process
    try:
        ros=start('ros',docker(f'roslaunch tron2_hw nandu1_mujoco_policy.launch task:={TASK} '+shlex.quote('model_bundle:='+str(a.bundle))+' '+shlex.quote('telemetry_output:='+str(a.output/'tracking.csv'))))
        time.sleep(2)
        sim_started=time.monotonic()
        sim=start('simulator',[sys.executable,str(ROOT/'tron2_mujoco_sim/simulator.py'),'--task',TASK,'--model-bundle',str(a.bundle),'--ip','127.0.0.1','--headless','--duration',str(a.duration)])
        deadline=time.monotonic()+20
        # C++ rosconsole stdout may be block buffered in redirected logs. A
        # synchronized telemetry row proves inference/control is actually live.
        tracking=a.output/'tracking.csv'
        while not tracking.exists() or len(tracking.read_text().splitlines())<3:
            if sim.poll() is not None or ros.poll() is not None or time.monotonic()>deadline:raise RuntimeError('ROS controller did not enter POLICY; inspect logs')
            time.sleep(.2)
        command=start('commands',docker(f'rosrun tron2_controllers nandu1_command.py --task {TASK} --mode steps --height 0.88564531185'))
        orderly_shutdown=False;lifecycle=None
        deadline=time.monotonic()+a.duration+30
        while sim.poll() is None:
            if ros.poll() is not None and not orderly_shutdown:raise RuntimeError('ROS exited before simulation completion')
            if command.poll() not in (None,0):raise RuntimeError('Step command node failed')
            if time.monotonic()>deadline:raise RuntimeError('Simulation timeout')
            if lifecycle is None and command.poll()==0 and time.monotonic()-sim_started>=103:
                lifecycle=start('lifecycle',docker(f'rosrun tron2_controllers nandu1_command.py --task {TASK} --mode lifecycle --height 0.88564531185'))
            # Stop the subscriber/controller first while physics still runs.
            # Stopping the simulator first would intentionally trip the 100ms
            # sensor watchdog and must not be mistaken for a policy failure.
            if not orderly_shutdown and time.monotonic()-sim_started>=a.duration-6 and command.poll()==0:
                if lifecycle is None or lifecycle.poll()!=0:raise RuntimeError('Lifecycle probe did not complete')
                subprocess.run(docker('rostopic pub -1 /nandu1/set_mode std_msgs/String "data: PASSIVE"'),stdout=subprocess.DEVNULL,check=True,timeout=10)
                orderly_shutdown=True;os.killpg(ros.pid,signal.SIGINT);ros.wait(timeout=10)
            time.sleep(.2)
        code=sim.returncode
        if code:raise RuntimeError(f'Simulator exited {code}; inspect logs')
        if command.poll()!=0:raise RuntimeError('Full step command sequence did not finish successfully')
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGINT)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
        for stream in streams:stream.close()
    subprocess.run([sys.executable,str(ROOT/'tron2_mujoco_sim/scripts/analyze_nandu1_ros_tracking.py'),
        '--csv',str(a.output/'tracking.csv'),'--bundle',str(a.bundle),'--output',str(a.output/'report.json')],check=True)
    # Inspect logs after process shutdown (all buffered ROS output is flushed).
    if 'NANDU1 FAULT' in (a.output/'ros.log').read_text():raise RuntimeError('Controller FAULT; inspect logs and report')
    report=json.loads((a.output/'report.json').read_text())
    with (a.output/'tracking.csv').open() as stream:rows=list(csv.DictReader(stream))
    marker=next(json.loads(line) for line in (a.output/'lifecycle.log').read_text().splitlines() if line.startswith('{'))
    expired=[r for r in rows if marker['expiry_last_publish_wall']+.65<float(r['tracking_stamp'])<marker['controller_reset_wall']]
    after_reset=[r for r in rows if marker['controller_reset_wall']+.04<float(r['tracking_stamp'])<marker['controller_reset_wall']+.2]
    if not expired or any(abs(float(r['external_intent_1']))>1e-6 for r in expired):raise RuntimeError('Command expiry did not zero vx intent')
    if not after_reset or min(abs(float(r['relative_heading'])) for r in after_reset)>.02:raise RuntimeError('Controller reset did not reset heading integration')
    report['lifecycle']={'command_expiry_verified':True,'controller_reset_verified':True,'markers':marker}
    hard=['torque_outside_limits','position_target_outside_limits','position_state_outside_limits_005_rad_tolerance','velocity_state_outside_limits','velocity_projection_events']
    if any(report[key]!=0 for key in hard):raise RuntimeError('Safety gate failed; inspect report counters')
    if report['wall_duration_s']<a.duration-20 or not 45<=report['policy_hz_average']<=55 or report['max_policy_gap_s']>.1:
        raise RuntimeError('Sustained controller telemetry/frequency gate failed')
    if len(report['command_plateaus'])<9:raise RuntimeError('Step coverage incomplete')
    report['deployment_acceptance']='passed; tracking accuracy is reported separately'
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'acceptance':str(a.output),'command_process_exit':command.returncode},indent=2))

if __name__=='__main__':main()
