"""Owned, finite test-only SDK+ROS lifecycle probe (not a hardware entry)."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

SIM=Path(__file__).resolve().parents[1];ROOT=SIM.parent
sys.path.insert(0,str(SIM))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from nandu1_adaptive_contract import TASK
    from nandu1_process import _stop
    spec=importlib.util.spec_from_file_location('acceptance',SIM/'scripts/check_nandu1_adaptive_ros.py')
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe);probe.preflight(a.output)
    a.output.mkdir(parents=True);container='nandu1-lifecycle-'+uuid.uuid4().hex;children=[];streams=[]
    def start(name,cmd,env=None):
        f=(a.output/(name+'.log')).open('w');streams.append(f)
        child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(child);return child
    try:
        ros=start('ros',[str(ROOT/'tron2_rl_deploy_ros/scripts/nandu1_noetic.sh'),'run','roslaunch','tron2_hw','nandu1_adaptive_mujoco_policy.launch',
            'task:='+TASK,'model_bundle:='+str(a.bundle),'telemetry_output:='+str(a.output/'tracking.csv')],dict(os.environ,NANDU1_CONTAINER_NAME=container))
        sim=start('simulator',[sys.executable,str(SIM/'simulator.py'),'--task',TASK,'--model-bundle',str(a.bundle),'--ip','127.0.0.1','--headless','--duration','25'])
        deadline=time.monotonic()+20
        while not (a.output/'tracking.csv').exists() or sum(1 for _ in (a.output/'tracking.csv').open())<3:
            if time.monotonic()>deadline or ros.poll() is not None or sim.poll() is not None:raise RuntimeError('Lifecycle startup failed')
            time.sleep(.1)
        script=ROOT/'tron2_rl_deploy_ros/tron2_controllers/test/nandu1_adaptive_ros_lifecycle.py'
        command='source /opt/ros/noetic/setup.bash && source /ws/devel/setup.bash && python3 '+shlex.quote(str(script))+' --output '+shlex.quote(str(a.output/'report.json'))
        test=start('probe',['docker','exec',container,'bash','-lc',command]);status=test.wait(timeout=12)
        if status:raise RuntimeError('Lifecycle probe failed; inspect '+str(a.output))
        events=[json.loads(line) for line in (a.output/'tracking.events.jsonl').read_text().splitlines()]
        if not any(e['mode']=='PASSIVE' for e in events) or not any(e['mode']=='FAULT' and 'Nonfinite' in e['reason'] for e in events):
            raise AssertionError('Missing durable PASSIVE/FAULT evidence')
        print((a.output/'report.json').read_text(),flush=True)
    finally:
        for child in children:_stop(child)
        subprocess.run(['docker','stop','--time','3',container],capture_output=True,timeout=8,check=False)
        for stream in streams:stream.close()


if __name__=='__main__':main()
