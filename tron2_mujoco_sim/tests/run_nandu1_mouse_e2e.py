"""Owned local GUI+ROS+SDK+MuJoCo acceptance; no physical robot support."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

SIM=Path(__file__).resolve().parents[1];ROOT=SIM.parent
sys.path.insert(0,str(SIM))
from nandu1_process import supervise,launch_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    telemetry=a.output/'tracking.csv';report=a.output/'gui.json'
    container='nandu1-mouse-test-'+uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='.mouse-test-',dir=SIM) as d:
        endpoint=Path(d)/'control.sock';session=uuid.uuid4().hex
        ros=[str(ROOT/'tron2_rl_deploy_ros/scripts/nandu1_noetic.sh'),'run','roslaunch','tron2_hw',launch_file(a.task),
             'task:='+a.task,'model_bundle:='+str(a.bundle),'telemetry_output:='+str(telemetry),
             'command_source:=mouse','mouse_socket:='+str(endpoint),'mouse_session:='+session]
        sim=[sys.executable,str(SIM/'simulator.py'),'--task',a.task,'--model-bundle',str(a.bundle),'--ip','127.0.0.1','--duration','30']
        gui=[sys.executable,str(SIM/'tests/nandu1_mouse_gui_probe.py'),'--report',str(report),'--task',a.task,
             '--bundle',str(a.bundle),'--socket',str(endpoint),'--session',session,'--container',container]
        def ready():
            if not telemetry.exists():return False
            with telemetry.open() as f:return all(f.readline().strip() for _ in range(3))
        def cleanup():
            subprocess.run(['docker','stop','--time','3',container],capture_output=True,timeout=8,check=False)
        result=supervise(ros,sim,ready,ros_env=dict(os.environ,NANDU1_CONTAINER_NAME=container),
                         cleanup=cleanup,gui_command=gui,gui_ready=endpoint.exists)
    records=json.loads(report.read_text())
    if result!=0 or len(records)!=6 or any('error' in r for r in records):
        raise RuntimeError('Mouse E2E failed: '+json.dumps(records))
    guard=json.loads(report.with_suffix('.guard.json').read_text())[0]
    if guard.get('returncode',0)==0 or 'Active mouse command source' not in guard.get('output',''):
        raise RuntimeError('Competing keyboard/steps source was not rejected: '+json.dumps(guard))
    import csv
    with telemetry.open() as f:rows=list(csv.DictReader(f))
    if len(rows)<50:raise RuntimeError('Insufficient real inference telemetry')
    commands=[[float(r[f'command_{i}']) for i in range(3)] for r in rows]
    if not (max(c[1] for c in commands)>.01 and min(c[1] for c in commands)<-.01
            and max(c[0] for c in commands)>.02 and min(c[0] for c in commands)<-.02):
        raise RuntimeError('Missing actual positive/negative yaw/vx response')
    summary=dict(task=a.task,telemetry_rows=len(rows),result='pass',source_guard='pass',gui=records)
    (a.output/'report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
