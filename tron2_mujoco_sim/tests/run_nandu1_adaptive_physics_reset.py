"""Acceptance-only: deliver R to the real viewer callback, verify epoch reset."""
import argparse
import csv
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import uuid

SIM=Path(__file__).resolve().parents[1];ROOT=SIM.parent
sys.path.insert(0,str(SIM))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--sim-probe',action='store_true')
    a=p.parse_args()
    from nandu1_adaptive_contract import TASK
    if a.sim_probe:
        import mujoco.viewer
        launch=mujoco.viewer.launch_passive
        def viewer(*args,**kw):
            handle=launch(*args,**kw)
            for delay in (3.,6.):
                timer=threading.Timer(delay,lambda:kw['key_callback'](82));timer.daemon=True;timer.start()
            return handle
        mujoco.viewer.launch_passive=viewer
        sys.argv=['simulator.py','--task',TASK,'--model-bundle',str(a.bundle),'--ip','127.0.0.1','--duration','10']
        runpy.run_path(str(SIM/'simulator.py'),run_name='__main__');return
    if a.output is None:p.error('--output required')
    from nandu1_process import supervise
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('acceptance',SIM/'scripts/check_nandu1_adaptive_ros.py');m=module_from_spec(spec);spec.loader.exec_module(m)
    m.preflight(a.output);a.output.mkdir(parents=True)
    container='nandu1-reset-test-'+uuid.uuid4().hex;telemetry=a.output/'tracking.csv'
    ros=[str(ROOT/'tron2_rl_deploy_ros/scripts/nandu1_noetic.sh'),'run','roslaunch','tron2_hw','nandu1_adaptive_mujoco_policy.launch',
         'task:='+TASK,'model_bundle:='+str(a.bundle),'telemetry_output:='+str(telemetry)]
    sim=[sys.executable,str(Path(__file__).resolve()),'--bundle',str(a.bundle),'--sim-probe']
    def ready():
        if not telemetry.exists():return False
        with telemetry.open() as f:return all(f.readline().strip() for _ in range(3))
    def cleanup():subprocess.run(['docker','stop','--time','3',container],capture_output=True,timeout=8,check=False)
    status=supervise(ros,sim,ready,ros_env=dict(os.environ,NANDU1_CONTAINER_NAME=container),cleanup=cleanup)
    with telemetry.open() as f:rows=list(csv.DictReader(f))
    epochs={r['epoch'] for r in rows};priming=[r for r in rows if float(r['policy_step'])==0]
    # The logger can start after initial policy priming. Both later resets must
    # be observed; never infer a missing startup frame is a controller failure.
    if status or len(epochs)!=3 or len(priming)<2:raise AssertionError('Missing actual physical epoch reset')
    for row in priming:
        if any(abs(float(row[f'{name}_{i}']))>1e-7 for name,n in [('dynamic',12),('hidden',64),('eta',8)] for i in range(n)):
            raise AssertionError('Physical reset leaked temporal state')
    result=dict(passed=True,epochs=sorted(epochs),samples=len(rows),priming_frames=len(priming),
        input='test delivers keycode82 to production MuJoCo viewer callback, not global keyboard injection')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
