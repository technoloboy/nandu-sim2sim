"""Bounded supervision of this launcher's two local child process groups."""
import os
import signal
import subprocess
import time

def _stop(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGINT)
    except ProcessLookupError:return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)

def supervise(ros_command,sim_command,ready,timeout=20,ros_env=None,cleanup=None,gui_command=None,gui_ready=None):
    processes=[]
    previous={}
    def interrupted(signum,_):raise KeyboardInterrupt
    try:
        for number in (signal.SIGINT,signal.SIGTERM):
            previous[number]=signal.signal(number,interrupted)
        gui=None
        if gui_command is not None:
            gui=subprocess.Popen(gui_command,start_new_session=True);processes.append(gui)
            deadline=time.monotonic()+timeout
            while True:
                if gui.poll() is not None:raise RuntimeError(f'GUI exited during startup (status {gui.returncode})')
                if gui_ready is not None and gui_ready():break
                if time.monotonic()>deadline:raise RuntimeError('GUI not ready before startup deadline')
                time.sleep(.02)
        ros=subprocess.Popen(ros_command,start_new_session=True,env=ros_env);processes.append(ros)
        sim=subprocess.Popen(sim_command,start_new_session=True);processes.append(sim)
        deadline=time.monotonic()+timeout;is_ready=False
        while True:
            if ros.poll() is not None:raise RuntimeError(f'ROS exited before simulation (status {ros.returncode})')
            if not is_ready:is_ready=bool(ready())
            if sim.poll() is not None:
                if not is_ready:raise RuntimeError(f'Simulation exited before controller ready (status {sim.returncode})')
                return sim.returncode
            if gui is not None and gui.poll() is not None:
                if not is_ready or gui.returncode!=0:
                    raise RuntimeError(f'GUI exited unexpectedly (status {gui.returncode})')
                return 0
            if not is_ready and time.monotonic()>deadline:raise RuntimeError('Controller not ready before startup deadline')
            time.sleep(.02)
    finally:
        # A second Ctrl+C must not abort cleanup of owned child groups/container.
        for number in previous:
            signal.signal(number,signal.SIG_IGN)
        cleanup_errors=[]
        try:
            for process in processes:
                try:_stop(process)
                except Exception as exc:cleanup_errors.append(exc)
            if cleanup is not None:
                try:cleanup()
                except Exception as exc:cleanup_errors.append(exc)
        finally:
            for number,handler in previous.items():signal.signal(number,handler)
        if cleanup_errors:
            raise RuntimeError('Owned session cleanup failed: '+str(cleanup_errors[0])) from cleanup_errors[0]

def parse_options(argv=None):
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument('--task',required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--telemetry',type=Path,required=True)
    p.add_argument('--command-source',choices=('external','mouse'),default='external')
    p.add_argument('sim_args',nargs=argparse.REMAINDER)
    args=p.parse_args(argv)
    if args.command_source=='mouse' and any(v.split('=')[0]=='--headless' for v in args.sim_args):
        p.error('mouse command source cannot be combined with --headless; use external for collection')
    return args

def launch_file(task):
    from nandu1_adaptive_contract import deployment_profile,TASK
    from nandu1_contract import V1312_TASK
    from nandu1_adaptive_balance_contract import TASK as balance
    canonical=deployment_profile(task)['task_id']
    from nandu1_ablation_contract import PROFILES
    if canonical in PROFILES:return PROFILES[canonical].LAUNCH
    if canonical==balance:return 'nandu1_adaptive_balance_mujoco_policy.launch'
    if canonical==V1312_TASK:return 'nandu1_network_e_mujoco_policy.launch'
    return ('nandu1_adaptive_mujoco_policy.launch' if canonical==TASK
            else 'nandu1_mujoco_policy.launch')

def main():
    import contextlib
    import fcntl
    from pathlib import Path
    import socket
    import sys
    import tempfile
    import uuid
    args=parse_options();root=Path(__file__).resolve().parent.parent
    bundle=args.bundle.resolve();telemetry=args.telemetry.resolve()
    # Create requested output directories as the host owner, not Docker root.
    telemetry.parent.mkdir(parents=True,exist_ok=True)
    if args.command_source=='mouse':
        from nandu1_mouse_gui import probe_display
        probe_display()
    lock=(root/'tron2_mujoco_sim/.nandu1_launch.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another NANDU1 unified launcher is active') from None
    for kind,port in ((socket.SOCK_STREAM,11311),(socket.SOCK_DGRAM,15012)):
        with socket.socket(socket.AF_INET,kind) as sock:
            try:sock.bind(('127.0.0.1',port))
            except OSError:raise RuntimeError(f'Local sim2sim port {port} is already in use') from None
    began=time.time()
    # A header and two synchronized telemetry rows prove actual inference/control.
    def ready():
        if not telemetry.exists() or telemetry.stat().st_mtime<began:return False
        with telemetry.open() as stream:return all(stream.readline().strip() for _ in range(3))
    extra=args.sim_args[1:] if args.sim_args[:1]==['--'] else args.sim_args
    # Docker's daemon can outlive a killed CLI. Only stop this invocation's UUID,
    # never any unrelated ROS, user container, or shared project service.
    container_name='nandu1-local-'+uuid.uuid4().hex
    def cleanup_container():
        subprocess.run(['docker','stop','--time','3',container_name],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=8,check=False)
        remaining=subprocess.run(['docker','inspect','--format','{{.State.Running}}',container_name],
            capture_output=True,text=True,timeout=5,check=False)
        if remaining.returncode==0 and remaining.stdout.strip()=='true':
            raise RuntimeError(f'Owned ROS container did not stop: {container_name}')
    try:
        with contextlib.ExitStack() as owned:
            gui_command=gui_ready=None;mouse_args=[]
            if args.command_source=='mouse':
                directory=owned.enter_context(tempfile.TemporaryDirectory(prefix='.mouse-',dir=root/'tron2_mujoco_sim'))
                endpoint=Path(directory)/'control.sock';session=uuid.uuid4().hex
                gui_command=[sys.executable,str(root/'tron2_mujoco_sim/nandu1_mouse_gui.py'),
                    '--task',args.task,'--bundle',str(bundle),'--socket',str(endpoint),'--session',session]
                gui_ready=endpoint.exists
                mouse_args=['command_source:=mouse','mouse_socket:='+str(endpoint),'mouse_session:='+session]
            return supervise([str(root/'tron2_rl_deploy_ros/scripts/nandu1_noetic.sh'),'run','roslaunch','tron2_hw',launch_file(args.task),
                'task:='+args.task,'model_bundle:='+str(bundle),'telemetry_output:='+str(telemetry),*mouse_args],
                [sys.executable,str(root/'tron2_mujoco_sim/simulator.py'),'--task',args.task,'--model-bundle',str(bundle),'--ip','127.0.0.1','--command-timeout','.1',*extra],ready,
                ros_env=dict(os.environ,NANDU1_CONTAINER_NAME=container_name),cleanup=cleanup_container,
                gui_command=gui_command,gui_ready=gui_ready)
    except KeyboardInterrupt:return 130
    finally:lock.close()

if __name__=='__main__':raise SystemExit(main())
