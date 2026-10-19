import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def test_ros_early_exit_is_failure_and_sim_child_is_reaped(tmp_path):
    from nandu1_process import supervise
    pidfile=tmp_path/'sim.pid'
    sim=[sys.executable,'-c',f'import os,time;open({str(pidfile)!r},"w").write(str(os.getpid()));time.sleep(30)']
    ros=[sys.executable,'-c','import time;time.sleep(.15);raise SystemExit(9)']
    with pytest.raises(RuntimeError,match='ROS.*9'):supervise(ros,sim,lambda:True,timeout=1)
    import os
    with pytest.raises(ProcessLookupError):os.kill(int(pidfile.read_text()),0)

def test_no_ready_telemetry_times_out_and_reaps_both_children():
    from nandu1_process import supervise
    child=[sys.executable,'-c','import time;time.sleep(30)']
    with pytest.raises(RuntimeError,match='ready'):supervise(child,child,lambda:False,timeout=.1)

def test_normal_sim_exit_returns_status_and_stops_ros():
    from nandu1_process import supervise
    ros=[sys.executable,'-c','import time;time.sleep(30)']
    sim=[sys.executable,'-c','import time;time.sleep(.1)']
    assert supervise(ros,sim,lambda:True,timeout=1)==0

def test_container_cleanup_runs_after_startup_failure(tmp_path):
    from nandu1_process import supervise
    cleaned=[]
    ros=[sys.executable,'-c','import time;time.sleep(.1);raise SystemExit(9)']
    sim=[sys.executable,'-c','import time;time.sleep(30)']
    with pytest.raises(RuntimeError,match='ROS.*9'):
        supervise(ros,sim,lambda:True,cleanup=lambda:cleaned.append(True))
    assert cleaned==[True]

def test_ros_receives_owned_container_identity(tmp_path):
    from nandu1_process import supervise
    import os
    destination=tmp_path/'name'
    ros=[sys.executable,'-c',f'import os,time;open({str(destination)!r},"w").write(os.environ["NANDU1_CONTAINER_NAME"]);time.sleep(30)']
    sim=[sys.executable,'-c','import time;time.sleep(.15)']
    assert supervise(ros,sim,lambda:destination.exists(),ros_env=dict(os.environ,NANDU1_CONTAINER_NAME='nandu1-local-test'))==0
    assert destination.read_text()=='nandu1-local-test'

def test_term_to_supervisor_cleans_both_process_groups(tmp_path):
    import subprocess,time,signal,os
    root=Path(__file__).resolve().parents[1]
    children=[]
    for role in ('ros','sim'):
        children.append([sys.executable,'-c',f'import os,time;open({str(tmp_path/(role+".pid"))!r},"w").write(str(os.getpid()));time.sleep(30)'])
    code=f'import sys;sys.path.insert(0,{str(root)!r});from nandu1_process import supervise\ntry:supervise({children[0]!r},{children[1]!r},lambda:True)\nexcept KeyboardInterrupt:raise SystemExit(130)'
    process=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+3
        while not all((tmp_path/(role+'.pid')).exists() and
                      (tmp_path/(role+'.pid')).read_text().strip().isdigit() for role in ('ros','sim')):
            if time.monotonic()>deadline:raise RuntimeError('Children not started')
            time.sleep(.01)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=3)==130
        for role in ('ros','sim'):
            with pytest.raises(ProcessLookupError):os.kill(int((tmp_path/(role+'.pid')).read_text()),0)
    finally:
        if process.poll() is None:process.kill();process.wait()

def test_gui_close_reaps_only_owned_children(tmp_path):
    from nandu1_process import supervise
    import os
    children=[]
    for role in ('ros','sim'):
        children.append([sys.executable,'-c',f'import os,time;open({str(tmp_path/(role+".pid"))!r},"w").write(str(os.getpid()));time.sleep(30)'])
    gui=[sys.executable,'-c','import time;time.sleep(.2)']
    assert supervise(*children,lambda:True,gui_command=gui,gui_ready=lambda:True)==0
    for role in ('ros','sim'):
        with pytest.raises(ProcessLookupError):os.kill(int((tmp_path/(role+'.pid')).read_text()),0)

def test_gui_startup_crash_does_not_spawn_ros_or_sim(tmp_path):
    from nandu1_process import supervise
    marker=tmp_path/'bad-start'
    child=[sys.executable,'-c',f'open({str(marker)!r},"w").write("unexpected")']
    gui=[sys.executable,'-c','raise SystemExit(9)']
    with pytest.raises(RuntimeError,match='GUI'):
        supervise(child,child,lambda:True,gui_command=gui,gui_ready=lambda:False,timeout=1)
    assert not marker.exists()

def test_gui_crash_after_ready_is_failure_and_cleans_owned_session(tmp_path):
    from nandu1_process import supervise
    cleaned=[]
    child=[sys.executable,'-c','import time;time.sleep(30)']
    gui=[sys.executable,'-c','import time;time.sleep(.1);raise SystemExit(7)']
    with pytest.raises(RuntimeError,match='GUI.*7'):
        supervise(child,child,lambda:True,gui_command=gui,gui_ready=lambda:True,
                  cleanup=lambda:cleaned.append(True))
    assert cleaned==[True]

def test_gui_socket_deadline_does_not_start_ros_or_sim(tmp_path):
    from nandu1_process import supervise
    marker=tmp_path/'bad-start';cleaned=[]
    child=[sys.executable,'-c',f'open({str(marker)!r},"w").write("unexpected")']
    gui=[sys.executable,'-c','import time;time.sleep(30)']
    with pytest.raises(RuntimeError,match='GUI not ready'):
        supervise(child,child,lambda:True,gui_command=gui,gui_ready=lambda:False,
                  timeout=.1,cleanup=lambda:cleaned.append(True))
    assert not marker.exists()
    assert cleaned==[True]

def test_mouse_headless_rejected_but_external_preserves_args(tmp_path):
    from nandu1_process import parse_options
    base=['--task','Isaac-NANDU1-Blind-Flat-v1-2','--bundle',str(tmp_path),'--telemetry',str(tmp_path/'test.csv')]
    args=parse_options(base+['--','--headless','--duration','1'])
    assert args.command_source=='external'
    assert args.sim_args==['--','--headless','--duration','1']
    with pytest.raises(SystemExit) as error:
        parse_options(base+['--command-source','mouse','--','--headless'])
    assert error.value.code==2

def test_mouse_entry_help_is_available_without_docker_or_display():
    import subprocess
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['bash',str(root/'scripts/run_nandu1_mouse.sh'),'--help'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'command-source' in result.stdout

def test_repeated_termination_cannot_interrupt_owned_cleanup(tmp_path):
    import subprocess,time,signal,os
    root=Path(__file__).resolve().parents[1]
    children=[]
    for role in ('ros','sim'):
        children.append([sys.executable,'-c',f'import os,time,signal;signal.signal(signal.SIGINT,signal.SIG_IGN);open({str(tmp_path/(role+".pid"))!r},"w").write(str(os.getpid()));time.sleep(30)'])
    code=f'import sys;sys.path.insert(0,{str(root)!r});from nandu1_process import supervise\ntry:supervise({children[0]!r},{children[1]!r},lambda:True)\nexcept KeyboardInterrupt:raise SystemExit(130)'
    process=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+3
        while not all((tmp_path/(role+'.pid')).exists() and
                      (tmp_path/(role+'.pid')).read_text().strip().isdigit() for role in ('ros','sim')):
            if time.monotonic()>deadline:raise RuntimeError('Children not started')
            time.sleep(.01)
        process.send_signal(signal.SIGTERM);time.sleep(.1);process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15)==130
        for role in ('ros','sim'):
            with pytest.raises(ProcessLookupError):os.kill(int((tmp_path/(role+'.pid')).read_text()),0)
    finally:
        if process.poll() is None:process.kill();process.wait()
        for role in ('ros','sim'):
            if (tmp_path/(role+'.pid')).exists():
                try:os.kill(int((tmp_path/(role+'.pid')).read_text()),signal.SIGKILL)
                except ProcessLookupError:pass
