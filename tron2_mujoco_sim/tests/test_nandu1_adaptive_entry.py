import sys
from pathlib import Path
import pytest
from nandu1_mouse_protocol import command_interface
from nandu1_process import parse_options
import nandu1_contract as base

TASK='Isaac-NANDU1-Blind-Flat-v1-3-1-2-6'


def test_adaptive_commands_route_to_tcp_without_changing_legacy_tasks():
    assert command_interface(TASK)==('tcp','/nandu1/v1_3_1_2_6/command')
    assert command_interface(TASK.replace('Flat-v','Flat-Play-v'))==command_interface(TASK)
    assert len(base.TASKS)==7
    assert command_interface(base.V131_TASK)==('tcp','/nandu1/v1_3_1/command')
    assert command_interface(base.TASK)==('body','/nandu1/command')


def test_supervisor_selects_new_launch_only_for_new_task():
    import nandu1_process
    assert hasattr(nandu1_process,'launch_file'), 'task-aware launch dispatch missing'
    assert nandu1_process.launch_file(TASK)=='nandu1_adaptive_mujoco_policy.launch'
    assert nandu1_process.launch_file(base.TASK)=='nandu1_mujoco_policy.launch'


def test_new_task_mouse_headless_refused_external_preserved(tmp_path):
    argv=['--task',TASK,'--bundle',str(tmp_path),'--telemetry',str(tmp_path/'data.csv')]
    assert parse_options(argv+['--','--headless']).command_source=='external'
    with pytest.raises(SystemExit):parse_options(argv+['--command-source','mouse','--','--headless'])
