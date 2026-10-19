"""The ROS command boundary must preserve executable and bundle paths."""
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['nandu1_mujoco_policy.launch',
    'nandu1_network_e_mujoco_policy.launch', 'nandu1_adaptive_mujoco_policy.launch'])
def test_launch_validator_commands_preserve_space_containing_paths(name):
    root = ET.parse(ROOT / 'tron2_rl_deploy_ros/tron2_hw/launch' / name).getroot()
    executable = '/tmp/new checkout/tron2_rl_deploy_ros/tron2_controllers/scripts/validate_nandu1_bundle.py'
    bundle = '/tmp/new checkout/deployment bundles/formal model'
    commands = [n.get('command') for n in root.findall('param') if n.get('command')]
    assert len(commands) == 3
    for command in commands:
        command = command.replace('$(find tron2_controllers)', '/tmp/new checkout/tron2_rl_deploy_ros/tron2_controllers')
        command = command.replace('$(arg model_bundle)', bundle).replace('$(arg task)', 'Isaac-NANDU1-Blind-Flat-v1-2')
        command = command.replace('$(arg robot_ip)', '127.0.0.1')
        argv = shlex.split(command)
        assert argv[0] == executable
        assert argv[argv.index('--bundle') + 1] == bundle
