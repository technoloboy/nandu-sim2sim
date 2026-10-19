"""Catch task propagation and distinct external-message semantics."""
import importlib.util
from pathlib import Path
import shlex
import unittest
import xml.etree.ElementTree as ET
import xmlrpc.client

DEPLOY=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('validator',DEPLOY/'tron2_controllers/scripts/validate_nandu1_bundle.py')
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)

class V14LaunchTest(unittest.TestCase):
    def test_task_is_passed_to_every_validator_and_controller(self):
        tree=ET.parse(DEPLOY/'tron2_hw/launch/nandu1_mujoco_policy.launch')
        for param in tree.findall('.//param'):
            if 'validate_nandu1_bundle.py' in param.get('command',''):
                # Resolve substitutions before tokenizing: ROS substitutes the
                # entire argument value, including any spaces, before exec.
                command = param.get('command').replace('$(arg task)', 'example-task')
                tokens = shlex.split(command)
                self.assertEqual(tokens[tokens.index('--task') + 1], 'example-task')
            if param.get('name')=='tron2_controller/task':self.assertEqual(param.get('value'),'$(arg task)')
    def test_v14_validated_ros_projection(self):
        from nandu1_bundle import load_bundle
        manifest,_=load_bundle(DEPLOY.parent/'deployment_bundles/nandu1_v1_4_model_24000','Isaac-NANDU1-Blind-Flat-Play-v1-4')
        params=validator.ros_parameters(manifest)
        xmlrpc.client.dumps((params,),allow_none=False)
        self.assertEqual(params['task']['evaluation_stage'],4)
        self.assertEqual(params['task']['task_id'],'Isaac-NANDU1-Blind-Flat-v1-4')
    def test_v14_message_does_not_relabel_body_velocity(self):
        text=(DEPLOY/'tron2_controllers/msg/NANDU1V14Command.msg').read_text()
        self.assertIn('up3_vx_base_heading',text);self.assertNotIn('up3_vx_base_body',text)

if __name__=='__main__':unittest.main()
