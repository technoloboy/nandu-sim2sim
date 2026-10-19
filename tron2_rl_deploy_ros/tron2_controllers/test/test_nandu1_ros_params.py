import importlib.util
from pathlib import Path
import unittest
import xmlrpc.client

spec=importlib.util.spec_from_file_location('nandu_validator',Path(__file__).resolve().parents[1]/'scripts/validate_nandu1_bundle.py')
validator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)

class RosParamsTest(unittest.TestCase):
    def test_null_bundle_metadata_is_not_sent_to_ros(self):
        manifest={'task':{'initial_up3_height':.8856,'joint_position_lower':[-3]*13,
                          'joint_position_upper':[3]*13,'action_clipping':None}}
        projected=validator.ros_parameters(manifest)
        xmlrpc.client.dumps((projected,),allow_none=False)
        self.assertEqual(len(projected['task']),3)
        self.assertIsNone(manifest['task']['action_clipping'])

if __name__=='__main__':unittest.main()
