import importlib.util
import os
from pathlib import Path
import select
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy  # Preserve the C extension across patch.dict(sys.modules) cleanup.

class KeyboardTest(unittest.TestCase):
    def test_new_tasks_select_the_correct_command_coordinates(self):
        spec=importlib.util.spec_from_file_location('nandu_keys',Path(__file__).resolve().parents[1]/'scripts/nandu1_command.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'rospy':SimpleNamespace(),'tron2_controllers.msg':SimpleNamespace(NANDU1Command=object)}):
            spec.loader.exec_module(module)
        self.assertEqual(module.command_interface('Isaac-NANDU1-Blind-Flat-Play-v1-2-1'),('body','/nandu1/v1_2_1/command'))
        self.assertEqual(module.command_interface('Isaac-NANDU1-Blind-Flat-v1-3-1'),('tcp','/nandu1/v1_3_1/command'))
        self.assertEqual(module.command_interface('Isaac-NANDU1-Blind-Flat-v1-4-1'),('heading','/nandu1/v1_4_1/command'))
        msg=SimpleNamespace()
        module.assign_command(msg,'tcp',[.2,.1,1.])
        self.assertEqual((msg.relative_heading_rate,msg.tcp_vx_base_body,msg.tcp_height_world),(.2,.1,1.))

    def test_burst_keys_remain_visible_to_select(self):
        spec=importlib.util.spec_from_file_location('nandu_keys',Path(__file__).resolve().parents[1]/'scripts/nandu1_command.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'rospy':SimpleNamespace(),'tron2_controllers.msg':SimpleNamespace(NANDU1Command=object)}):
            spec.loader.exec_module(module)
        read,write=os.pipe()
        try:
            os.write(write,b'Wx')
            self.assertEqual(module.read_keyboard_byte(read),'w')
            self.assertTrue(select.select([read],[],[],0)[0])
            self.assertEqual(module.read_keyboard_byte(read),'x')
        finally:os.close(read);os.close(write)

if __name__=='__main__':unittest.main()
