"""Asset/CLI mapping regression only; does not claim SDK/ROS runtime regression."""
from pathlib import Path
import sys
import mujoco
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulator import load_variant_config

@pytest.mark.parametrize('variant',['SF_TRON2A','WF_TRON2A'])
def test_legacy_ten_joint_asset_still_loads(variant):
    root=Path(__file__).resolve().parents[1]
    joints,sensors,path=load_variant_config(str(root),variant)
    model=mujoco.MjModel.from_xml_path(path)
    assert len(joints)==len(sensors)==model.nu==10
    for name in joints:
        assert mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name+'_Joint')>=0
