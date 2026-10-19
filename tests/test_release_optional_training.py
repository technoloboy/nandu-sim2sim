"""A runtime-only checkout must not import optional USD build dependencies."""
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_missing_training_assets_are_checked_before_importing_usd_builder(tmp_path, monkeypatch):
    test = runpy.run_path(str(ROOT / 'tron2_mujoco_sim/tests/test_nandu1_bundle.py'))
    monkeypatch.setitem(sys.modules, 'nandu1_asset', None)
    with pytest.raises(pytest.skip.Exception, match='Optional training USD/URDF'):
        test['test_asset_matches_source_and_fk'](tmp_path)
