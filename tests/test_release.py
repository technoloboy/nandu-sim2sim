"""Release contract: complete deployable bundles, no local-only artifacts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tron2_mujoco_sim'))
BUNDLES = [
    'nandu1_v1_1_model_30000', 'nandu1_v1_2_model_40000',
    'nandu1_v1_2_1_model_29000', 'nandu1_v1_3_model_30000',
    'nandu1_v1_3_1_model_30000', 'nandu1_v1_3_1_2_model_27500',
    'nandu1_v1_3_1_2_2_model_29500', 'nandu1_v1_3_1_2_3_model_11000',
    'nandu1_v1_3_1_2_5_model_10500', 'nandu1_v1_3_1_2_6_model_23500',
    'nandu1_v1_3_1_2_7_model_10000', 'nandu1_v1_4_model_24000',
    'nandu1_v1_4_1_model_14000',
]
C7_BUNDLES = [f'nandu1_v1_6_C7_P_1_R{i}_model_30000' for i in range(3)]


@pytest.mark.parametrize('version', ['R0', 'R1', 'R2'])
def test_c7_release_bundle_is_complete_and_preserves_experimental_status(version):
    path = ROOT / 'deployment_bundles' / f'nandu1_v1_6_C7_P_1_{version}_model_30000'
    assert path.is_dir(), 'C7 release bundle missing'
    from nandu1_c7_sim2sim import validate_bundle
    manifest = validate_bundle(path, version)
    assert manifest['checkpoint_iteration'] == 30000
    assert manifest['checkpoint_sha256']
    assert manifest['release_status'] == 'experimental_pending_Isaac_and_feasibility_validation'
    assert not any('__pycache__' in Path(f).parts or f.endswith('.pyc') for f in manifest['files'])


@pytest.mark.parametrize('name', BUNDLES)
def test_release_bundle_is_sealed_complete_and_cache_free(name):
    from nandu1_bundle import load_bundle
    p = ROOT / 'deployment_bundles' / name
    assert p.is_dir(), 'Official release bundle missing: ' + name
    manifest, _ = load_bundle(p)
    assert not any('__pycache__' in Path(f).parts or f.endswith('.pyc') for f in manifest['files'])
    assert manifest['checkpoint']['sha256']


def test_runtime_interpreter_selection_is_portable():
    result = subprocess.run(
        ['bash', str(ROOT / 'tron2_mujoco_sim/scripts/nandu1_python.sh'), '--print'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    selected = Path(result.stdout.strip())
    assert selected.is_file()
    assert 'cambot-train' not in result.stdout or selected.resolve() == Path(sys.executable).resolve()


def test_launcher_uses_active_python_instead_of_original_workstation(tmp_path):
    # The shell/Python process boundary is the dependency under test. Exit
    # before starting ROS; no docker or original environment is required.
    executable = tmp_path / 'python3'
    executable.write_text('#!/bin/sh\nprintf "selected-active-python:%s\\n" "$1"\nexit 23\n')
    executable.chmod(0o755)
    result = subprocess.run(['bash', str(ROOT / 'tron2_mujoco_sim/scripts/run_nandu1.sh'),
        '--task', 'Isaac-NANDU1-Blind-Flat-v1-3-1-2-2'],
        env=dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH']),
        capture_output=True, text=True)
    assert result.returncode == 23, result.stdout + result.stderr
    assert 'selected-active-python:' + str(ROOT / 'tron2_rl_deploy_ros/tron2_controllers/scripts/validate_nandu1_bundle.py') in result.stdout


def test_catalogue_identifies_all_weights_and_does_not_claim_f_is_stable():
    catalogue = json.loads((ROOT / 'deployment_bundles/catalogue.json').read_text())
    assert {b['directory'] for b in catalogue['bundles']} == set(BUNDLES + C7_BUNDLES)
    f = next(b for b in catalogue['bundles'] if b['directory'] == 'nandu1_v1_3_1_2_7_model_10000')
    assert f['stability'] == 'not_accepted_zero_command_instability'
    for b in catalogue['bundles']:
        p = ROOT / 'deployment_bundles' / b['directory'] / b.get('manifest_file', 'manifest.yaml')
        assert hashlib.sha256(p.read_bytes()).hexdigest() == b['manifest_sha256']
        if b['directory'] in C7_BUNDLES:
            assert b['stability'] == 'experimental_pending_Isaac_and_feasibility_validation'
