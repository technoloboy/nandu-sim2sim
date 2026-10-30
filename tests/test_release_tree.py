"""Publication policy rejects accidental data, secrets and extra model variants."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_policy', ROOT / 'scripts/check_release_tree.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_json_manifest_allows_only_sealed_c7_payloads():
    prefix = 'deployment_bundles/nandu1_v1_6_C7_P_1_R0_model_30000/'
    contents = {
        'deployment_bundles/catalogue.json': json.dumps({'bundles': [{
            'directory': 'nandu1_v1_6_C7_P_1_R0_model_30000', 'manifest_file': 'manifest.json'}]}).encode(),
        prefix + 'manifest.json': json.dumps({'files': {'encoder.onnx': 'hash'}}).encode(),
    }
    allowed = policy.payload_paths(contents)
    assert not policy.violations([prefix + 'manifest.json', prefix + 'encoder.onnx'], allowed)
    assert policy.violations([prefix + 'unlisted.onnx'], allowed)


@pytest.mark.parametrize('path', [
    'tron2_mujoco_sim/outputs/tracking.csv', 'tron2_mujoco_sim/nandu-sim2sim.md',
    'tron2_mujoco_sim/docs/report.md', 'tron2_mujoco_sim/__pycache__/a.pyc',
    '.env', 'deployment_bundles/old_initial_export/policy.onnx', 'weights/policy.onnx',
    'output/report.json', 'cache/session.json', '.venv/private.json', 'tron2_mujoco_sim/run.log',
    'deployment_bundles/nandu1_v1_2_model_40000/model_99999.pt',
    'tron2_rl_deploy_ros/tron2_controllers/config/SF_TRON2A/policy/experimental.onnx',
])
def test_rejects_local_or_unapproved_artifact(path):
    assert policy.violations([path])


def test_accepts_original_licenses_and_selected_runtime_files():
    assert not policy.violations(['tron2_mujoco_sim/LICENSE', 'tron2_rl_deploy_ros/NOTICE',
        'tron2_mujoco_sim/README.md', 'deployment_bundles/nandu1_v1_2_model_40000/policy.onnx',
        'tron2_rl_deploy_ros/tron2_controllers/test/golden/wfyg_math.csv'])


def test_detects_credential_pattern_without_printing_the_value(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, 'ROOT', tmp_path)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    f = tmp_path / 'config.pem'
    f.write_text('-----BEGIN ' + 'OPENSSH PRIVATE KEY-----\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    f.write_text('safe working tree\n')
    with pytest.raises(SystemExit, match='credential-like content'):
        policy.main()


@pytest.mark.parametrize('name', ['settings.conf', 'settings.launch', 'extensionless'])
def test_scans_staged_credentials_independent_of_extension(tmp_path, monkeypatch, name):
    monkeypatch.setattr(policy, 'ROOT', tmp_path)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / name).write_text('-----BEGIN ' + 'RSA PRIVATE KEY-----\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    with pytest.raises(SystemExit, match='credential-like content'):
        policy.main()


def test_unstaged_secret_is_not_mistaken_for_published_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, 'ROOT', tmp_path)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    f = tmp_path / 'settings.conf'
    f.write_text('safe staged data\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    f.write_text('-----BEGIN ' + 'RSA PRIVATE KEY-----\n')
    policy.main()
