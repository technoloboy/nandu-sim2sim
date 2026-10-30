"""Real frozen-bundle integration, not a mock controller or modified policy."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from nandu1_c7_physics import C7Simulation, bundle_for
from nandu1_c7_outer_loop import OuterLoopConfig

ROOT = Path(__file__).resolve().parents[1]


def config():
    return OuterLoopConfig(warmup_s=0., ramp_s=0., feedback_tau=0., velocity_kp=.5,
                           velocity_correction_rate=1., velocity_ki=0., height_ki=0.)


@pytest.mark.parametrize('version', ['R1', 'R2'])
def test_non_r0_truth_mode_is_rejected_before_loading_models(tmp_path, version):
    with pytest.raises(ValueError, match='R0'):
        C7Simulation(tmp_path, version, outer_loop='truth')


def test_outer_config_requires_truth_mode():
    with pytest.raises(ValueError, match='outer'):
        C7Simulation(bundle_for('R0'), 'R0', outer_config=config())


@pytest.mark.parametrize('stage', range(5))
def test_every_course_stage_preserves_original_command_limits(stage):
    torch.set_num_threads(1)
    sim = C7Simulation(bundle_for('R0'), 'R0', stage=stage, outer_loop='truth')
    previous = sim.controller.processor.command.numpy()[0].copy()
    for _ in range(4):
        row = sim.step([10., 10., 10.])
        assert np.all(row['command'] >= sim.outer.lower-1e-6)
        assert np.all(row['command'] <= sim.outer.upper+1e-6)
        assert np.all(np.abs(row['command']-previous) <= sim.outer.rate_limits*.02+1e-6)
        previous = row['command']


def test_truth_mode_preserves_reference_tensor_intent_and_actual_issued_rate():
    torch.set_num_threads(1)
    sim = C7Simulation(bundle_for('R0'), 'R0', outer_loop='truth', outer_config=config())
    request = np.array([0., 0., .9], dtype=np.float32)
    previous = sim.controller.processor.command.numpy()[0].copy()
    issued_step = sim.controller.issued_step
    captured = []
    def capture(policy, history, command, rate, step_id, dt=.02, intent=None):
        captured.append((command.numpy()[0].copy(), rate.numpy()[0].copy(), intent.numpy()[0].copy()))
        return issued_step(policy, history, command, rate, step_id, dt, intent=intent)
    sim.controller.issued_step = capture
    for _ in range(30):
        pre = sim.physics.truth()
        row = sim.step(request)
        np.testing.assert_array_equal(row['request'], request)
        np.testing.assert_array_equal(captured[-1][2], request)
        np.testing.assert_array_equal(row['reference_command'], sim.controller.processor.command.numpy()[0])
        np.testing.assert_allclose(captured[-1][1], (row['command']-previous)/.02, atol=1e-5)
        np.testing.assert_array_equal(captured[-1][0], row['command'])
        np.testing.assert_array_equal(row['tcp_velocity_world_at_command'], pre['tcp_velocity_api'])
        np.testing.assert_array_equal(row['base_quaternion_at_command'], pre['base_quaternion'])
        assert row['wheel_clipped_pd_steps'].shape == (2,)
        assert 'outer_candidate_correction' in row and 'outer_safety_override' in row
        assert 'outer_command_amplitude_limited' in row and 'outer_command_slew_limited' in row
        previous = row['command'].copy()
    assert np.max(np.abs(row['outer_correction'])) > 0
    assert np.all(np.isfinite(row['outer_integral']))


def test_reset_recreates_first_outer_step_and_clears_bias():
    torch.set_num_threads(1)
    sim = C7Simulation(bundle_for('R0'), 'R0', outer_loop='truth', outer_config=config())
    initial = sim.step([0., 0., .9])
    for _ in range(8):sim.step([0., .1, 1.])
    sim.reset()
    after = sim.step([0., 0., .9])
    for name in ['command', 'reference_command', 'outer_integral', 'outer_correction', 'action', 'tcp']:
        np.testing.assert_array_equal(after[name], initial[name])


def test_invalid_truth_is_rejected_before_advancing_frozen_processor():
    sim = C7Simulation(bundle_for('R0'), 'R0', outer_loop='truth')
    read_truth = sim.physics.truth
    def invalid_truth():
        row = read_truth(); row['tcp_velocity_api'][0] = np.nan
        return row
    sim.physics.truth = invalid_truth
    with pytest.raises(ValueError):sim.step([0., 0., .9])
    assert sim.controller.processor.last_step is None
    sim.physics.truth = read_truth
    assert sim.step([0., 0., .9])['command_time'] == 0.


def test_mouse_status_distinguishes_reference_final_and_actual_without_display():
    import tkinter as tk
    from nandu1_c7_mouse import MousePanel
    interp = tk.Tcl()
    panel = object.__new__(MousePanel)
    panel.status = tk.StringVar(master=interp)
    panel.outer_mode = 'truth'
    panel.update(dict(request=[0., 0., .9], command=[0., .01, .901],
        reference_command=[0., 0., .9], outer_saturated=[False]*3, outer_ready=True,
        yaw_rate=0., tcp_velocity_body=[.001, 0., 0.], tcp=[0., 0., .9001], time=10.), 1.)
    text = panel.status.get()
    assert 'Joy request' in text and 'Reference' in text and 'Actor command' in text and 'MuJoCo truth' in text
    assert '+0.010' in text and 'outer=truth' in text


def test_disabled_runtime_matches_original_frozen_source_snapshot():
    # Immutable runtime from release commit 272a947; ships with the tests so
    # fresh checkouts never skip this check or compare the adapter to itself.
    old = ROOT/'tests/fixtures/nandu1_c7_pre_outer_sim2sim.py'
    assert old.is_file(), 'Independent pre-outer runtime fixture is missing'
    assert hashlib.sha256(old.read_bytes()).hexdigest() == 'b86a77ba4ddc2df1d7af5c244cf239e642d8998dce618a9220791072bc363b74'
    spec = importlib.util.spec_from_file_location('c7_pre_outer_runtime', old)
    baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    torch.set_num_threads(1)
    a = baseline.C7Simulation(bundle_for('R0'), 'R0')
    b = C7Simulation(bundle_for('R0'), 'R0', outer_loop='off', physics_profile='legacy')
    for k in range(50):
        request = [0.1 if k > 25 else 0., .1 if k > 10 else 0., .9]
        before, after = a.step(request), b.step(request)
        for name in before:np.testing.assert_array_equal(after[name], before[name], err_msg=name)
        np.testing.assert_array_equal(a.physics.data.qpos, b.physics.data.qpos)


def test_cli_truth_mode_records_identity_and_request_reference_final(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'scripts/play_nandu1_c7.py'), '--version', 'R0',
        '--outer-loop', 'truth', '--headless', '--duration', '.1', '--output', str(tmp_path/'run')],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    meta = json.loads((tmp_path/'run/manifest.json').read_text())
    assert meta['outer_loop']['mode'] == 'truth'
    assert meta['outer_loop']['sim_truth_feedback'] is True
    assert meta['outer_loop']['config']['velocity_kp'] >= 0
    assert meta['clocks_s'] == dict(policy=.02, pd=.005, physics=.005)
    assert meta['physics']['integrator'] == 'mjINT_EULER'
    assert meta['physics']['profile'] == 'robojudo'
    assert meta['source_sha256']['nandu1_c7_outer_loop.py'] == hashlib.sha256((ROOT/'nandu1_c7_outer_loop.py').read_bytes()).hexdigest()
    with np.load(tmp_path/'run/trajectory.npz') as data:
        assert data['reference_command'].shape == data['request'].shape == (5, 3)
        assert data['tcp_velocity_world_at_command'].shape == (5, 3)
        np.testing.assert_allclose(data['time']-data['command_time'], .02, atol=1e-10)
