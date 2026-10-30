"""Hand-derived physical metrics must not score against compensated commands."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1]/'scripts/evaluate_nandu1_c7_outer.py'
    assert path.is_file(), 'Paired evaluator missing'
    spec = importlib.util.spec_from_file_location('c7_outer_eval', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def recording(vx=-.01, duration=20.):
    t = np.arange(round(duration/.02))*.02
    vel = np.full(len(t), vx) if np.ndim(vx) == 0 else np.asarray(vx)
    x = np.r_[0., np.cumsum(vel*.02)]
    reference = np.tile([0., 0., .9], (len(t), 1))
    return dict(command_time=t, time=t+.02,
        reference_command=reference, request=reference.copy(), command=reference+np.array([0., .1, .1]),
        tcp_velocity_body_at_command=np.column_stack([vel, np.zeros((len(t), 2))]),
        tcp_velocity_world_at_command=np.column_stack([vel, np.zeros((len(t), 2))]),
        tcp_at_command=np.column_stack([x[:-1], np.zeros(len(t)), np.full(len(t), .9)]),
        tcp=np.column_stack([x[1:], np.zeros(len(t)), np.full(len(t), .9)]),
        yaw_rate_at_command=np.zeros(len(t)), base_quaternion_at_command=np.tile([1., 0., 0., 0.], (len(t), 1)),
        outer_correction=np.zeros((len(t), 3)), outer_integral=np.zeros((len(t), 3)),
        outer_saturated=np.zeros((len(t), 3), bool), integral=np.zeros((len(t), 3)),
        wheel_clipped_pd_steps=np.zeros((len(t), 2)))


def test_constant_drift_has_correct_rms_net_path_and_reference_error():
    result = module().tracking_metrics(recording(), settle_s=0.)
    assert result['vx_mean_m_s'] == pytest.approx(-.01)
    assert result['vx_rms_m_s'] == pytest.approx(.01)
    assert result['vx_rmse_reference_m_s'] == pytest.approx(.01)
    assert result['height_rmse_reference_m'] == pytest.approx(0.)
    assert result['tcp_net_displacement_m'][0] == pytest.approx(-.2)
    assert result['xy_path_length_m'] == pytest.approx(.2)
    assert result['max_20s_xy_drift_m'] == pytest.approx(.2)


def test_opposite_motions_do_not_cancel_rms_or_path_length():
    data = recording(np.r_[np.full(500, .01), np.full(500, -.01)])
    result = module().tracking_metrics(data, settle_s=0.)
    assert result['vx_mean_m_s'] == pytest.approx(0., abs=1e-12)
    assert result['vx_rms_m_s'] == pytest.approx(.01)
    assert result['xy_path_length_m'] == pytest.approx(.2)
    assert result['tcp_net_displacement_m'][0] == pytest.approx(0., abs=1e-12)


def test_settled_metrics_do_not_include_startup_or_one_step_future_truth():
    data = recording(); data['tcp_velocity_body_at_command'][:500, 0] = 10.
    result = module().tracking_metrics(data, settle_s=10.)
    assert result['samples'] == 500
    assert result['vx_mean_m_s'] == pytest.approx(-.01)
    assert result['max_20s_xy_drift_m'] is None


def test_invalid_or_short_trajectories_are_explicit_not_silently_zero_error():
    with pytest.raises(ValueError):module().tracking_metrics(recording(duration=1.), settle_s=10.)
    data = recording(); data['tcp_at_command'][0, 2] = np.nan
    with pytest.raises(ValueError):module().tracking_metrics(data, settle_s=0.)


def test_stage_protocols_retain_positive_negative_and_release_requests():
    protocol = module().protocol_for('vx', .9)
    requests = np.array([entry['command'] for entry in protocol['schedule']])
    assert {-.3, -.1, 0., .1, .3}.issubset(set(requests[:, 1]))
    assert protocol['duration_s'] > protocol['schedule'][-1]['time']


def test_response_delay_requires_continuous_band_and_scores_uncorrected_reference():
    t = np.arange(501)*.02
    actual = np.where(t < 2., 0., .1)
    result = module().response_metrics(t, np.full(len(t), .1), actual, .1, .005)
    assert result['settle_delay_from_event_s'] == pytest.approx(2.)
    assert result['settle_delay_from_reference_s'] == pytest.approx(2.)
    assert result['overshoot_abs'] == pytest.approx(0.)
    assert result['iae_reference'] == pytest.approx(.199)


def test_response_oscillation_is_not_declared_settled_by_last_sample():
    t = np.arange(501)*.02
    actual = .1+.04*np.sin(2*np.pi*t)
    result = module().response_metrics(t, np.full(len(t), .1), actual, .1, .005)
    assert result['settle_delay_from_event_s'] is None


def test_safety_extrema_and_constraints_include_startup_not_only_settled_window():
    data = recording()
    data['base_quaternion_at_command'][0] = [np.cos(.15), np.sin(.15), 0., 0.]
    data['outer_integral'][0, 1] = .1
    data['outer_saturated'][0, 1] = True
    data['outer_safety_override'] = np.zeros((1000, 3), bool)
    data['outer_safety_override'][0, 1] = True
    m = module().tracking_metrics(data, settle_s=10.)
    assert m['samples'] == 500
    assert m['max_base_tilt_deg'] == pytest.approx(np.degrees(.3))
    assert m['max_abs_outer_integral'][1] == pytest.approx(.1)
    assert m['outer_saturation_fraction'][1] == pytest.approx(.001)
    assert m['command_safety_override_fraction'][1] == pytest.approx(.001)
    assert m['safety_window'] == 'full_completed_run_including_startup'


def test_completed_endpoint_pose_is_included_in_safety_maximum():
    data = recording()
    data['base_quaternion'] = data['base_quaternion_at_command'].copy()
    data['base_quaternion'][-1] = [np.cos(.2), np.sin(.2), 0., 0.]
    assert module().tracking_metrics(data)['max_base_tilt_deg'] == pytest.approx(np.degrees(.4))


def test_short_run_keeps_full_metrics_even_when_no_settled_window_exists(tmp_path):
    p = dict(name='short', duration_s=.1, schedule=[dict(time=0., command=[0., 0., .9])])
    m = module().run_case(p, 'off', module().OuterLoopConfig(), tmp_path/'short')
    assert m['outcome'] == 'duration_complete'
    assert 'metrics_full' in m and m['metrics_full']['samples'] == 5
    assert 'metrics_error' in m
