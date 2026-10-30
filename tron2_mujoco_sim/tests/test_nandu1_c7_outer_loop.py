"""Truth outer feedback signs, frames, bounded state and time invariants."""
import copy
import importlib
import json

import numpy as np
import pytest


def api():
    return importlib.import_module('nandu1_c7_outer_loop')


def truth(velocity=(0., 0., 0.), height=.9, quaternion=(1., 0., 0., 0.)):
    return dict(tcp=np.array([0., 0., height]), tcp_velocity_api=np.array(velocity),
                base_quaternion=np.array(quaternion), yaw_rate=0.)


def controller(**overrides):
    fields = dict(warmup_s=0., ramp_s=0., feedback_tau=0., velocity_ki=0., height_ki=0.,
                  velocity_correction_rate=100., height_correction_rate=100.)
    fields.update(overrides)
    c = api().TruthOuterLoop(api().OuterLoopConfig(**fields),
        [-1.57, -1., .4], [1.57, 1., 1.4], [2., 100., 100.])
    c.reset([0., 0., .9])
    return c


@pytest.mark.parametrize('velocity,quaternion', [
    ((0., .2, 0.), (2**-.5, 0., 0., 2**-.5)),
    ((0., 0., -.2), (2**-.5, 0., 2**-.5, 0.)),
])
def test_world_feedback_is_projected_to_full_base_not_world_x(velocity, quaternion):
    c = controller(velocity_kp=.5, height_kp=0., height_damping=0.)
    row = c.update([0., 0., .9], truth(velocity, quaternion=quaternion), 0)
    assert row['feedback'][1] == pytest.approx(.2)
    assert row['command'][1] == pytest.approx(-.1)


def test_feedback_corrects_negative_speed_and_low_height_with_positive_commands():
    c = controller(velocity_kp=.5, height_kp=.25, height_damping=0.)
    row = c.update([0., 0., .9], truth((-.1, 0., 0.), .8), 0)
    np.testing.assert_allclose(row['command'], [0., .05, .925], atol=1e-12)


def test_height_damping_uses_world_vertical_not_rotated_body_vertical():
    c = controller(velocity_kp=0., height_kp=0., height_damping=.1)
    row = c.update([0., 0., .9], truth((0., 0., .2), quaternion=(2**-.5, 0., 2**-.5, 0.)), 0)
    assert row['command'][2] == pytest.approx(.88)


def test_zero_error_does_not_create_commands_or_mutate_inputs():
    c = controller()
    reference = np.array([0., 0., .9]); measured = truth(); before = copy.deepcopy(measured)
    row = c.update(reference, measured, 0)
    np.testing.assert_array_equal(row['command'], reference)
    np.testing.assert_array_equal(row['integral'], [0., 0., 0.])
    for k in measured:np.testing.assert_array_equal(measured[k], before[k])
    row['command'][1] = 999
    np.testing.assert_array_equal(c.update(reference, measured, 0)['command'], reference)


def test_duplicate_step_does_not_integrate_and_reset_removes_bias():
    c = controller(velocity_kp=0., velocity_ki=1.)
    first = c.update([0., 0., .9], truth((-.1, 0., 0.)), 0)
    assert first['integral'][1] == pytest.approx(.002)
    np.testing.assert_array_equal(c.update([0., 0., .9], truth((-.1, 0., 0.)), 0)['integral'], first['integral'])
    c.reset([0., 0., .9])
    np.testing.assert_array_equal(c.update([0., 0., .9], truth(), 0)['integral'], np.zeros(3))


@pytest.mark.parametrize('step', [-1, 2, True, .5])
def test_invalid_or_skipped_steps_are_rejected_without_changing_state(step):
    c = controller()
    c.update([0., 0., .9], truth(), 0)
    with pytest.raises(ValueError):c.update([0., 0., .9], truth(), step)
    np.testing.assert_array_equal(c.update([0., 0., .9], truth(), 1)['integral'], np.zeros(3))


def test_amplitude_slew_and_correction_limits_all_constrain_final_output():
    cfg = api().OuterLoopConfig(warmup_s=0., ramp_s=0., feedback_tau=0.,
        velocity_kp=100., velocity_ki=1., height_kp=100., velocity_correction_limit=.03,
        height_correction_limit=.02, velocity_correction_rate=.1, height_correction_rate=.05)
    c = api().TruthOuterLoop(cfg, [-.2, -.08, .78], [.2, .08, 1.04], [2., .6, .05])
    c.reset([0., 0., .9])
    previous = np.array([0., 0., .9])
    for step in range(150):
        r = c.update([0., .08, 1.04], truth((-.5, 0., 0.), .5), step)
        assert np.all(r['command'] <= [.2, .08, 1.04])
        assert np.all(r['command'] >= [-.2, -.08, .78])
        assert np.all(np.abs(r['command'] - previous) <= np.array([2., .6, .05])*.02 + 1e-12)
        np.testing.assert_allclose(r['rate'], (r['command']-previous)/.02, atol=1e-12)
        assert abs(r['integral'][1]) <= cfg.velocity_integral_limit
        previous = r['command']
    assert r['saturated'].any()


def test_saturated_integral_can_unwind_when_error_changes_direction():
    c = controller(velocity_kp=0., velocity_ki=.5, velocity_correction_limit=.01)
    for step in range(200):row = c.update([0., 0., .9], truth((-.5, 0., 0.)), step)
    initial = row['integral'][1]
    assert initial > 0
    for step in range(200, 220):row = c.update([0., 0., .9], truth((.5, 0., 0.)), step)
    assert row['integral'][1] < initial
    assert row['command'][1] < 0


def test_zero_integral_gain_really_disables_integral_even_during_saturation():
    c = controller(velocity_kp=100., velocity_ki=0., height_kp=100., height_ki=0.)
    for step in range(40):
        row = c.update([0., 1., 1.4], truth((-.5, 0., 0.), .5), step)
        np.testing.assert_array_equal(row['integral'], np.zeros(3))


def test_clipped_correction_does_not_accumulate_a_hidden_slew_tail():
    c = controller(height_kp=1., height_ki=0., height_damping=0., height_correction_rate=.001)
    c.reset([0., 0., 1.4])
    for step in range(100):c.update([0., 0., 1.4], truth(height=1.), step)
    row = c.update([0., 0., 1.39], truth(height=1.39), 100)
    assert row['command'][2] == pytest.approx(1.39)


def test_velocity_bias_adaptation_freezes_during_turning_but_keeps_learned_trim():
    c = controller(velocity_kp=0., velocity_ki=.1)
    for step in range(20):row = c.update([0., 0., .9], truth((-.1, 0., 0.)), step)
    learned = row['integral'][1]
    assert learned > 0
    for step in range(20, 120):row = c.update([.6, 0., .9], truth((-.1, 0., 0.)), step)
    assert row['integral'][1] == pytest.approx(learned)
    assert not row['adapting'][1]
    row = c.update([0., 0., .9], truth((-.1, 0., 0.)), 120)
    assert row['integral'][1] > learned and row['adapting'][1]


def test_command_safety_can_override_candidate_slew_and_is_explicitly_reported():
    c = controller(velocity_kp=0., velocity_ki=.1, velocity_correction_rate=.01)
    c.reset([0., .99, .9])
    for step in range(100):row = c.update([0., .99, .9], truth(), step)
    previous = row['correction'][1]
    row = c.update([0., .999, .9], truth(), 100)
    assert 'candidate_correction' in row and 'correction_anchor' in row
    assert row['command'][1] == pytest.approx(1.)
    assert row['correction'][1] == pytest.approx(.001)
    assert abs(row['correction'][1]-previous)/.02 > .01
    assert abs(row['candidate_correction'][1]-row['correction_anchor'][1]) <= .01*.02+1e-12
    assert row['safety_override'][1] and row['command_amplitude_limited'][1]


def test_finite_arithmetic_overflow_is_rejected_without_committing_state():
    c = controller(velocity_kp=1e308)
    with pytest.raises((ValueError, FloatingPointError)):
        c.update([0., 0., .9], truth((-2., 0., 0.)), 0)
    np.testing.assert_array_equal(c.update([0., 0., .9], truth(), 0)['command'], [0., 0., .9])


def test_warmup_does_not_integrate_or_suddenly_apply_feedback():
    c = controller(warmup_s=.1, ramp_s=.1, velocity_kp=.5, velocity_ki=1.)
    for step in range(5):
        r = c.update([0., 0., .9], truth((-.1, 0., 0.)), step)
        assert r['command'][1] == 0 and r['integral'][1] == 0 and not r['ready']
    r = c.update([0., 0., .9], truth((-.1, 0., 0.)), 5)
    assert r['command'][1] == 0
    for step in range(6, 12):r = c.update([0., 0., .9], truth((-.1, 0., 0.)), step)
    assert r['ready'] and r['integral'][1] > 0


@pytest.mark.parametrize('kwargs', [{'velocity_kp': -1.}, {'height_ki': float('nan')},
    {'feedback_tau': True}, {'warmup_s': '1'}, {'velocity_integral_limit': float('inf')}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):api().OuterLoopConfig(**kwargs)


def test_unknown_json_fields_are_rejected(tmp_path):
    p = tmp_path/'config.json'; p.write_text(json.dumps({'unknown_gain': .1}))
    with pytest.raises(ValueError):api().OuterLoopConfig.from_file(p)


@pytest.mark.parametrize('bad', [dict(tcp=np.array([0., 0., np.nan])),
    dict(base_quaternion=np.zeros(4)), dict(base_quaternion=np.array([1e308, 1e308, 0., 0.])),
    dict(tcp_velocity_api=np.ones(2)), dict(yaw_rate=np.inf)])
def test_invalid_truth_cannot_poison_later_valid_update(bad):
    c = controller(); m = truth(); m.update(bad)
    with pytest.raises(ValueError):c.update([0., 0., .9], m, 0)
    np.testing.assert_array_equal(c.update([0., 0., .9], truth(), 0)['command'], [0., 0., .9])
