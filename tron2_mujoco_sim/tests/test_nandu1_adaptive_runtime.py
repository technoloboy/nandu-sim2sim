"""Regress state advancement, priming and previous-issued action semantics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[2] / 'deployment_bundles/nandu1_v1_3_1_2_6_model_23500'


def runtime():
    assert importlib.util.find_spec('nandu1_adaptive_runtime'), 'stateful E/r2 runtime missing'
    from nandu1_adaptive_runtime import AdaptiveRuntime
    return AdaptiveRuntime(BUNDLE, initial_height=.9, stage=4)


def obs():
    x = np.zeros(49, np.float32)
    x[5] = x[11] = -1
    return x


def test_priming_repetition_and_reset_are_not_extra_gru_steps():
    r = runtime()
    a = r.step(obs(), [0, 0, .9], step_id=0)
    np.testing.assert_array_equal(a['dynamic'], np.zeros(12))
    np.testing.assert_array_equal(a['history'], np.tile(obs(), 5))
    assert a['actor_input'].shape == (71,)
    b = r.step(obs(), [0, 0, .9], step_id=1)
    repeated = r.step(obs(), [0, .8, 1.2], step_id=1)
    np.testing.assert_array_equal(b['hidden'], repeated['hidden'])
    np.testing.assert_array_equal(b['commands'], repeated['commands'])
    np.testing.assert_array_equal(b['observation'][36:], a['action'])
    with pytest.raises(ValueError): r.step(obs(), [0, 0, .9], step_id=0)
    r.reset(initial_height=.9)
    np.testing.assert_array_equal(r.step(obs(), [0, 0, .9], step_id=0)['action'], a['action'])


def test_command_precedes_memory_like_r2_post_physics_observation():
    r = runtime()
    r.step(obs(), [0, 0, .9], step_id=0)
    a = r.step(obs(), [0, .2, 1.], step_id=1)
    assert a['commands'][1] == pytest.approx(.012, abs=1e-7)
    assert a['height_rate'] == pytest.approx(.5, abs=1e-5)
    assert a['dynamic'][10] == pytest.approx(1., abs=2e-6)
    # Height motion gates accumulation, but must not erase the error LPF.
    assert a['dynamic'][11] == 0
    r.report_interval_clipping(action_clipped=True, target_clipped=False)
    frozen = r.step(obs(), [0, 0, 1.], step_id=2)
    assert frozen['dynamic'][11] == a['dynamic'][11]


def test_nonfinite_input_rejected_without_advancing_success_cache():
    r = runtime()
    first = r.step(obs(), [0, 0, .9], step_id=0)
    bad = obs(); bad[0] = np.nan
    with pytest.raises(ValueError): r.step(bad, [0, 0, .9], step_id=1)
    np.testing.assert_array_equal(r.step(obs(), [0, 0, .9], step_id=0)['action'], first['action'])
