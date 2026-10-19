import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_history_and_action_mapping():
    import nandu1_contract as c
    history = c.History()
    first = history.append(np.ones(49))
    assert first.shape == (245,)
    np.testing.assert_array_equal(first, np.ones(245))
    second = history.append(np.full(49, 2))
    np.testing.assert_array_equal(second[:196], np.ones(196))
    np.testing.assert_array_equal(second[196:], np.full(49, 2))
    q, dq = c.decode(np.arange(13, dtype=float))
    assert dq[4] == 11 and dq[9] == 12
    assert q[10] == pytest.approx(-0.15 + 8 * 0.25)
    assert q[5] == pytest.approx(0.475 + 4 * 0.25)
    history.reset()
    np.testing.assert_array_equal(history.append(np.full(49, 3)), np.full(245, 3))


def test_observation_frames_and_finite_rejection():
    import nandu1_contract as c
    obs = c.observe([1, 0, 0, 0], [4, 8, 12], [1, 0, 0, 0], [0, 4, 0],
                    c.DEFAULT_Q, np.arange(13), np.zeros(13))
    np.testing.assert_allclose(obs[:12], [1, 2, 3, 0, 0, -1, 0, 1, 0, 0, 0, -1])
    np.testing.assert_allclose(obs[12:23], 0)
    np.testing.assert_allclose(obs[23:36], np.arange(13)*0.05)
    with pytest.raises(ValueError):
        c.decode(np.full(13, np.nan))


@pytest.mark.parametrize('bundle_name', ['nandu1_v1_2_model_40000', 'nandu1_v1_4_model_24000'])
@pytest.mark.parametrize('sign', [-1, 1])
def test_position_targets_use_inclusive_physical_limits_in_both_versions(bundle_name, sign):
    import json
    import nandu1_contract as c
    bundle = Path(__file__).resolve().parents[2] / 'deployment_bundles' / bundle_name
    task = json.loads((bundle / 'manifest.yaml').read_text())['task']
    lower, upper = np.asarray(task['joint_position_lower']), np.asarray(task['joint_position_upper'])
    action = np.r_[np.full(11, sign * 1000.), 17., -19.]
    q, dq = c.decode(action)
    raw = q.copy()
    clipped = c.clip_position_targets(q, lower, upper)
    np.testing.assert_array_equal(clipped[c.POSITION_IDS], (lower if sign < 0 else upper)[c.POSITION_IDS])
    np.testing.assert_array_equal(clipped[[4, 9]], [0., 0.])
    np.testing.assert_array_equal(dq[[4, 9]], [17., -19.])
    np.testing.assert_array_equal(q, raw)  # no mutation of raw targets/action


def test_position_clip_preserves_boundary_interior_and_nonposition_targets():
    import nandu1_contract as c
    lower, upper = c.DEFAULT_Q - .5, c.DEFAULT_Q + .5
    for endpoint in [lower, upper]:
        np.testing.assert_array_equal(c.clip_position_targets(endpoint, lower, upper), endpoint)
    q = c.DEFAULT_Q.copy()
    q[[4, 9]] = [100., -100.]
    np.testing.assert_array_equal(c.clip_position_targets(q, lower, upper), q)


@pytest.mark.parametrize('invalid', ['nan_target', 'infinite_target', 'nan_limit', 'reversed_limit', 'wrong_size'])
def test_position_clip_does_not_hide_invalid_data(invalid):
    import nandu1_contract as c
    q = c.DEFAULT_Q.copy()
    lower, upper = q - .5, q + .5
    if invalid == 'nan_target': q[10] = np.nan
    if invalid == 'infinite_target': q[10] = np.inf
    if invalid == 'nan_limit': lower[10] = np.nan
    if invalid == 'reversed_limit': lower[10] = upper[10] + 1
    if invalid == 'wrong_size': q = q[:12]
    with pytest.raises(ValueError):
        c.clip_position_targets(q, lower, upper)
