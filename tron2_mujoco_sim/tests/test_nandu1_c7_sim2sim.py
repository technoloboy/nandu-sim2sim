"""C7 bundle identity/integrity and actual closed-loop state regressions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from nandu1_c7_sim2sim import C7Simulation, bundle_for, validate_bundle


def fixture_bundle(tmp_path):
    files = {}
    for name in ("encoder.onnx", "actor.onnx", "processor/__init__.py",
                 "processor/deployment.py", "processor/mujoco_adapter.py",
                 "simulator_sources/nandu1_contract.py", "simulator_sources/nandu1_v131.py",
                 "assets/robot.xml"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
        files[name] = hashlib.sha256(b"fixture").hexdigest()
    meta = dict(task_name="Isaac-NANDU1-Blind-Flat-v1-6-C7-P-1-R0",
                checkpoint_schema="nandu_drl_v1_6_C7_P_1_R0_online_r4",
                policy_dim=49, history_dim=490, encoder_dim=7, actor_dim=86,
                action_dim=13, control_dt=.02, checkpoint_iteration=30000,
                sim2sim_format="nandu_c7_r012_mujoco_v1", files=files)
    (tmp_path / "manifest.json").write_text(json.dumps(meta))
    return tmp_path


def test_wrong_task_rejected_before_network_loading(tmp_path):
    with pytest.raises(ValueError, match="task"):
        validate_bundle(fixture_bundle(tmp_path), "R1")


def test_tampered_simulator_dependency_rejected(tmp_path):
    bundle = fixture_bundle(tmp_path)
    (bundle / "simulator_sources/nandu1_contract.py").write_text("modified")
    with pytest.raises(ValueError, match="integrity"):
        validate_bundle(bundle, "R0")


def test_manifest_cannot_hash_files_outside_bundle(tmp_path):
    bundle = fixture_bundle(tmp_path / "bundle")
    (tmp_path / "outside").write_bytes(b"fixture")
    meta = json.loads((bundle / "manifest.json").read_text())
    meta["files"]["../outside"] = hashlib.sha256(b"fixture").hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="path"):
        validate_bundle(bundle, "R0")


def test_missing_physics_dependency_rejected(tmp_path):
    bundle = fixture_bundle(tmp_path)
    meta = json.loads((bundle / "manifest.json").read_text())
    del meta["files"]["processor/mujoco_adapter.py"]
    (bundle / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="missing"):
        validate_bundle(bundle, "R0")


@pytest.mark.parametrize("version", ["R0", "R1", "R2"])
def test_real_bundle_executes_and_reset_clears_policy_history(version):
    path = bundle_for(version)
    if not path.is_dir():
        pytest.skip("Real 30000 bundle has not yet been exported")
    sim = C7Simulation(path, version)
    assert sim.policy.shape == (1, 49)
    assert sim.history.shape == (1, 490)
    initial = sim.physics.data.qpos.copy()
    assert np.allclose(sim.physics.model.jnt_range[sim.physics.ids][[3, 8]], [[.2, 1.2], [.2, 1.2]])
    assert sim.physics.model.jnt_range[sim.physics.ids[12], 1] == pytest.approx(2.24)
    assert sim.physics.model.opt.timestep * sim.physics.pd_substeps == pytest.approx(.005)
    for _ in range(4):
        row = sim.step([0., 0., .9])
    assert np.isfinite(row["action"]).all()
    # Feedback must be the interval-mean executed wheel increment, not the requested packet.
    np.testing.assert_allclose(sim.policy.numpy()[0, -13:], sim.physics.executed_action, atol=1e-6)
    np.testing.assert_allclose(sim.history.numpy()[0, -49:], sim.policy.numpy()[0])
    assert sim.controller.complete_motor_records
    sim.reset()
    np.testing.assert_allclose(sim.physics.data.qpos, initial)
    assert sim.step_index == 0
    assert not sim.policy.numpy()[0, -13:].any()
    np.testing.assert_allclose(sim.history.numpy().reshape(10, 49), np.repeat(sim.policy.numpy(), 10, axis=0))
    assert sim.controller.state.integral.abs().max().item() == 0


def test_nonfinite_request_cannot_poison_controller_state():
    path = bundle_for("R0")
    if not path.is_dir():
        pytest.skip("Real bundle has not yet been exported")
    sim = C7Simulation(path, "R0")
    with pytest.raises(ValueError, match="finite"):
        sim.step([0., float("nan"), .9])
    assert sim.step_index == 0


def test_mouse_circle_maps_forward_left_and_retains_height():
    from nandu1_c7_mouse import pointer_command
    np.testing.assert_allclose(pointer_command(0, -100, 100, 4, .8), [0., 1., .8])
    np.testing.assert_allclose(pointer_command(-100, 0, 100, 4, .8), [1.57, 0., .8])
    np.testing.assert_allclose(pointer_command(0, 0, 100, 4, .8), [0., 0., .8])
    np.testing.assert_allclose(pointer_command(1000, 1000, 100, 0, .8),
                               [-.2/np.sqrt(2), -.08/np.sqrt(2), .8])


def test_logged_pose_and_next_observation_use_completed_physics_timestamp():
    import mujoco
    path = bundle_for("R0")
    if not path.is_dir():
        pytest.skip("Real bundle has not yet been exported")
    sim = C7Simulation(path, "R0")
    row = sim.step([0., 0., .9])
    observed = sim.policy.numpy().copy()
    # mj_step integrates qpos last; derived body/sensor tables otherwise lag one physics tick.
    mujoco.mj_forward(sim.physics.model, sim.physics.data)
    np.testing.assert_allclose(row["tcp"], sim.physics.truth()["tcp"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(observed[0], sim.physics.measurement(sim.physics.executed_action), rtol=0, atol=1e-8)


def test_standard_ground_changes_display_without_changing_physics_or_bundle():
    path = bundle_for("R0")
    if not path.is_dir():
        pytest.skip("Real bundle has not yet been exported")
    original_digest = hashlib.sha256((path / "assets/robot.xml").read_bytes()).hexdigest()
    plain = C7Simulation(path, "R0")
    decorated = C7Simulation(path, "R0", standard_scene=True)
    a, b = plain.physics.model, decorated.physics.model
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "jnt_range",
                 "geom_type", "geom_size", "geom_pos", "geom_friction", "geom_contype",
                 "geom_conaffinity", "dof_armature", "actuator_ctrlrange"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert b.ntex > a.ntex and b.nmat > a.nmat
    for _ in range(5):
        plain.step([0., 0., .9])
        decorated.step([0., 0., .9])
    np.testing.assert_array_equal(plain.physics.data.qpos, decorated.physics.data.qpos)
    assert hashlib.sha256((path / "assets/robot.xml").read_bytes()).hexdigest() == original_digest
