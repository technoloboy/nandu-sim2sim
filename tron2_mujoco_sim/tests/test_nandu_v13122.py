"""New v13122 deployment state and fixed-TCP MuJoCo behavior."""
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MATH_SOURCE = ROOT.parent / "tron2_rl_lab/rsl_rl/rsl_rl/nandu_adaptive_balance/math.py"
if not MATH_SOURCE.is_file():MATH_SOURCE = ROOT.parent / 'deployment_bundles/nandu1_v1_3_1_2_2_model_29500/adaptive_math.py'
XML = ROOT.parent / "deployment_bundles/nandu1_v1_3_1_model_30000/assets/robot.xml"


def module():
    assert importlib.util.find_spec("nandu1_v13122") is not None, "v13122 runtime missing"
    return importlib.import_module("nandu1_v13122")


class Encoder(torch.nn.Module):
    def forward(self, history):
        return history[:, -49:-42]


class Context(torch.nn.Module):
    def forward(self, observation, previous_action, hidden, eta):
        feature = torch.cat((observation[:, :36], previous_action), -1)
        next_hidden = hidden * .9 + torch.nn.functional.pad(feature, (0, 15))
        raw_eta = next_hidden[:, :8].tanh()
        next_eta = eta + .0625 * (raw_eta - eta)
        return next_hidden, next_eta, raw_eta


class Policy(torch.nn.Module):
    def forward(self, raw):
        # A real, deterministic actor with state-dependent output and no mock.
        return torch.tanh(raw[:, :13])


@pytest.fixture
def bundle(tmp_path):
    metadata = dict(task="Isaac-NANDU1-Blind-Flat-v1-3-1-2-2",
                    checkpoint_schema="nandu_network_v1_3_1_2_2_E_adaptive_balance_r1",
                    control_dt=.02, curriculum_stage=4, file_sha256={},
                    interface=dict(policy=49, history=245, encoder=7, commands=3,
                                   actor_raw=70, actor_features=73, critic=123, critic_with_commands=126, actions=13))
    (tmp_path / "deployment_metadata.json").write_text(json.dumps(metadata))
    shutil.copyfile(MATH_SOURCE, tmp_path / "adaptive_math.py")
    torch.jit.trace(Encoder(), torch.zeros(1, 245)).save(str(tmp_path / "encoder.pt"))
    torch.jit.trace(Policy(), torch.zeros(1, 70)).save(str(tmp_path / "policy.pt"))
    torch.jit.trace(Context(), (torch.zeros(1, 49), torch.zeros(1, 13), torch.zeros(1, 64), torch.zeros(1, 8))).save(str(tmp_path / "context_step.pt"))
    return tmp_path


def observation():
    x = np.zeros(49, dtype=np.float32)
    x[5] = x[11] = -1.
    x[0] = .02
    return x


def test_runtime_exact_actor_order_initial_state_guard_and_reset(bundle):
    m = module()
    runtime = m.AdaptiveRuntime(bundle, initial_height=.8)
    obs = observation()
    first = runtime.step(obs, [0., .2, .9], step_id=0)
    assert first["actor_input"].shape == (70,)
    np.testing.assert_allclose(first["actor_input"][7:56], obs)
    np.testing.assert_array_equal(first["actor_input"][59:70], 0.)
    frozen = runtime.state_dict()
    repeated = runtime.step(obs + 1., [1., 1., 1.], step_id=0)
    np.testing.assert_array_equal(repeated["action"], first["action"])
    np.testing.assert_array_equal(runtime.state_dict()["hidden"], frozen["hidden"])
    second = runtime.step(obs, [0., .2, .9], step_id=1)
    np.testing.assert_allclose(second["observation"][36:], first["action"])
    assert np.linalg.norm(second["hidden"]) > 0
    assert np.linalg.norm(second["eta"]) > 0
    np.testing.assert_allclose(second["actor_input"][61:69], second["eta"])
    assert second["actor_input"][69] == pytest.approx(second["height_rate"] / .5)
    runtime.reset(initial_height=.6)
    state = runtime.state_dict()
    assert not state["hidden"].any() and not state["eta"].any()
    assert not state["previous_issued_action"].any()
    np.testing.assert_allclose(runtime.processor.command.numpy(), [[0., 0., .6]])


def test_runtime_rejects_old_schema_and_tampered_bundle(bundle):
    m = module()
    path = bundle / "deployment_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["checkpoint_schema"] = "old-E-schema"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="schema"):
        m.AdaptiveRuntime(bundle, initial_height=.8)
    metadata["checkpoint_schema"] = m.SCHEMA
    metadata["file_sha256"] = {"policy.pt": "0" * 64}
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash"):
        m.AdaptiveRuntime(bundle, initial_height=.8)


def test_runtime_context_command_memory_conformance_1000_steps(bundle):
    m = module()
    runtime = m.AdaptiveRuntime(bundle, initial_height=.8)
    math_spec = importlib.util.spec_from_file_location("training_adaptive_math", MATH_SOURCE)
    reference = importlib.util.module_from_spec(math_spec)
    math_spec.loader.exec_module(reference)
    processor = reference.CausalCommandProcessor(1)
    processor.reset(height=.8)
    memory = reference.SlowErrorMemory(1)
    hidden, eta = torch.zeros(1, 64), torch.zeros(1, 8)
    previous = torch.zeros(1, 13)
    context = Context()
    for step in range(1000):
        obs = observation()
        obs[0] = .025 * np.sin(step * .01)
        desired = np.array([.2 if step % 300 < 100 else 0., .1 * np.cos(step * .03), .8 + .03 * np.sin(step * .02)])
        out = runtime.step(obs, desired, step_id=step)
        policy = torch.from_numpy(obs).reshape(1, 49)
        policy[:, 36:] = previous
        command = processor.step(torch.as_tensor(desired, dtype=torch.float32).reshape(1, 3), .02,
                                 gravity_b=policy[:, 3:6], gyro_b=policy[:, :3] / .25, step_id=step)
        if step:
            hidden, eta, _ = context(policy, previous, hidden, eta)
            up = -policy[:, 3:6] / policy[:, 3:6].norm(dim=-1, keepdim=True)
            world_yaw = (policy[:, :3] / .25 * up).sum(-1)
            mem = memory.step_errors(command[:, 1] - policy[:, 3], command[:, 0] - world_yaw,
                                     torch.as_tensor(desired, dtype=torch.float32).reshape(1, 3), processor.height_rate,
                                     tilt=up[:, 2].acos(), dt=.02, step_id=step)
            np.testing.assert_allclose(out["memory"], mem.numpy()[0], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(out["commands"], command.numpy()[0], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(out["hidden"], hidden.numpy()[0], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(out["eta"], eta.numpy()[0], rtol=2e-6, atol=2e-7)
        previous = torch.from_numpy(out["action"]).reshape(1, 13)


def test_mujoco_fixed_tcp_sensor_only_observation_and_nominal_pd():
    pytest.importorskip("mujoco")
    if not XML.exists():
        pytest.skip("Audited original NANDU1 asset absent")
    m = module()
    import mujoco
    import nandu1_contract as c
    physics = m.MujocoPhysics(XML)
    assert physics.model.opt.timestep == .005
    np.testing.assert_allclose(physics.data.qpos[physics.qadr], c.DEFAULT_Q)
    physics.data.qvel[physics.vadr[10]] = .3
    mujoco.mj_forward(physics.model, physics.data)
    truth = physics.truth()
    up = physics.bids["up3"]
    expected = physics.data.xpos[up] + physics.data.xmat[up].reshape(3, 3) @ np.array(m.TCP_LOCAL)
    np.testing.assert_allclose(truth["tcp_position_world"], expected)
    before = physics.observe(np.zeros(13))
    physics.data.qvel[:3] += [.2, -.1, 0.]
    mujoco.mj_forward(physics.model, physics.data)
    np.testing.assert_allclose(physics.observe(np.zeros(13)), before)
    physics.apply(c.DEFAULT_Q + 100., np.zeros(13))
    np.testing.assert_array_equal(physics.data.ctrl[physics.aids[:4]], c.EFFORT[:4])
    for _ in range(4):
        physics.step()
    assert physics.data.time == pytest.approx(.02)


@pytest.mark.parametrize("payload", [( .67148, 0., -.275), (6., 5., -.35)])
def test_mujoco_tcp_transport_velocity_matches_independent_point_jacobian(payload):
    mujoco = pytest.importorskip("mujoco")
    if not XML.exists():
        pytest.skip("Audited original NANDU1 asset absent")
    m = module()
    physics = m.MujocoPhysics(XML, up3_total_mass=payload[0], rear_mass=payload[1], rear_x=payload[2])
    physics.data.qvel[:6] = [.12, -.08, .03, .4, -.3, .2]
    physics.data.qvel[physics.vadr[10:13]] = [.7, -.2, .3]
    mujoco.mj_forward(physics.model, physics.data)
    truth = physics.truth()
    jacp, jacr = np.zeros((3, physics.model.nv)), np.zeros((3, physics.model.nv))
    mujoco.mj_jac(physics.model, physics.data, jacp, jacr,
                  truth["tcp_position_world"], physics.bids["up3"])
    independent_velocity = jacp @ physics.data.qvel
    np.testing.assert_allclose(truth["tcp_velocity_world"], independent_velocity, atol=1e-11, rtol=1e-11)
    rotation = physics.data.xmat[physics.bids["base_Link"]].reshape(3, 3)
    np.testing.assert_allclose(truth["encoder_label"][3:6], rotation.T @ independent_velocity, atol=1e-11, rtol=1e-11)


def test_new_cli_help_is_independent_of_isaac_and_ros():
    import subprocess
    result = subprocess.run([sys.executable, str(ROOT / "scripts/play_nandu_v13122.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--bundle" in result.stdout and "--model-xml" in result.stdout and "--steps" in result.stdout
