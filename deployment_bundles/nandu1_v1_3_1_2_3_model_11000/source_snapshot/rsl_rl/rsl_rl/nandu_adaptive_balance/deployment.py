"""Strict CPU export of the v13122 Actor and explicit 50Hz Student state."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import torch
from torch import nn

from .adaptation import AdaptiveBalanceLearner, StudentContext
from .contract import AdaptiveBalanceExperiment, checkpoint_experiment
from .runner import build_models
from ..nandu_experiments.contract import Experiment


class ContextStep(nn.Module):
    """Deployment graph output order is hidden64, filtered8, raw8."""
    def __init__(self, student: StudentContext):
        super().__init__()
        self.student = student

    def forward(self, observation: torch.Tensor, previous_issued_action: torch.Tensor,
                hidden: torch.Tensor, filtered_eta: torch.Tensor):
        features = torch.cat((observation[:, :36], previous_issued_action), -1)
        raw, next_hidden, next_eta = self.student(features, hidden, filtered_eta, .02)
        return next_hidden, next_eta, raw


class Pipeline(nn.Module):
    def __init__(self, encoder, actor, student):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.context_step = ContextStep(student)

    def forward(self, history: torch.Tensor, observation: torch.Tensor,
                commands: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        return self.actor(torch.cat((self.encoder(history), observation, commands, dynamic), -1))


def load_pipeline(path, _state=None):
    state = torch.load(path, map_location="cpu", weights_only=False) if _state is None else _state
    exp = checkpoint_experiment(state)
    if not isinstance(exp, AdaptiveBalanceExperiment):
        raise ValueError("Adaptive export requires an adaptive-balance checkpoint")
    exp.validate_checkpoint(state)
    adaptation_config = state["adaptation_state_dict"]["config"]
    if adaptation_config["dt"] != .02 or adaptation_config["latent_tau"] != .3:
        raise ValueError("Adaptive export requires the fixed 50Hz, tau=.3 runtime profile")
    cfg = state["train_cfg"]
    encoder, actor_critic = build_models(exp, cfg.get("seed", 42), cfg["encoder"], cfg["policy"])
    learner = AdaptiveBalanceLearner(adaptation_config)
    learner.validate_checkpoint_state(state["adaptation_state_dict"])
    encoder.load_state_dict(state["encoder_state_dict"], strict=True)
    actor_critic.load_state_dict(state["model_state_dict"], strict=True)
    learner.load_checkpoint_state(state["adaptation_state_dict"])
    return exp, state, Pipeline(encoder, actor_critic.actor, learner.student).eval()


_PROCESSING_REFERENCE = '''"""Standalone v13122 reference; no training package or simulator imports.

Command step runs once before the next physical interval. Observe runs once
after that interval, using its previous clipped issued action. Read properties
freely; they never advance state. Clear every temporal state for new physics.
"""
import importlib.util
from pathlib import Path
import torch

_spec = importlib.util.spec_from_file_location("exported_v13122_math", Path(__file__).with_name("adaptive_math.py"))
_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_math)
CausalCommandProcessor = _math.CausalCommandProcessor
SlowErrorMemory = _math.SlowErrorMemory
CONTROL_DT = .02
STUDENT_TAU = .3


class StatefulProcessor:
    def __init__(self, num_envs=1, device="cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.commands = CausalCommandProcessor(num_envs, self.device)
        self.memory = SlowErrorMemory(num_envs, self.device)
        self.hidden = torch.zeros(num_envs, 64, device=self.device)
        self.filtered_eta = torch.zeros(num_envs, 8, device=self.device)
        self.previous_issued_action = torch.zeros(num_envs, 13, device=self.device)
        self.raw_intent = torch.zeros(num_envs, 3, device=self.device)
        self._dynamic = torch.zeros(num_envs, 11, device=self.device)
        self._command_step_id = None
        self._context_step_id = None

    @property
    def dynamic(self):
        return self._dynamic.detach().clone()

    @property
    def command(self):
        return self.commands.command

    @torch.no_grad()
    def reset(self, env_ids=None, height=.9):
        self.commands.reset(env_ids, height=height)
        self.memory.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for name in ("hidden", "filtered_eta", "previous_issued_action", "raw_intent", "_dynamic"):
            value = getattr(self, name).clone()
            value[ids] = 0.
            setattr(self, name, value)
        if env_ids is None:
            self._command_step_id = None
            self._context_step_id = None

    @torch.no_grad()
    def command_step(self, goal, observation, stage=4, step_id=None):
        if step_id is not None and self._command_step_id is not None:
            if step_id < self._command_step_id:
                raise ValueError("command step_id must not move backwards")
            if step_id == self._command_step_id:
                return self.command
        self.raw_intent = goal.detach().to(self.device).clone()
        applied = self.commands.step(self.raw_intent, CONTROL_DT, step_id=step_id,
            gravity_b=observation[:, 3:6], gyro_b=observation[:, :3] / .25, stage=stage)
        if step_id is not None:
            self._command_step_id = step_id
        return applied

    @torch.no_grad()
    def observe(self, observation, encoded, previous_issued_action, context_model,
                reset_mask=None, known_clipping=False, step_id=None):
        if step_id is not None and self._context_step_id is not None:
            if step_id < self._context_step_id:
                raise ValueError("context step_id must not move backwards")
            if step_id == self._context_step_id:
                return self.dynamic
        observation = observation.detach().to(self.device)
        encoded = encoded.detach().to(self.device)
        issued = previous_issued_action.detach().to(self.device).clone()
        reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if reset_mask is not None:
            reset = reset_mask.reshape(-1).to(device=self.device, dtype=torch.bool)
            ids = reset.nonzero().flatten()
            self.memory.reset(ids)
            self.hidden = torch.where(reset[:, None], torch.zeros_like(self.hidden), self.hidden)
            self.filtered_eta = torch.where(reset[:, None], torch.zeros_like(self.filtered_eta), self.filtered_eta)
            issued[reset] = 0.
        # Reset rows are priming observations, with no preceding physical step.
        valid = torch.isfinite(observation[:, :36]).all(-1) & torch.isfinite(issued).all(-1) & ~reset
        next_hidden, next_eta, _ = context_model(torch.nan_to_num(observation), torch.nan_to_num(issued),
                                                 self.hidden, self.filtered_eta)
        self.hidden = torch.where(valid[:, None], next_hidden, self.hidden)
        self.filtered_eta = torch.where(valid[:, None], next_eta, self.filtered_eta)
        self.previous_issued_action = issued
        gravity = observation[:, 3:6]
        norm = gravity.norm(dim=-1)
        up = -gravity / norm[:, None].clamp_min(1e-6)
        yaw = (observation[:, :3] / .25 * up).sum(-1)
        tilt = up[:, 2].clamp(-1., 1.).acos()
        good_measurement = valid & torch.isfinite(encoded).all(-1) & (norm > 1e-6) & ~reset
        hdot = self.commands.height_rate
        applied = self.commands.command
        memory = self.memory.step_errors(applied[:, 1] - encoded[:, 3], applied[:, 0] - yaw,
            self.raw_intent, hdot, valid=good_measurement, tilt=tilt, clipped=known_clipping,
            dt=CONTROL_DT, step_id=step_id)
        self._dynamic = torch.cat((memory, self.filtered_eta, hdot[:, None] / .5), -1)
        self._dynamic[reset] = 0.
        if step_id is not None:
            self._context_step_id = step_id
        return self.dynamic
'''


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_source(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_bundle(checkpoint, output, onnx=True, _loaded=None):
    """Export only inference networks, plus an audited standalone state reference."""
    import numpy as np
    exp, state, pipeline = load_pipeline(checkpoint) if _loaded is None else _loaded
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new export directory; existing bundles are not overwritten")
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[3]
    mathdir = root / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/NANDU1"
    baseline = _load_source(mathdir / "nandu1_v1_3_1_contract.py").export_metadata()
    shutil.copyfile(Path(__file__).with_name("math.py"), output / "adaptive_math.py")
    for name in ("nandu1_v1_3_1_contract.py", "nandu1_v1_3_1_math.py"):
        shutil.copyfile(mathdir / name, output / name)
    ablation = hasattr(exp, "height_memory_enabled")
    if ablation:
        shutil.copyfile(Path(__file__).with_name("height_memory.py"), output / "adaptive_height_memory.py")
        shutil.copyfile(Path(__file__).with_name("ablation_processing_reference.py"), output / "processing_reference.py")
    else:
        (output / "processing_reference.py").write_text(_PROCESSING_REFERENCE, encoding="utf-8")
    generator = torch.Generator().manual_seed(13122)
    history = torch.randn(8, 245, generator=generator) * .1
    observation = torch.randn(8, 49, generator=generator) * .1
    observation[:, 5] = -1.
    commands = torch.zeros(8, 3)
    commands[:, 2] = torch.linspace(.4, 1.4, 8)
    dynamic = torch.randn(8, exp.interface["actor_raw"] - 59, generator=generator) * .1
    issued = torch.randn(8, 13, generator=generator).clamp(-1., 1.)
    hidden = torch.randn(8, 64, generator=generator) * .1
    filtered_eta = torch.randn(8, 8, generator=generator) * .1
    with torch.no_grad():
        encoded = pipeline.encoder(history)
        raw = torch.cat((encoded, observation, commands, dynamic), -1)
        expected_actions = pipeline(history, observation, commands, dynamic)
        expected_context = pipeline.context_step(observation, issued, hidden, filtered_eta)
    metrics = {}
    graphs = (
        ("encoder", pipeline.encoder, (history,), ["history"], ["encoded"]),
        ("policy", pipeline.actor, (raw,), ["raw_actor_input"], ["actions"]),
        ("actor_pipeline", pipeline, (history, observation, commands, dynamic),
         ["history", "observation", "commands", "dynamic"], ["actions"]),
        ("context_step", pipeline.context_step, (observation, issued, hidden, filtered_eta),
         ["observation", "previous_issued_action", "hidden", "filtered_eta"],
         ["next_hidden", "next_filtered_eta", "raw_eta"]))
    for name, model, values, names, outputs in graphs:
        scripted = torch.jit.script(model)
        scripted.save(str(output / f"{name}.pt"))
        with torch.no_grad():
            reference = model(*values)
            actual = torch.jit.load(str(output / f"{name}.pt"))(*values)
        reference = reference if isinstance(reference, tuple) else (reference,)
        actual = actual if isinstance(actual, tuple) else (actual,)
        for left, right in zip(actual, reference):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
        metrics[name + "_jit_max_error"] = max(float((left - right).abs().max())
                                               for left, right in zip(actual, reference))
        if onnx:
            import onnx as onnxlib
            torch.onnx.export(model, tuple(x[:1] for x in values), str(output / f"{name}.onnx"),
                input_names=names, output_names=outputs, opset_version=17, dynamo=False,
                dynamic_axes={key: {0: "batch"} for key in names + outputs})
            onnxlib.checker.check_model(onnxlib.load(str(output / f"{name}.onnx")))
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(output / f"{name}.onnx"), providers=["CPUExecutionProvider"])
                metrics["onnx_validation_engine"] = "onnxruntime"
            except ImportError:
                from onnx.reference import ReferenceEvaluator
                session = ReferenceEvaluator(str(output / f"{name}.onnx"))
                metrics["onnx_validation_engine"] = "onnx_reference_evaluator"
            result = session.run(None, {key: value.numpy() for key, value in zip(names, values)})
            for left, right in zip(result, reference):
                np.testing.assert_allclose(left, right.numpy(), atol=1e-5, rtol=1e-4)
            metrics[name + "_onnx_max_error"] = max(float(np.abs(left - right.numpy()).max())
                                                    for left, right in zip(result, reference))
    np.savez(output / "conformance.npz", history=history.numpy(), observation=observation.numpy(),
             commands=commands.numpy(), dynamic=dynamic.numpy(), encoded=encoded.numpy(), raw_actor_input=raw.numpy(),
             actions=expected_actions.numpy(), previous_issued_action=issued.numpy(), hidden=hidden.numpy(),
             filtered_eta=filtered_eta.numpy(), next_hidden=expected_context[0].numpy(),
             next_filtered_eta=expected_context[1].numpy(), raw_eta=expected_context[2].numpy())
    metadata = dict(task=exp.task, checkpoint_schema=exp.schema, experiment=exp.manifest(),
        interface=exp.interface, baseline_contract=baseline, control_dt=.02,
        checkpoint_sha256=_sha256(checkpoint),
        curriculum_stage=int(state["environment_state_dict"][exp.command_name]["stage"]),
        physical_command_units=["rad/s", "m/s", "m"], encoder_output_frame="base_body",
        actor_velocity_frame="body", tcp_local_m=[-.066473051764, .161623719479, -.000621484902],
        observation_order=["base_gyro_scaled_0.25:3", "base_gravity:3", "up3_gyro_scaled_0.25:3", "up3_gravity:3",
                           "relative_joint_position:11", "joint_velocity_scaled_0.05:13", "previous_raw_action:13"],
        history_order="oldest_to_newest_5_policy49_frames; clear_on_new_physics_or_episode_reset",
        raw_actor_input_order="Encoder7,policy49,applied_commands3,normalized_memory2,filtered_student8,hdot_over_0.5",
        actor_features_order="original_E62_in_original_order_then_normalized_memory2_filtered_student8_hdot_over_0.5",
        dynamic_input_order=["memory_vx_over_0.10", "memory_yaw_over_0.20"]
                            + [f"filtered_eta_{index}" for index in range(8)] + ["hdot_over_0.5"],
        context_step=dict(input_order=["observation", "previous_issued_action", "hidden", "filtered_eta"],
            output_order=["next_hidden", "next_filtered_eta", "raw_eta"], hidden_width=64, latent_width=8,
            student_input="noisy_policy_first36_previous_software_clipped_issued13", lpf_tau_s=.3,
            reset="zero_hidden64_filtered_eta8_previous_issued13_and_dynamic11; skip_episode_priming_observation",
            timing="advance_once_after_successful_physical_step_then_cache_for_actor_and_ppo"),
        feature_transform_included=True, encoder_actor_gradient=False, adaptation_actor_gradient=False,
        stateful_processing_reference="processing_reference.py", runtime_math="adaptive_math.py",
        runtime_support="50Hz explicit Student/command/memory state; no ROS schema support or hardware validation claimed",
        validation=metrics)
    if ablation:
        from .height_memory import HeightSlowErrorMemory
        metadata.update(
            height_rate_limits_m_s=list(exp.height_rates),
            height_memory=dict(enabled=exp.height_memory_enabled, profile=dict(HeightSlowErrorMemory.PROFILE),
                error="applied_height_minus_detached_encoder_height",
                runtime_source="adaptive_height_memory.py",
                reset="zero_selected_rows_on_episode_reset_and_all_rows_on_new_physics; skip_priming_observation",
                timing="advance_once_after_successful_physical_step; getters_and_duplicate_step_ids_do_not_advance"),
            raw_actor_input_order=metadata["raw_actor_input_order"] + ",normalized_memory_height1",
            actor_features_order=metadata["actor_features_order"] + "_then_normalized_memory_height1",
            dynamic_input_order=metadata["dynamic_input_order"] + ["memory_height_over_0.05"])
        metadata["context_step"]["reset"] = (
            "zero_hidden64_filtered_eta8_previous_issued13_and_dynamic12_including_height_memory; "
            "skip_episode_priming_observation")
    metadata["file_sha256"] = {path.name: _sha256(path) for path in output.iterdir() if path.is_file()}
    (output / "deployment_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def export_for_play(checkpoint):
    """Reuse only a verified complete new-schema bundle; preserve prior exports."""
    checkpoint = Path(checkpoint).resolve()
    identity = torch.load(checkpoint, map_location="meta", weights_only=False)
    exp = checkpoint_experiment(identity)
    if not isinstance(exp, AdaptiveBalanceExperiment):
        raise ValueError("Adaptive export requires an adaptive-balance checkpoint")
    output = checkpoint.parent / "exported"
    digest = _sha256(checkpoint)
    required = {f"{name}.{suffix}" for name in ("encoder", "policy", "actor_pipeline", "context_step")
                for suffix in ("pt", "onnx")}
    if hasattr(exp, "height_memory_enabled"):
        required.update(("processing_reference.py", "adaptive_math.py", "adaptive_height_memory.py"))
    try:
        metadata = json.loads((output / "deployment_metadata.json").read_text())
        files = metadata["file_sha256"]
        if (metadata["checkpoint_schema"] == exp.schema and metadata.get("task") == exp.task
                and metadata["checkpoint_sha256"] == digest and required.issubset(files)
                and all(Path(name).name == name and _sha256(output / name) == expected
                        for name, expected in files.items())):
            return output
    except (OSError, ValueError, KeyError, TypeError):
        pass
    with tempfile.TemporaryDirectory(prefix=".adaptive-play-export-", dir=checkpoint.parent) as staging:
        bundle = Path(staging) / "exported"
        export_bundle(checkpoint, bundle, onnx=True)
        backup = None
        if output.exists():
            backup = output.with_name("exported.backup-" + uuid.uuid4().hex)
            output.rename(backup)
        try:
            bundle.rename(output)
        except BaseException:
            if backup is not None:
                backup.rename(output)
            raise
    return output
