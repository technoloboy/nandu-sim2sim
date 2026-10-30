"""Isolated C7 R0/R1/R2 runtime; no ROS, SDK sockets or live training imports."""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = ROOT.parent / "deployment_bundles"
VERSIONS = ("R0", "R1", "R2")


def bundle_for(version):
    if version not in VERSIONS:
        raise ValueError("Choose R0, R1 or R2")
    return BUNDLE_ROOT / f"nandu1_v1_6_C7_P_1_{version}_model_30000"


def validate_bundle(directory, version):
    """Verify all frozen code/assets before importing any executable bundle code."""
    if version not in VERSIONS:
        raise ValueError("Unsupported task version")
    directory = Path(directory).resolve()
    meta = json.loads((directory / "manifest.json").read_text())
    task = f"Isaac-NANDU1-Blind-Flat-v1-6-C7-P-1-{version}"
    if meta.get("task_name") != task or meta.get("checkpoint_schema") != f"nandu_drl_v1_6_C7_P_1_{version}_online_r4":
        raise ValueError("C7 bundle task/schema mismatch")
    if (meta.get("sim2sim_format") != "nandu_c7_r012_mujoco_v1"
            or meta.get("checkpoint_iteration") != 30000):
        raise ValueError("Expected a frozen 30000-iteration C7 Sim2Sim bundle")
    for name, value in dict(policy_dim=49, history_dim=490, encoder_dim=7,
                            actor_dim=86, action_dim=13, control_dt=.02).items():
        if meta.get(name) != value:
            raise ValueError(f"C7 network/control interface mismatch: {name}")
    required = {"encoder.onnx", "actor.onnx", "processor/__init__.py",
                "processor/deployment.py", "processor/mujoco_adapter.py",
                "simulator_sources/nandu1_contract.py", "simulator_sources/nandu1_v131.py",
                "assets/robot.xml"}
    files = meta.get("files", {})
    if required - files.keys():
        raise ValueError(f"C7 bundle missing dependencies: {sorted(required-files.keys())}")
    for name, expected in files.items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory):
            raise ValueError(f"Unsafe bundle path: {name}")
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle integrity mismatch: {name}")
    return meta


def _package(directory):
    name = "_nandu_c7_" + hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, directory / "processor/__init__.py",
            submodule_search_locations=[str(directory / "processor")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return name


@contextmanager
def _physics_sources(directory):
    """Scope the two legacy absolute imports without replacing another runtime's modules."""
    names = ("nandu1_contract", "nandu1_v131")
    previous = {name: sys.modules.pop(name, None) for name in names}
    old_path = sys.path.copy()
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


class C7Simulation:
    """One 50 Hz closed loop; returned rows retain issued/executed actions separately."""
    def __init__(self, directory, version, stage=None, *, up3_mass=.67148,
                 rear_mass=0., rear_x=-.275, standard_scene=False,
                 outer_loop='off', outer_config=None):
        if outer_loop not in ('off', 'truth'):
            raise ValueError('Unsupported outer loop mode')
        if outer_loop == 'truth' and version != 'R0':
            raise ValueError('Truth outer loop is available only for C7 R0')
        if outer_config is not None and outer_loop != 'truth':
            raise ValueError('outer_config requires truth outer loop mode')
        self.outer_mode, self.outer = outer_loop, None
        self.directory = Path(directory).resolve()
        self.metadata = validate_bundle(self.directory, version)
        self.version = version
        self.stage = self.metadata["training_stage"] if stage is None else stage
        if type(self.stage) is not int or self.stage not in range(5):
            raise ValueError("Stage must be an integer from 0 to 4")
        package = _package(self.directory)
        deployment = importlib.import_module(package + ".deployment")
        physics = importlib.import_module(package + ".mujoco_adapter")
        self.mujoco = importlib.import_module("mujoco")
        self.controller = deployment.DeploymentController.from_bundle(self.directory)
        if outer_loop == 'truth':
            from nandu1_c7_outer_loop import OuterLoopConfig, TruthOuterLoop
            cfg = (OuterLoopConfig() if outer_config is None else outer_config
                   if isinstance(outer_config, OuterLoopConfig) else OuterLoopConfig.from_file(outer_config))
            commands = importlib.import_module(package + '.commands')
            low_h, high_h = commands.HEIGHT_LIMITS[self.stage]
            yaw, vx = commands.YAW_LIMITS[self.stage], commands.VX_LIMITS[self.stage]
            self.outer = TruthOuterLoop(cfg, [-yaw, -vx, low_h], [yaw, vx, high_h],
                                        [2., .6, commands.HEIGHT_RATES[self.stage]])
        source_xml = self.directory / "assets/robot.xml"
        if standard_scene:
            from nandu1_c7_scene import standard_scene_xml
            scene = standard_scene_xml(source_xml)
        else:
            scene = nullcontext(source_xml)
        self.scene_style = "standard_checkerboard_v1" if standard_scene else "bundle_original"
        with _physics_sources(self.directory / "simulator_sources"), scene as xml:
            self.physics = physics.Physics(self.directory / "simulator_sources",
                xml, up3_mass, rear_mass, rear_x,
                spec=self.controller.spec)
        self.reset()

    def reset(self):
        self.physics.reset()
        self.policy = torch.from_numpy(self.physics.measurement(np.zeros(13)))[None]
        self.history = self.policy[:, None].repeat(1, 10, 1).flatten(1)
        self.initial_height = float(self.physics.truth()["tcp"][2])
        self.controller.reset(self.policy, self.history,
            initial_height=torch.tensor([self.initial_height], dtype=torch.float32))
        # Physics has been recreated/reset, so previous saturation packets cannot survive.
        self.controller.wheel_records = []
        self.controller.complete_motor_records = True
        self.step_index = 0
        if self.outer is not None:
            self.outer.reset(self.controller.processor.command.numpy()[0])
            self.outer_previous_issued = self.controller.processor.command.numpy()[0].copy()

    def outer_metadata(self):
        return dict(mode=self.outer_mode, sim_truth_feedback=self.outer is not None,
                    config=asdict(self.outer.config) if self.outer is not None else None,
                    zero_mode='velocity_suppression_height_hold_no_world_position_anchor',
                    constraint_priority='hard_command_bounds_and_slew_before_candidate_correction_slew',
                    correction_rate_semantics='candidate_increment_relative_to_previous_applied_trim; safety_may_override_applied_correction')

    @torch.no_grad()
    def step(self, request):
        request = np.asarray(request, dtype=np.float32)
        if request.shape != (3,) or not np.isfinite(request).all():
            raise ValueError("Expected three finite command values [yaw, vx, height]")
        if not torch.isfinite(self.policy).all():
            raise FloatingPointError("Non-finite physical observation")
        start = self.physics.truth()
        estimate = self.controller.encoder(self.history).numpy()[0].copy()
        intent = torch.from_numpy(request)[None]
        if self.outer is None:
            action = self.controller.step(self.policy, self.history,
                intent, self.stage, self.step_index).numpy()[0]
            reference = self.controller.processor.command.numpy()[0].copy()
            reference_rate = self.controller.processor.rate.numpy()[0].copy()
            outer = dict(correction=np.zeros(3), integral=np.zeros(3), saturated=np.zeros(3, dtype=bool),
                         requested_correction=np.zeros(3), candidate_correction=np.zeros(3), correction_anchor=np.zeros(3),
                         command_amplitude_limited=np.zeros(3, dtype=bool), command_slew_limited=np.zeros(3, dtype=bool),
                         safety_override=np.zeros(3, dtype=bool),
                         ready=False, adapting=np.zeros(3, dtype=bool),
                         feedback=np.r_[start['yaw_rate'], start['tcp_velocity_body'][0], start['tcp'][2]])
            final_command, final_rate = reference.copy(), reference_rate.copy()
        else:
            from nandu1_c7_outer_loop import truth_feedback
            truth_feedback(start)  # fail before mutating the frozen processor
            self.controller.processor.advance(intent, self.policy[:, 3:6], self.policy[:, :3]/.25,
                                               .02, self.stage, self.step_index)
            reference = self.controller.processor.command.numpy()[0].copy()
            reference_rate = self.controller.processor.rate.numpy()[0].copy()
            outer = self.outer.update(reference, start, self.step_index)
            # The actor consumes float32. Derivative is based on its actual
            # rounded command, not an unrounded shadow trajectory.
            final_command = outer['command'].astype(np.float32)
            final_rate = ((final_command.astype(float)-self.outer_previous_issued)/.02).astype(np.float32)
            self.outer_previous_issued = final_command.copy()
            outer['correction'] = final_command.astype(float)-reference
            action = self.controller.issued_step(self.policy, self.history,
                torch.from_numpy(final_command)[None], torch.from_numpy(final_rate)[None],
                self.step_index, intent=intent).numpy()[0]
        self.physics.control(action, packet_context=self.controller.packet_context)
        # mj_step integrates qpos after updating derived body/sensor tables. Refresh
        # at the completed control boundary; no extra integration or output filter.
        self.mujoco.mj_forward(self.physics.model, self.physics.data)
        self.controller.wheel_records = self.physics.motor_records
        self.controller.complete_motor_records = True
        feedback = self.physics.executed_action.copy()
        row = self.physics.truth()
        if not all(np.isfinite(row[k]).all() for k in ("tcp", "joint_position", "joint_velocity", "torque")):
            raise FloatingPointError("Non-finite MuJoCo state")
        self.policy = torch.from_numpy(self.physics.measurement(feedback))[None]
        self.history = torch.cat((self.history.reshape(1, 10, 49)[:, 1:], self.policy[:, None]), 1).flatten(1)
        row.update(time=float(self.physics.data.time), command_time=float(self.step_index*.02),
                   request=request.copy(), command=final_command.copy(), command_rate=final_rate.copy(),
                   reference_command=reference, reference_rate=reference_rate,
                   outer_correction=outer['correction'], outer_integral=outer['integral'],
                   outer_saturated=outer['saturated'], outer_ready=outer['ready'], outer_feedback=outer['feedback'],
                   outer_adapting=outer['adapting'],
                   outer_requested_correction=outer['requested_correction'], outer_candidate_correction=outer['candidate_correction'],
                   outer_correction_anchor=outer['correction_anchor'], outer_safety_override=outer['safety_override'],
                   outer_command_amplitude_limited=outer['command_amplitude_limited'],
                   outer_command_slew_limited=outer['command_slew_limited'],
                   tcp_velocity_world_at_command=start['tcp_velocity_api'].copy(),
                   tcp_velocity_body_at_command=start['tcp_velocity_body'].copy(),
                   yaw_rate_at_command=start['yaw_rate'], base_quaternion_at_command=start['base_quaternion'].copy(),
                   wheel_clipped_pd_steps=self.physics.wheel_state.clipped_steps.numpy()[0].copy(),
                   wheel_requested_torque_last_pd=self.physics.wheel_state.requested_torque.numpy()[0].copy(),
                   wheel_emitted_torque_last_pd=self.physics.wheel_state.motor_torque.numpy()[0].copy(),
                   estimate_at_command=estimate, tcp_at_command=start["tcp"],
                   action=action.copy(), executed_action=feedback,
                   integral=self.controller.state.integral.numpy()[0].copy(),
                   integral_permission=self.controller.state.permission.numpy()[0].copy())
        self.step_index += 1
        return row

    def terminal_reason(self):
        """Record failures, do not reset/delete samples or change policy actions."""
        up = self.physics.data.xmat[self.physics.base].reshape(3, 3)[2, 2]
        if up < np.cos(np.deg2rad(65.)):
            return "base_tilt_over_65deg"
        return None

