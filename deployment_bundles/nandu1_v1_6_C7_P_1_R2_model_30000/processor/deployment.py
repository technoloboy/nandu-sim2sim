"""Portable stateful controller. No Isaac, Critic, reference library or solver."""
import json
from pathlib import Path
import torch
from .commands import CommandProcessor
from .state import ControllerState
from .observations import actor_input
from .contract import DRLTaskSpec


def read_bundle_metadata(directory, expected_spec=None):
    """Validate task and physical identity before loading networks or physics."""
    meta = json.loads((Path(directory) / "manifest.json").read_text())
    spec = DRLTaskSpec.from_metadata(meta)
    spec.validate_metadata(meta)
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("Deployment task mismatch with requested task")
    if spec.physics_revision:
        from .profiles import physical_profile_for
        if meta.get("physical_profile") != physical_profile_for(spec):
            raise ValueError(f"Physical profile missing or incompatible with C7-P-{spec.physics_revision}")
        dependencies = {"processor/c7_p1.py"}
        if spec.physics_revision == 2:
            dependencies.update(("processor/profiles.py", "processor/c7_p2.py"))
        if dependencies - meta.get("files", {}).keys():
            raise ValueError("Physical profile processor dependency missing")
    elif "physical_profile" in meta:
        raise ValueError("Physical profile is incompatible with the original task")
    if spec.stable_numerics:
        from .actions import action_interface_for
        if (meta.get("action_interface") != action_interface_for(spec).metadata()
                or meta.get("raw_action_clip") != (None if spec.revision_profile else "per_axis_action_interface")
                or meta.get("actor_output_semantics") != "raw_distribution_mean"):
            raise ValueError("Action interface missing or incompatible with numerical revision 3")
        if "processor/actions.py" not in meta.get("files", {}):
            raise ValueError("Action interface processor dependency missing")
        if spec.revision_profile:
            required={'processor/profiles.py','processor/c7_p1_r4_actions.py',
                'processor/c7_p1_r4_state.py','processor/c7_p1_r4_wheels.py','processor/wheel_pd.py'}
            if required-meta.get('files',{}).keys():
                raise ValueError('R4 packet/controller processor dependency missing')
    elif "action_interface" in meta or meta.get("raw_action_clip") is not None:
        raise ValueError("Action interface is incompatible with the legacy numerical revision")
    return spec, meta


class DeploymentController:
    def __init__(self, spec, encoder, actor, num_envs=1, device="cpu"):
        self.spec, self.encoder, self.actor = spec, encoder, actor
        self.device = device
        self.processor = CommandProcessor(num_envs, device)
        self.state = ControllerState(num_envs, device, spec.use_integral)
        if spec.revision_profile:
            from .c7_p1_r4_state import R4ControllerState
            self.state=R4ControllerState(num_envs,device,spec.use_integral)
        self.packet_context=None
        self.wheel_records=[]
        self.complete_motor_records=False
        self.hidden = torch.zeros(num_envs, 32, device=device) if spec.recurrent else None
        self.clipped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.last_step, self.last_action = None, None
        self.action_interface = None
        if spec.stable_numerics:
            from .actions import action_interface_for
            self.action_interface = action_interface_for(spec,device)

    @classmethod
    def from_checkpoint(cls, path, num_envs=1, device="cpu", expected_spec=None):
        from .checkpoint import validate_checkpoint
        from .networks import build_models
        data = torch.load(path, map_location="cpu", weights_only=False)
        spec = DRLTaskSpec.from_metadata(data)
        if expected_spec is not None and spec != expected_spec:
            raise ValueError("Deployment task mismatch with requested task")
        validate_checkpoint(data, spec)
        encoder, actor, _ = build_models(spec, device=device)
        encoder.load_state_dict(data["encoder"]); actor.load_state_dict(data["actor"])
        encoder.eval(); actor.eval()
        return cls(spec, encoder, actor, num_envs, device)

    @classmethod
    def from_bundle(cls, directory, num_envs=1, expected_spec=None):
        import onnxruntime as ort
        import hashlib
        directory = Path(directory)
        spec, meta = read_bundle_metadata(directory, expected_spec)
        for filename, digest in meta["files"].items():
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Export integrity mismatch: {filename}")
        options = ort.SessionOptions(); options.intra_op_num_threads = 1
        enc = ort.InferenceSession(str(directory / "encoder.onnx"), options, providers=["CPUExecutionProvider"])
        act = ort.InferenceSession(str(directory / "actor.onnx"), options, providers=["CPUExecutionProvider"])
        def encoder(h): return torch.from_numpy(enc.run(None, {"history": h.detach().cpu().numpy()})[0])
        def actor(x, hidden):
            values = {"features": x.detach().cpu().numpy()}
            if spec.recurrent: values["hidden"] = hidden.detach().cpu().numpy()
            outputs = act.run(None, values)
            return (torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1]) if spec.recurrent else None,
                    torch.from_numpy(outputs[1]) if spec.revision_profile else None)
        return cls(spec, encoder, actor, num_envs)

    @torch.no_grad()
    def reset(self, policy, history, ids=None, initial_height=None):
        if ids is None:
            ids = torch.arange(len(policy), device=policy.device)
            self.last_step = self.state.last_step = self.processor.last_step = None
        estimate = self.encoder(history.reshape(-1, self.spec.history_dim))
        self.state.reset(ids, policy[ids], estimate[ids])
        height = estimate[ids, 6] if initial_height is None else initial_height
        self.processor.reset(ids, height)
        self.clipped[ids] = False
        if self.hidden is not None: self.hidden[ids] = 0
        if self.packet_context is not None:
            for v in self.packet_context.values():v[ids]=0
        for record in self.wheel_records:
            for v in record.values():v[ids]=0

    @torch.no_grad()
    def issued_step(self, policy, history, command, rate, step_id, dt=.02, intent=None):
        """Isaac command manager has already processed the request exactly once."""
        if step_id == self.last_step: return self.last_action
        estimate = self.encoder(history.reshape(-1, self.spec.history_dim))
        if self.spec.revision_profile:
            if intent is None:raise ValueError('R4 requires raw user intent separately from heading correction')
            from .c7_p1_r4_state import packet_records
            self.state.record_issued_action(policy[:,-13:])
            self.state.advance(policy,estimate,command,rate,
                packet_records(policy,None if self.complete_motor_records else self.packet_context,self.wheel_records),
                dt,step_id,intent=intent)
        else:self.state.advance(policy, estimate, command, rate, self.clipped, dt, step_id)
        features = actor_input(policy, estimate, command, self.state)
        action, hidden, auxiliary = self.actor(features, self.hidden)
        if self.spec.revision_profile:
            mapping=(self.actor.integral_mapping(features) if hasattr(self.actor,'integral_mapping') else auxiliary)
            if not isinstance(mapping,torch.Tensor) or mapping.shape!=(len(policy),13,3):
                raise ValueError('R4 export missing packet integral allocation map')
            self.packet_context=dict(mapping=mapping.detach().clone(),integral=features[:,62:65].clone(),latent=action.clone())
        if self.action_interface is not None:
            if not torch.isfinite(action).all():
                raise FloatingPointError("Non-finite raw action in numerical revision 3")
            issued = self.action_interface.map(action)
            self.clipped.copy_((issued != action).any(-1))
            action = issued
        if hidden is not None: self.hidden.copy_(hidden)
        if not self.spec.revision_profile:self.state.record_issued_action(action)
        self.last_step, self.last_action = step_id, action
        return action

    @torch.no_grad()
    def step(self, policy, history, request, stage, step_id, dt=.02):
        command = self.processor.advance(request, policy[:, 3:6], policy[:, :3] / .25,
                                          dt, stage, step_id)
        return self.issued_step(policy, history, command, self.processor.rate, step_id, dt,intent=request)


def isaac_issued_step(controller,env,term,obs):
    """Shared ordinary/joystick/evaluation hook, no privileged features forwarded."""
    if controller.spec.revision_profile:
        wheel=env._nandu17_wheel_actuator.state
        controller.wheel_records=term.saturation_records()
        controller.complete_motor_records=True
    action=controller.issued_step(obs['policy'],obs['obsHistory'],term.command,
        term.processor.rate,env.common_step_counter,intent=term.target_command)
    if controller.spec.revision_profile:term.capture_controller_packet(controller.packet_context)
    return action
