"""Pure Torch adaptation for the independent E v1.3.1.2.2 experiment.

The environment owns parameter normalization and cached same-pose geometry.
Only ``StudentContext`` and the actor adapter are needed for deployment. PPO
must consume the detached context produced during collection, never rerun it.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


CHECKPOINT_SCHEMA = "nandu_network_v1_3_1_2_2_E_adaptive_balance_r1"


@dataclass(frozen=True)
class AdaptationConfig:
    dt: float = .02
    latent_tau: float = .3
    ema_rate: float = .01
    ring_capacity: int = 128
    sequence_length: int = 64
    burn_in: int = 16
    sequences_per_iteration: int = 128
    minibatch_size: int = 32
    epochs: int = 2
    learning_rate: float = 1.e-4
    max_grad_norm: float = 1.

    def __post_init__(self):
        positive = ("dt", "latent_tau", "learning_rate", "max_grad_norm")
        for key in positive:
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not 0. < value < float("inf"):
                raise ValueError(f"{key} must be finite and positive")
        for key in ("ring_capacity", "sequence_length", "sequences_per_iteration", "minibatch_size", "epochs"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.burn_in) is not int or not 0 <= self.burn_in < self.sequence_length:
            raise ValueError("burn_in must leave at least one loss frame")
        if not 0. < self.ema_rate <= 1.:
            raise ValueError("ema_rate must lie in (0, 1]")


def build_student_input(policy_obs: torch.Tensor, previous_issued_action: torch.Tensor) -> torch.Tensor:
    """Use the noisy first36 and software-clipped issued13 before actuator delay."""
    if policy_obs.shape[-1] != 49 or previous_issued_action.shape[-1] != 13:
        raise ValueError("Student input requires policy49 and previous issued action13")
    if policy_obs.shape[:-1] != previous_issued_action.shape[:-1]:
        raise ValueError("Student observation and action batch dimensions differ")
    return torch.cat((policy_obs[..., :36], previous_issued_action), dim=-1)


def build_adaptive_actor_input(old_e_features: torch.Tensor, memory: torch.Tensor,
                               eta: torch.Tensor, hdot: torch.Tensor) -> torch.Tensor:
    """Keep the original E62 prefix and append normalized detached dynamic11."""
    if old_e_features.shape[-1] != 62 or memory.shape[-1] != 2 or eta.shape[-1] != 8:
        raise ValueError("Adaptive actor requires original E62, memory2 and student eta8")
    if hdot.dim() == old_e_features.dim() - 1:
        hdot = hdot.unsqueeze(-1)
    if hdot.shape[-1] != 1:
        raise ValueError("Height rate must have one channel")
    if any(value.shape[:-1] != old_e_features.shape[:-1] for value in (memory, eta, hdot)):
        raise ValueError("Adaptive actor batch dimensions differ")
    scale = memory.new_tensor((.10, .20))
    return torch.cat((old_e_features, memory.detach() / scale, eta.detach(), hdot.detach() / .5), -1)


class StudentContext(nn.Module):
    """Exportable causal GRU49→64→tanh8, with explicit recurrent and LPF state."""
    def __init__(self, tau: float = .3):
        super().__init__()
        if tau <= 0.:
            raise ValueError("Student LPF tau must be positive")
        self.tau = float(tau)
        self.gru = nn.GRUCell(49, 64)
        self.projection = nn.Linear(64, 8)

    @torch.jit.export
    def step(self, features: torch.Tensor, hidden: torch.Tensor, filtered_eta: torch.Tensor,
             dt: float = .02, reset_mask: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.dim() != 2 or features.size(-1) != 49:
            raise ValueError("Student context_step requires [batch,49]")
        if hidden.size(0) != features.size(0) or hidden.size(-1) != 64:
            raise ValueError("Student hidden state requires [batch,64]")
        if filtered_eta.size(0) != features.size(0) or filtered_eta.size(-1) != 8:
            raise ValueError("Student LPF state requires [batch,8]")
        if dt <= 0.:
            raise ValueError("Student context_step dt must be positive")
        if reset_mask is not None:
            reset = reset_mask.reshape(-1, 1).to(dtype=torch.bool)
            hidden = torch.where(reset, torch.zeros_like(hidden), hidden)
            filtered_eta = torch.where(reset, torch.zeros_like(filtered_eta), filtered_eta)
        next_hidden = self.gru(features, hidden)
        raw_eta = torch.tanh(self.projection(next_hidden))
        next_eta = filtered_eta + (dt / (self.tau + dt)) * (raw_eta - filtered_eta)
        return raw_eta, next_hidden, next_eta

    def forward(self, features: torch.Tensor, hidden: torch.Tensor, filtered_eta: torch.Tensor,
                dt: float = .02, reset_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.step(features, hidden, filtered_eta, dt, reset_mask)


class AdaptationSequenceRing:
    """Frame ring shared across environments, sampled as time-major sequences.

    Samples end at any cached frame, including short or failed episodes. Left
    padding is invalid and cannot advance the recurrent state. ``reset`` means
    reset *before* this frame; initial hidden/LPF are recorded before that step.
    No poses, FK, or simulator objects are retained.
    """
    _WIDTHS = {"student_obs": 49, "teacher_params": 20, "counterfactual_teacher_params": 20,
               "balance_target": 2, "counterfactual_balance_target": 2, "hidden": 64, "filtered_eta": 8}
    _MASKS = ("valid", "reset", "balance_valid", "counterfactual_balance_valid")

    def __init__(self, capacity: int = 128, device=None):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Ring capacity must be a positive integer")
        self.capacity = capacity
        self.device = torch.device(device) if device is not None else None
        self.num_envs = 0
        self.position = 0
        self.size = 0
        self.storage: Dict[str, torch.Tensor] = {}

    def __len__(self):
        return self.size

    def clear(self):
        self.position = self.size = 0
        self.num_envs = 0
        self.storage.clear()

    @torch.no_grad()
    def append(self, frame: Mapping[str, torch.Tensor]):
        converted = {}
        obs = frame["student_obs"]
        if obs.dim() != 2 or obs.shape[-1] != 49 or obs.shape[0] < 1:
            raise ValueError("Cached student_obs must be [num_envs,49]")
        count = obs.shape[0]
        if self.num_envs and count != self.num_envs:
            raise ValueError("Ring environment count changed; clear it after rebuilding the scene")
        device = self.device if self.device is not None else obs.device
        for key, width in self._WIDTHS.items():
            value = frame[key]
            if value.shape != (count, width):
                raise ValueError(f"Cached {key} must be [{count},{width}]")
            converted[key] = value.detach().to(device=device, dtype=obs.dtype)
        for key in self._MASKS:
            fallback = (frame.get("balance_valid", frame["valid"]) if key == "counterfactual_balance_valid"
                        else frame["valid"] if key != "reset" else None)
            value = frame.get(key, fallback)
            if value is None or value.numel() != count:
                raise ValueError(f"Cached {key} must have one mask per environment")
            converted[key] = value.detach().reshape(count).to(device=device, dtype=torch.bool)
        if not self.storage:
            self.num_envs = count
            self.device = torch.device(device)
            self.storage = {key: torch.empty((self.capacity,) + tuple(value.shape),
                                             device=device, dtype=value.dtype)
                            for key, value in converted.items()}
        for key, value in converted.items():
            self.storage[key][self.position].copy_(value)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    @torch.no_grad()
    def sample(self, num_sequences: int, sequence_length: int = 64, burn_in: int = 16,
               *, end_indices: Optional[torch.Tensor] = None,
               env_indices: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        if not self.size:
            raise ValueError("Cannot sample an empty adaptation ring")
        if num_sequences < 1 or not 0 <= burn_in < sequence_length:
            raise ValueError("Invalid sequence count, length or burn-in")
        if end_indices is None:
            end_indices = torch.randint(self.size, (num_sequences,), device=self.device, generator=generator)
        else:
            end_indices = end_indices.to(device=self.device, dtype=torch.long)
        if env_indices is None:
            env_indices = torch.randint(self.num_envs, (num_sequences,), device=self.device, generator=generator)
        else:
            env_indices = env_indices.to(device=self.device, dtype=torch.long)
        if end_indices.shape != (num_sequences,) or env_indices.shape != (num_sequences,):
            raise ValueError("Sequence sample indices must match num_sequences")
        if bool(((end_indices < 0) | (end_indices >= self.size)).any()):
            raise ValueError("Sequence end index is outside cached frames")
        if bool(((env_indices < 0) | (env_indices >= self.num_envs)).any()):
            raise ValueError("Sequence environment index is outside the scene")
        offsets = torch.arange(sequence_length, device=self.device) - sequence_length + 1
        times = end_indices.unsqueeze(0) + offsets.unsqueeze(1)
        present = times >= 0
        oldest = (self.position - self.size) % self.capacity
        physical = (times.clamp_min(0) + oldest) % self.capacity
        batch = {key: value[physical, env_indices.unsqueeze(0)] for key, value in self.storage.items()}
        for key in self._MASKS:
            batch[key] = batch[key] & present
        initial_index = (times[0].clamp_min(0) + oldest) % self.capacity
        batch["initial_hidden"] = self.storage["hidden"][initial_index, env_indices].clone()
        batch["initial_filtered_eta"] = self.storage["filtered_eta"][initial_index, env_indices].clone()
        batch.pop("hidden")
        batch.pop("filtered_eta")
        batch["loss_mask"] = batch["valid"] & (torch.arange(sequence_length, device=self.device)[:, None] >= burn_in)
        return batch

    def state_dict(self):
        # Save only initialized slots; unused ring memory must never enter a file.
        used = self.capacity if self.size == self.capacity else self.size
        return dict(capacity=self.capacity, num_envs=self.num_envs, position=self.position, size=self.size,
                    storage={key: value[:used].detach().clone() for key, value in self.storage.items()})

    @staticmethod
    def validate_state_dict(state):
        capacity, size, count = state["capacity"], state["size"], state["num_envs"]
        position = state["position"]
        if (type(capacity) is not int or type(size) is not int or type(count) is not int
                or capacity < 1 or not 0 <= size <= capacity or count < 0
                or type(position) is not int or not 0 <= position < capacity):
            raise ValueError("Invalid adaptation ring checkpoint dimensions")
        if not size:
            if state["storage"] or count or position:
                raise ValueError("Empty adaptation ring checkpoint contains stale state")
            return
        if count < 1 or (size < capacity and position != size):
            raise ValueError("Invalid adaptation ring checkpoint cursor")
        widths = AdaptationSequenceRing._WIDTHS
        masks = AdaptationSequenceRing._MASKS
        if set(state["storage"]) != set(widths) | set(masks):
            raise ValueError("Adaptation ring checkpoint fields differ")
        for key, value in state["storage"].items():
            shape = (size, count, widths[key]) if key in widths else (size, count)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"Invalid adaptation ring checkpoint tensor {key}")
            if key in masks and value.dtype != torch.bool:
                raise ValueError(f"Invalid adaptation ring checkpoint mask {key}")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.clear()
        self.capacity = state["capacity"]
        self.num_envs, self.position, self.size = state["num_envs"], state["position"], state["size"]
        for key, value in state["storage"].items():
            device = self.device if self.device is not None else value.device
            target = torch.empty((self.capacity,) + tuple(value.shape[1:]), dtype=value.dtype, device=device)
            target[:self.size].copy_(value)
            self.storage[key] = target
        if self.storage:
            self.device = next(iter(self.storage.values())).device


def _mlp(widths, tanh=False):
    layers = []
    for index, (source, target) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(source, target))
        if index < len(widths) - 2:
            layers.append(nn.ELU())
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _masked_mean(value, mask):
    # Sanitize inputs before networks as well; multiplying a NaN by zero is NaN.
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


class AdaptiveBalanceLearner(nn.Module):
    """Independent teacher/student/aux optimizer; no actor or Encoder parameters."""
    _MODULE_NAMES = ("teacher", "decoder", "student", "aux_head", "ema_teacher")
    _METRICS = ("loss", "reconstruction", "latent", "teacher_balance", "student_balance", "grad_norm")

    def __init__(self, config=None, device="cpu", num_envs=None):
        super().__init__()
        self.config = (config if isinstance(config, AdaptationConfig)
                       else AdaptationConfig(**dict(config or {})))
        self.device = torch.device(device)
        self.teacher = _mlp((20, 64, 32, 8), tanh=True)
        self.decoder = _mlp((8, 32, 64, 20))
        self.student = StudentContext(self.config.latent_tau)
        self.aux_head = _mlp((57, 128, 64, 2))
        self.ema_teacher = deepcopy(self.teacher).requires_grad_(False)
        self.ema_teacher.eval()
        self.to(self.device)
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        self.ring = AdaptationSequenceRing(self.config.ring_capacity, self.device)
        self.hidden = torch.zeros((num_envs or 0, 64), device=self.device)
        self.filtered_eta = torch.zeros((num_envs or 0, 8), device=self.device)
        self.updates = 0

    @torch.no_grad()
    def reset(self, env_ids=None):
        if env_ids is None:
            self.hidden = torch.zeros_like(self.hidden)
            self.filtered_eta = torch.zeros_like(self.filtered_eta)
        else:
            # Collection can run in inference_mode, followed by a reset outside
            # that context. Clone before writing so inference tensors stay legal.
            self.hidden = self.hidden.clone()
            self.filtered_eta = self.filtered_eta.clone()
            self.hidden[env_ids] = 0.
            self.filtered_eta[env_ids] = 0.

    @torch.no_grad()
    def context_step(self, policy_obs, previous_issued_action=None, reset_mask=None):
        features = (build_student_input(policy_obs, previous_issued_action)
                    if previous_issued_action is not None else policy_obs)
        features = features.detach().to(self.device)
        if not self.hidden.shape[0]:
            self.hidden = features.new_zeros((features.shape[0], 64))
            self.filtered_eta = features.new_zeros((features.shape[0], 8))
        if self.hidden.shape[0] != features.shape[0]:
            raise ValueError("Student scene size changed; construct a new learner or clear temporal state")
        if reset_mask is not None:
            reset_mask = reset_mask.to(device=self.device)
        _, self.hidden, self.filtered_eta = self.student.step(features, self.hidden, self.filtered_eta,
                                                              self.config.dt, reset_mask)
        return self.filtered_eta.detach().clone()

    def append_frame(self, frame):
        self.ring.append(frame)

    record = append_frame

    @torch.no_grad()
    def update_ema(self):
        for target, source in zip(self.ema_teacher.parameters(), self.teacher.parameters()):
            target.lerp_(source, self.config.ema_rate)

    def unroll_student(self, observations, initial_hidden, initial_filtered_eta, reset, valid, burn_in=None):
        """Burn in without gradient, then backpropagate only within the loss suffix."""
        burn_in = self.config.burn_in if burn_in is None else burn_in
        hidden, filtered = initial_hidden.detach(), initial_filtered_eta.detach()
        raw_outputs, filtered_outputs = [], []
        for time in range(observations.shape[0]):
            if time == burn_in:
                hidden, filtered = hidden.detach(), filtered.detach()
            with torch.set_grad_enabled(torch.is_grad_enabled() and time >= burn_in):
                reset_now = reset[time].unsqueeze(-1)
                hidden = torch.where(reset_now, torch.zeros_like(hidden), hidden)
                filtered = torch.where(reset_now, torch.zeros_like(filtered), filtered)
                good = valid[time].unsqueeze(-1)
                safe_observation = torch.where(good, observations[time], torch.zeros_like(observations[time]))
                raw, next_hidden, next_filtered = self.student.step(safe_observation, hidden, filtered, self.config.dt)
                hidden = torch.where(good, next_hidden, hidden)
                filtered = torch.where(good, next_filtered, filtered)
                raw_outputs.append(torch.where(good, raw, torch.zeros_like(raw)))
                filtered_outputs.append(filtered)
        return torch.stack(raw_outputs), torch.stack(filtered_outputs)

    def compute_loss(self, batch):
        """Actual Student; actual + independently sampled same-pose CF Teacher."""
        obs = batch["student_obs"].detach()
        valid = batch["valid"] & torch.isfinite(obs).all(-1)
        loss_mask = batch["loss_mask"] & valid
        raw, filtered = self.unroll_student(obs, batch["initial_hidden"], batch["initial_filtered_eta"],
                                             batch["reset"], valid)
        safe_obs = torch.where(valid.unsqueeze(-1), obs, torch.zeros_like(obs))
        params = batch["teacher_params"].detach()
        cf_params = batch["counterfactual_teacher_params"].detach()
        actual_mask = loss_mask & torch.isfinite(params).all(-1)
        cf_mask = loss_mask & torch.isfinite(cf_params).all(-1)
        safe_params = torch.where(actual_mask.unsqueeze(-1), params, torch.zeros_like(params))
        safe_cf_params = torch.where(cf_mask.unsqueeze(-1), cf_params, torch.zeros_like(cf_params))
        teacher_eta, cf_eta = self.teacher(safe_params), self.teacher(safe_cf_params)
        reconstruction = .5 * (_masked_mean((self.decoder(teacher_eta) - safe_params).square(), actual_mask)
                               + _masked_mean((self.decoder(cf_eta) - safe_cf_params).square(), cf_mask))
        with torch.no_grad():
            teacher_target = self.ema_teacher(safe_params)
        latent = _masked_mean((raw - teacher_target).square(), actual_mask)
        target = batch["balance_target"].detach()
        cf_target = batch["counterfactual_balance_target"].detach()
        balance_mask = actual_mask & batch["balance_valid"] & torch.isfinite(target).all(-1)
        cf_balance_mask = cf_mask & batch["counterfactual_balance_valid"] & torch.isfinite(cf_target).all(-1)
        scale = target.new_tensor((.08, .5))
        target = torch.where(balance_mask.unsqueeze(-1), target, torch.zeros_like(target)) / scale
        cf_target = torch.where(cf_balance_mask.unsqueeze(-1), cf_target, torch.zeros_like(cf_target)) / scale
        teacher_prediction = self.aux_head(torch.cat((safe_obs, teacher_eta), -1))
        cf_prediction = self.aux_head(torch.cat((safe_obs, cf_eta), -1))
        student_prediction = self.aux_head(torch.cat((safe_obs, filtered), -1))
        teacher_balance = .5 * (_masked_mean(F.smooth_l1_loss(teacher_prediction, target, reduction="none"), balance_mask)
                                + _masked_mean(F.smooth_l1_loss(cf_prediction, cf_target, reduction="none"), cf_balance_mask))
        student_balance = _masked_mean(F.smooth_l1_loss(student_prediction, target, reduction="none"), balance_mask)
        total = reconstruction + latent + .25 * teacher_balance + .25 * student_balance
        return dict(loss=total, reconstruction=reconstruction, latent=latent,
                    teacher_balance=teacher_balance, student_balance=student_balance)

    def update(self, *, optimize=True):
        metrics = {name: 0. for name in self._METRICS}
        metrics["updated"] = 0
        evaluated = 0
        if not optimize:
            metrics['frozen'] = 1
        if not len(self.ring):
            return metrics
        cfg = self.config
        batch = self.ring.sample(cfg.sequences_per_iteration, cfg.sequence_length, cfg.burn_in)
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        for _ in range(cfg.epochs):
            order = torch.randperm(cfg.sequences_per_iteration, device=self.device)
            for start in range(0, cfg.sequences_per_iteration, cfg.minibatch_size):
                indices = order[start:start + cfg.minibatch_size]
                part = {key: (value[indices] if key.startswith("initial_") else value[:, indices])
                        for key, value in batch.items()}
                usable = (part["loss_mask"] & torch.isfinite(part["student_obs"]).all(-1)
                          & (torch.isfinite(part["teacher_params"]).all(-1)
                             | torch.isfinite(part["counterfactual_teacher_params"]).all(-1)))
                if not bool(usable.any()):
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(optimize):
                    losses = self.compute_loss(part)
                if not bool(torch.isfinite(losses["loss"])):
                    continue
                norm = 0.
                if optimize:
                    losses["loss"].backward()
                    norm = nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                    if not bool(torch.isfinite(norm)):
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    self.optimizer.step()
                    self.update_ema()
                    self.updates += 1
                    metrics["updated"] += 1
                evaluated += 1
                for key, value in losses.items():
                    metrics[key] += float(value.detach())
                metrics["grad_norm"] += float(norm)
        if evaluated:
            for key in self._METRICS:
                metrics[key] /= evaluated
        metrics['evaluated'] = evaluated
        return metrics

    def checkpoint_state(self):
        return dict(schema=CHECKPOINT_SCHEMA, config=asdict(self.config), updates=self.updates,
                    modules={name: deepcopy(getattr(self, name).state_dict()) for name in self._MODULE_NAMES},
                    optimizer=deepcopy(self.optimizer.state_dict()), ring=self.ring.state_dict(),
                    runtime=dict(hidden=self.hidden.detach().clone(), filtered_eta=self.filtered_eta.detach().clone()))

    def validate_checkpoint_state(self, state, restore_temporal=False):
        """Validate all components before mutating any live learning state."""
        required = {"schema", "config", "updates", "modules", "optimizer", "ring", "runtime"}
        if not isinstance(state, Mapping) or not required.issubset(state):
            raise ValueError("Incomplete adaptation checkpoint")
        if state["schema"] != CHECKPOINT_SCHEMA:
            raise ValueError("Adaptation checkpoint schema mismatch; old experiments require fresh initialization")
        if AdaptationConfig(**state["config"]) != self.config:
            raise ValueError("Adaptation checkpoint configuration differs")
        if type(state["updates"]) is not int or state["updates"] < 0:
            raise ValueError("Invalid adaptation checkpoint update count")
        if set(state["modules"]) != set(self._MODULE_NAMES):
            raise ValueError("Adaptation checkpoint module set differs")
        for name in self._MODULE_NAMES:
            expected = getattr(self, name).state_dict()
            actual = state["modules"][name]
            if set(expected) != set(actual):
                raise ValueError(f"Adaptation checkpoint {name} parameter set differs")
            for key, value in actual.items():
                if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError(f"Invalid adaptation checkpoint {name}.{key}")
        optimizer = state["optimizer"]
        groups = optimizer.get("param_groups", [])
        expected_groups = self.optimizer.state_dict()["param_groups"]
        if len(groups) != len(expected_groups):
            raise ValueError("Adaptation optimizer checkpoint parameter groups differ")
        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        ids = []
        for group, expected in zip(groups, expected_groups):
            if not set(expected).issubset(group) or len(group.get("params", [])) != len(expected["params"]):
                raise ValueError("Adaptation optimizer checkpoint parameter count differs")
            for key in ("lr", "eps", "weight_decay"):
                value = group[key]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.:
                    raise ValueError(f"Invalid adaptation optimizer checkpoint {key}")
            if (not isinstance(group["betas"], (tuple, list)) or len(group["betas"]) != 2
                    or any(not isinstance(value, (int, float)) or not 0. <= value < 1.
                           for value in group["betas"])):
                raise ValueError("Invalid adaptation optimizer checkpoint betas")
            ids.extend(group["params"])
        if len(set(ids)) != len(ids) or set(optimizer.get("state", {})) - set(ids):
            raise ValueError("Invalid adaptation optimizer checkpoint parameter identities")
        for identity, parameter in zip(ids, params):
            parameter_state = optimizer.get("state", {}).get(identity, {})
            if parameter_state and not {"step", "exp_avg", "exp_avg_sq"}.issubset(parameter_state):
                raise ValueError("Incomplete adaptation optimizer checkpoint state")
            for key, value in parameter_state.items():
                if isinstance(value, torch.Tensor):
                    if not bool(torch.isfinite(value).all()) or (key != "step" and value.shape != parameter.shape):
                        raise ValueError("Invalid adaptation optimizer checkpoint tensor")
        AdaptationSequenceRing.validate_state_dict(state["ring"])
        if state["ring"]["capacity"] != self.config.ring_capacity:
            raise ValueError("Adaptation checkpoint ring capacity differs")
        runtime = state["runtime"]
        if set(runtime) != {"hidden", "filtered_eta"}:
            raise ValueError("Adaptation checkpoint runtime state differs")
        h, eta = runtime["hidden"], runtime["filtered_eta"]
        if (not isinstance(h, torch.Tensor) or not isinstance(eta, torch.Tensor) or h.dim() != 2
                or eta.dim() != 2 or h.shape[-1] != 64 or eta.shape != (h.shape[0], 8)
                or not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(eta).all())):
            raise ValueError("Invalid adaptation checkpoint temporal state")
        if restore_temporal and self.hidden.shape[0] not in (0, h.shape[0]):
            raise ValueError("Cannot restore temporal state into a different physics scene size")

    def load_checkpoint_state(self, state, restore_temporal=False):
        self.validate_checkpoint_state(state, restore_temporal)
        for name in self._MODULE_NAMES:
            getattr(self, name).load_state_dict(state["modules"][name])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
        if restore_temporal:
            self.ring.load_state_dict(state["ring"])
            self.hidden = state["runtime"]["hidden"].detach().to(self.device).clone()
            self.filtered_eta = state["runtime"]["filtered_eta"].detach().to(self.device).clone()
        else:
            self.ring.clear()
            self.reset()


# Short alias retained for runner and downstream users of the explicit API.
AdaptationLearner = AdaptiveBalanceLearner
