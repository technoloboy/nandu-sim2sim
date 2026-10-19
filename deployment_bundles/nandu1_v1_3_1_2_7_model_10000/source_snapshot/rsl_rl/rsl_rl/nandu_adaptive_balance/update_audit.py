"""Small, fixed-input policy diagnostics across independent optimizer stages.

This module owns tensor snapshots only. It neither advances rollout context nor
changes optimizer settings. An inference callback must be deterministic and use
explicit input state: replay ``student.step`` from captured hidden/filtered-eta
inputs, or replay a sequence from its captured entry state. Reading the live
cached eta after adaptation would hide Student parameter changes. Teacher
outputs likewise mean evaluation on the same captured teacher parameters.
Callbacks must not call the stateful runtime ``context_step``, update memory or
integrators, sample actions, or mutate model buffers. Use direct Actor means,
explicit distribution standard deviations, and pure Student/Teacher forwards.
"""
from typing import Callable, Dict, Mapping, Tuple

import torch


TensorMap = Mapping[str, torch.Tensor]
Inference = Callable[[TensorMap], TensorMap]


class UpdateAudit:
    """Compare each stage with its own detached, bounded before snapshot.

    ``raw_context`` tensors are batch-first and share their leading sample
    dimension. For sequence replay, each sample can contain an entire sequence
    and its entry state; callbacks return one output vector per sample.

    Outputs require ``actor_mean[B,13]`` and ``actor_std[B,13]`` (a shared
    ``[13]`` or ``[1,13]`` std is accepted). Optional outputs are ``encoder[B,7]``
    and ``student_raw``, ``eta``, ``teacher`` of shape ``[B,8]``. Every output must
    be recomputed with current parameters on the fixed inputs. In particular,
    ``eta`` is the Student's replayed filtered output, not live cached features.

    Distances are the sample mean of vector L2 changes. Action order is leg8,
    arm3, wheel2; KL is summed across joints in the before-to-after direction.
    Each metric has a ``*_finite_count``. No usable pairs yields NaN, never a
    silent zero; ``finite_samples`` counts rows valid across all output fields.
    Capturing an existing stage replaces its baseline without retaining older
    iterations. Repeated measurements always use that stage's before snapshot.
    """

    _WIDTHS = {"actor_mean": 13, "actor_std": 13, "encoder": 7,
               "student_raw": 8, "eta": 8, "teacher": 8}

    def __init__(self, max_samples: int = 128):
        if type(max_samples) is not int or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        self.max_samples = max_samples
        self._stages: Dict[str, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = {}

    @torch.no_grad()
    def capture(self, stage: str, raw_context: TensorMap, infer: Inference) -> None:
        """Snapshot a held-out input batch and evaluate before-stage outputs."""
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a nonempty string")
        if not raw_context:
            raise ValueError("raw_context must contain batch-first tensors")
        count = None
        for name, value in raw_context.items():
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise ValueError(f"raw_context {name} must be a batch-first tensor")
            if count is None:
                count = value.shape[0]
            if value.shape[0] != count or count < 1:
                raise ValueError("raw_context batch dimensions must match and be nonempty")
        count = min(count, self.max_samples)
        context = {name: value[:count].detach().clone() for name, value in raw_context.items()}
        before = self._evaluate(context, infer, count)
        self._stages[stage] = context, before

    @torch.no_grad()
    def measure(self, stage: str, infer: Inference) -> Dict[str, float]:
        """Reevaluate the same fixed inputs and return scalar diagnostic values."""
        context, before = self._stages[stage]
        count = before["actor_mean"].shape[0]
        after = self._evaluate(context, infer, count)
        if set(before) != set(after):
            raise ValueError("Before and after output fields must match")
        if any(before[name].device != after[name].device for name in before):
            raise ValueError("Before and after output devices must match")

        finite = torch.ones(count, dtype=torch.bool, device=before["actor_mean"].device)
        for name in before:
            finite &= torch.isfinite(before[name]).all(-1) & torch.isfinite(after[name]).all(-1)
        finite &= (before["actor_std"] > 0.).all(-1) & (after["actor_std"] > 0.).all(-1)
        finite_count = float(finite.sum().item())
        result = dict(samples=float(count), finite_samples=finite_count,
                      nonfinite_samples=float(count) - finite_count)

        for name, columns in (("mean_action_l2", slice(None)),
                              ("leg_mean_action_l2", slice(0, 8)),
                              ("arm_mean_action_l2", slice(8, 11)),
                              ("wheel_mean_action_l2", slice(11, 13))):
            self._distance(result, name, before["actor_mean"][:, columns], after["actor_mean"][:, columns])
        for name in ("encoder", "student_raw", "eta", "teacher"):
            if name in before:
                self._distance(result, name + "_l2", before[name], after[name])

        mean0, mean1 = before["actor_mean"].double(), after["actor_mean"].double()
        std0, std1 = before["actor_std"].double(), after["actor_std"].double()
        good = (torch.isfinite(mean0).all(-1) & torch.isfinite(mean1).all(-1)
                & torch.isfinite(std0).all(-1) & torch.isfinite(std1).all(-1)
                & (std0 > 0.).all(-1) & (std1 > 0.).all(-1))
        mean0, mean1, std0, std1 = mean0[good], mean1[good], std0[good], std1[good]
        kl = (std1.log() - std0.log() +
              (std0.square() + (mean0 - mean1).square()) / (2. * std1.square()) - .5).sum(-1)
        self._record(result, "normal_kl", kl)
        return result

    @classmethod
    def _evaluate(cls, context: TensorMap, infer: Inference, count: int) -> Dict[str, torch.Tensor]:
        # A callback gets scratch copies, so in-place operations cannot alter
        # live tensors or the held-out inputs used by the next measurement.
        values = infer({name: value.detach().clone() for name, value in context.items()})
        if not isinstance(values, Mapping) or not {"actor_mean", "actor_std"}.issubset(values):
            raise ValueError("Inference outputs require actor_mean and actor_std")
        if set(values) - set(cls._WIDTHS):
            raise ValueError("Inference output fields are not supported")
        outputs = {}
        for name, value in values.items():
            width = cls._WIDTHS[name]
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise ValueError(f"Inference {name} must be a floating tensor")
            if name == "actor_std" and value.shape in ((width,), (1, width)):
                value = value.reshape(1, width).expand(count, width)
            if value.shape != (count, width):
                raise ValueError(f"Inference {name} must have shape [{count},{width}]")
            outputs[name] = value.detach().clone()
        if any(value.device != outputs["actor_mean"].device for value in outputs.values()):
            raise ValueError("Inference outputs must share a device")
        return outputs

    @classmethod
    def _distance(cls, result: Dict[str, float], name: str,
                  before: torch.Tensor, after: torch.Tensor) -> None:
        good = torch.isfinite(before).all(-1) & torch.isfinite(after).all(-1)
        delta = after[good].double() - before[good].double()
        cls._record(result, name, torch.linalg.vector_norm(delta, dim=-1))

    @staticmethod
    def _record(result: Dict[str, float], name: str, values: torch.Tensor) -> None:
        values = values[torch.isfinite(values)]
        result[name + "_finite_count"] = float(values.numel())
        result[name] = float(values.mean().item()) if values.numel() else float("nan")
