"""Bounded-memory pooled curriculum statistics (no simulator dependency)."""
import torch


STATISTICS_SCHEMA = "pooled_height_histogram_1mm_no_step_v1"


class HeightErrorHistogram:
    """All time/env samples, 1 mm bins up to 4 m plus conservative overflow.

    Quantiles use linear rank interpolation and bin upper edges. Within range
    the approximation overestimates torch.quantile by at most 1 mm. Overflow
    uses the largest observed error, not clipping; NaN/Inf count as infinity.
    """
    def __init__(self, device):
        self.counts = torch.zeros(4001, dtype=torch.int64, device=device)
        self.maximum = torch.zeros((), device=device)

    def update(self, errors):
        values = torch.nan_to_num(errors.detach().abs(), nan=float('inf'), posinf=float('inf'))
        indices = (values.clamp(max=4.) * 1000).long().clamp(max=4000)
        self.counts.add_(torch.bincount(indices.flatten(), minlength=4001))
        self.maximum.copy_(torch.maximum(self.maximum, values.max()))

    def quantile(self):
        total = int(self.counts.sum())
        if total == 0:
            return 0.
        rank = .95 * (total - 1)
        lo = int(rank)
        fraction = rank - lo
        ranks = torch.tensor([lo + 1, min(lo + 2, total)], device=self.counts.device)
        bins = torch.searchsorted(self.counts.cumsum(0), ranks)
        values = torch.where(bins < 4000, (bins + 1).float() / 1000., self.maximum)
        low, high = values.tolist()
        # Avoid 0 * inf at an exact rank.
        return low if fraction == 0. else (1. - fraction) * low + fraction * high

    def clear(self):
        self.counts.zero_()
        self.maximum.zero_()

    def state_dict(self):
        return {"counts": self.counts.detach().cpu().clone(), "maximum": self.maximum.detach().cpu().clone()}

    def load_state_dict(self, state):
        counts = torch.as_tensor(state["counts"], device=self.counts.device, dtype=torch.int64)
        if counts.shape != self.counts.shape or bool((counts < 0).any()):
            raise ValueError("invalid height-error histogram")
        self.counts.copy_(counts)
        self.maximum.copy_(torch.as_tensor(state["maximum"], device=self.maximum.device))


def sampling_coverage(height, vx, yaw, boundary):
    """Remove obsolete step-mode component and renormalize remaining weights."""
    return (.35 * height + .25 * vx + .20 * yaw + .10 * boundary) / .90


def window_state(term):
    return {name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for name, value in vars(term).items() if name.startswith("window_")}


def clear_window(term):
    for name, value in vars(term).items():
        if name.startswith("window_"):
            if isinstance(value, torch.Tensor):
                value.zero_()
            else:
                setattr(term, name, type(value)(0))
    term.height_error_histogram.clear()
    term._coverage_stride = 0


def restore_window(term, state):
    clear_window(term)
    for name, value in state.items():
        if not name.startswith("window_") or not hasattr(term, name):
            raise ValueError(f"unknown curriculum window field: {name}")
        destination = getattr(term, name)
        if isinstance(destination, torch.Tensor):
            value = torch.as_tensor(value, device=destination.device, dtype=destination.dtype)
            if value.shape != destination.shape:
                raise ValueError(f"invalid curriculum window shape: {name}")
            destination.copy_(value)
        else:
            setattr(term, name, type(destination)(value))
