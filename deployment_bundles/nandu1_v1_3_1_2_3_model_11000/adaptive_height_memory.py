"""Isolated, causal height-error memory for the B–F v13122 ablations."""
from __future__ import annotations

import math
from types import MappingProxyType

import torch

if __package__:
    from .math import _BatchState, _as, _dt, quiet_gate
else:
    # Deployment bundles load this source directly beside adaptive_math.py.
    import importlib.util
    from pathlib import Path

    _math_path = Path(__file__).with_name("adaptive_math.py")
    if not _math_path.is_file():
        _math_path = Path(__file__).with_name("math.py")
    _math_spec = importlib.util.spec_from_file_location("exported_height_memory_math", _math_path)
    _math_module = importlib.util.module_from_spec(_math_spec)
    _math_spec.loader.exec_module(_math_module)
    _BatchState, _as, _dt, quiet_gate = (
        _math_module._BatchState, _math_module._as, _math_module._dt, _math_module.quiet_gate
    )


class HeightSlowErrorMemory(_BatchState):
    """One detached Actor feature from applied minus Encoder-estimated height.

    ``height_error`` is in metres, ``height_rate`` in m/s, and ``tilt`` in
    radians. The filtered error integrates into ``z`` in metre-seconds. Only
    height motion gates accumulation; yaw and vx intent are not inputs.
    Invalid measurements, tilt beyond 45 degrees, or known software clipping
    freeze both filter and integral. Calls sharing a step ID are idempotent.
    """

    SCHEMA = "nandu_v13122_height_slow_error_memory_r1"
    PROFILE = MappingProxyType(dict(
        error_lpf_tau_s=.5,
        error_clip_m=.10,
        integral_clip_m_s=.10,
        actor_scale_m_s=.05,
        quiet_rate_low_m_s=.005,
        quiet_rate_high_m_s=.02,
        quiet_leak_tau_s=10.,
        moving_leak_tau_s=1.,
        freeze_tilt_rad=math.pi / 4,
    ))

    def __init__(self, num_envs, device="cpu", dtype=torch.float32, enabled=True):
        super().__init__(num_envs, device, dtype)
        self.enabled = bool(enabled)
        self.error_lpf = self._zeros(1)
        self.z = self._zeros(1)
        self._reference = self.z

    def normalized(self):
        """Return an independent ``(num_envs, 1)`` Actor input, bounded ±2."""
        if not self.enabled:
            return torch.zeros_like(self.z)
        return self.z / self.PROFILE["actor_scale_m_s"]

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        self.error_lpf[ids] = 0.
        self.z[ids] = 0.
        if env_ids is None:
            self._last_step_id = None

    @torch.no_grad()
    def step_errors(self, height_error, height_rate, valid=True, tilt=0.,
                    clipped=False, dt=.02, *, step_id=None):
        if self._repeat(step_id):
            return self.normalized()
        dt = _dt(dt)
        if not self.enabled:
            self.error_lpf.zero_()
            self.z.zero_()
            self._finish(step_id)
            return self.normalized()

        error, rate, tilt, valid, clipped = (
            self._vector(value) for value in (height_error, height_rate, tilt, valid, clipped)
        )
        update = valid.bool() & ~clipped.bool()
        for value in (error, rate, tilt, valid, clipped):
            update &= torch.isfinite(value)
        update &= tilt.abs() <= self.PROFILE["freeze_tilt_rad"]

        cap = self.PROFILE["error_clip_m"]
        safe_error = torch.nan_to_num(error).clamp(-cap, cap)[:, None]
        alpha = dt / (self.PROFILE["error_lpf_tau_s"] + dt)
        filtered = self.error_lpf + alpha * (safe_error - self.error_lpf)
        self.error_lpf.copy_(torch.where(update[:, None], filtered, self.error_lpf))

        g = quiet_gate(torch.nan_to_num(rate), self.PROFILE["quiet_rate_low_m_s"],
                       self.PROFILE["quiet_rate_high_m_s"])[:, None]
        leakage = g / self.PROFILE["quiet_leak_tau_s"] + (1. - g) / self.PROFILE["moving_leak_tau_s"]
        next_z = self.z + dt * (g * self.error_lpf - leakage * self.z)
        cap = self.PROFILE["integral_clip_m_s"]
        self.z.copy_(torch.where(update[:, None], next_z.clamp(-cap, cap), self.z))
        self._finish(step_id)
        return self.normalized()

    def state_dict(self):
        """Snapshot the fixed profile and temporal state without sharing tensors."""
        return dict(schema=self.SCHEMA, num_envs=self.num_envs, enabled=self.enabled,
                    profile=dict(self.PROFILE), error_lpf=self.error_lpf.clone(),
                    z=self.z.clone(), last_step_id=self._last_step_id)

    @torch.no_grad()
    def load_state_dict(self, state, *, same_world=False):
        """Resume only compatible profiles; a new physical scene clears memory."""
        if state.get("schema") != self.SCHEMA:
            raise ValueError("incompatible height memory state schema")
        if state.get("profile") != dict(self.PROFILE):
            raise ValueError("incompatible height memory profile")
        if state.get("enabled") is not self.enabled:
            raise ValueError("incompatible height memory enabled profile")
        if not same_world or state.get("num_envs") != self.num_envs:
            self.reset()
            return

        step_id = state.get("last_step_id")
        if step_id is not None and (isinstance(step_id, bool) or not isinstance(step_id, int)):
            raise ValueError("invalid height memory state step_id")
        try:
            values = [_as(state[key], self.z) for key in ("error_lpf", "z")]
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("invalid height memory temporal state") from exc
        caps = (self.PROFILE["error_clip_m"], self.PROFILE["integral_clip_m_s"])
        if any(value.shape != self.z.shape or not torch.isfinite(value).all()
               or (value.abs() > cap).any() for value, cap in zip(values, caps)):
            raise ValueError("invalid height memory temporal state")
        if not self.enabled and any(torch.count_nonzero(value) for value in values):
            raise ValueError("disabled height memory state must be zero")
        self.error_lpf.copy_(values[0])
        self.z.copy_(values[1])
        self._last_step_id = step_id
