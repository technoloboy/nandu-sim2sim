"""Deployable Torch command processing; this module has no simulator dependency.

High-level inputs are heading-change rate, Heading forward speed, absolute
height. Outputs are world-Z yaw rate, Heading speed, absolute applied height.
Only IMU gravity and angular velocity are used to estimate relative heading.
"""
from __future__ import annotations

import math

import torch


STAGE_LIMITS = ((0.20, 0.08), (0.50, 0.20), (0.85, 0.40), (1.20, 0.70), (1.57, 1.00))
STAGE_HEIGHT_RANGES = ((0.78, 1.04), (0.68, 1.12), (0.58, 1.22), (0.48, 1.32), (0.40, 1.40))
HEIGHT_RATE_LIMITS = (0.05, 0.10, 0.20, 0.35, 0.50)
SMALL_STEP_LIMITS = (0.03, 0.05, 0.10, 0.10, 0.10)
LARGE_STEP_LIMITS = (0.0, 0.10, 0.18, 0.30, 0.45)
HEIGHT_MODE_PROBABILITIES = ((.80, .20, .0), (.70, .25, .05), (.60, .25, .15), (.60, .25, .15), (.60, .25, .15))
DT_MIN = 1.0e-6
DT_MAX = 0.1


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _heading_frame(gravity_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthonormal Heading axes expressed in the body frame, including a pole fallback."""
    norm = torch.linalg.vector_norm(gravity_b, dim=-1, keepdim=True)
    up_default = torch.zeros_like(gravity_b)
    up_default[..., 2] = 1
    up = torch.where(norm > 1e-6, -gravity_b / norm.clamp_min(1e-6), up_default)
    ex = torch.zeros_like(up)
    ex[..., 0] = 1
    forward = ex - up[..., :1] * up
    fallback = torch.zeros_like(up)
    fallback[..., 1] = 1
    fallback = fallback - up[..., 1:2] * up
    forward_norm = torch.linalg.vector_norm(forward, dim=-1, keepdim=True)
    forward = torch.where(forward_norm > 1e-6, forward, fallback)
    forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(1e-6)
    left = torch.linalg.cross(up, forward, dim=-1)
    return forward, left, up


def body_to_heading(velocity: torch.Tensor, gravity_b: torch.Tensor) -> torch.Tensor:
    """Rotate (...,3) body vectors into Heading coordinates using the same-frame IMU gravity.

    Leading dimensions broadcast, so one gravity sample may transform both
    Encoder velocity heads. The gravity magnitude need not be exactly one.
    """
    if velocity.shape[-1:] != (3,) or gravity_b.shape[-1:] != (3,):
        raise ValueError("velocity and gravity must end in dimension 3")
    gravity_b = gravity_b.to(device=velocity.device, dtype=velocity.dtype)
    forward, left, up = _heading_frame(gravity_b)
    return torch.stack(tuple((velocity * axis).sum(-1) for axis in (forward, left, up)), dim=-1)


class CommandProcessor:
    """Batched state machine for mode/axis sampling, yaw hold, and bounded height targets.

    ``mode`` is the number of active axes (0..3); ``height_mode`` is slew/small
    step/large step (0..2). ``is_step`` marks the action update that applies a
    queued finite step, and its physical ``height_rate`` is zero. ``trend`` uses
    the applied command difference and remains signed and bounded for steps.
    Reset holds the supplied height indefinitely until resample/set_external.
    Sampling and IMU integration happen only in ``step``, not set_external.
    """

    STAGE_LIMITS = STAGE_LIMITS
    STAGE_HEIGHT_RANGES = STAGE_HEIGHT_RANGES
    HEIGHT_RATE_LIMITS = HEIGHT_RATE_LIMITS
    SMALL_STEP_LIMITS = SMALL_STEP_LIMITS
    LARGE_STEP_LIMITS = LARGE_STEP_LIMITS
    HEIGHT_MODE_PROBABILITIES = HEIGHT_MODE_PROBABILITIES
    DT_MIN = DT_MIN
    DT_MAX = DT_MAX
    SCHEMA = "nandu1_v1_4_command_processor_v1"
    _STATE_FIELDS = (
        "commands", "goal", "trend", "height_rate", "is_step", "mode", "active_axes",
        "height_mode", "mode_time_left", "axis_time_left", "heading", "heading_target",
        "_was_turning", "_pending_step", "_external",
    )

    def __init__(self, num_envs: int, device: str | torch.device, dtype: torch.dtype = torch.float32):
        if int(num_envs) != num_envs or num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not dtype.is_floating_point:
            raise ValueError("commands require a floating dtype")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.heading_kp = 1.5
        self.yaw_deadzone = 1e-5
        self._generator = torch.Generator(device=self.device).manual_seed(torch.initial_seed())
        self.commands = torch.zeros(self.num_envs, 3, device=self.device, dtype=dtype)
        self.goal = torch.zeros_like(self.commands)
        self.trend = torch.zeros_like(self.commands)
        self.height_rate = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self.is_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.active_axes = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.bool)
        self.height_mode = torch.zeros_like(self.mode)
        self.mode_time_left = torch.full_like(self.height_rate, float("inf"))
        self.axis_time_left = torch.full_like(self.commands, float("inf"))
        self.heading = torch.zeros_like(self.height_rate)
        self.heading_target = torch.zeros_like(self.height_rate)
        self._was_turning = torch.zeros_like(self.is_step)
        self._pending_step = torch.zeros_like(self.is_step)
        self._external = torch.zeros_like(self.is_step)
        self.reset(None, .9)

    def _ids(self, ids) -> torch.Tensor:
        all_ids = torch.arange(self.num_envs, device=self.device)
        if ids is None:
            return all_ids
        if isinstance(ids, slice):
            return all_ids[ids]
        selected = torch.as_tensor(ids, device=self.device)
        if selected.dtype == torch.bool:
            if selected.shape != (self.num_envs,):
                raise ValueError("boolean ids must match num_envs")
            return selected.nonzero(as_tuple=False).flatten()
        selected = selected.to(torch.long).reshape(-1)
        if ((selected < 0) | (selected >= self.num_envs)).any():
            raise ValueError("environment ids out of range")
        return selected

    @staticmethod
    def _stage(stage: int) -> int:
        if int(stage) != stage or stage not in range(5):
            raise ValueError("stage must be 0..4")
        return int(stage)

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, device=self.device, dtype=self.dtype, generator=self._generator)

    def reset(self, ids, height) -> None:
        """Clear only selected environments and hold their supplied absolute height."""
        ids = self._ids(ids)
        height = torch.as_tensor(height, device=self.device, dtype=self.dtype).flatten()
        if height.numel() == self.num_envs and height.numel() != ids.numel():
            height = height[ids]
        if height.numel() not in (1, ids.numel()) or not torch.isfinite(height).all():
            raise ValueError("height must be finite and scalar or match selected environments")
        height = height.clamp(.40, 1.40)
        for name in self._STATE_FIELDS:
            if name != "_external":
                getattr(self, name)[ids] = 0
        self.commands[ids, 2] = height
        self.goal[ids, 2] = height
        self.mode_time_left[ids] = float("inf")
        self.axis_time_left[ids] = float("inf")

    def _sample_axes(self, selected: torch.Tensor, stage: int) -> None:
        yaw_limit, vx_limit = STAGE_LIMITS[stage]
        lo, hi = STAGE_HEIGHT_RANGES[stage]
        values = self._rand(self.num_envs, 3)
        self.goal[:, 0] = torch.where(selected[:, 0], (2 * values[:, 0] - 1) * yaw_limit, self.goal[:, 0])
        self.goal[:, 1] = torch.where(selected[:, 1], (2 * values[:, 1] - 1) * vx_limit, self.goal[:, 1])
        rate_p, small_p, _ = HEIGHT_MODE_PROBABILITIES[stage]
        choice = self._rand(self.num_envs)
        height_mode = (choice >= rate_p).long() + (choice >= rate_p + small_p).long()
        amplitude = self._rand(self.num_envs)
        small = .01 + amplitude * (SMALL_STEP_LIMITS[stage] - .01)
        large = SMALL_STEP_LIMITS[stage] + amplitude * (LARGE_STEP_LIMITS[stage] - SMALL_STEP_LIMITS[stage])
        direction = torch.where(self._rand(self.num_envs) < .5, -1., 1.)
        step_target = self.commands[:, 2] + direction * torch.where(height_mode == 1, small, large)
        target = torch.where(height_mode == 0, lo + (hi - lo) * values[:, 2], step_target).clamp(lo, hi)
        self.goal[:, 2] = torch.where(selected[:, 2], target, self.goal[:, 2])
        self.height_mode[:] = torch.where(selected[:, 2], height_mode, self.height_mode)
        self._pending_step[:] = torch.where(selected[:, 2], height_mode != 0, self._pending_step)
        self.axis_time_left[:] = torch.where(selected, 4 + 4 * self._rand(self.num_envs, 3), self.axis_time_left)

    def resample(self, ids, stage: int) -> None:
        """Choose mode first, then targets/timers of its active axes; inactive height is retained."""
        stage = self._stage(stage)
        ids = self._ids(ids)
        if ids.numel() == 0:
            return
        u = self._rand(ids.numel())
        mode = (u >= .2).long() + (u >= .5).long() + (u >= .8).long()
        chosen_axis = (self._rand(ids.numel()) * 3).long()
        axis_index = torch.arange(3, device=self.device)[None, :]
        active = ((mode[:, None] == 1) & (axis_index == chosen_axis[:, None])) | (
            (mode[:, None] == 2) & (axis_index != chosen_axis[:, None])) | (mode[:, None] == 3)
        self.mode[ids] = mode
        self.active_axes[ids] = active
        dwell = self._rand(ids.numel())
        self.mode_time_left[ids] = torch.where(mode == 0, 8 + 12 * dwell, 4 + 4 * dwell)
        self.axis_time_left[ids] = float("inf")
        self.goal[ids, :2] = 0
        self._pending_step[ids] = False
        self.height_mode[ids] = 0
        self._external[ids] = False
        selected = torch.zeros_like(self.active_axes)
        selected[ids] = active
        self._sample_axes(selected, stage)

    def set_external(self, command: torch.Tensor, stage: int) -> None:
        """Queue 3D high-level commands (broadcast or N×3), applying no time integration."""
        stage = self._stage(stage)
        command = torch.as_tensor(command, device=self.device, dtype=self.dtype)
        if command.shape not in ((3,), (1, 3), (self.num_envs, 3)) or not torch.isfinite(command).all():
            raise ValueError("external command must be finite 3D or num_envs by 3")
        goal = command.expand(self.num_envs, 3).clone()
        yaw_limit, vx_limit = STAGE_LIMITS[stage]
        goal[:, 0].clamp_(-yaw_limit, yaw_limit)
        goal[:, 1].clamp_(-vx_limit, vx_limit)
        goal[:, 2].clamp_(*STAGE_HEIGHT_RANGES[stage])
        self.goal.copy_(goal)
        self._external[:] = True
        self.active_axes[:, :2] = goal[:, :2].abs() > self.yaw_deadzone
        self.active_axes[:, 2] = (goal[:, 2] - self.commands[:, 2]).abs() > 1e-6
        self.mode.copy_(self.active_axes.sum(-1))
        self.height_mode.zero_()
        self._pending_step.zero_()
        self.mode_time_left[:] = float("inf")
        self.axis_time_left[:] = float("inf")

    def step(self, dt: float, gravity_b: torch.Tensor, gyro_b: torch.Tensor, stage: int) -> torch.Tensor:
        """Advance exactly one physical action period and return the live N×3 command tensor."""
        dt = float(dt)
        if not math.isfinite(dt) or not DT_MIN <= dt <= DT_MAX:
            raise ValueError(f"dt must be finite in [{DT_MIN}, {DT_MAX}]")
        stage = self._stage(stage)
        gravity_b = torch.as_tensor(gravity_b, device=self.device, dtype=self.dtype)
        gyro_b = torch.as_tensor(gyro_b, device=self.device, dtype=self.dtype)
        if gravity_b.shape != (self.num_envs, 3) or gyro_b.shape != (self.num_envs, 3):
            raise ValueError("IMU inputs must have shape (num_envs,3)")
        if not torch.isfinite(gravity_b).all() or not torch.isfinite(gyro_b).all():
            raise ValueError("IMU inputs must be finite")
        previous = self.commands.clone()
        self.mode_time_left -= dt
        self.axis_time_left -= dt
        self.resample(((self.mode_time_left <= 0) & ~self._external).nonzero(as_tuple=False).flatten(), stage)
        due = (self.axis_time_left <= 0) & self.active_axes & ~self._external[:, None]
        if due.any():
            self._sample_axes(due, stage)
        yaw_limit, vx_limit = STAGE_LIMITS[stage]
        lo, hi = STAGE_HEIGHT_RANGES[stage]
        self.goal[:, 0].clamp_(-yaw_limit, yaw_limit)
        self.goal[:, 1].clamp_(-vx_limit, vx_limit)
        self.goal[:, 2].clamp_(lo, hi)
        # Heading rate of the projected body-forward axis, expressed entirely in
        # body IMU coordinates. This includes roll/pitch coupling, unlike gyro_z.
        _, left, up = _heading_frame(gravity_b)
        projection = (1 - up[:, 0].square()).clamp_min(1e-4).sqrt()
        ex = torch.zeros_like(gyro_b)
        ex[:, 0] = 1
        heading_rate = (torch.linalg.cross(gyro_b, ex, dim=-1) * left).sum(-1) / projection
        heading_rate = torch.where(projection > .01, heading_rate, (gyro_b * up).sum(-1))
        self.heading[:] = _wrap_angle(self.heading + heading_rate * dt)
        joystick = self.goal[:, 0]
        turning = joystick.abs() > self.yaw_deadzone
        released = self._was_turning & ~turning
        self.heading_target[:] = torch.where(released, self.heading, self.heading_target)
        self.heading_target[:] = _wrap_angle(self.heading_target + torch.where(turning, joystick * dt, 0.))
        heading_error = _wrap_angle(self.heading_target - self.heading)
        # The outer loop requests Euler Heading rate. The public actuator
        # target is angular velocity projected on world Z. Their difference
        # is the current instantaneous tilt-rotation contribution from IMU.
        tilt_coupling = (gyro_b * up).sum(-1) - heading_rate
        raw_yaw = joystick + self.heading_kp * heading_error + tilt_coupling
        applied_yaw = raw_yaw.clamp(-yaw_limit, yaw_limit)
        # Back calculation removes target integration that the actuator limit
        # cannot realize; the feedforward term remains the requested turn rate.
        saturated = raw_yaw != applied_yaw
        self.heading_target[:] = _wrap_angle(torch.where(saturated,
            self.heading + (applied_yaw - tilt_coupling - joystick) / self.heading_kp, self.heading_target))
        self._was_turning.copy_(turning)
        self.commands[:, 0] = applied_yaw
        self.commands[:, 1] = self.goal[:, 1]
        delta = (self.goal[:, 2] - self.commands[:, 2]).clamp(-HEIGHT_RATE_LIMITS[stage] * dt,
                                                                    HEIGHT_RATE_LIMITS[stage] * dt)
        # A shrinking evaluation-stage envelope can require an immediate clamp;
        # that bounded discontinuity is also a step, never a velocity reference.
        self.is_step.copy_(self._pending_step | (previous[:, 2] < lo) | (previous[:, 2] > hi))
        self.commands[:, 2] = torch.where(self.is_step, self.goal[:, 2], self.commands[:, 2] + delta).clamp(lo, hi)
        self._pending_step.zero_()
        self.height_rate[:] = torch.where(self.is_step, 0., (self.commands[:, 2] - previous[:, 2]) / dt)
        scales = self.commands.new_tensor((yaw_limit, vx_limit, HEIGHT_RATE_LIMITS[stage]))
        self.trend.copy_(((self.commands - previous) / dt / scales).clamp(-1, 1))
        return self.commands

    def state_dict(self) -> dict:
        """Return detached copies, including sampling RNG and all per-environment state."""
        state = {name: getattr(self, name).detach().clone() for name in self._STATE_FIELDS}
        state.update(schema=self.SCHEMA, num_envs=self.num_envs, heading_kp=self.heading_kp,
                     yaw_deadzone=self.yaw_deadzone, rng_state=self._generator.get_state().clone())
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore compatible state; differing batch size resets ephemeral environment state.

        Configuration and RNG survive a count change, but episodes from unrelated
        environment indices must not be copied/truncated into the new batch.
        The receiving batch keeps its current applied height as its reset hold.
        """
        if state.get("schema") != self.SCHEMA:
            raise ValueError("incompatible command processor schema")
        kp = float(state.get("heading_kp", self.heading_kp))
        deadzone = float(state.get("yaw_deadzone", self.yaw_deadzone))
        if not math.isfinite(kp) or kp <= 0 or not math.isfinite(deadzone) or deadzone < 0:
            raise ValueError("invalid heading configuration")
        compatible = state.get("num_envs") == self.num_envs
        converted = {}
        if compatible:
            for name in self._STATE_FIELDS:
                target = getattr(self, name)
                source = torch.as_tensor(state[name], device=self.device, dtype=target.dtype)
                if source.shape != target.shape:
                    raise ValueError(f"incompatible state shape for {name}")
                converted[name] = source
        self.heading_kp, self.yaw_deadzone = kp, deadzone
        if compatible:
            for name, source in converted.items():
                getattr(self, name).copy_(source)
        else:
            self.reset(None, self.commands[:, 2].clone())
        if "rng_state" in state:
            self._generator.set_state(torch.as_tensor(state["rng_state"], device="cpu", dtype=torch.uint8))
