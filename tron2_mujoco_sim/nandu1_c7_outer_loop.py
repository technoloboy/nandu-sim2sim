"""Optional simulation-truth command correction; never imported by a bundle."""
from dataclasses import asdict, dataclass, fields
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OuterLoopConfig:
    velocity_kp: float = 0.
    velocity_ki: float = .1
    height_kp: float = 0.
    height_ki: float = .02
    height_damping: float = 0.
    feedback_tau: float = .5
    velocity_integral_limit: float = .12
    height_integral_limit: float = .05
    velocity_correction_limit: float = .20
    height_correction_limit: float = .08
    velocity_correction_rate: float = .01
    height_correction_rate: float = .001
    warmup_s: float = 1.
    ramp_s: float = 1.
    antiwindup_gain: float = 2.
    velocity_adapt_yaw_limit: float = .1

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Outer configuration {name} must be finite and nonnegative')

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError('Unknown outer configuration fields or non-object JSON')
        return cls(**data)


def _vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{name} must have {size} finite values')
    return a.copy()


def truth_feedback(truth):
    """World TCP velocity projected onto the full-base X axis (wxyz)."""
    position = _vector(truth['tcp'], 3, 'TCP position')
    velocity = _vector(truth['tcp_velocity_api'], 3, 'World TCP velocity')
    q = _vector(truth['base_quaternion'], 4, 'Base quaternion')
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:raise ValueError('Invalid base quaternion')
    w, x, y, z = q / norm
    body_x_world = np.array([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)])
    yaw_rate = truth['yaw_rate']
    if not isinstance(yaw_rate, Real) or not math.isfinite(yaw_rate):
        raise ValueError('Invalid world yaw angular velocity')
    return np.array([yaw_rate, body_x_world @ velocity, position[2]]), velocity


class TruthOuterLoop:
    """Resettable 50Hz PI with output-unit integrals and cascade anti-windup."""
    def __init__(self, config, lower, upper, rate_limits, dt=.02):
        if not isinstance(config, OuterLoopConfig) or dt != .02:
            raise ValueError('Expected OuterLoopConfig and 50Hz period')
        self.config, self.dt = config, dt
        self.lower = _vector(lower, 3, 'Command lower limits')
        self.upper = _vector(upper, 3, 'Command upper limits')
        self.rate_limits = _vector(rate_limits, 3, 'Command rate limits')
        if (self.lower >= self.upper).any() or (self.rate_limits <= 0).any():
            raise ValueError('Invalid command limits')
        self.kp = np.array([config.velocity_kp, config.height_kp])
        self.ki = np.array([config.velocity_ki, config.height_ki])
        self.i_limit = np.array([config.velocity_integral_limit, config.height_integral_limit])
        self.c_limit = np.array([config.velocity_correction_limit, config.height_correction_limit])
        self.c_rate = np.array([config.velocity_correction_rate, config.height_correction_rate])
        self.reset([0., 0., (self.lower[2]+self.upper[2])/2])

    def reset(self, initial_command):
        command = _vector(initial_command, 3, 'Initial command')
        self.previous = np.clip(command, self.lower, self.upper)
        self.integral = np.zeros(2)
        self.correction = np.zeros(2)
        self.filtered_velocity = None
        self.last_step, self.last_result = None, None

    @staticmethod
    def _copy(result):
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in result.items()}

    @np.errstate(over='raise', invalid='raise', divide='raise')
    def update(self, reference, truth, step_id):
        reference = _vector(reference, 3, 'Command reference')
        feedback, world_velocity = truth_feedback(truth)
        if isinstance(step_id, bool) or not isinstance(step_id, (int, np.integer)) or step_id < 0:
            raise ValueError('Invalid outer step index')
        if self.last_step is not None and step_id == self.last_step:
            return self._copy(self.last_result)
        expected = 0 if self.last_step is None else self.last_step + 1
        if step_id != expected:raise ValueError('Outer control steps must be consecutive')
        # Validate first; invalid input cannot poison integrators/clock/filter.
        previous_velocity = self.filtered_velocity
        alpha = self.dt/(self.config.feedback_tau+self.dt)
        filtered = world_velocity.copy() if previous_velocity is None else previous_velocity + alpha*(world_velocity-previous_velocity)
        filtered_feedback, _ = truth_feedback(dict(truth, tcp_velocity_api=filtered))
        error = reference[1:] - filtered_feedback[1:]
        elapsed = step_id*self.dt - self.config.warmup_s
        ramp = 0. if elapsed < 0 else (1. if self.config.ramp_s == 0 else min(1., elapsed/self.config.ramp_s))
        ready = ramp >= 1.
        candidate_i = self.integral.copy()
        # This state estimates a slow straight-line trim, not the reversible
        # kinematic/dynamic error of a turning TCP. Keep trim during turns, but
        # do not learn turning error as a persistent straight-line bias.
        adapting = np.array([ready and abs(reference[0]) <= self.config.velocity_adapt_yaw_limit, ready])
        candidate_i = np.clip(candidate_i+self.ki*error*self.dt*adapting, -self.i_limit, self.i_limit)
        raw_delta = self.kp*error + candidate_i
        raw_delta[1] -= self.config.height_damping*filtered[2]
        raw_delta *= ramp
        delta = np.clip(raw_delta, -self.c_limit, self.c_limit)
        delta = self.correction + np.clip(delta-self.correction, -self.c_rate*self.dt, self.c_rate*self.dt)
        unconstrained = reference.copy(); unconstrained[1:] += delta
        bounded = np.clip(unconstrained, self.lower, self.upper)
        command = self.previous + np.clip(bounded-self.previous, -self.rate_limits*self.dt, self.rate_limits*self.dt)
        command = np.clip(command, self.lower, self.upper)
        actual_delta = command[1:] - reference[1:]
        if ready:
            candidate_i += self.config.antiwindup_gain*self.dt*(actual_delta-raw_delta)
            candidate_i = np.clip(candidate_i, -self.i_limit, self.i_limit)
        candidate_i[self.ki == 0] = 0.
        raw_command = reference.copy(); raw_command[1:] += raw_delta
        result = dict(command=command, rate=(command-self.previous)/self.dt,
            correction=np.r_[0., actual_delta], integral=np.r_[0., candidate_i],
            requested_correction=np.r_[0., raw_delta], candidate_correction=np.r_[0., delta],
            correction_anchor=np.r_[0., self.correction],
            command_amplitude_limited=np.abs(bounded-unconstrained)>1e-7,
            command_slew_limited=np.abs(command-bounded)>1e-7,
            safety_override=np.abs(command-unconstrained)>1e-7,
            saturated=np.abs(command-raw_command)>1e-7, feedback=feedback,
            filtered_velocity_world=filtered, ready=ready, adapting=np.r_[False, adapting])
        if any(not np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError('Non-finite outer controller arithmetic; update not committed')
        self.previous = command.copy()
        # Track what the downstream actuator/reference bounds really allowed;
        # an unissued correction must not reappear when the limit releases.
        self.correction = np.clip(actual_delta, -self.c_limit, self.c_limit)
        self.integral = candidate_i
        self.filtered_velocity = filtered.copy()
        self.last_step, self.last_result = int(step_id), self._copy(result)
        return self._copy(result)
