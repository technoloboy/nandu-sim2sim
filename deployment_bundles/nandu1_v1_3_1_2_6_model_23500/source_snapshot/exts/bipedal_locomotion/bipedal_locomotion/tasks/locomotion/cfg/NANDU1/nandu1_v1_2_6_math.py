"""Persistent installation error, distinct from per-control-step sensor noise."""
from __future__ import annotations

import math
import torch


STAGE_AXIS_LIMITS_DEG = (0., .25, .5, 1., 2.)
SENSORS = ('base_imu', 'up3_imu')


def fixed_rpy(value):
    if value is None:
        return None
    if len(value) != 3 or not all(math.isfinite(float(x)) for x in value):
        raise ValueError('IMU mount RPY requires three finite angles in degrees')
    return tuple(float(x) for x in value)


def sensor_to_body_rotation(rpy_deg):
    """R_body_sensor = Rz(yaw) Ry(pitch) Rx(roll), all batch rows independent."""
    roll, pitch, yaw = torch.deg2rad(rpy_deg).unbind(-1)
    cr, sr, cp, sp, cy, sy = roll.cos(), roll.sin(), pitch.cos(), pitch.sin(), yaw.cos(), yaw.sin()
    return torch.stack((
        cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
        sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
        -sp, cp*sr, cp*cr,
    ), dim=-1).reshape(*rpy_deg.shape[:-1], 3, 3)


class IMUInstallationMeasurements:
    """Two episode-fixed mounts and one measurement per IMU/channel/control tick.

    The cache is separate from the legacy joint-noise cache: a partial physics
    reset may clear that cache globally, but must not re-noise other environments'
    already-read IMU measurements in the same tick. No simulator data is written.
    """
    def __init__(self, num_envs, device, dtype=torch.float32):
        self.num_envs, self.device = num_envs, device
        self.rpy_deg = {name: torch.zeros(num_envs, 3, device=device, dtype=dtype) for name in SENSORS}
        self.rotations = {name: torch.eye(3, device=device, dtype=dtype).repeat(num_envs, 1, 1) for name in SENSORS}
        self.stage_at_reset = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._cache_step = None
        self._cache = {}
        self._dirty_ids = {}

    def reset(self, env_ids, stage, randomize=True, base_rpy_deg=None, up3_rpy_deg=None):
        if isinstance(stage, bool) or stage not in range(5):
            raise ValueError('IMU installation stage must be 0..4')
        fixed = dict(zip(SENSORS, (fixed_rpy(base_rpy_deg), fixed_rpy(up3_rpy_deg))))
        ids = torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, slice):
            ids = ids[env_ids]
        elif env_ids is not None:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not len(ids):
            return
        bound = STAGE_AXIS_LIMITS_DEG[stage] if randomize else 0.
        for name in SENSORS:
            angles = self.rpy_deg[name].new_zeros(len(ids), 3)
            if fixed[name] is not None:
                angles[:] = angles.new_tensor(fixed[name])
            elif bound > 0:
                angles.uniform_(-bound, bound)
            # Zero-error resets do not consume RNG, preserving legacy noise.
            self.rpy_deg[name][ids] = angles
            self.rotations[name][ids] = sensor_to_body_rotation(angles)
        self.stage_at_reset[ids] = stage
        for key in self._cache:
            previous = self._dirty_ids.get(key)
            self._dirty_ids[key] = ids if previous is None else torch.unique(torch.cat((previous, ids)))

    def rotate(self, sensor_name, vectors, env_ids=None):
        if sensor_name not in self.rotations:
            raise ValueError('Unknown installation-randomized IMU: ' + sensor_name)
        rotation = self.rotations[sensor_name]
        if env_ids is not None:
            rotation = rotation[env_ids]
        rotation = rotation.to(device=vectors.device, dtype=vectors.dtype)
        return torch.bmm(rotation.transpose(-1,-2), vectors.unsqueeze(-1)).squeeze(-1)

    def observe(self, sensor_name, kind, true_vectors, step, noise_amplitude=0.):
        if kind not in ('gyro', 'gravity'):
            raise ValueError('IMU installation handles only gyro/gravity channels')
        if self._cache_step != step:
            self._cache_step, self._cache, self._dirty_ids = step, {}, {}
        key = (sensor_name, kind)
        if key not in self._cache:
            value = self.rotate(sensor_name, true_vectors)
            if noise_amplitude > 0:
                value = value + torch.empty_like(value).uniform_(-noise_amplitude, noise_amplitude)
            self._cache[key] = value
        elif key in self._dirty_ids:
            ids = self._dirty_ids.pop(key)
            value = self.rotate(sensor_name, true_vectors[ids], ids)
            if noise_amplitude > 0:
                value = value + torch.empty_like(value).uniform_(-noise_amplitude, noise_amplitude)
            self._cache[key][ids] = value
        return self._cache[key]
