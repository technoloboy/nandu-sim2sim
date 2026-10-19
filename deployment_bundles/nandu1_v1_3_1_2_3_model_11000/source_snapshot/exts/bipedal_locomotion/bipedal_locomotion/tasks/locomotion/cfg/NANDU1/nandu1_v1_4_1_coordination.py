"""Reward-target and action-history math; no simulator or action filtering."""
import torch
from .nandu1_v1_4_coordination import allocate, StepLatch, target_jerk


def vertical_reference(height_rate, height_error, actual_vz, previous_low, dt, vmax):
    raw = height_rate + 1.5*height_error - .25*(actual_vz-height_rate)
    reference = raw.clamp(-vmax, vmax)
    trial = previous_low + dt/(.8+dt)*(reference-previous_low)
    low = torch.minimum(torch.maximum(trial, torch.minimum(torch.zeros_like(reference), reference)),
                        torch.maximum(torch.zeros_like(reference), reference))
    return reference, low, raw != reference, reference-low


class ActionHistory:
    """Raw sums over 13 joints, with reset action zero as first-order predecessor."""
    SCHEMA = 'nandu1_v1_4_1_action_history_v1'
    def __init__(self, num_envs, device, dtype=torch.float32):
        self.raw = torch.zeros(3, num_envs, 13, device=device, dtype=dtype)
        self.physical = torch.zeros(4, num_envs, 13, device=device, dtype=dtype)
        self.count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.rate_groups = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self.smoothness_groups = torch.zeros_like(self.rate_groups)
        self.physical_delta_groups = torch.zeros_like(self.rate_groups)
        self.jerk = torch.zeros_like(self.raw[0])

    def reset(self, ids):
        self.raw[:, ids] = 0
        self.physical[:, ids] = 0
        self.count[ids] = 0
        self.rate_groups[ids] = self.smoothness_groups[ids] = self.physical_delta_groups[ids] = 0
        self.jerk[ids] = 0

    @staticmethod
    def _groups(values):
        return torch.stack([values[:, :8].square().sum(-1), values[:, 8:11].square().sum(-1),
                            values[:, 11:].square().sum(-1)], -1)

    def update(self, action, dt):
        d1 = action-self.raw[-1]
        d2 = action-2*self.raw[-1]+self.raw[-2]
        self.rate_groups.copy_(self._groups(d1))
        self.smoothness_groups.copy_(self._groups(d2)*(self.count >= 2)[:, None])
        physical = action*action.new_tensor([.25]*11+[1.]*2)
        self.physical_delta_groups.copy_(self._groups(physical-self.physical[-1]))
        self.raw[:-1] = self.raw[1:].clone()
        self.raw[-1] = action
        self.physical[:-1] = self.physical[1:].clone()
        self.physical[-1] = physical
        self.count.add_(1)
        self.jerk.copy_(target_jerk(self.physical, dt)*(self.count >= 4)[:, None])
        return self.rate_groups.sum(-1), self.smoothness_groups.sum(-1)

    def state_dict(self):
        return dict(schema=self.SCHEMA, raw=self.raw.detach().clone(), physical=self.physical.detach().clone(),
                    count=self.count.detach().clone())

    def load_state_dict(self, state):
        if state.get('schema') != self.SCHEMA:
            raise ValueError('incompatible action history schema')
        if tuple(state['count'].shape) != tuple(self.count.shape):
            self.reset(slice(None))
            return
        for name in ('raw', 'physical', 'count'):
            target = getattr(self, name)
            source = torch.as_tensor(state[name], device=target.device, dtype=target.dtype)
            if source.shape != target.shape:
                raise ValueError(f'incompatible action history shape for {name}')
            target.copy_(source)


class StaticDiagnostics:
    """Cumulative stage-bin exposure and measured motion, independent of success."""
    _SUM_FIELDS = ('preparation_seconds', 'hold_seconds', 'bias_sum', 'velocity_square_sum',
                   'attitude_square_sum', 'vertical_acceleration_square_sum', 'drift_square_sum', 'samples',
                   'qualified_samples', 'height_not_reached_samples', 'action_rate_sum', 'action_smoothness_sum',
                   'failures', 'resets')
    _ENV_FIELDS = ('bin', 'origin', 'was_holding', 'previous_vz', 'valid')

    def __init__(self, num_envs, device, dtype=torch.float32):
        for name in self._SUM_FIELDS:
            setattr(self, name, torch.zeros(50, device=device, dtype=dtype))
        self.bin = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.origin = torch.zeros(num_envs, 2, device=device, dtype=dtype)
        self.was_holding = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.previous_vz = torch.zeros(num_envs, device=device, dtype=dtype)
        self.valid = torch.zeros_like(self.was_holding)

    def update(self, bins, holding, bias, vz, attitude, xy, dt, *, action_rate=None, action_smoothness=None):
        active = bins >= 0
        hold = active & holding
        entered = hold & (~self.was_holding | (self.bin != bins))
        self.origin[entered] = xy[entered]
        drift = (xy-self.origin).square().sum(-1)
        acceleration = torch.where(self.valid & (self.bin == bins), (vz-self.previous_vz)/dt, 0.)
        if action_rate is None:
            action_rate = torch.zeros_like(vz)
        if action_smoothness is None:
            action_smoothness = torch.zeros_like(vz)
        qualified = hold & (bias.abs() <= .04) & (vz.abs() <= .08) & (attitude.abs() <= .1745329252)
        values = dict(preparation_seconds=(active & ~hold)*dt, hold_seconds=hold*dt,
                      bias_sum=bias*hold, velocity_square_sum=vz.square()*hold,
                      attitude_square_sum=attitude.square()*hold, vertical_acceleration_square_sum=acceleration.square()*hold,
                      drift_square_sum=drift*hold, samples=hold.to(vz.dtype), qualified_samples=qualified.to(vz.dtype),
                      height_not_reached_samples=(hold & (bias.abs() > .04)).to(vz.dtype),
                      action_rate_sum=action_rate*hold, action_smoothness_sum=action_smoothness*hold)
        for name, value in values.items():
            getattr(self, name).scatter_add_(0, bins[active], value[active].to(vz.dtype))
        self.bin.copy_(bins)
        self.was_holding.copy_(hold)
        self.previous_vz.copy_(vz)
        self.valid.copy_(active)
        return acceleration, drift.sqrt()

    def record_reset(self, ids, failed):
        bins = self.bin[ids]
        active = bins >= 0
        self.resets.scatter_add_(0, bins[active], torch.ones_like(bins[active], dtype=self.resets.dtype))
        self.failures.scatter_add_(0, bins[active], failed[active].to(self.failures.dtype))
        self.clear_episode_memory(ids)

    def clear_episode_memory(self, ids=slice(None)):
        """Retain pooled diagnostics but forget geometry from the former scene."""
        self.bin[ids] = -1
        self.origin[ids] = 0
        self.previous_vz[ids] = 0
        self.was_holding[ids] = self.valid[ids] = False

    def summary(self):
        """One small host transfer for all 5×10 bins, with explicit exposure."""
        denominator = self.samples.clamp_min(1.)
        columns = dict(samples=self.samples, observed=(self.samples > 0).to(self.samples.dtype),
                       preparation_seconds=self.preparation_seconds, hold_seconds=self.hold_seconds,
                       failures=self.failures, resets=self.resets, height_bias=self.bias_sum/denominator,
                       height_velocity_rms=(self.velocity_square_sum/denominator).sqrt(),
                       attitude_rms=(self.attitude_square_sum/denominator).sqrt(),
                       vertical_acceleration_rms=(self.vertical_acceleration_square_sum/denominator).sqrt(),
                       drift_rms=(self.drift_square_sum/denominator).sqrt(),
                       qualified_fraction=self.qualified_samples/denominator,
                       height_not_reached_fraction=self.height_not_reached_samples/denominator,
                       action_rate=self.action_rate_sum/denominator,
                       action_smoothness=self.action_smoothness_sum/denominator)
        rows = torch.stack(tuple(columns.values()), -1).detach().cpu().tolist()
        return {f'V141/static_stage_{i//10}_bin_{i%10}/{name}':value
                for i, row in enumerate(rows) for name, value in zip(columns, row)}

    def state_dict(self):
        return {name:getattr(self, name).detach().clone() for name in self._SUM_FIELDS+self._ENV_FIELDS}

    def load_state_dict(self, state):
        for name in self._SUM_FIELDS+self._ENV_FIELDS:
            dest = getattr(self, name)
            value = torch.as_tensor(state[name], device=dest.device, dtype=dest.dtype)
            if dest.shape == value.shape:
                dest.copy_(value)
