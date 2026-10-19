"""Scheme B math: GPU batch costs and a slow, audit-rate weight controller."""
import math
import torch


def mix_hold_commands(sampled, issued_height, uniform, probability=0.15):
    if not 0.0 <= probability <= 1.0:
        raise ValueError('hold probability must be in [0,1]')
    hold = uniform < probability
    result = sampled.clone()
    result[hold, :2] = 0.0
    result[hold, 2] = issued_height[hold]
    return result, hold


def hold_phase(age):
    fraction = ((age - 0.5) / 0.5).clamp(0., 1.)
    return fraction.square() * (3. - 2. * fraction)


def static_intent_gate(command):
    limits = command.new_tensor([.03, .02])
    widths = command.new_tensor([.07, .06])
    x = ((command[:, :2].abs() - limits) / widths).clamp(0., 1.)
    return (1. - x.square() * (3. - 2. * x)).prod(dim=-1)


def channel_motion_cost(base_velocity, base_angular_velocity, up3_velocity, wheel_velocity):
    # 2*Huber matches the original square locally. Each measurement component
    # owns its cap; a saturated wheel cannot hide improving base/up3 motion.
    result = torch.zeros_like(base_velocity[:, 0])
    for value, scale in ((base_velocity, .10), (base_angular_velocity, .15),
                         (up3_velocity, .10), (wheel_velocity, 1.0)):
        z = (value / scale).abs()
        result += torch.where(z <= 1., z.square(), 2. * z - 1.).clamp(max=8.).sum(dim=-1)
    return result


class SlowRewardAdapter:
    """CPU work only at an audit/rollout boundary, never an optimizer target.

    active is immutable during rollout+PPO; audit changes pending only.
    Reason codes: 0 updated, 1 warmup, 2 insufficient, 3 tracking/safety guard,
    4 non-finite, 5 too early. A guard does not hide the original safety score.
    """
    def __init__(self):
        self.active = [1., 1.]
        self.pending = [1., 1.]
        self.ema = [0., 0.]
        self.valid_audits = 0
        self.last_iteration = -250
        self.reference_tracking = None
        self.stage = -1
        self.reason = 1

    def set_stage(self, stage):
        if self.stage != stage:
            self.stage = int(stage)
            self.valid_audits = 0
            self.reference_tracking = None
            self.ema = [0., 0.]

    def begin_rollout(self):
        self.active = self.pending.copy()

    def audit(self, *, iteration, raw, good, bad, samples, environments, tracking, safety_veto):
        if iteration - self.last_iteration < 250:
            self.reason = 5
            return
        self.last_iteration = int(iteration)
        if samples < 256 or environments < 32:
            self.reason = 2
            return
        if not all(math.isfinite(float(x)) for x in (*raw, *tracking)):
            self.reason = 4
            return
        if any(b <= g for g, b in zip(good, bad)):
            raise ValueError('adaptive bad bounds must exceed targets')
        if safety_veto:
            self.reason = 3
            return
        if self.reference_tracking is None:
            self.reference_tracking = list(tracking)
        floors = (.02, .02, .01)
        guard = safety_veto or any(
            value > 1.10 * max(reference, floor)
            for value, reference, floor in zip(tracking, self.reference_tracking, floors)
        )
        if guard:
            self.reason = 3
            return
        self.valid_audits += 1
        for i in range(2):
            error = max(-1., min(1., (float(raw[i]) - good[i]) / (bad[i] - good[i])))
            if abs(error) < .05:
                error = 0.
            self.ema[i] = .9 * self.ema[i] + .1 * error
        if self.valid_audits <= 2:
            self.reason = 1
            return
        self.pending = [max(.5, min(3., value * math.exp(.03 * error)))
                        for value, error in zip(self.active, self.ema)]
        self.reason = 0

    def state_dict(self):
        return dict(active=self.active.copy(), pending=self.pending.copy(), ema=self.ema.copy(),
                    valid_audits=self.valid_audits, last_iteration=self.last_iteration,
                    reference_tracking=None if self.reference_tracking is None else self.reference_tracking.copy(),
                    stage=self.stage, reason=self.reason)

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError('incomplete adaptive reward state')
        for key in ('active', 'pending', 'ema'):
            values = state[key]
            if len(values) != 2 or not all(math.isfinite(float(x)) for x in values):
                raise ValueError('invalid adaptive vector')
            low, high = (-1., 1.) if key == 'ema' else (.5, 3.)
            if any(not low <= x <= high for x in values):
                raise ValueError('adaptive vector outside bounds')
        reference = state['reference_tracking']
        if reference is not None and (len(reference) != 3 or not all(math.isfinite(x) and x >= 0 for x in reference)):
            raise ValueError('invalid tracking reference')
        for key in ('valid_audits', 'last_iteration', 'stage', 'reason'):
            if isinstance(state[key], bool) or not isinstance(state[key], int):
                raise ValueError('invalid adaptive counter')
        if state['valid_audits'] < 0 or not -1 <= state['stage'] <= 4:
            raise ValueError('invalid adaptive stage/count')
        self.active, self.pending, self.ema = (list(state[k]) for k in ('active','pending','ema'))
        self.reference_tracking = None if reference is None else list(reference)
        for key in ('valid_audits','last_iteration','stage','reason'):
            setattr(self, key, state[key])
