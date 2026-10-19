"""Pure tensor reward/allocation math, deliberately isolated from v1.3."""
import torch


class StepLatch:
    """Give one stateful update owner to each control step."""
    def __init__(self):
        self.step = -1

    def begin(self, step):
        if self.step == step:
            return False
        self.step = step
        return True


def low_gate(value, low, high):
    x = ((value.abs() - low) / (high - low)).clamp(0., 1.)
    return 1. - x.square() * (3. - 2. * x)


def static_gates(command, height_error, height_rate):
    intent = low_gate(command[:, 0], .03, .10) * low_gate(command[:, 1], .02, .08)
    intent = intent * low_gate(height_rate, .01, .04)
    return intent, intent * low_gate(height_error, .02, .06)


def lateral_hold_gate(command):
    return low_gate(command[:, 0], .03, .10) * low_gate(command[:, 1], .02, .08)


def hold_huber(error, scale, cap, hold_gate):
    z = (error / scale).abs()
    huber = torch.where(z <= 1., .5 * z.square(), z - .5).clamp_max(cap)
    return huber * (1. + .75 * hold_gate)


def allocate(reference, low_frequency, alpha_base, arm_high_frequency_availability):
    high = reference - low_frequency
    beta = arm_high_frequency_availability.clamp(0., 1.)
    base = alpha_base * low_frequency + (1. - beta) * high
    return base, reference - base


def applied_intent(goal, mode, active_axes):
    """Copy high-level metadata at application, never alias the next queued input."""
    # Rollouts use inference_mode, but manual reset/checkpoint restore may not.
    # Persistent state must remain writable in both contexts.
    with torch.inference_mode(False):
        return dict(goal=goal.detach().clone(), mode=mode.detach().clone(), active_axes=active_axes.detach().clone())


def target_jerk(history, dt):
    """history is oldest→newest physical targets (4,N,13); units rad/s^3."""
    position = (history[3, :, :11] - 3 * history[2, :, :11]
                + 3 * history[1, :, :11] - history[0, :, :11]) / dt ** 3
    wheel = (history[3, :, 11:] - 2 * history[2, :, 11:] + history[1, :, 11:]) / dt ** 2
    return torch.cat((position, wheel), dim=-1)
