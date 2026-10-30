"""Deployable controller state only. Truth-based diagnostics live elsewhere."""
import math
import torch


def physical_errors(policy, estimate, command):
    gyro = policy[:, :3] / .25
    up = -torch.nn.functional.normalize(policy[:, 3:6], dim=-1, eps=1e-8)
    yaw = (gyro * up).sum(-1)
    return torch.stack((command[:, 0] - yaw, command[:, 1] - estimate[:, 3],
                        command[:, 2] - estimate[:, 6]), -1)


def configuration(policy, estimate):
    return torch.cat((policy[:, 12:23], policy[:, 3:6], policy[:, 9:12], estimate[:, 6:7]), -1)


class ControllerState:
    def __init__(self, num_envs, device="cpu", use_integral=False):
        self.use_integral = use_integral
        self.integral = torch.zeros(num_envs, 3, device=device)
        self.error_lp = torch.zeros_like(self.integral)
        self.slow = torch.zeros(num_envs, 18, device=device)
        self.rate = torch.zeros_like(self.integral)
        self.stop_clock = torch.zeros(num_envs, 1, device=device)
        self.previous_action = torch.zeros(num_envs, 13, device=device)
        self.previous_previous_action = torch.zeros_like(self.previous_action)
        self.last_step = None

    @torch.no_grad()
    def reset(self, ids, policy, estimate):
        self.integral[ids] = 0; self.error_lp[ids] = 0
        self.slow[ids] = configuration(policy, estimate)
        self.rate[ids] = 0; self.stop_clock[ids] = 0
        self.previous_action[ids] = 0; self.previous_previous_action[ids] = 0

    @torch.no_grad()
    def advance(self, policy, estimate, command, rate, clipped, dt, step_id, valid=None):
        if step_id == self.last_step: return
        if self.last_step is not None and step_id < self.last_step:
            raise ValueError("Controller step counter went backwards")
        finite = torch.isfinite(policy).all(-1) & torch.isfinite(estimate).all(-1)
        finite &= torch.isfinite(command).all(-1)
        upright = -policy[:, 5] / policy[:, 3:6].norm(dim=-1).clamp_min(1e-8) > math.cos(math.pi / 4)
        ok = finite & upright
        if valid is not None: ok &= valid
        errors = physical_errors(policy, estimate, command)
        scaled = torch.nan_to_num(errors / errors.new_tensor([.10, .05, .05])).clamp(-3, 3)
        self.error_lp += torch.where(ok[:, None], dt / (.25 + dt) * (scaled - self.error_lp), 0.)
        gate = (1 - rate.abs() / rate.new_tensor([.5, .2, .02])).clamp(0, 1)
        if self.use_integral:
            unwind = self.integral * self.error_lp < 0
            allowed = ok[:, None] & (~clipped[:, None] | unwind)
            self.integral += dt * gate * self.error_lp * allowed
            self.integral.clamp_(-2, 2)
        else:
            self.integral.zero_()
        measured = torch.nan_to_num(configuration(policy, estimate))
        self.slow += torch.where(finite[:, None], dt / (.30 + dt) * (measured - self.slow), 0.)
        stopped = (command[:, 0].abs() < .03) & (command[:, 1].abs() < .02) & (rate[:, 2].abs() < .01)
        self.stop_clock.copy_(torch.where(stopped[:, None], self.stop_clock + dt, 0.))
        self.rate.copy_(rate)
        self.last_step = step_id

    @torch.no_grad()
    def record_issued_action(self, action):
        self.previous_previous_action.copy_(self.previous_action)
        self.previous_action.copy_(action)

    def snapshot(self):
        return torch.cat((self.integral, self.error_lp, self.slow, self.stop_clock,
                          self.previous_previous_action), -1).detach().clone()

    def state_dict(self):
        return {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in vars(self).items()}
