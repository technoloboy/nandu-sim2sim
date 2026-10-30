"""Causal references. Fixed tensor operations, no mechanism or trajectory solver."""
import math
import torch

HEIGHT_LIMITS = ((.78, 1.04), (.68, 1.12), (.58, 1.22), (.48, 1.32), (.4, 1.4))
HEIGHT_RATES = (.05, .10, .20, .35, .50)
YAW_LIMITS = (.20, .50, .85, 1.20, 1.57)
VX_LIMITS = (.08, .20, .40, .70, 1.)


def wrap(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class CommandProcessor:
    """One update per physical control step; all state is explicit and resettable."""
    def __init__(self, num_envs, device="cpu"):
        self.command = torch.zeros(num_envs, 3, device=device)
        self.filtered = torch.zeros_like(self.command)
        self.request = torch.zeros_like(self.command)
        self.rate = torch.zeros_like(self.command)
        self.heading = self.command[:, 0].clone()
        self.heading_target = self.heading.clone()
        self.was_turning = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.last_step = None

    def reset(self, ids, height):
        for value in (self.command, self.filtered, self.request, self.rate):
            value[ids] = 0
        for value in (self.command, self.filtered, self.request):
            value[ids, 2] = height
        self.heading[ids] = 0
        self.heading_target[ids] = 0
        self.was_turning[ids] = False

    @torch.no_grad()
    def advance(self, request, gravity, gyro, dt, stage, step_id):
        if not .001 <= dt <= .1 or stage not in range(5):
            raise ValueError("Invalid control period/stage")
        if step_id == self.last_step:
            return self.command
        if self.last_step is not None and step_id < self.last_step:
            raise ValueError("Step counter moved backwards")
        previous = self.command.clone()
        self.request.copy_(request)
        target = request.clone()
        target[:, 0].clamp_(-YAW_LIMITS[stage], YAW_LIMITS[stage])
        target[:, 1].clamp_(-VX_LIMITS[stage], VX_LIMITS[stage])
        target[:, 2].clamp_(*HEIGHT_LIMITS[stage])
        alpha = target.new_tensor([dt / (.10 + dt), dt / (.15 + dt), dt / (.15 + dt)])
        bound = target.new_tensor([2., .6, HEIGHT_RATES[stage]]) * dt
        self.filtered += ((target - self.filtered) * alpha).clamp(-bound, bound)
        # The same noisy IMU vectors feed the policy and the relative heading loop.
        up = -torch.nn.functional.normalize(gravity, dim=-1, eps=1e-8)
        ex = torch.zeros_like(up); ex[:, 0] = 1
        ey = torch.zeros_like(up); ey[:, 1] = 1
        forward = ex - up[:, :1] * up
        fallback = ey - up[:, 1:2] * up
        forward = torch.where(forward.norm(dim=-1, keepdim=True) < 1e-5, fallback, forward)
        forward = torch.nn.functional.normalize(forward, dim=-1, eps=1e-8)
        left = torch.linalg.cross(up, forward)
        projection = (1 - up[:, 0].square()).clamp_min(0).sqrt()
        world_rate = (gyro * up).sum(-1)
        heading_rate = (torch.linalg.cross(gyro, ex) * left).sum(-1) / projection.clamp_min(1e-5)
        heading_rate = torch.where(projection < 1e-5, world_rate, heading_rate)
        self.heading.copy_(wrap(self.heading + heading_rate * dt))
        turning = target[:, 0].abs() > 1e-5  # release uses intent, not the filter tail
        released = self.was_turning & ~turning
        self.heading_target.copy_(torch.where(released, self.heading, self.heading_target))
        intent = torch.where(turning, self.filtered[:, 0], 0.)
        self.heading_target.copy_(wrap(self.heading_target + intent * dt))
        coupling = world_rate - heading_rate
        raw = intent + 1.5 * wrap(self.heading_target - self.heading) + coupling
        limited = raw.clamp(-YAW_LIMITS[stage], YAW_LIMITS[stage])
        applied = previous[:, 0] + (limited - previous[:, 0]).clamp(-2 * dt, 2 * dt)
        # Back-calculate only while turning. Released anchor is invariant under saturation.
        saturated = turning & ((applied - raw).abs() > 1e-7)
        backcalc = wrap(self.heading + (applied - coupling - intent) / 1.5)
        self.heading_target.copy_(torch.where(saturated, backcalc, self.heading_target))
        self.command[:, 0] = applied
        self.command[:, 1:] = self.filtered[:, 1:]
        self.rate.copy_((self.command - previous) / dt)
        self.was_turning.copy_(turning)
        self.last_step = step_id
        return self.command

    def context(self, remaining):
        return torch.cat((self.request, self.filtered[:, :1],
                          wrap(self.heading_target - self.heading)[:, None],
                          self.was_turning[:, None].to(self.command), remaining[:, None]), -1)

    def state_dict(self):
        return {name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in vars(self).items()}

    def load_state_dict(self, data):
        for name, value in data.items():
            current = getattr(self, name)
            if isinstance(current, torch.Tensor):
                if current.shape != value.shape:
                    raise ValueError(f"Command state shape mismatch: {name}")
                current.copy_(value)
            else:
                setattr(self, name, value)


class SegmentSampler:
    """80% mixed-axis exploration; 20% static segments timed after reference arrival."""
    def __init__(self, num_envs, device="cpu", seed=0):
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.request = torch.zeros(num_envs, 3, device=self.device)
        self.static = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.remaining = torch.zeros(num_envs, device=self.device)
        self.arrived = torch.zeros_like(self.static)
        self.axis_code = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.height_bin = torch.zeros_like(self.axis_code)
        self.bin_cursor = 0
        self.exposure = torch.zeros(2, device=self.device)

    def random(self, shape):
        return torch.rand(shape, device=self.device, generator=self.generator)

    def resample(self, ids, stage, issued, force_static=False):
        n = len(ids)
        if not n:
            return
        static = self.random((n,)) < .2
        if force_static: static[:] = True
        code = torch.randint(1, 8, (n,), device=self.device, generator=self.generator)
        bins = (torch.arange(n, device=self.device) + self.bin_cursor) % 8
        self.bin_cursor = (self.bin_cursor + n) % 8
        lo, hi = HEIGHT_LIMITS[stage]
        h = lo + (bins + self.random((n,))) / 8 * (hi - lo)
        candidate = torch.stack(((2 * self.random((n,)) - 1) * YAW_LIMITS[stage],
                                 (2 * self.random((n,)) - 1) * VX_LIMITS[stage],
                                 lo + self.random((n,)) * (hi - lo)), -1)
        candidate[:, 0] *= ((code & 1) != 0) & ~static
        candidate[:, 1] *= ((code & 2) != 0) & ~static
        candidate[:, 2] = torch.where((code & 4) != 0, candidate[:, 2], issued[ids, 2])
        candidate[:, 2] = torch.where(static, h, candidate[:, 2])
        self.request[ids] = candidate
        self.static[ids] = static
        self.axis_code[ids] = code
        self.height_bin[ids] = bins
        self.remaining[ids] = torch.where(static, 8 + 12 * self.random((n,)), 4 + 4 * self.random((n,)))
        self.arrived[ids] = False

    def advance(self, issued, rate, dt):
        reached = (issued[:, 2] - self.request[:, 2]).abs() <= .001
        reached &= rate[:, 2].abs() <= .002
        self.arrived |= reached
        self.remaining -= dt * (~self.static | self.arrived)
        self.exposure += torch.stack(((~self.static).sum(), self.static.sum())).to(self.exposure) * dt
        return (self.remaining <= 0).nonzero(as_tuple=False).flatten()

    def state_dict(self):
        return dict(generator=self.generator.get_state(), bin_cursor=self.bin_cursor,
                    **{k: v.clone() for k, v in vars(self).items() if isinstance(v, torch.Tensor)})


def scheduled_stage(step, total_steps):
    if total_steps <= 0: raise ValueError("Stage schedule needs a positive fixed horizon")
    return min(4, max(0, int(5 * step / total_steps)))
