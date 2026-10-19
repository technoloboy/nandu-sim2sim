"""Simulator-free v1.4.1 commands; heading control remains the v1.4 implementation."""
import torch
try:
    from . import nandu1_v1_4_math as parent
except ImportError:
    import nandu1_v1_4_math as parent

STAGE_LIMITS = parent.STAGE_LIMITS
STAGE_HEIGHT_RANGES = parent.STAGE_HEIGHT_RANGES
HEIGHT_RATE_LIMITS = parent.HEIGHT_RATE_LIMITS
body_to_heading = parent.body_to_heading


class CommandProcessor(parent.CommandProcessor):
    SCHEMA = 'nandu1_v1_4_1_command_processor_v1'
    TREND_SCALES = (10., 5.)
    TREND_TIME_CONSTANT = .08
    HEIGHT_CUE_SCALE = .50
    STATIC_BINS = 10
    _EXTRA_FIELDS = ('static_bin', 'static_stage', 'static_hold_duration', 'static_arrived')

    def __init__(self, num_envs, device, dtype=torch.float32):
        super().__init__(num_envs, device, dtype)
        self.static_bin = torch.full_like(self.mode, -1)
        self.static_stage = torch.full_like(self.mode, -1)
        self.static_hold_duration = torch.zeros_like(self.height_rate)
        self.static_arrived = torch.zeros_like(self.is_step)
        self._static_cursor = 0
        self._last_tick = None

    def reset(self, ids, height):
        super().reset(ids, height)
        if hasattr(self, 'static_bin'):
            ids = self._ids(ids)
            self.static_bin[ids] = -1
            self.static_stage[ids] = -1
            self.static_hold_duration[ids] = 0
            self.static_arrived[ids] = False
            # A new whole batch may start its clock at the previous tick. Keep
            # the guard for partial resets so untouched environments cannot
            # integrate a second time in that same control tick.
            if ids.numel() >= self.num_envs and torch.unique(ids).numel() == self.num_envs:
                self._last_tick = None

    def resample(self, ids, stage):
        super().resample(ids, stage)
        ids = self._ids(ids)
        self.static_bin[ids] = -1
        self.static_stage[ids] = -1
        self.static_arrived[ids] = False
        self.static_hold_duration[ids] = 0
        static = ids[self.mode[ids] == 0]
        if static.numel():
            lo, hi = self.STAGE_HEIGHT_RANGES[stage]
            bins = (torch.arange(static.numel(), device=self.device) + self._static_cursor) % self.STATIC_BINS
            self._static_cursor = (self._static_cursor + static.numel()) % self.STATIC_BINS
            self.static_bin[static] = bins
            self.static_stage[static] = stage
            self.goal[static, 2] = lo + (hi-lo) * (bins + self._rand(static.numel())) / self.STATIC_BINS
            self.static_hold_duration[static] = self.mode_time_left[static]
            self.mode_time_left[static] = float('inf')
            self.height_mode[static] = 0
            self._pending_step[static] = False

    def set_external(self, command, stage):
        super().set_external(command, stage)
        self.static_bin.fill_(-1)
        self.static_stage.fill_(-1)
        self.static_arrived.zero_()
        self.static_hold_duration.zero_()

    def step(self, dt, gravity_b, gyro_b, stage, *, tick=None):
        if tick is not None and tick == self._last_tick:
            return self.commands
        previous = self.commands.clone()
        old_trend = self.trend[:, :2].clone()
        result = super().step(dt, gravity_b, gyro_b, stage)
        gain = dt / (self.TREND_TIME_CONSTANT + dt)
        cue = torch.tanh((self.commands[:, :2]-previous[:, :2]) / dt /
                         self.commands.new_tensor(self.TREND_SCALES))
        self.trend[:, :2] = (1-gain)*old_trend + gain*cue
        self.trend[:, 2] = (self.height_rate / self.HEIGHT_CUE_SCALE).clamp(-1, 1)
        # Promotion widens the interval without resampling all environments.
        # Keep a segment in the height interval from which it was sampled.
        # An explicit evaluation-stage shrink may clamp its goal; classify the
        # actual goal within that saved interval rather than the new stage.
        sampled_ranges = self.commands.new_tensor(self.STAGE_HEIGHT_RANGES)[self.static_stage.clamp_min(0)]
        actual_bin = ((self.goal[:,2]-sampled_ranges[:,0]) /
                      (sampled_ranges[:,1]-sampled_ranges[:,0])*self.STATIC_BINS).long().clamp(0, self.STATIC_BINS-1)
        self.static_bin.copy_(torch.where(self.static_stage >= 0, actual_bin, self.static_bin))
        arrived = (self.mode == 0) & (self.static_bin >= 0) & ~self.static_arrived & ~self._external
        arrived &= (self.commands[:, 2]-self.goal[:, 2]).abs() <= 1e-6
        self.static_arrived[arrived] = True
        self.mode_time_left[arrived] = self.static_hold_duration[arrived]
        self._last_tick = tick
        return result

    def state_dict(self):
        state = super().state_dict()
        state.update({k: getattr(self, k).detach().clone() for k in self._EXTRA_FIELDS})
        state.update(static_cursor=self._static_cursor, last_tick=self._last_tick)
        return state

    def load_state_dict(self, state):
        if state.get('schema') != self.SCHEMA:
            raise ValueError('incompatible command processor schema')
        converted = {}
        if state.get('num_envs') == self.num_envs:
            for name in self._EXTRA_FIELDS:
                target = getattr(self, name)
                value = torch.as_tensor(state[name], device=self.device, dtype=target.dtype)
                if value.shape != target.shape:
                    raise ValueError(f'incompatible state shape for {name}')
                converted[name] = value
        super().load_state_dict(state)
        for name, value in converted.items():
            getattr(self, name).copy_(value)
        self._static_cursor = int(state['static_cursor']) % self.STATIC_BINS
        self._last_tick = state.get('last_tick') if converted else None
