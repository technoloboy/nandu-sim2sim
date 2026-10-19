"""Standalone v13122 reference; no training package or simulator imports.

Command step runs once before the next physical interval. Observe runs once
after that interval, using its previous clipped issued action. Read properties
freely; they never advance state. Clear every temporal state for new physics.
"""
import importlib.util
from pathlib import Path
import torch

_spec = importlib.util.spec_from_file_location("exported_v13122_math", Path(__file__).with_name("adaptive_math.py"))
_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_math)
CausalCommandProcessor = _math.CausalCommandProcessor
SlowErrorMemory = _math.SlowErrorMemory
CONTROL_DT = .02
STUDENT_TAU = .3


class StatefulProcessor:
    def __init__(self, num_envs=1, device="cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.commands = CausalCommandProcessor(num_envs, self.device)
        self.memory = SlowErrorMemory(num_envs, self.device)
        self.hidden = torch.zeros(num_envs, 64, device=self.device)
        self.filtered_eta = torch.zeros(num_envs, 8, device=self.device)
        self.previous_issued_action = torch.zeros(num_envs, 13, device=self.device)
        self.raw_intent = torch.zeros(num_envs, 3, device=self.device)
        self._dynamic = torch.zeros(num_envs, 11, device=self.device)
        self._command_step_id = None
        self._context_step_id = None

    @property
    def dynamic(self):
        return self._dynamic.detach().clone()

    @property
    def command(self):
        return self.commands.command

    @torch.no_grad()
    def reset(self, env_ids=None, height=.9):
        self.commands.reset(env_ids, height=height)
        self.memory.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for name in ("hidden", "filtered_eta", "previous_issued_action", "raw_intent", "_dynamic"):
            value = getattr(self, name).clone()
            value[ids] = 0.
            setattr(self, name, value)
        if env_ids is None:
            self._command_step_id = None
            self._context_step_id = None

    @torch.no_grad()
    def command_step(self, goal, observation, stage=4, step_id=None):
        if step_id is not None and self._command_step_id is not None:
            if step_id < self._command_step_id:
                raise ValueError("command step_id must not move backwards")
            if step_id == self._command_step_id:
                return self.command
        self.raw_intent = goal.detach().to(self.device).clone()
        applied = self.commands.step(self.raw_intent, CONTROL_DT, step_id=step_id,
            gravity_b=observation[:, 3:6], gyro_b=observation[:, :3] / .25, stage=stage)
        if step_id is not None:
            self._command_step_id = step_id
        return applied

    @torch.no_grad()
    def observe(self, observation, encoded, previous_issued_action, context_model,
                reset_mask=None, known_clipping=False, step_id=None):
        if step_id is not None and self._context_step_id is not None:
            if step_id < self._context_step_id:
                raise ValueError("context step_id must not move backwards")
            if step_id == self._context_step_id:
                return self.dynamic
        observation = observation.detach().to(self.device)
        encoded = encoded.detach().to(self.device)
        issued = previous_issued_action.detach().to(self.device).clone()
        reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if reset_mask is not None:
            reset = reset_mask.reshape(-1).to(device=self.device, dtype=torch.bool)
            ids = reset.nonzero().flatten()
            self.memory.reset(ids)
            self.hidden = torch.where(reset[:, None], torch.zeros_like(self.hidden), self.hidden)
            self.filtered_eta = torch.where(reset[:, None], torch.zeros_like(self.filtered_eta), self.filtered_eta)
            issued[reset] = 0.
        # Reset rows are priming observations, with no preceding physical step.
        valid = torch.isfinite(observation[:, :36]).all(-1) & torch.isfinite(issued).all(-1) & ~reset
        next_hidden, next_eta, _ = context_model(torch.nan_to_num(observation), torch.nan_to_num(issued),
                                                 self.hidden, self.filtered_eta)
        self.hidden = torch.where(valid[:, None], next_hidden, self.hidden)
        self.filtered_eta = torch.where(valid[:, None], next_eta, self.filtered_eta)
        self.previous_issued_action = issued
        gravity = observation[:, 3:6]
        norm = gravity.norm(dim=-1)
        up = -gravity / norm[:, None].clamp_min(1e-6)
        yaw = (observation[:, :3] / .25 * up).sum(-1)
        tilt = up[:, 2].clamp(-1., 1.).acos()
        good_measurement = valid & torch.isfinite(encoded).all(-1) & (norm > 1e-6) & ~reset
        hdot = self.commands.height_rate
        applied = self.commands.command
        memory = self.memory.step_errors(applied[:, 1] - encoded[:, 3], applied[:, 0] - yaw,
            self.raw_intent, hdot, valid=good_measurement, tilt=tilt, clipped=known_clipping,
            dt=CONTROL_DT, step_id=step_id)
        self._dynamic = torch.cat((memory, self.filtered_eta, hdot[:, None] / .5), -1)
        self._dynamic[reset] = 0.
        if step_id is not None:
            self._context_step_id = step_id
        return self.dynamic
