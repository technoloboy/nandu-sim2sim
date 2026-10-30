"""Portable C7-P-1 r3 raw-Gaussian to executable-action contract.

The policy distribution is unchanged. Its raw sample is used for PPO likelihoods;
only the finite issued value is sent to the simulator and fed back as action
history. Constants are audited against the canonical URDF/default configuration
in tests and against every actual articulation at startup. No runtime XML/FK.
"""
from __future__ import annotations

import torch


JOINT_NAMES = tuple(
    f"{part}_{side}_Joint" for side in ("L", "R")
    for part in ("proximal_pitch", "proximal_roll", "proximal_yaw", "knee")
) + ("arm1", "arm2", "arm3", "wheel_L_Joint", "wheel_R_Joint")


def action_interface_for(spec, device="cpu"):
    if spec.revision_profile:
        from .c7_p1_r4_actions import R4ActionInterface
        return R4ActionInterface(device)
    return ActionInterface(device)
POSITION_DEFAULT = (-.475, 0., 0., .95, .475, 0., 0., .95, -.15, -1.78, 1.93)
# NANDU1/urdf/robot.urdf, with only the C7-P-1 knee/arm3 upper bounds applied.
POSITION_LOWER = (
    -2.60054058547155, -.261799387799149, -3.66519142918809, -.261799387799149,
    -3.14159265358979, -3.19395248, -3.66519142918809, -.261799387799149,
    -2.443460952792061, -2.2689280275926285, -1.5707963267948966,
)
POSITION_UPPER = (
    3.14159265358979, 3.19395248, 1.48352986419518, 1.2,
    2.60054058547155, .261799387799149, 1.48352986419518, 1.2,
    2.356194490192345, 2.2689280275926285, 2.24,
)


class ActionInterface:
    """Per-joint legal target box, preserving the original affine action map."""
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.scale = torch.tensor([.25] * 11 + [1., 1.], device=self.device)
        self.offset = torch.tensor(POSITION_DEFAULT + (0., 0.), device=self.device)
        self.lower = torch.tensor(
            tuple((q - offset) / .25 for q, offset in zip(POSITION_LOWER, POSITION_DEFAULT))
            + (-40., -40.), device=self.device)
        self.upper = torch.tensor(
            tuple((q - offset) / .25 for q, offset in zip(POSITION_UPPER, POSITION_DEFAULT))
            + (40., 40.), device=self.device)

    def map(self, raw: torch.Tensor) -> torch.Tensor:
        """Return issued normalized targets; never replace the saved PPO sample."""
        if raw.ndim < 1 or raw.shape[-1] != 13 or not raw.is_floating_point():
            raise ValueError("C7-P-1 action must be a floating tensor [...,13]")
        # Do not turn infinite distribution output into apparently valid targets.
        if not torch.isfinite(raw).all():
            raise ValueError("C7-P-1 raw action is non-finite")
        return torch.maximum(torch.minimum(raw, self.upper.to(raw)), self.lower.to(raw))

    def metadata(self) -> dict:
        """A fresh JSON record shared by training, native/ONNX and MuJoCo."""
        return {
            "schema": "nandu_c7_p1_issued_action_v1",
            "joint_names": list(JOINT_NAMES),
            "mapping": "componentwise_clip_after_gaussian_sampling",
            "distribution_log_probability": "unclipped_gaussian_sample",
            "feedback": "issued_action",
            "scale": [.25] * 11 + [1., 1.],
            "offset": list(POSITION_DEFAULT) + [0., 0.],
            "lower": [(q - offset) / .25 for q, offset in zip(POSITION_LOWER, POSITION_DEFAULT)] + [-40., -40.],
            "upper": [(q - offset) / .25 for q, offset in zip(POSITION_UPPER, POSITION_DEFAULT)] + [40., 40.],
            "position_hard_lower_rad": list(POSITION_LOWER),
            "position_hard_upper_rad": list(POSITION_UPPER),
            "wheel_velocity_limit_rad_s": 40.,
            "position_source": "NANDU1/urdf/robot.urdf_plus_C7_P_1_physical_revision",
            "default_source": "assets/config/nandu1_cfg.py; unchanged_action_zero",
            "position_target": "default_q+0.25*issued[:11]",
            "wheel_target": "issued[11:] rad/s",
        }

    @staticmethod
    def _require_close(actual, expected, label, *, atol=1e-5):
        actual = torch.as_tensor(actual)
        expected = torch.as_tensor(expected, device=actual.device, dtype=actual.dtype)
        if not torch.isfinite(actual).all() or not torch.isclose(actual, expected, atol=atol, rtol=0.).all():
            raise ValueError(f"C7-P-1 action contract mismatch: {label}")

    def validate_articulation(self, robot, action_manager=None) -> None:
        """Validate all environments after the instance-local limit startup event."""
        names = list(robot.joint_names)
        if len(names) != len(set(names)) or any(name not in names for name in JOINT_NAMES):
            raise ValueError("C7-P-1 action contract mismatch: articulation joint names")
        position_ids = [names.index(name) for name in JOINT_NAMES[:11]]
        wheel_ids = [names.index(name) for name in JOINT_NAMES[11:]]
        d = robot.data
        hard = torch.tensor(list(zip(POSITION_LOWER, POSITION_UPPER)),
                            device=d.joint_pos_limits.device, dtype=d.joint_pos_limits.dtype)
        for cache_name in ("joint_pos_limits", "default_joint_pos_limits"):
            self._require_close(getattr(d, cache_name)[:, position_ids], hard, cache_name)
        mid = hard.mean(-1)
        half = .45 * (hard[:, 1] - hard[:, 0])
        self._require_close(d.soft_joint_pos_limits[:, position_ids],
                            torch.stack((mid - half, mid + half), -1), "soft_joint_pos_limits")
        self._require_close(d.default_joint_pos[:, position_ids], POSITION_DEFAULT, "default_joint_pos")
        self._require_close(d.joint_vel_limits[:, wheel_ids], 40., "wheel joint_vel_limits")
        if action_manager is not None:
            if list(action_manager.active_terms) != ["joint_position", "wheel_velocity"]:
                raise ValueError("C7-P-1 action contract mismatch: action term order")
            for name, expected_names, scale, offset in (
                ("joint_position", JOINT_NAMES[:11], .25, POSITION_DEFAULT),
                ("wheel_velocity", JOINT_NAMES[11:], 1., 0.),
            ):
                term = action_manager.get_term(name)
                if tuple(term._joint_names) != expected_names or term.cfg.clip is not None:
                    raise ValueError(f"C7-P-1 action contract mismatch: {name} joints/clip")
                self._require_close(term._scale, scale, name + " scale")
                self._require_close(term._offset, offset, name + " offset")


class IssuedActionCosts:
    """Once-per-transition 2*Huber differences with episode-local history.

    Reset's zero action is the first-difference baseline. Two actual preceding
    actions are required for a second difference; another episode never supplies
    that history. Reward getters read ``rate``/``smoothness`` without advancing.
    """
    def __init__(self, num_envs: int, device="cpu"):
        self.previous = torch.zeros(num_envs, 13, device=device)
        self.previous_previous = torch.zeros_like(self.previous)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.rate = torch.zeros(num_envs, device=device)
        self.smoothness = torch.zeros_like(self.rate)
        self.last_step = None

    @staticmethod
    def _cost(delta):
        absolute = delta.abs()
        return torch.where(absolute <= 1., delta.square(), 2. * absolute - 1.).sum(-1)

    @torch.no_grad()
    def advance(self, issued: torch.Tensor, step: int) -> None:
        if self.last_step == step:
            return
        if issued.shape != self.previous.shape:
            raise ValueError("issued action history shape mismatch")
        self.rate.copy_(self._cost(issued - self.previous))
        second = issued - 2. * self.previous + self.previous_previous
        self.smoothness.copy_(torch.where(self.count >= 2, self._cost(second), 0.))
        self.previous_previous.copy_(self.previous)
        self.previous.copy_(issued)
        self.count.add_(1).clamp_max_(2)
        self.last_step = step

    @torch.no_grad()
    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.previous[ids] = 0.
        self.previous_previous[ids] = 0.
        self.count[ids] = 0
        self.rate[ids] = self.smoothness[ids] = 0.
