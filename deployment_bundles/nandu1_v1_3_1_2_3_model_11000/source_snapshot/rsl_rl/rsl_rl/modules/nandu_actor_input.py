"""Pure Torch actor input contract shared by training, inference, and export."""
import torch
from torch import nn


def body_velocity_to_heading(velocity: torch.Tensor, projected_gravity: torch.Tensor) -> torch.Tensor:
    """Resolve Body vectors on horizontal forward/left/up axes from observed gravity.

    Heading forward is the horizontal projection of Body +X. A near-vertical
    Body +X uses Body +Y to select a deterministic horizontal direction. Zero
    gravity falls back to upright; neither branch needs privileged attitude.
    """
    norm = projected_gravity.norm(p=2, dim=-1, keepdim=True)
    upright = torch.zeros_like(projected_gravity)
    upright[..., 2] = 1.0
    up = torch.where(norm > 1.0e-6, -projected_gravity / norm.clamp_min(1.0e-6), upright)
    body_x = torch.zeros_like(up)
    body_x[..., 0] = 1.0
    forward = body_x - (body_x * up).sum(dim=-1, keepdim=True) * up
    body_y = torch.zeros_like(up)
    body_y[..., 1] = 1.0
    fallback = body_y - (body_y * up).sum(dim=-1, keepdim=True) * up
    forward = torch.where(forward.norm(p=2, dim=-1, keepdim=True) > 1.0e-6, forward, fallback)
    forward = forward / forward.norm(p=2, dim=-1, keepdim=True).clamp_min(1.0e-6)
    left = torch.cross(up, forward, dim=-1)
    return torch.stack(((velocity * forward).sum(-1), (velocity * left).sum(-1),
                        (velocity * up).sum(-1)), dim=-1)


def build_actor_input(
    encoder_out: torch.Tensor,
    policy_obs: torch.Tensor,
    command_obs: torch.Tensor,
    actor_velocity_frame: str = "body",
) -> torch.Tensor:
    """Build Body7+policy49+commands or Heading7+policy49+commands6.

    Heading mode enforces detached Encoder features even when an encoder was
    accidentally constructed with output_detach=False. Legacy Body mode keeps
    the encoder's historical detach setting.
    """
    features = encoder_out
    if actor_velocity_frame == "heading":
        if encoder_out.shape[-1] != 7 or policy_obs.shape[-1] != 49 or command_obs.shape[-1] != 6:
            raise ValueError("Heading actor requires Encoder7, policy49, commands6")
        features = encoder_out.detach()
        gravity = policy_obs[..., 3:6]
        features = torch.cat((body_velocity_to_heading(features[..., :3], gravity),
                              body_velocity_to_heading(features[..., 3:6], gravity),
                              features[..., 6:7]), dim=-1)
    elif actor_velocity_frame != "body":
        raise ValueError("actor_velocity_frame must be body or heading")
    return torch.cat((features, policy_obs, command_obs), dim=-1)


class NanduActorInput(nn.Module):
    """Scriptable adapter for a composite exported encoder/actor policy."""
    def __init__(self, actor_velocity_frame: str = "body"):
        super().__init__()
        self.actor_velocity_frame = actor_velocity_frame

    def forward(self, encoder_out: torch.Tensor, policy_obs: torch.Tensor,
                command_obs: torch.Tensor) -> torch.Tensor:
        return build_actor_input(encoder_out, policy_obs, command_obs, self.actor_velocity_frame)
