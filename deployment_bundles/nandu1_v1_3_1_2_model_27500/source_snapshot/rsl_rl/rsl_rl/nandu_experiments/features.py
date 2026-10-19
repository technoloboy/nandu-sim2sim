"""Deployable task features from measurements and detached estimates only."""
import torch


def raw_tracking_errors(x: torch.Tensor) -> torch.Tensor:
    # Layout: Encoder7, policy49, physical commands3 (+ baseline cues3).
    gravity = x[..., 10:13]
    norm = torch.linalg.vector_norm(gravity, dim=-1, keepdim=True)
    fallback = torch.zeros_like(gravity)
    fallback[..., 2] = 1.
    up_body = torch.where(norm > 1e-6, -gravity/norm.clamp_min(1e-6), fallback)
    yaw_world = ((x[..., 7:10] / .25) * up_body).sum(-1)
    return torch.stack((x[..., 56]-yaw_world, x[..., 57]-x[..., 3], x[..., 58]-x[..., 6]), -1)


def tracking_errors(x: torch.Tensor) -> torch.Tensor:
    scales = torch.tensor([.25, .12, .10],dtype=x.dtype,device=x.device)
    return (raw_tracking_errors(x)/scales).clamp(-8.,8.)


def smooth_activity(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    t = ((value.abs()-low)/(high-low)).clamp(0.,1.)
    return t.square()*(3.-2.*t)


def activity_features(x: torch.Tensor, commands: int) -> torch.Tensor:
    e = raw_tracking_errors(x)
    yaw = 1.-(1.-smooth_activity(x[...,56],.03,.10))*(1.-smooth_activity(e[...,0],.03,.10))
    vx = 1.-(1.-smooth_activity(x[...,57],.02,.08))*(1.-smooth_activity(e[...,1],.02,.08))
    height = smooth_activity(e[...,2],.02,.06)
    if commands == 6:
        height = 1.-(1.-height)*(1.-smooth_activity(x[...,61]*.50,.01,.04))
    return torch.stack((yaw,vx,height),-1)
