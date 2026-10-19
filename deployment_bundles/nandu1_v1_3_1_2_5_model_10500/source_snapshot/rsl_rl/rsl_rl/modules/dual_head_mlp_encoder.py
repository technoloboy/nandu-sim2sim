"""Shared-trunk encoder with independent velocity and height regression heads."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .mlp_encoder import get_activation


class DualHeadMLPEncoder(nn.Module):
    is_mlp_encoder = True
    is_vae = False

    def __init__(
        self,
        num_input_dim: int,
        num_output_dim: int,
        hidden_dims: list[int] = [256, 128],
        activation: str = "elu",
        orthogonal_init: bool = False,
        output_detach: bool = False,
        velocity_output_dim: int = 6,
        height_output_dim: int = 1,
        height_initial_bias: float = 0.9,
        output_initial_bias=None,
        **kwargs,
    ):
        super().__init__()
        del output_initial_bias
        if kwargs:
            print(f"DualHeadMLPEncoder ignored arguments: {sorted(kwargs)}")
        if velocity_output_dim + height_output_dim != num_output_dim:
            raise ValueError(
                "velocity_output_dim + height_output_dim must equal num_output_dim: "
                f"{velocity_output_dim} + {height_output_dim} != {num_output_dim}"
            )
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer")

        self.num_input_dim = num_input_dim
        self.num_output_dim = num_output_dim
        self.output_detach = output_detach

        layers: list[nn.Module] = []
        previous_dim = num_input_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(previous_dim, hidden_dim)
            if orthogonal_init:
                nn.init.orthogonal_(linear.weight, np.sqrt(2))
                nn.init.constant_(linear.bias, 0.0)
            layers.extend((linear, get_activation(activation)))
            previous_dim = hidden_dim
        self.shared_trunk = nn.Sequential(*layers)
        self.velocity_head = nn.Linear(previous_dim, velocity_output_dim)
        self.height_head = nn.Linear(previous_dim, height_output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(self.velocity_head.weight, 0.01)
            nn.init.orthogonal_(self.height_head.weight, 0.01)
        nn.init.constant_(self.velocity_head.bias, 0.0)
        nn.init.constant_(self.height_head.weight, 0.0)
        nn.init.constant_(self.height_head.bias, height_initial_bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.shared_trunk(inputs)
        return torch.cat((self.velocity_head(features), self.height_head(features)), dim=-1)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        self.encoder_out = self.forward(inputs)
        return self.encoder_out.detach() if self.output_detach else self.encoder_out

    def get_encoder_out(self) -> torch.Tensor:
        return self.encoder_out

    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(inputs)
