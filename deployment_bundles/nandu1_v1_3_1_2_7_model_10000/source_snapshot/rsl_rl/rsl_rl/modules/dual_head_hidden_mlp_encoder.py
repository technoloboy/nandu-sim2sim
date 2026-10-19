"""v1.2.2: one shared history trunk and a private 128D layer in each head."""
import torch.nn as nn

from .dual_head_mlp_encoder import DualHeadMLPEncoder
from .mlp_encoder import get_activation


class DualHeadHiddenMLPEncoder(DualHeadMLPEncoder):
    """Keep the existing 7D state/gradient interface; add nonlinear head capacity."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        activation = kwargs.get("activation", "elu")
        for name in ("velocity_head", "height_head"):
            output = getattr(self, name)
            hidden = nn.Linear(output.in_features, 128)
            if kwargs.get("orthogonal_init", False):
                nn.init.orthogonal_(hidden.weight, 2.0 ** 0.5)
                nn.init.constant_(hidden.bias, 0.0)
            # The confirmed shared feature size is 128; fail instead of silently
            # changing the head interface for an incompatible runner config.
            if output.in_features != 128:
                raise ValueError("v1.2.2 requires a shared trunk ending in 128 features")
            setattr(self, name, nn.Sequential(hidden, get_activation(activation), output))
