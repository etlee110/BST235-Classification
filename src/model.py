import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple fully-connected network. Swap this out for your architecture."""

    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
