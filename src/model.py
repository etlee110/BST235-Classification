import torch
import torch.nn as nn


def _conv_block(in_channels, out_channels):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d (2×2)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class CNN(nn.Module):
    """3-block CNN for 32×32 RGB images (binary classification: airplane vs truck).

    Spatial resolution after each block (32×32 input):
        Block 1 (32 ch)  → 16×16
        Block 2 (64 ch)  → 8×8
        Block 3 (128 ch) → 4×4
    Flattened: 128 × 4 × 4 = 2048 features.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
