"""Localization modules
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels)
        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 4),
            nn.Sigmoid()           # outputs in [0, 1]
        )
        # Image size assumed fixed at 224x224 per VGG11 paper
        self.img_size = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            format in original image pixel space (not normalized values).
        """
        features = self.encoder(x)
        bbox = self.regressor(features)
        # Scale from [0,1] to pixel coordinates [0, img_size]
        bbox = bbox * self.img_size
        return bbox