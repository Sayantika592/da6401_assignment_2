"""Segmentation model
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels)
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512+512, 512, 3, padding=1),   # 512 from decoder (upsample)+ 512 from encoder (skip connection)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256+256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128+128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def decode(self, x5: torch.Tensor, feats: dict) -> torch.Tensor:
        """Decoder-only forward pass using pre-computed encoder features.
 
        This allows the multitask model to share the encoder and only
        call the decoder separately.
 
        Args:
            x5: Bottleneck feature tensor [B, 512, 7, 7].
            feats: Dict with keys 'block1'..'block4' from VGG11Encoder.
 
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        x1 = feats["block1"]   # 64 x 112 x 112
        x2 = feats["block2"]   # 128 x 56 x 56
        x3 = feats["block3"]   # 256 x 28 x 28
        x4 = feats["block4"]   # 512 x 14 x 14
 
        d1 = self.up1(x5)                     # 512 x 14 x 14
        d1 = torch.cat([d1, x4], dim=1)       # 1024 x 14 x 14
        d1 = self.conv1(d1)                    # 512 x 14 x 14
 
        d2 = self.up2(d1)                      # 256 x 28 x 28
        d2 = torch.cat([d2, x3], dim=1)       # 512 x 28 x 28
        d2 = self.conv2(d2)                    # 256 x 28 x 28
 
        d3 = self.up3(d2)                      # 128 x 56 x 56
        d3 = torch.cat([d3, x2], dim=1)       # 256 x 56 x 56
        d3 = self.conv3(d3)                    # 128 x 56 x 56
 
        d4 = self.up4(d3)                      # 64 x 112 x 112
        d4 = torch.cat([d4, x1], dim=1)       # 128 x 112 x 112
        d4 = self.conv4(d4)                    # 64 x 112 x 112
 
        d5 = self.up5(d4)                      # 64 x 224 x 224
        out = self.final(d5)                   # num_classes x 224 x 224
 
        return out
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
 
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        x5, feats = self.encoder(x, return_features=True)
        return self.decode(x5, feats)