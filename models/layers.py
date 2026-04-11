"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        if not 0<=p<=1:
            raise ValueError("p must be between 0 and 1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        # TODO: implement dropout.
        if not self.training or self.p==0:
            return x
        if self.p==1:
            return torch.zeros_like(x)
        mask = torch.rand_like(x)
        mask = (mask > self.p).float()
        x=x*mask/(1-self.p)
        return x    
        
