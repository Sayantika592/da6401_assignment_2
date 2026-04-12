"""Unified multi-task model
"""

import torch
import torch.nn as nn
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()

        import os
        import gdown

        # download only if file not already present
        if not os.path.exists(classifier_path):
            gdown.download(id="1tH-vQsxz3rHdARMJ0mLeBxEctt4xWand", output=classifier_path, quiet=False)

        if not os.path.exists(localizer_path):
            gdown.download(id="1Ub0xiHf4-BmQeSqf1IoIZhFkGISBOoRU", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="1X8CRoMmvSwy1_1jwqJ_iDkRKXhB8u1b0", output=unet_path, quiet=False)

        self.classifier = VGG11Classifier(num_breeds, in_channels)
        self.localizer = VGG11Localizer(in_channels)
        self.segmenter = VGG11UNet(seg_classes, in_channels)

        # checkpoints loaded
        device = torch.device("cpu")
        self._load_checkpoint(self.classifier, classifier_path, device)
        self._load_checkpoint(self.localizer, localizer_path, device)
        self._load_checkpoint(self.segmenter, unet_path, device)

        self.encoder = self.classifier.encoder
        
        self.cls_head = self.classifier.classifier # classifier head
        self.box_head = self.localizer.regressor # regressor head
        self.loc_img_size = self.localizer.img_size # pixel scaling factor
        self.seg_decoder = self.segmenter # segmentation decoder

    def _load_checkpoint(self, model, path, device):
        """Load checkpoint handling both plain state_dict and dict-wrapped format."""
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        features, skip_feats = self.encoder(x, return_features=True)
        cls_out = self.cls_head(features) # classification head
        box_out = self.box_head(features) # localization head
        box_out = box_out * self.loc_img_size # scale to pixel coordinates
        seg_out = self.seg_decoder.decode(features, skip_feats) # segmentation decoder (encoder features+skip connections)
        return {
            "classification": cls_out,
            "localization": box_out,
            "segmentation": seg_out
        }
