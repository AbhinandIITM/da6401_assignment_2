"""Unified multi-task model with automated weight downloading for submission."""

import torch
import torch.nn as nn
import os
import gdown

from .classification import ClassificationHead
from .localization import BoundingBoxHead
from .segmentation import UNetDecoder
from .vgg11 import VGG11Encoder


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
    ):
        # --- Submission Automated Download Block ---
        classifier_path = "checkpoints/classifier.pth"
        localizer_path = "checkpoints/localizer.pth"
        unet_path = "checkpoints/unet.pth"
        
        os.makedirs("checkpoints", exist_ok=True)
        
        # Download checkpoints from Google Drive using IDs provided in Step 2
        gdown.download(id="YOUR_CLASSIFIER_ID_HERE", output=classifier_path, quiet=False)
        gdown.download(id="YOUR_LOCALIZER_ID_HERE", output=localizer_path, quiet=False)
        gdown.download(id="YOUR_UNET_ID_HERE", output=unet_path, quiet=False)
        # --------------------------------------------

        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.output_channels,
            num_classes=num_breeds,
            dropout_p=dropout_p,
            use_batchnorm=use_batchnorm,
        )
        self.localization_head = BoundingBoxHead(in_channels=self.encoder.output_channels)
        self.segmentation_head = UNetDecoder(num_classes=seg_classes, use_batchnorm=use_batchnorm)

    def forward(self, x: torch.Tensor):
        """Return outputs for all three tasks in a single forward pass."""
        bottleneck, features = self.encoder(x, return_features=True)
        return {
            "classification": self.classification_head(bottleneck),
            "localization": self.localization_head(bottleneck),
            "segmentation": self.segmentation_head(bottleneck, features),
        }