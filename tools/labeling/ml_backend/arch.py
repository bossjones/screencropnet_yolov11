"""EfficientNet-B0 single-box localization model (vendored).

Copied verbatim (behaviourally) from
``pytorch-lab/screencropnet/arch.py`` so the Label Studio ML backend is
self-contained for ``uvx``/Docker runs and does not depend on the pytorch-lab
checkout being importable. The ``ScreenCropNetV1_378_epochs.pth`` checkpoint was
trained against this exact architecture.
"""

from __future__ import annotations

import timm
import torch.nn as nn


class ObjLocModel(nn.Module):
    """EfficientNet-B0 backbone with ``num_classes=4`` regressing xyxy boxes."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=4
        )

    def forward(self, images, gt_bboxes=None):
        bboxes_logits = self.backbone(images)
        if gt_bboxes is not None:
            loss = nn.MSELoss()(bboxes_logits, gt_bboxes)
            return bboxes_logits, loss
        return bboxes_logits
