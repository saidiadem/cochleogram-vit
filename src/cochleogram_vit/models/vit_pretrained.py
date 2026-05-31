"""
ImageNet-pretrained Vision Transformer for cochleogram classification (transfer learning).

Rationale
---------
Training a ViT from scratch on ~6,900 ICBHI cochleograms is data-starved: validation
loss hovers near chance and per-class predictions are unstable. Since the cochleograms
are rendered as viridis RGB images, they are directly compatible with ImageNet-pretrained
backbones. This wraps a `timm` pretrained ViT and fine-tunes it for the 4 ICBHI classes.

The wrapper keeps the dataset unchanged by handling, internally:
  - resize from the cochleogram size (128) to the backbone's expected size (224), and
  - normalization with the backbone's own pretrained mean/std (read from its config).
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CochleogramViTPretrained(nn.Module):
    """Fine-tunable ImageNet-pretrained ViT for 3-channel cochleogram images.

    Args:
        model_name:      timm model id (default 'vit_base_patch16_224').
        num_classes:     Output classes (4 for ICBHI).
        pretrained:      Load ImageNet weights (downloads on first use).
        freeze_backbone: If True, train only the classifier head.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes, in_chans=3,
            drop_rate=drop_rate, drop_path_rate=drop_path_rate,
        )

        # Read expected input size + normalization from the backbone's own config,
        # so this stays correct if model_name changes.
        cfg = self.backbone.pretrained_cfg
        self.img_size = int(cfg["input_size"][-1])
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        self.register_buffer("norm_mean", mean)
        self.register_buffer("norm_std", std)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("head"):
                    p.requires_grad = False

        self._log_parameter_count()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) viridis RGB in [0, 1].
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
        x = (x - self.norm_mean) / self.norm_std
        return self.backbone(x)

    def head_parameters(self) -> Iterator[nn.Parameter]:
        """Classifier-head params only (for a higher-LR group when fine-tuning)."""
        return self.backbone.get_classifier().parameters()

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        """Backbone params (everything except the classifier head)."""
        head_ids = {id(p) for p in self.head_parameters()}
        for p in self.backbone.parameters():
            if id(p) not in head_ids:
                yield p

    def _log_parameter_count(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[CochleogramViTPretrained:{self.model_name}] Parameters — "
            f"total: {total:,}  trainable: {trainable:,}  "
            f"(frozen_backbone={self.freeze_backbone})"
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "CochleogramViTPretrained":
        m = cfg["model"]
        return cls(
            model_name=m.get("model_name", "vit_base_patch16_224"),
            num_classes=m["num_classes"],
            pretrained=m.get("pretrained", True),
            freeze_backbone=m.get("freeze_backbone", False),
            drop_rate=m.get("drop_rate", 0.0),
            drop_path_rate=m.get("drop_path_rate", 0.0),
        )
