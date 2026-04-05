"""
Vision Transformer (ViT) adapted for cochleogram-based respiratory sound classification.

Architecture overview
---------------------
Cochleagram (1 × H × W)
    │
    ▼
Patch Embedding (non-overlapping patches → token sequence)
    │
    ▼
[CLS] token prepended + Positional Embedding added
    │
    ▼
Transformer Encoder (depth × Multi-Head Self-Attention + FFN)
    │
    ▼
CLS token → MLP Classification Head → logits (n_classes)

The core ViT is taken from `vit-pytorch` (lucidrains/vit-pytorch).
This module wraps it with:
  - A configurable dict / YAML-friendly constructor.
  - Weight initialization logging.
  - Optional label-smoothed cross-entropy loss helper.

Reference:
  Dosovitskiy et al., "An Image is Worth 16x16 Words:
  Transformers for Image Recognition at Scale", ICLR 2021.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from vit_pytorch import ViT


class CochleogramViT(nn.Module):
    """
    ViT classifier that ingests single-channel cochleagram images.

    Args:
        image_size:   Height (= width) of the square cochleagram input.
        patch_size:   Side length of each image patch (must divide image_size).
        num_classes:  Number of output classes (4 for ICBHI: normal/crackle/wheeze/both).
        dim:          Embedding / hidden dimension of the transformer.
        depth:        Number of transformer encoder blocks.
        heads:        Number of attention heads per block.
        mlp_dim:      Hidden dimension of the FFN inside each block.
        dropout:      Dropout applied inside attention and FFN.
        emb_dropout:  Dropout applied to patch + positional embeddings.
        channels:     Number of input channels (1 for grayscale cochleagram).
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        num_classes: int = 4,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        channels: int = 1,
    ):
        super().__init__()

        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )

        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            channels=channels,
        )

        self._log_parameter_count()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Cochleagram tensor of shape (B, C, H, W).
               Typically (B, 1, 128, 128).
        Returns:
            logits: Tensor of shape (B, num_classes).
        """
        return self.vit(x)

    def _log_parameter_count(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CochleogramViT] Parameters — total: {total:,}  trainable: {trainable:,}")

    @classmethod
    def from_config(cls, cfg: dict) -> "CochleogramViT":
        """Construct from the 'model' sub-dict of the YAML config."""
        m = cfg["model"]
        return cls(
            image_size=m["image_size"],
            patch_size=m["patch_size"],
            num_classes=m["num_classes"],
            dim=m["dim"],
            depth=m["depth"],
            heads=m["heads"],
            mlp_dim=m["mlp_dim"],
            dropout=m["dropout"],
            emb_dropout=m["emb_dropout"],
            channels=m["channels"],
        )


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing — helpful for small medical datasets
    where overconfidence is a concern.

    Args:
        smoothing: Label smoothing factor ε ∈ [0, 1). 0 = standard CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # One-hot + smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(-1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss
