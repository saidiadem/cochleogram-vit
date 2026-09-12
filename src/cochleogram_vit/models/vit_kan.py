"""
Vision Transformer with KAN (Kolmogorov-Arnold Network) at the FFN tail.

Architecture overview
---------------------
Identical to ``CochleogramViT`` (see ``vit.py``) except that, in the
transformer blocks specified by ``kan_blocks``, the second ``Linear`` of
the FeedForward sub-module is replaced by an ``efficient_kan.KAN`` layer.

Default ``kan_blocks=(-1,)`` swaps only the final block — close to the CLS
head, bounded extra params, easy ablation against the baseline ViT.

The trailing ``Dropout`` after the FFN is kept (spline coefficients on a
small dataset benefit from stochastic regularization).
"""

from __future__ import annotations

from typing import Iterable, Iterator

import torch
import torch.nn as nn
from efficient_kan import KAN
from vit_pytorch import ViT


class CochleogramViTKAN(nn.Module):
    """ViT variant with KAN replacing the FFN tail of selected blocks.

    Args:
        image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
        dropout, emb_dropout, channels:
            Standard ViT hyperparameters — see ``CochleogramViT``.
        kan_blocks:
            Indices (negative allowed) of transformer blocks whose FFN
            tail should be replaced. Defaults to ``(-1,)`` (last block).
        grid_size, spline_order:
            ``efficient_kan.KAN`` spline hyperparameters.
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
        kan_blocks: Iterable[int] = (-1,),
        grid_size: int = 5,
        spline_order: int = 3,
        kan_dropout: float | None = None,
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

        blocks = self.vit.transformer.layers
        assert len(blocks) == depth, (
            f"vit-pytorch layer layout changed: expected {depth} blocks, "
            f"got {len(blocks)}"
        )

        tail_dropout = dropout if kan_dropout is None else kan_dropout
        self._kan_blocks = tuple(int(i) % depth for i in kan_blocks)
        for idx in self._kan_blocks:
            ff = blocks[idx][1]
            # ff.net = LN, Linear(dim→mlp), GELU, Dropout, Linear(mlp→dim), Dropout
            assert isinstance(ff.net[4], nn.Linear), (
                "Unexpected FeedForward layout in vit-pytorch"
            )
            ff.net = nn.Sequential(
                ff.net[0], ff.net[1], ff.net[2], ff.net[3],
                KAN([mlp_dim, dim], grid_size=grid_size, spline_order=spline_order),
                nn.Dropout(tail_dropout),
            )

        self._log_parameter_count()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

    def kan_parameters(self) -> Iterator[nn.Parameter]:
        """Yield params of the KAN sub-modules (for a separate LR group)."""
        for idx in self._kan_blocks:
            yield from self.vit.transformer.layers[idx][1].net[4].parameters()

    def _log_parameter_count(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        kan = sum(p.numel() for p in self.kan_parameters())
        print(
            f"[CochleogramViTKAN] Parameters — total: {total:,}  "
            f"trainable: {trainable:,}  KAN: {kan:,}  "
            f"(kan_blocks={self._kan_blocks})"
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "CochleogramViTKAN":
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
            kan_blocks=m.get("kan_blocks", (-1,)),
            grid_size=m.get("grid_size", 5),
            spline_order=m.get("spline_order", 3),
            kan_dropout=m.get("kan_dropout", None),
        )
