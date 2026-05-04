"""
Training and evaluation loop for CochleogramViT.

Design decisions:
  - No heavy framework dependency (no Lightning) — keeps the code transparent.
  - TensorBoard logging via SummaryWriter.
  - Cosine annealing LR schedule with optional linear warmup.
  - Checkpoints saved as {save_dir}/epoch_{N:03d}.pt with best model tracking.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cochleogram_vit.training.metrics import MetricTracker


class Trainer:
    """
    Manages the full training lifecycle.

    Args:
        model:         CochleogramViT (or any nn.Module with matching I/O).
        train_loader:  DataLoader for the training split.
        val_loader:    DataLoader for the validation split.
        cfg:           Full config dict (see configs/default.yaml).
        device:        torch.device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        t_cfg = cfg["training"]
        self.epochs = t_cfg["epochs"]
        self.log_every = cfg["logging"]["log_every"]
        self.save_every = cfg["logging"]["save_every"]
        self.save_dir = Path(cfg["logging"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping configuration
        # User-configurable via cfg["training"]["early_stopping"] = True/False
        # patience: number of consecutive epochs without sufficient improvement
        # rel_improve: required improvement threshold expressed as fraction (e.g. 0.25)
        self.early_stop_enabled = t_cfg.get("early_stopping", False)
        self.early_stop_patience = int(t_cfg.get("early_stopping_patience", 10))
        self.early_stop_rel = float(t_cfg.get("early_stopping_rel", 0.25))
        self._early_no_improve = 0
        self._early_best_val_loss = float("inf")

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=t_cfg["learning_rate"],
            weight_decay=t_cfg["weight_decay"],
        )

        # LR schedule: linear warmup → cosine annealing
        warmup_epochs = t_cfg.get("warmup_epochs", 0)
        cosine = CosineAnnealingLR(self.optimizer, T_max=self.epochs - warmup_epochs, eta_min=1e-6)
        if warmup_epochs > 0:
            warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            self.scheduler = cosine

        # TensorBoard
        log_dir = cfg["logging"]["log_dir"]
        self.writer = SummaryWriter(log_dir=log_dir)

        self.best_icbhi_score = 0.0
        self.best_epoch = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run the full training loop."""
        for epoch in range(1, self.epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.epochs}  |  LR: {self._current_lr():.2e}")

            train_metrics = self._run_epoch(epoch, training=True)
            val_metrics = self._run_epoch(epoch, training=False)

            self.scheduler.step()

            self._log_metrics(epoch, train_metrics, split="train")
            self._log_metrics(epoch, val_metrics, split="val")

            print(
                f"  Train → loss: {train_metrics['loss']:.4f}  "
                f"acc: {train_metrics['accuracy']:.3f}  "
                f"ICBHI: {train_metrics['icbhi_score']:.3f}"
            )
            print(
                f"  Val   → loss: {val_metrics['loss']:.4f}  "
                f"acc: {val_metrics['accuracy']:.3f}  "
                f"ICBHI: {val_metrics['icbhi_score']:.3f}"
            )

            if val_metrics["icbhi_score"] > self.best_icbhi_score:
                self.best_icbhi_score = val_metrics["icbhi_score"]
                self.best_epoch = epoch
                self._save_checkpoint(epoch, tag="best")
                print(f"  ** New best ICBHI score: {self.best_icbhi_score:.4f} (epoch {epoch})")

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

            # -------------------------
            # Early stopping (optional)
            # -------------------------
            # Interpretation/assumption: we require the validation loss to decrease
            # by at least `early_stop_rel * train_loss` compared to the previous
            # best validation loss. If this does not happen for `patience`
            # consecutive epochs, stop training early.
            if self.early_stop_enabled:
                cur_val_loss = val_metrics.get("loss", float("inf"))
                cur_train_loss = train_metrics.get("loss", float("inf"))

                # Amount of improvement (positive if val loss decreased)
                improvement = self._early_best_val_loss - cur_val_loss
                required = self.early_stop_rel * (cur_train_loss + 1e-8)

                if improvement > required:
                    # Sufficient improvement: reset counter and update best
                    self._early_best_val_loss = min(self._early_best_val_loss, cur_val_loss)
                    self._early_no_improve = 0
                else:
                    self._early_no_improve += 1
                    print(f"  EarlyStopping: no sufficient val-loss improvement ({self._early_no_improve}/{self.early_stop_patience})")

                if self._early_no_improve >= self.early_stop_patience:
                    print(f"\nEarly stopping triggered. No sufficient validation loss improvement for {self.early_stop_patience} epochs.")
                    break

        self.writer.close()
        print(f"\nTraining complete. Best ICBHI score: {self.best_icbhi_score:.4f} at epoch {self.best_epoch}.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int, training: bool) -> dict[str, float]:
        self.model.train(training)
        loader = self.train_loader if training else self.val_loader
        tracker = MetricTracker()
        global_step = (epoch - 1) * len(self.train_loader)

        with torch.set_grad_enabled(training):
            for batch_idx, (inputs, targets) in enumerate(tqdm(loader, leave=False)):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    if (batch_idx + 1) % self.log_every == 0:
                        step = global_step + batch_idx
                        self.writer.add_scalar("train/batch_loss", loss.item(), step)

                tracker.update(logits.detach(), targets.detach(), loss.item())

        return tracker.compute()

    def _log_metrics(self, epoch: int, metrics: dict[str, float], split: str) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"{split}/{key}", value, epoch)

    def _save_checkpoint(self, epoch: int, tag: Optional[str] = None) -> None:
        name = f"epoch_{epoch:03d}" if tag is None else f"{tag}"
        path = self.save_dir / f"{name}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_icbhi_score": self.best_icbhi_score,
                "config": self.cfg,
            },
            path,
        )
        print(f"  Checkpoint saved → {path}")

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
