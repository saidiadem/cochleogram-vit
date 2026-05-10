# src/cochleogram_vit/training/trainer.py
"""
Training and evaluation loop for CochleogramViT.
One Trainer instance per fold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cochleogram_vit.training.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        class_weights: np.ndarray,
        fold: int,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.fold = fold

        t_cfg = cfg["training"]
        self.epochs = t_cfg["epochs"]
        self.save_dir = Path(cfg["logging"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Loss with softened class weights
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=t_cfg["learning_rate"],
            weight_decay=t_cfg["weight_decay"],
        )

        # LR schedule: linear warmup → cosine decay
        warmup_epochs = t_cfg.get("warmup_epochs", 4)
        epochs = self.epochs

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            denom = epochs - warmup_epochs
            if denom == 0:
                return 0.0
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / denom))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # TensorBoard — one run per fold
        log_dir = cfg["logging"]["log_dir"]
        self.writer = SummaryWriter(log_dir=f"{log_dir}/fold_{fold}")

        self.best_score = 0.0
        self.best_model_state = None
        self.best_epoch = 1

    def fit(self) -> dict:
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            for cochleograms, labels in tqdm(
                self.train_loader,
                desc=f"  Fold {self.fold} Epoch {epoch+1}/{self.epochs}",
                leave=False,
            ):
                cochleograms = cochleograms.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(cochleograms)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(self.train_loader)

            # Validation phase
            val_loss, val_preds, val_labels = self._evaluate()
            metrics = compute_metrics(
                np.array(val_labels),
                np.array(val_preds),
            )
            epoch_score = metrics["score"]

            # Track best checkpoint
            if epoch_score > self.best_score:
                self.best_score = epoch_score
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # TensorBoard
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/score", epoch_score, epoch)
            self.writer.add_scalar("val/sensitivity", metrics["sensitivity"], epoch)
            self.writer.add_scalar("val/specificity", metrics["specificity"], epoch)
            self.writer.add_scalar("lr", current_lr, epoch)

            print(
                f"  Epoch {epoch+1:>2}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Score: {epoch_score*100:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

        print(f"\n  Best checkpoint at epoch {self.best_epoch} "
              f"with Score: {self.best_score*100:.2f}%")

        # Load best model and run final evaluation
        self.model.load_state_dict(self.best_model_state)
        _, all_preds, all_labels = self._evaluate()
        final_metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
        )

        # Save best model for this fold to disk
        self._save_checkpoint()

        print(f"\n  --- Fold {self.fold} Results ---")
        print(f"  Accuracy:    {final_metrics['accuracy']*100:.2f}%")
        print(f"  Sensitivity: {final_metrics['sensitivity']*100:.2f}%")
        print(f"  Specificity: {final_metrics['specificity']*100:.2f}%")
        print(f"  Precision:   {final_metrics['precision']*100:.2f}%")
        print(f"  Score:       {final_metrics['score']*100:.2f}%")
        print(
            f"  TP={final_metrics['TP']}  FN={final_metrics['FN']}  "
            f"TN={final_metrics['TN']}  FP={final_metrics['FP']}  "
            f"FN_wrong_type={final_metrics['FN_wrong_type']}"
        )

        self.writer.close()

        return {
            **final_metrics,
            "fold": self.fold,
            "best_epoch": self.best_epoch,
            "preds": all_preds,
            "labels": all_labels,
        }

    def _evaluate(self) -> tuple[float, list, list]:
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for cochleograms, labels in self.val_loader:
                cochleograms = cochleograms.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(cochleograms)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(self.val_loader)
        return val_loss, all_preds, all_labels

    def _save_checkpoint(self) -> None:
        path = self.save_dir / f"best_fold{self.fold}.pt"
        torch.save(
            {
                "fold": self.fold,
                "best_epoch": self.best_epoch,
                "best_score": self.best_score,
                "model_state_dict": self.best_model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.cfg,
            },
            path,
        )
        print(f"  Checkpoint saved → {path}")