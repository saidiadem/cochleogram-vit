# src/cochleogram_vit/training/metrics.py
"""
Evaluation metrics for ICBHI classification.

Metric definitions follow the paper exactly:
  TP = adventitious correctly classified (exact subtype match)
  TN = normal correctly classified
  FP = normal incorrectly classified as adventitious
  FN = adventitious incorrectly classified as normal  (paper definition)
  FN_wrong_type = adventitious predicted as wrong adventitious subtype
                  (stored for analysis, not used in score)

  Sensitivity = TP / (TP + FN)
  Specificity = TN / (TN + FP)
  Precision   = TP / (TP + FP)
  Accuracy    = (TP + TN) / total_samples
  Score       = (Sensitivity + Specificity) / 2
"""

from __future__ import annotations

import numpy as np
import torch


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute all metrics for one fold or aggregated results.

    Args:
        y_true: Ground truth labels (0=normal, 1=crackles, 2=wheezes, 3=both)
        y_pred: Predicted labels

    Returns:
        Dict with sensitivity, specificity, precision, accuracy, score,
        TP, FN, FN_wrong_type, TN, FP.
    """
    TP = int(np.sum((y_true != 0) & (y_pred == y_true)))
    FN = int(np.sum((y_true != 0) & (y_pred == 0)))                               # paper def: adventitious → normal
    FN_wrong_type = int(np.sum((y_true != 0) & (y_pred != 0) & (y_pred != y_true))) # subtype confusion, stored only
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred != 0)))

    assert FN + FN_wrong_type + TP == int(np.sum(y_true != 0)), \
        "Adventitious decomposition mismatch"

    total = len(y_true)
    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    precision   = TP / (TP + FP + 1e-8)
    accuracy    = (TP + TN) / total
    score       = (sensitivity + specificity) / 2.0

    return {
        "sensitivity":   sensitivity,
        "specificity":   specificity,
        "precision":     precision,
        "accuracy":      accuracy,
        "score":         score,
        "TP":            TP,
        "FN":            FN,
        "FN_wrong_type": FN_wrong_type,
        "TN":            TN,
        "FP":            FP,
    }


class MetricTracker:
    """Accumulates predictions over an epoch for TensorBoard logging."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._preds: list[int] = []
        self._targets: list[int] = []
        self._loss_sum = 0.0
        self._n_batches = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float):
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        self._preds.extend(preds)
        self._targets.extend(targets.cpu().numpy().tolist())
        self._loss_sum += loss
        self._n_batches += 1

    def compute(self) -> dict:
        metrics = compute_metrics(
            np.array(self._targets),
            np.array(self._preds),
        )
        metrics["loss"] = self._loss_sum / max(self._n_batches, 1)
        return metrics