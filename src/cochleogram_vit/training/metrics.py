"""
Evaluation metrics for ICBHI classification.

The ICBHI challenge uses a specific scoring metric:
    Score = (Sensitivity + Specificity) / 2
where both are computed in a 4-class multi-class setting.

We also report standard sklearn metrics for completeness.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def icbhi_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    """
    Compute the official ICBHI challenge score.

    Score = mean over classes of (TP_k / (TP_k + FN_k + FP_k + TN_k ...))
    Simplified as average of per-class sensitivity and specificity.

    Returns a scalar in [0, 1].
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    per_class_sensitivity = []
    per_class_specificity = []

    for k in range(n_classes):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        per_class_sensitivity.append(sensitivity)
        per_class_specificity.append(specificity)

    score = (np.mean(per_class_sensitivity) + np.mean(per_class_specificity)) / 2.0
    return float(score)


class MetricTracker:
    """Accumulates predictions over an epoch and computes all metrics at once."""

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

    def compute(self) -> dict[str, float]:
        y_pred = np.array(self._preds)
        y_true = np.array(self._targets)

        acc = (y_pred == y_true).mean()
        score = icbhi_score(y_true, y_pred)
        avg_loss = self._loss_sum / max(self._n_batches, 1)

        return {
            "loss": avg_loss,
            "accuracy": float(acc),
            "icbhi_score": score,
        }

    def classification_report(self) -> str:
        target_names = ["normal", "crackle", "wheeze", "both"]
        return classification_report(
            self._targets, self._preds, target_names=target_names, zero_division=0
        )
