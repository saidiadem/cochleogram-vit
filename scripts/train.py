# scripts/train.py
"""
End-to-end training entry point.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import os

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler

from cochleogram_vit.data.dataset import CochleogramDataset
from cochleogram_vit.models.vit import CochleogramViT
from cochleogram_vit.models.baseline_cnn import BaselineCNN
from cochleogram_vit.training.trainer import Trainer
from cochleogram_vit.utils.config import get_device, load_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train CochleogramViT on ICBHI 2017.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])
    print(f"[train] Using device: {device}")

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    dataset = CochleogramDataset(
        data_dir=cfg["data"]["processed_dir"],
        metadata_path=cfg["data"]["metadata_path"],
    )
    print(f"[train] Total samples: {len(dataset)}")

    # ------------------------------------------------------------------ #
    # Class weights (softened ^0.75, renormalized)
    # ------------------------------------------------------------------ #
    labels = dataset.metadata["label"].values
    raw_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2, 3]),
        y=labels,
    )
    class_weights = raw_weights ** 0.75
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"[train] Class weights (softened ^0.75, renormalized):")
    for i, name in enumerate(["Normal", "Crackles", "Wheezes", "Both"]):
        print(f"         {name:<10}: {class_weights[i]:.4f}")

    # ------------------------------------------------------------------ #
    # GroupKFold by patient ID
    # ------------------------------------------------------------------ #
    metadata = dataset.metadata.copy()
    metadata["patient_id"] = metadata["npy_path"].apply(
        lambda x: os.path.basename(x).split("_")[0]
    )
    groups = metadata["patient_id"].values
    n_splits = cfg["training"]["n_splits"]
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    all_preds_total = []
    all_labels_total = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(metadata, groups=groups)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{n_splits}")
        print(f"{'='*60}")

        # Per-fold seed
        seed_everything(cfg["training"]["seed"] + fold)

        # WeightedRandomSampler for training
        train_labels = metadata["label"].values[train_idx]
        sample_weights = torch.tensor(
            [class_weights[l] for l in train_labels],
            dtype=torch.float,
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        # SubsetRandomSampler for validation (real distribution, no balancing)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            dataset,
            batch_size=cfg["training"]["batch_size"],
            sampler=train_sampler,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            dataset,
            batch_size=cfg["training"]["batch_size"],
            sampler=val_sampler,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=(device.type == "cuda"),
        )

        print(f"  Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")

        # ------------------------------------------------------------------ #
        # Model — re-initialized every fold
        # ------------------------------------------------------------------ #
        model_name = cfg["model"].get("name", "CochleogramViT")
        if model_name == "BaselineCNN":
            model = BaselineCNN(
                in_channels=cfg["model"].get("channels", 3),
                num_classes=cfg["model"]["num_classes"],
            )
        else:
            model = CochleogramViT.from_config(cfg)
        model = model.to(device)

        # ------------------------------------------------------------------ #
        # Trainer
        # ------------------------------------------------------------------ #
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            class_weights=class_weights,
            fold=fold + 1,
        )
        fold_result = trainer.fit()

        fold_results.append(fold_result)
        all_preds_total.extend(fold_result["preds"])
        all_labels_total.extend(fold_result["labels"])

    # ------------------------------------------------------------------ #
    # Aggregated results across all folds
    # ------------------------------------------------------------------ #
    from cochleogram_vit.training.metrics import compute_metrics

    print(f"\n{'='*60}")
    print("AGGREGATED 10-FOLD RESULTS")
    print(f"{'='*60}")

    agg = compute_metrics(
        np.array(all_labels_total),
        np.array(all_preds_total),
    )
    print(f"  Accuracy:    {agg['accuracy']*100:.2f}%")
    print(f"  Sensitivity: {agg['sensitivity']*100:.2f}%")
    print(f"  Specificity: {agg['specificity']*100:.2f}%")
    print(f"  Precision:   {agg['precision']*100:.2f}%")
    print(f"  Score:       {agg['score']*100:.2f}%")
    print(f"  TP={agg['TP']}  FN={agg['FN']}  TN={agg['TN']}  FP={agg['FP']}  FN_wrong_type={agg['FN_wrong_type']}")

    # Macro average (mean ± std across folds)
    scores = [f["score"] for f in fold_results]
    print(f"\n  Macro Score: {np.mean(scores)*100:.2f}% ± {np.std(scores)*100:.2f}%")
    print(f"\n{'='*60}")
    print("Training complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()