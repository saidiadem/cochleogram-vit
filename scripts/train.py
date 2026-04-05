"""
scripts/train.py
----------------
End-to-end training entry point.

Usage:
    python scripts/train.py --config configs/default.yaml

The script:
  1. Loads config and sets seeds.
  2. Builds ICBHIDataset with CochleogramTransform (on-the-fly generation)
     OR loads pre-generated cochleograms from data/processed/ (faster).
  3. Splits into train/val using a fixed seed.
  4. Instantiates CochleogramViT from config.
  5. Runs training via Trainer.
"""

import argparse

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from cochleogram_vit.data.dataset import ICBHIDataset, build_icbhi_dataframe
from cochleogram_vit.models.vit import CochleogramViT
from cochleogram_vit.preprocessing.cochleogram import CochleogramTransform
from cochleogram_vit.training.trainer import Trainer
from cochleogram_vit.utils.config import get_device, load_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train CochleogramViT on ICBHI 2017.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume from.")
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
    raw_dir = cfg["data"]["raw_dir"]
    print(f"[train] Building dataset from: {raw_dir}")
    df = build_icbhi_dataframe(raw_dir)
    print(f"[train] Total cycles: {len(df)}")

    transform = CochleogramTransform(
        sr=cfg["data"]["sample_rate"],
        n_filters=cfg["cochleogram"]["n_filters"],
        low_lim=cfg["cochleogram"]["low_lim"],
        high_lim=cfg["cochleogram"]["high_lim"],
        sample_factor=cfg["cochleogram"]["sample_factor"],
        downsample=cfg["cochleogram"].get("downsample"),
        output_size=cfg["model"]["image_size"],
    )

    dataset = ICBHIDataset(
        dataframe=df,
        target_sr=cfg["data"]["sample_rate"],
        clip_duration=cfg["data"]["clip_duration"],
        transform=transform,
    )

    # Train / val split (stratified)
    indices = list(range(len(dataset)))
    labels = df["label"].tolist()
    val_frac = cfg["training"]["val_split"]
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_frac,
        stratify=labels,
        random_state=cfg["training"]["seed"],
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    print(f"[train] Train: {len(train_set)}  Val: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = CochleogramViT.from_config(cfg)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[train] Resumed from: {args.resume} (epoch {checkpoint['epoch']})")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
