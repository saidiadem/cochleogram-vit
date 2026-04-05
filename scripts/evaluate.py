"""
scripts/evaluate.py
-------------------
Load a trained checkpoint and run evaluation on the full dataset or a test split.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
"""

import argparse

import torch
from torch.utils.data import DataLoader

from cochleogram_vit.data.dataset import ICBHIDataset, build_icbhi_dataframe
from cochleogram_vit.models.vit import CochleogramViT
from cochleogram_vit.preprocessing.cochleogram import CochleogramTransform
from cochleogram_vit.training.metrics import MetricTracker
from cochleogram_vit.utils.config import get_device, load_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CochleogramViT checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["training"]["seed"])
    device = get_device(cfg["training"]["device"])

    # Dataset
    df = build_icbhi_dataframe(cfg["data"]["raw_dir"])
    transform = CochleogramTransform(
        sr=cfg["data"]["sample_rate"],
        n_filters=cfg["cochleogram"]["n_filters"],
        low_lim=cfg["cochleogram"]["low_lim"],
        high_lim=cfg["cochleogram"]["high_lim"],
        sample_factor=cfg["cochleogram"]["sample_factor"],
        output_size=cfg["model"]["image_size"],
    )
    dataset = ICBHIDataset(df, cfg["data"]["sample_rate"], cfg["data"]["clip_duration"], transform)
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # Model
    model = CochleogramViT.from_config(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[evaluate] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Evaluation
    criterion = torch.nn.CrossEntropyLoss()
    tracker = MetricTracker()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            tracker.update(logits, targets, loss.item())

    metrics = tracker.compute()
    print("\n===== Evaluation Results =====")
    print(f"  Loss       : {metrics['loss']:.4f}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  ICBHI Score: {metrics['icbhi_score']:.4f}")
    print("\n" + tracker.classification_report())


if __name__ == "__main__":
    main()
