"""
scripts/preprocess.py
---------------------
Scan the raw ICBHI dataset, generate cochleagrams for every respiratory cycle,
and save them to disk as .npy files alongside a metadata CSV.

Usage:
    python scripts/preprocess.py --config configs/default.yaml

Output layout (under data/processed/):
    cochleagrams/
        <recording_id>_<cycle_idx>.npy   -- shape (128, 128) float32
    metadata.csv                          -- wav_path, start, end, label, npy_path
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from cochleogram_vit.data.dataset import build_icbhi_dataframe
from cochleogram_vit.preprocessing.cochleogram import CochleogramTransform
from cochleogram_vit.utils.config import load_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess ICBHI dataset into cochleograms.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N cycles (for quick testing).")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["training"]["seed"])

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    cg_dir = processed_dir / "cochleograms"
    cg_dir.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess] Scanning raw dataset at: {raw_dir}")
    df = build_icbhi_dataframe(raw_dir)
    print(f"[preprocess] Found {len(df)} respiratory cycles.")

    if args.limit:
        df = df.head(args.limit)
        print(f"[preprocess] Limited to {len(df)} cycles.")

    sr = cfg["data"]["sample_rate"]
    clip_duration = cfg["data"]["clip_duration"]
    cg_cfg = cfg["cochleogram"]

    transform = CochleogramTransform(
        sr=sr,
        n_filters=cg_cfg["n_filters"],
        low_lim=cg_cfg["low_lim"],
        high_lim=cg_cfg["high_lim"],
        sample_factor=cg_cfg["sample_factor"],
        downsample=cg_cfg.get("downsample"),
        output_size=cfg["model"]["image_size"],
    )

    npy_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating cochleograms"):
        # Load and slice audio
        waveform, orig_sr = torchaudio.load(row["wav_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != sr:
            waveform = torchaudio.transforms.Resample(orig_sr, sr)(waveform)

        start_s = int(row["start"] * sr)
        end_s = int(row["end"] * sr)
        clip = waveform[:, start_s:end_s]

        clip_len = int(clip_duration * sr)
        if clip.shape[-1] < clip_len:
            clip = torch.nn.functional.pad(clip, (0, clip_len - clip.shape[-1]))
        else:
            clip = clip[..., :clip_len]

        # Generate cochleogram
        cg_tensor = transform(clip)  # (1, H, W)
        cg_np = cg_tensor.squeeze(0).numpy()  # (H, W)

        # Derive a clean file name from the source wav
        rec_id = Path(row["wav_path"]).stem
        npy_name = f"{rec_id}_{idx:05d}.npy"
        npy_path = cg_dir / npy_name
        np.save(npy_path, cg_np)
        npy_paths.append(str(npy_path))

    df = df.reset_index(drop=True)
    df["npy_path"] = npy_paths
    meta_path = processed_dir / "metadata.csv"
    df.to_csv(meta_path, index=False)

    print(f"\n[preprocess] Done.")
    print(f"  Cochleograms saved to : {cg_dir}")
    print(f"  Metadata CSV saved to : {meta_path}")
    print(f"  Label distribution    :\n{df['label_name'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
