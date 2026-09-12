# scripts/precompute_rgb.py
"""
scripts/precompute_rgb.py
--------------------------
Post-processing step to be run ONCE after cochleogram generation
(i.e. after scripts/preprocess.py has been run).

What it does:
    Converts grayscale cochleograms (.npy, shape: H x W, values in [0, 1])
    to RGB cochleograms (.npy, shape: 3 x H x W, values in [0, 1]) by
    applying the Viridis colormap.

Usage:
    python scripts/precompute_rgb.py

After running this script, update configs/default.yaml:
    processed_dir: "data/processed/cochleograms_rgb"
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from cochleogram_vit.utils.config import load_config


def main():
    cfg = load_config("configs/default.yaml")

    data_dir      = cfg["data"]["processed_dir"]
    output_dir    = data_dir.rstrip("/") + "_rgb"
    metadata_path = cfg["data"]["metadata_path"]

    print(f"[precompute_rgb] Input:    {data_dir}")
    print(f"[precompute_rgb] Output:   {output_dir}")
    print(f"[precompute_rgb] Metadata: {metadata_path}")

    os.makedirs(output_dir, exist_ok=True)
    viridis  = plt.get_cmap("viridis")
    metadata = pd.read_csv(metadata_path)

    skipped   = 0
    converted = 0

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Converting"):
        fname    = os.path.basename(row["npy_path"])
        in_path  = os.path.join(data_dir, fname)
        out_path = os.path.join(output_dir, fname)

        if os.path.exists(out_path):
            skipped += 1
            continue

        gray    = np.load(in_path)
        colored = viridis(gray)
        rgb     = np.ascontiguousarray(
            colored[:, :, :3].transpose(2, 0, 1)
        ).astype(np.float32)
        np.save(out_path, rgb)
        converted += 1

    print(f"\n[precompute_rgb] Done.")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped} (already existed)")
    print(f"  Output:    {output_dir}")
    print(f"\n  Next step: update configs/default.yaml:")
    print(f'    processed_dir: "{output_dir}"')


if __name__ == "__main__":
    main()