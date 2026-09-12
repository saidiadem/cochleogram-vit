"""
ICBHI 2017 Challenge Dataset loader.

Two dataset classes are provided:

1. CochleogramDataset — loads pre-generated RGB cochleograms from disk.
   Use this after running scripts/precompute_rgb.py.
   This is the recommended and faster option.

2. ICBHIDataset — loads raw .wav files and applies a transform on the fly.
   Use this if you want to generate cochleograms during training.

Directory layout for CochleogramDataset:
    data/processed/
        cochleograms_rgb/    ← RGB .npy files (shape: 3 x H x W, float32)
        metadata.csv         ← columns: npy_path, label

Labels map:
    0 → normal   (crackles=0, wheezes=0)
    1 → crackle  (crackles=1, wheezes=0)
    2 → wheeze   (crackles=0, wheezes=1)
    3 → both     (crackles=1, wheezes=1)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

LABEL_MAP = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


# ------------------------------------------------------------------ #
# CochleogramDataset — pre-generated RGB .npy files (used in training)
# ------------------------------------------------------------------ #

class CochleogramDataset(Dataset):
    """
    Loads pre-generated RGB cochleograms from .npy files.
    Expects files to be already in (3, H, W) float32 format,
    as produced by scripts/precompute_rgb.py.

    Args:
        data_dir:      Path to folder containing RGB .npy cochleogram files.
        metadata_path: Path to metadata.csv with columns: npy_path, label.
        transform:     Optional callable applied to the tensor after loading.
    """

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        transform: Optional[Callable] = None,
    ):
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[idx]
        npy_path = os.path.join(self.data_dir, os.path.basename(row["npy_path"]))

        # Already (3, H, W) float32 — no colormap conversion needed
        rgb = np.load(npy_path)
        tensor = torch.from_numpy(rgb)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, int(row["label"])


# ------------------------------------------------------------------ #
# ICBHIDataset — raw .wav files with on-the-fly transform
# ------------------------------------------------------------------ #

def _parse_annotation_file(txt_path: Path) -> list[tuple[float, float, int, int]]:
    """Return a list of (start, end, crackle, wheeze) tuples from an annotation .txt."""
    cycles = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            start, end, crackle, wheeze = (
                float(parts[0]), float(parts[1]),
                int(parts[2]), int(parts[3]),
            )
            cycles.append((start, end, crackle, wheeze))
    return cycles


def build_icbhi_dataframe(raw_dir: str | Path) -> pd.DataFrame:
    """
    Scan `raw_dir` and build a DataFrame with one row per respiratory cycle.
    Columns: wav_path, start, end, label (int), label_name (str)
    """
    raw_dir = Path(raw_dir)
    rows = []

    for txt_path in sorted(raw_dir.glob("*.txt")):
        if "diagnosis" in txt_path.name or "label" in txt_path.name.lower():
            continue
        wav_path = txt_path.with_suffix(".wav")
        if not wav_path.exists():
            continue

        for start, end, crackle, wheeze in _parse_annotation_file(txt_path):
            label = LABEL_MAP.get((crackle, wheeze), 0)
            rows.append({
                "wav_path": str(wav_path),
                "start": start,
                "end": end,
                "label": label,
                "label_name": CLASS_NAMES[label],
            })

    return pd.DataFrame(rows)


class ICBHIDataset(Dataset):
    """
    PyTorch Dataset for ICBHI 2017 respiratory sound cycles.
    Loads raw .wav files and applies a transform on the fly.

    Args:
        dataframe:     DataFrame produced by `build_icbhi_dataframe`.
        target_sr:     Target sample rate (audio is resampled if needed).
        clip_duration: Fixed clip length in seconds.
        transform:     Optional callable applied to the waveform tensor.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_sr: int = 22050,
        clip_duration: float = 5.0,
        transform: Optional[Callable] = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.target_sr = target_sr
        self.clip_samples = int(clip_duration * target_sr)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        waveform, sr = torchaudio.load(row["wav_path"])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.target_sr
            )
            waveform = resampler(waveform)

        start_sample = int(row["start"] * self.target_sr)
        end_sample   = int(row["end"]   * self.target_sr)
        clip = waveform[:, start_sample:end_sample]
        clip = self._fix_length(clip)

        if self.transform is not None:
            clip = self.transform(clip)

        return clip, int(row["label"])

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.shape[-1]
        if length < self.clip_samples:
            pad = self.clip_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[..., : self.clip_samples]
        return waveform

    @property
    def class_weights(self) -> torch.Tensor:
        counts = self.df["label"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(float)
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)