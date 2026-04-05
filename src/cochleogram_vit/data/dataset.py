"""
ICBHI 2017 Challenge Dataset loader.

Directory layout expected under `raw_dir`:
    <raw_dir>/
        *.wav          -- audio recordings
        *.txt          -- annotation files (one per recording)
        ICBHI_Challenge_diagnosis.txt  -- patient diagnosis metadata

Annotation file format (per respiratory cycle):
    <start_sec> <end_sec> <crackles> <wheezes>
    e.g.:  0.036  1.018  0  0

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


def _parse_annotation_file(txt_path: Path) -> list[tuple[float, float, int, int]]:
    """Return a list of (start, end, crackle, wheeze) tuples from an annotation .txt."""
    cycles = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            start, end, crackle, wheeze = float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])
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
            rows.append(
                {
                    "wav_path": str(wav_path),
                    "start": start,
                    "end": end,
                    "label": label,
                    "label_name": CLASS_NAMES[label],
                }
            )

    df = pd.DataFrame(rows)
    return df


class ICBHIDataset(Dataset):
    """
    PyTorch Dataset for ICBHI 2017 respiratory sound cycles.

    Args:
        dataframe:    DataFrame produced by `build_icbhi_dataframe`.
        target_sr:    Target sample rate (audio is resampled if needed).
        clip_duration: Fixed clip length in seconds. Shorter clips are zero-padded;
                       longer clips are truncated.
        transform:    Optional callable applied to the raw waveform tensor
                      (e.g. cochleogram conversion). Should return a tensor.
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

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Slice the respiratory cycle
        start_sample = int(row["start"] * self.target_sr)
        end_sample = int(row["end"] * self.target_sr)
        clip = waveform[:, start_sample:end_sample]

        # Pad or truncate to fixed length
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
        """Inverse-frequency weights for use with WeightedRandomSampler or CrossEntropyLoss."""
        counts = self.df["label"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(float)
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)
