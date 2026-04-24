"""
Cochleogram generation pipeline.

A cochleagram is an auditory spectrogram computed using a bank of gammatone
(or related) filters spaced on an Equivalent Rectangular Bandwidth (ERB)
scale — closely mimicking the frequency resolution of the human cochlea.

This module provides two backends:
  1. pycochleagram (McDermott Lab) — full biologically-inspired pipeline.
     Install: pip install git+https://github.com/mcdermottLab/pycochleagram.git
  2. Librosa mel-spectrogram fallback — used when pycochleagram is not available,
     producing a visually similar time-frequency representation.

Both backends output a 2-D numpy array of shape (n_filters, n_time_frames)
that is then wrapped into a (1, H, W) torch.Tensor suitable for ViT input.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch

try:
    from pycochleagram import cochleagram as _cgram_module  # pycochleagram public API

    _PYCOCHLEAGRAM_AVAILABLE = True
except ImportError:
    _PYCOCHLEAGRAM_AVAILABLE = False
    warnings.warn(
        "pycochleagram not found — using librosa mel-spectrogram fallback.\n"
        "Install the real cochleagram backend with:\n"
        "  pip install git+https://github.com/mcdermottLab/pycochleagram.git",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Backend 1: pycochleagram
# ---------------------------------------------------------------------------

def _cochleagram_pycochleagram(
    waveform: np.ndarray,
    sr: int,
    n_filters: int,
    low_lim: float,
    high_lim: float,
    sample_factor: int,
    downsample: Optional[int],
) -> np.ndarray:
    """
    Generate a cochleagram using the pycochleagram library.

    Steps (following McDermott Lab pipeline):
      1. Design a bank of cosine-windowed bandpass filters on an ERB scale.
      2. Apply filters via FFT convolution (fast for long signals).
      3. Extract the envelope (Hilbert transform) of each sub-band.
      4. Optionally downsample the temporal dimension.
      5. Apply a compression non-linearity (power law: x^0.3).

    Returns:
        2-D array of shape (n_filters, n_time_frames), values in [0, 1].
    """
    # pycochleagram expects a 1-D signal
    signal = waveform.squeeze()

    cg = _cgram_module.cochleagram(
        signal,
        sr,
        n=n_filters,
        low_lim=low_lim,
        hi_lim=high_lim,
        sample_factor=sample_factor,
        downsample=downsample,
        nonlinearity="power",  # applies x^0.3 compression
        strict=True,
        ret_mode="envs",
    )  # shape: (n_filters, n_time_frames)

    # Normalize to [0, 1]
    cg_min, cg_max = cg.min(), cg.max()
    if cg_max > cg_min:
        cg = (cg - cg_min) / (cg_max - cg_min)

    return cg.astype(np.float32)


# ---------------------------------------------------------------------------
# Backend 2: librosa mel-spectrogram fallback
# ---------------------------------------------------------------------------

def _cochleagram_librosa_fallback(
    waveform: np.ndarray,
    sr: int,
    n_filters: int,
    low_lim: float,
    high_lim: float,
) -> np.ndarray:
    """
    Approximate a cochleagram using a librosa mel-spectrogram.

    The mel scale is a perceptual frequency scale that closely approximates
    the ERB spacing used in true cochleagrams at the cost of some biological
    fidelity.

    Returns:
        2-D array of shape (n_filters, n_time_frames), values in [0, 1].
    """
    import librosa

    signal = waveform.squeeze()

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=n_filters,
        fmin=low_lim,
        fmax=high_lim,
        power=2.0,
    )

    # Log compression (approximates cochlear compressive non-linearity)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

    return log_mel.astype(np.float32)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class CochleogramTransform:
    """
    Callable transform: waveform tensor → cochleagram tensor.

    Designed to be passed as the `transform` argument to `ICBHIDataset`.

    Args:
        sr:            Sample rate of the input waveform.
        n_filters:     Number of frequency channels.
        low_lim:       Lower frequency bound (Hz).
        high_lim:      Upper frequency bound (Hz).
        sample_factor: pycochleagram temporal resolution multiplier.
        downsample:    pycochleagram temporal downsampling factor (None = no DS).
        output_size:   If given, resize cochleagram to (output_size, output_size)
                       using bilinear interpolation — required for fixed ViT input.
    """

    def __init__(
        self,
        sr: int = 22050,
        n_filters: int = 128,
        low_lim: int = 50,
        high_lim: int = 8000,
        sample_factor: int = 1,
        downsample: Optional[int] = None,
        output_size: Optional[int] = 128,
    ):
        self.sr = sr
        self.n_filters = n_filters
        self.low_lim = low_lim
        self.high_lim = high_lim
        self.sample_factor = sample_factor
        self.downsample = downsample
        self.output_size = output_size

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape (1, n_samples) — mono audio.
        Returns:
            Tensor of shape (1, output_size, output_size) or (1, n_filters, n_time).
        """
        audio_np = waveform.squeeze().numpy()

        if _PYCOCHLEAGRAM_AVAILABLE:
            cg = _cochleagram_pycochleagram(
                audio_np,
                self.sr,
                self.n_filters,
                self.low_lim,
                self.high_lim,
                self.sample_factor,
                self.downsample,
            )
        else:
            cg = _cochleagram_librosa_fallback(
                audio_np,
                self.sr,
                self.n_filters,
                self.low_lim,
                self.high_lim,
            )

        # (n_filters, n_time) → torch (1, n_filters, n_time)
        tensor = torch.from_numpy(cg).unsqueeze(0)

        if self.output_size is not None:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),           # (1, 1, H, W)
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)                        # (1, H, W)

        return tensor
