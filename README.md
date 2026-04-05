# Cochleogram-ViT — Abnormal Respiratory Sound Classification

**PFA (End of Studies Project)**  
*Implementation of a Transformer-Based Architecture Using Cochleograms for Abnormal Respiratory Sound Classification*

---

## Overview

This project implements and extends the method proposed in:

> *"Classification of Adventitious Sounds Combining Cochleogram and Vision Transformers"*

The core idea is to convert lung auscultation audio into **cochleograms** — biologically-inspired auditory spectrograms computed using a gammatone filter bank on an Equivalent Rectangular Bandwidth (ERB) scale — and feed these 2D representations into a **Vision Transformer (ViT)** for 4-class classification.

| Class | Label | Description |
|-------|-------|-------------|
| Normal | 0 | No adventitious sounds |
| Crackle | 1 | Discontinuous, explosive sounds (pneumonia, fibrosis) |
| Wheeze | 2 | Musical, continuous sounds (asthma, COPD) |
| Both | 3 | Crackle + wheeze present simultaneously |

---

## Project Structure

```
pfa/
├── configs/
│   └── default.yaml              # All hyperparameters in one place
├── data/
│   ├── raw/                      # ICBHI .wav + .txt files go here
│   └── processed/                # Generated cochleograms (.npy) + metadata.csv
├── notebooks/
│   └── 01_preprocessing_exploration.ipynb   # Step-by-step visual walkthrough
├── scripts/
│   ├── preprocess.py             # Batch-generate cochleograms from raw audio
│   ├── train.py                  # Full training loop
│   └── evaluate.py               # Evaluate a saved checkpoint
├── src/
│   └── cochleogram_vit/
│       ├── data/
│       │   └── dataset.py        # ICBHI Dataset + annotation parser
│       ├── preprocessing/
│       │   └── cochleogram.py    # CochleogramTransform (pycochleagram / librosa)
│       ├── models/
│       │   └── vit.py            # CochleogramViT wrapper around vit-pytorch
│       ├── training/
│       │   ├── trainer.py        # Training loop, optimizer, scheduler, TensorBoard
│       │   └── metrics.py        # ICBHI score, accuracy, classification report
│       └── utils/
│           └── config.py         # YAML loading, device selection, seeding
├── requirements.txt
└── pyproject.toml
```

---

## Architecture

```
Input: raw audio (.wav)
    │
    ▼
[CochleogramTransform]
  ├── Gammatone filter bank (ERB scale, pycochleagram)
  ├── Sub-band envelope extraction (Hilbert transform)
  ├── Power-law compression (x^0.3)
  └── Resize → (1 × 128 × 128)
    │
    ▼
[Vision Transformer — CochleogramViT]
  ├── Patch Embedding  (16×16 patches → 64 tokens)
  ├── [CLS] token + Positional Embedding
  ├── Transformer Encoder (6 blocks, 8 heads, dim=512)
  └── MLP Head → 4-class logits
    │
    ▼
Output: normal / crackle / wheeze / both
```

---

## Dataset — ICBHI 2017

1. Download from the official challenge page: https://bhichallenge.med.auth.gr/
2. Extract all `.wav` and `.txt` files into `data/raw/`.

Expected layout:
```
data/raw/
    101_1b1_Al_sc_Meditron.wav
    101_1b1_Al_sc_Meditron.txt
    ...
    ICBHI_Challenge_diagnosis.txt
```

Each `.txt` annotation file contains one row per respiratory cycle:
```
<start_sec>  <end_sec>  <crackles>  <wheezes>
```

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch (adjust for your CUDA version — see pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Install pycochleagram (not on PyPI)
pip install git+https://github.com/mcdermottLab/pycochleagram.git

# 5. Install the project package in editable mode
pip install -e .
```

> **No GPU?** Set `device: "cpu"` in `configs/default.yaml`. Training will be slower but fully functional. The librosa mel-spectrogram fallback is used if pycochleagram is not installed.

---

## Quickstart

### Step 1 — Explore the preprocessing pipeline (notebook)

```bash
jupyter notebook notebooks/01_preprocessing_exploration.ipynb
```

This notebook visualizes raw waveforms, mel-spectrograms, and cochleograms side-by-side for each of the 4 ICBHI classes.

### Step 2 — Batch preprocess the dataset

```bash
python scripts/preprocess.py --config configs/default.yaml
# For a quick sanity-check (first 50 cycles only):
python scripts/preprocess.py --config configs/default.yaml --limit 50
```

### Step 3 — Train

```bash
python scripts/train.py --config configs/default.yaml
```

Monitor training in TensorBoard:
```bash
tensorboard --logdir runs/
```

### Step 4 — Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --config configs/default.yaml
```

---

## Key Configuration Options (`configs/default.yaml`)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `data` | `sample_rate` | 22050 | Resample all audio to this SR |
| `data` | `clip_duration` | 5.0 s | Fixed clip length (pad/truncate) |
| `cochleogram` | `n_filters` | 128 | ERB filter bank channels |
| `cochleogram` | `low_lim` | 50 Hz | Lowest frequency |
| `cochleogram` | `high_lim` | 8000 Hz | Highest frequency |
| `model` | `image_size` | 128 | Cochleogram resize target |
| `model` | `patch_size` | 16 | ViT patch size |
| `model` | `depth` | 6 | Transformer encoder blocks |
| `model` | `heads` | 8 | Attention heads |
| `training` | `epochs` | 50 | Total training epochs |
| `training` | `batch_size` | 32 | Mini-batch size |
| `training` | `learning_rate` | 3e-4 | AdamW initial LR |

---

## Evaluation Metric

The ICBHI challenge defines the scoring metric as:

```
ICBHI Score = (mean Sensitivity + mean Specificity) / 2
```

computed in a 4-class setting. This is reported alongside standard accuracy and a full per-class classification report.

---

## References

- Dosovitskiy et al., *"An Image is Worth 16×16 Words"*, ICLR 2021
- *Classification of Adventitious Sounds Combining Cochleogram and Vision Transformers* (core paper)
- ICBHI 2017 Respiratory Sound Database
- [pycochleagram](https://github.com/mcdermottLab/pycochleagram) — McDermott Lab
- [vit-pytorch](https://github.com/lucidrains/vit-pytorch) — lucidrains
- [MVST](https://github.com/wentaoheunnc/MVST) — reference ViT + spectrogram codebase
