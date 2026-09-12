# PFA Report: Abnormal Respiratory Sound Classification

This report summarizes the experiments conducted for the PFA (End of Studies Project) on abnormal respiratory sound classification using deep learning models.

## 1. Comparative Table of Results

This table compares the performance of the different models and techniques evaluated. The primary metric is the ICBHI score (average of Sensitivity and Specificity).

| Experiment ID                  | Model       | Technique            | Per-Fold Score (Mean ± Std) | Pooled Score |
| ------------------------------ | ----------- | -------------------- | --------------------------- | ------------ |
| **ast-sam-soft0.25-stratified**| **AST**     | **SAM Optimizer**    | **68.19% ± 4.41%**          | **68.58%**   |
| ast-soft0.25-stratified        | AST         | Baseline             | 68.14% ± 4.08%              | 68.02%       |
| sampler-p05                    | ViT         | Weighted Sampler     | 64.27% ± 4.82%              | 64.31%       |
| loss-p025                      | ViT         | Weighted Loss (p=0.25)| 65.25% ± 5.49%              | 64.67%       |
| loss-p05                       | ViT         | Weighted Loss (p=0.5) | 64.86% ± 5.70%              | 63.21%       |
| mixup-a0.4-p1.0-soft0.25       | ViT         | MixUp (α=0.4)        | 61.84% ± 6.06%              | 62.26%       |
| librosa-aug                    | ViT         | Audio Augmentation   | 60.39% ± 3.69%              | 60.48%       |
| librosa-baseline               | ViT         | Baseline (librosa)   | 60.55% ± 2.76%              | 60.42%       |

**Conclusion**: The Audio Spectrogram Transformer (AST) model, particularly when combined with the Sharpness-Aware Minimization (SAM) optimizer, achieves the best performance, significantly outperforming the Vision Transformer (ViT) baselines.

## 2. Training Configurations

Here are the key hyperparameters and configurations for each major experiment.

### 2.1. AST (Audio Spectrogram Transformer)

*   **Model**: `MIT/ast-finetuned-audioset-10-10-0.4593`
*   **Input**: Single-channel cochleogram (128x128), per-sample normalized.
*   **Optimizer**: AdamW
*   **Learning Rate**: 5e-5
*   **Weight Decay**: 0.01
*   **Batch Size**: 8
*   **Epochs**: 30
*   **Loss**: Cross-Entropy with class weights (`soften_power=0.25`)
*   **Data Split**: StratifiedGroupKFold (10 folds)
*   **SAM**: For the `ast-sam` experiment, `rho` was set to `0.05`.

### 2.2. ViT (Vision Transformer) - Various Experiments

*   **Model**: Vision Transformer (from scratch)
*   **Input**: RGB Cochleogram (from `pycochleagram` with viridis colormap)
*   **Optimizer**: AdamW
*   **Learning Rate**: 1e-4
*   **Batch Size**: 16
*   **Epochs**: 30
*   **Data Split**: StratifiedGroupKFold (10 folds)

**Experiment-specific configurations:**
*   **Weighted Loss**: `imbalance='weighted_loss'` with `softpow` of 0.25 or 0.5.
*   **Weighted Sampler**: `imbalance='weighted_sampler'`.
*   **MixUp**: `mixup_alpha=0.4`, `mixup_prob=1.0`.
*   **Audio Augmentation**: Augmentations (e.g., noise, pitch shift) applied to the raw audio before cochleogram generation.

## 3. Architecture and Methodology

### 3.1. Preprocessing: Cochleogram Generation

Raw audio files (`.wav`) are transformed into 2D representations called cochleograms. This process mimics the human auditory system.

```mermaid
graph TD
    A[Raw Audio (.wav)] --> B{Gammatone Filter Bank};
    B --> C{Sub-band Envelope Extraction};
    C --> D{Power-law Compression};
    D --> E[Resize to 128x128];
    E --> F[Cochleogram Image];
```
*Figure 1: Cochleogram generation pipeline.*

For ViT models, the grayscale cochleogram is converted to an RGB image using the "viridis" colormap. For the AST model, the single-channel (grayscale) cochleogram is used directly, as this aligns better with its pre-training data.

### 3.2. Model Architectures

#### Vision Transformer (ViT)

The baseline model is a Vision Transformer. It processes the 2D cochleogram image by dividing it into patches and feeding them through a series of Transformer encoders.

```mermaid
graph TD
    A[Cochleogram Image (128x128)] --> B{Patch Embedding};
    B --> C{Add [CLS] Token & Positional Embeddings};
    C --> D[Transformer Encoder Blocks];
    D --> E{MLP Head};
    E --> F[4-Class Output];
```
*Figure 2: Vision Transformer (ViT) architecture.*

#### Audio Spectrogram Transformer (AST)

AST is a ViT pre-trained on AudioSet, a large-scale dataset of audio clips. This domain-specific pre-training provides a significant advantage for audio classification tasks. The positional embeddings of the pre-trained model were interpolated to match the dimensions of our cochleograms.

### 3.3. Data Imbalance Handling

The ICBHI dataset is imbalanced across the four classes (Normal, Crackle, Wheeze, Both). Two primary strategies were used to address this:

1.  **Weighted Cross-Entropy Loss**: The loss function is modified to give more weight to minority classes, forcing the model to pay more attention to them. The `soften_power` parameter controls the strength of this weighting.
2.  **Weighted Random Sampler**: The data loader samples instances from minority classes more frequently, creating more balanced batches during training.

## 4. Visualizations (Placeholder)

This section is a placeholder for the figures you requested. You will need to generate these using the notebooks and saved results (`.npz` and `.json` files).

### Confusion Matrix

*To generate, use the `pooled_preds` and `pooled_labels` from the experiment's results or the saved `.npz` files. Plot a confusion matrix for the best model (e.g., `ast-sam`).*

**(Placeholder for Confusion Matrix Figure)**

### Evolution of Metrics (Training & Validation Curves)

*To generate, you will need to log metrics (loss, accuracy, Se/Sp) during training for each epoch. The `runs/` directory contains TensorBoard logs which can be used to plot these curves.*

**(Placeholder for Training/Validation Loss Curve)**
**(Placeholder for Training/Validation Score Curve)**

### Sensitivity/Specificity (Se/Sp) Curves

*Similar to the metric evolution, these can be plotted from the validation results saved at each epoch.*

**(Placeholder for Validation Se/Sp Curve)**

### t-SNE Visualization of Embeddings

*To generate this, you need to extract the feature embeddings (e.g., the output of the final Transformer block) for the test set and use a tool like `sklearn.manifold.TSNE` to project them into 2D space. Color the points by their true class label.*

**(Placeholder for t-SNE Figure)**
*Figure 3. t-SNE visualization of the learned embeddings.*
