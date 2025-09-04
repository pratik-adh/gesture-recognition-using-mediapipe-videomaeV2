# Nepali Sign Language Gesture Recognition using VideoMAE

A deep learning pipeline for recognizing Nepali Sign Language (NSL) gestures from video sequences. This project leverages **MediaPipe** for hand region preprocessing and **VideoMAE** (Video Masked Autoencoders) for robust video classification. The system has been designed with carefully crafted augmentations and a stratified k-fold validation strategy to ensure high accuracy and generalization on limited gesture datasets.

---

## Overview

- **Goal**: Recognize 36 distinct Nepali Sign Language gestures.
- **Model**: VideoMAE with transformer-based spatiotemporal attention.
- **Preprocessing**: MediaPipe-based hand detection, cropping, and frame sampling.
- **Augmentation**: Brightness, contrast, saturation, blur, noise, and gamma variations.
- **Validation**: Stratified k-fold cross-validation for robust performance measurement.

---

## Dataset

- **Original size**: ~28 videos per class (36 classes).
- **After augmentation**: Expanded to ~130 videos per class.
- **Variations in raw data**: Different FPS (25–30), lengths, and resolutions (480×640 to 720×1280).

---

## Preprocessing

1. **Hand Detection**
   - MediaPipe Hands extracts bounding boxes of left and/or right hands.
2. **Cropping & Resizing**
   - Cropped hand regions are resized to `224×224` pixels.
3. **Frame Sampling**
   - Each video is uniformly sampled to **16 frames**.
4. **Normalization**
   - Pixel values normalized either to `[0,1]` or standardized with mean/std.
5. **Tensor Shape**
   - Single video after preprocessing: `(16, 224, 224, 3)` → (frames, height, width, channels).
   - Batched data: `(B, 16, 224, 224, 3)` → converted to `(B, T, C, H, W)` for VideoMAE.

> ℹ️ Hugging Face’s VideoMAE automatically permutes inputs internally:  
> `(B, T, C, H, W)` → `(B, C, T, H, W)`.

---

## Data Augmentation

To overcome the small dataset size and improve robustness, the following augmentations were applied:

- Brightness (high and low)
- Contrast (high and low)
- Saturation (high and low)
- Motion blur
- Gaussian blur
- Moderate noise
- Salt-and-pepper noise
- Gamma adjustment (high and low)

⚠️ **Excluded Augmentations**: Rotation and flipping were intentionally avoided because sign language gestures are highly orientation-dependent. Even slight variations could alter the meaning of gestures or cause overlaps between classes, leading to misinterpretation.

**Effect**: Dataset expanded from ~28 → ~130 videos per class.

---

## Model Architecture (VideoMAE)

1. **Input**: `(B, T, C, H, W)` → e.g. `(8, 16, 3, 224, 224)`.
2. **Tubelet Embedding**: Spatiotemporal patches (e.g. `2×16×16`).
3. **Transformer Encoder**: Spatial & temporal self-attention layers.
4. **CLS Token**: Extracted as a video-level embedding.
5. **Classification Head**: Linear layer → logits `(B, num_classes)`.

For NSL: `(B, 36)` output.

---

## Training & Validation

- **Validation Strategy**:

  - Stratified k-fold cross-validation (ensures balanced class distribution across folds).
  - Provides a reliable estimate of generalization performance compared to fixed splits.

- **Loss Function**: Cross-entropy.
- **Optimizer**: AdamW.
- **Learning Rate Scheduling**: Cosine decay with warmup.

---

## Usage

**1. Install Dependencies**:

```bash
pip install -r requirements.txt
```

**2. Preprocess Dataset**:

```bash
python preprocessing_videos.py
```

**3. Train Model**:

```bash
python train.py
```

**4. Run Inference**:

Run inference on the test dataset for recognizing and classifying the gestures.

```bash
pip inference_test.py
```
