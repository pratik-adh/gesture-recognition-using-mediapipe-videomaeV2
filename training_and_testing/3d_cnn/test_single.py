#!/usr/bin/env python3
"""
Working3DCNN Single Video Gesture Classifier
Provide a video file (MP4/AVI/NPZ), get the predicted gesture with confidence scores.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# ✏️  CONFIGURE YOUR PATHS HERE
# ─────────────────────────────────────────────
VIDEO_PATH = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test/BA/BA_11_aug_12.npz'
# VIDEO_PATH = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test/M_SHA/M_SHA_1_aug_06.npz'
# VIDEO_PATH   = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test/RA/RA_4_aug_19.npz'
MODEL_PATH = './../../kaggle/training_output_3dcnn/final_new/checkpoint_ep20.pth'

# Option A — list class names directly (must match training order):
CLASS_NAMES = [
    "BA",
    "BHA",
    "CHA",
    "CHHA",
    "DA",
    "DAA",
    "DHA",
    "DHAA",
    "D_SHA",
    "GA",
    "GHA",
    "GYA",
    "HA",
    "JA",
    "JHA",
    "KA",
    "KHA",
    "KSHA",
    "LA",
    "MA",
    "M_SHA",
    "NA",
    "NAA",
    "NGA",
    "PA",
    "PHA",
    "RA",
    "TA",
    "TAA",
    "THA",
    "THAA",
    "TRA",
    "T_SHA",
    "WA",
    "YA",
    "YAN",
]

# Option B — load from a text file (one class per line).
# Set to None to use CLASS_NAMES above, or provide a path to override:
CLASSES_FILE = None   # e.g. "classes.txt"

NUM_FRAMES = 16    # overridden automatically from checkpoint if possible
TOP_K      = 5    # how many top predictions to display

# ─────────────────────────────────────────────
# Model — exact copy of training architecture
# (Working3DCNN: 64 → 128 → 256 → 512)
# ─────────────────────────────────────────────
class Working3DCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.35):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 → 64
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            # Block 2: 64 → 128
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            # Block 3: 128 → 256
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            # Block 4: 256 → 512
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
# Auto-detect config from checkpoint shapes
# ─────────────────────────────────────────────
def infer_config(state_dict):
    """
    Reads dropout from classifier and channel counts from conv weights.
    Returns a dict with dropout and num_classes.
    Everything else (num_frames, img_size) is fixed by architecture.
    """
    # Last linear layer gives num_classes
    num_classes = int(state_dict['classifier.4.weight'].shape[0])
    return dict(num_classes=num_classes, dropout=0.35)


# ─────────────────────────────────────────────
# Video Loaders
# ─────────────────────────────────────────────
def load_npz_video(path: Path) -> np.ndarray:
    """Load frames from an NPZ file."""
    data   = np.load(path)
    frames = data['frames'] if 'frames' in data else data[data.files[0]]
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def load_mp4_video(path: Path) -> np.ndarray:
    """Load frames from an MP4/AVI/MOV/MKV file using OpenCV."""
    try:
        import cv2
    except ImportError:
        print("❌  OpenCV is required for MP4/AVI files:  pip install opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"❌  Cannot open video: {path}")
        sys.exit(1)

    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not raw_frames:
        print("❌  No frames could be read from the video.")
        sys.exit(1)

    return np.array(raw_frames, dtype=np.uint8)


def temporal_sample(frames: np.ndarray, num_frames: int) -> np.ndarray:
    """Uniformly sample `num_frames` from the frame array — identical to training."""
    n = len(frames)
    if n == 0:
        return np.zeros((num_frames, frames.shape[1], frames.shape[2], 3), dtype=np.uint8)
    if n <= num_frames:
        indices = list(range(n)) + [n - 1] * (num_frames - n)
    else:
        indices = np.linspace(0, n - 1, num_frames, dtype=int)
    return frames[indices]


def preprocess_frames(frames: np.ndarray, img_size: int) -> torch.Tensor:
    """
    Apply the same normalization as NPZDirectoryDataset:
        resize → float32 / 255 → Normalize(ImageNet mean/std)
    Returns tensor of shape (1, C, T, H, W)  — 3D-CNN expects (B, C, T, H, W).
    """
    try:
        import cv2
        resized = np.array([
            cv2.resize(f, (img_size, img_size)) for f in frames
        ])
    except ImportError:
        # Fallback: PIL resize
        from PIL import Image
        resized = np.array([
            np.array(Image.fromarray(f).resize((img_size, img_size))) for f in frames
        ])

    resized = resized.astype(np.float32) / 255.0
    mean    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized = (resized - mean) / std                           # (T, H, W, C)

    tensor = torch.from_numpy(resized).permute(3, 0, 1, 2)    # (C, T, H, W)
    return tensor.unsqueeze(0)                                 # (1, C, T, H, W)


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def predict(video_path: str,
            model_path: str,
            class_names: list,
            num_frames:  int  = 16,
            img_size:    int  = 112,
            top_k:       int  = 5):

    # Use CPU — MPS doesn't support 3D pooling operations on Mac
    device     = torch.device('cpu')
    video_path = Path(video_path)
    model_path = Path(model_path)

    # ── Validate inputs ───────────────────────
    if not video_path.exists():
        print(f"❌  Video not found: {video_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"❌  Model not found: {model_path}")
        sys.exit(1)
    if not class_names:
        print("❌  No class names provided.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print("  Working3DCNN — Single-Video Gesture Classifier")
    print(f"{'='*55}")
    print(f"  Video  : {video_path.name}")
    print(f"  Model  : {model_path.name}")
    print(f"  Device : {device}  (CPU enforced — MPS lacks 3D pool)")
    print(f"  Classes: {len(class_names)}")
    print(f"{'='*55}\n")

    # ── Load checkpoint ───────────────────────
    print("⏳ Loading model checkpoint...")
    raw = torch.load(model_path, map_location=device, weights_only=False)

    ckpt_epoch   = 'N/A'
    ckpt_val_acc = 'N/A'

    if isinstance(raw, dict):
        if 'model' in raw:
            # Training checkpoint format used by this codebase
            state_dict   = raw['model']
            ckpt_epoch   = raw.get('epoch',  'N/A')
            ckpt_val_acc = raw.get('vl_acc', 'N/A')
        elif 'model_state_dict' in raw:
            state_dict   = raw['model_state_dict']
            ckpt_epoch   = raw.get('epoch',        'N/A')
            ckpt_val_acc = raw.get('val_accuracy', 'N/A')
        elif 'state_dict' in raw:
            state_dict   = raw['state_dict']
        else:
            state_dict   = raw
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(raw)}")

    if isinstance(ckpt_epoch, int):
        print(f"   Epoch   : {ckpt_epoch}")
    if isinstance(ckpt_val_acc, float):
        print(f"   Val acc : {ckpt_val_acc * 100:.2f}%")

    # ── Auto-detect config ────────────────────
    print("⏳ Auto-detecting model config from checkpoint...")
    cfg = infer_config(state_dict)
    print(f"   num_classes : {cfg['num_classes']}")
    print(f"   dropout     : {cfg['dropout']}")
    print(f"   num_frames  : {num_frames}  (fixed by architecture)")
    print(f"   img_size    : {img_size}  (fixed by architecture)")

    # Sanity-check class list length
    if cfg['num_classes'] != len(class_names):
        print(f"\n⚠️  WARNING: checkpoint has {cfg['num_classes']} output classes "
              f"but {len(class_names)} class names were provided.")
        print(f"   Truncating / padding class list to match checkpoint.\n")
        if len(class_names) > cfg['num_classes']:
            class_names = class_names[:cfg['num_classes']]
        else:
            class_names += [f"class_{i}" for i in range(len(class_names), cfg['num_classes'])]

    model = Working3DCNN(
        num_classes=cfg['num_classes'],
        dropout=cfg['dropout'],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded  ({total_params:,} parameters)\n")

    # ── Load video frames ─────────────────────
    print("⏳ Loading video frames...")
    ext = video_path.suffix.lower()
    if ext == '.npz':
        frames = load_npz_video(video_path)
    elif ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
        frames = load_mp4_video(video_path)
    else:
        print(f"❌  Unsupported format '{ext}'. Supported: .npz  .mp4  .avi  .mov  .mkv")
        sys.exit(1)

    print(f"   Raw frames     : {len(frames)}")
    frames = temporal_sample(frames, num_frames)
    print(f"   Sampled frames : {len(frames)}\n")

    # ── Preprocess & infer ────────────────────
    print("⏳ Running inference...")
    pixel_values = preprocess_frames(frames, img_size).to(device)  # (1, C, T, H, W)

    with torch.no_grad():
        logits = model(pixel_values)               # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)[0]   # (num_classes,)

    probs_np   = probs.cpu().numpy()
    pred_idx   = int(np.argmax(probs_np))
    pred_label = class_names[pred_idx]
    confidence = probs_np[pred_idx] * 100

    # ── Print results ─────────────────────────
    print(f"\n{'='*55}")
    print("  PREDICTION RESULT")
    print(f"{'='*55}")
    print(f"  🏆 Gesture   : {pred_label}")
    print(f"  📊 Confidence: {confidence:.2f}%")
    print(f"{'='*55}")

    top_k       = min(top_k, len(class_names))
    top_indices = np.argsort(probs_np)[::-1][:top_k]

    print(f"\n  Top-{top_k} Predictions:")
    print(f"  {'Rank':<6} {'Class':<35} {'Confidence':>10}")
    print(f"  {'-'*53}")
    for rank, idx in enumerate(top_indices, 1):
        marker = " ◀" if rank == 1 else ""
        print(f"  {rank:<6} {class_names[idx]:<35} {probs_np[idx]*100:>9.2f}%{marker}")

    print(f"\n{'='*55}\n")

    return {
        'predicted_class':   pred_label,
        'predicted_index':   pred_idx,
        'confidence':        confidence,
        'all_probabilities': {class_names[i]: float(probs_np[i])
                              for i in range(len(class_names))},
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    # Resolve class names (CLASSES_FILE takes priority over CLASS_NAMES)
    class_names = []
    if CLASSES_FILE:
        with open(CLASSES_FILE) as f:
            class_names = [line.strip() for line in f if line.strip()]
    elif CLASS_NAMES:
        class_names = CLASS_NAMES
    else:
        print("❌  No class names provided. Fill in CLASS_NAMES or CLASSES_FILE above.")
        sys.exit(1)

    predict(
        video_path  = VIDEO_PATH,
        model_path  = MODEL_PATH,
        class_names = class_names,
        num_frames  = NUM_FRAMES,
        img_size    = 112,
        top_k       = TOP_K,
    )


if __name__ == "__main__":
    main()