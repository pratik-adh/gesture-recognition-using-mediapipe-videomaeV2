#!/usr/bin/env python3
"""
VideoMAE Single Video Gesture Classifier
Provide a video file (MP4/AVI/NPZ), get the predicted gesture with confidence scores.

Usage:
    python predict_gesture.py --video path/to/video.mp4 --model path/to/best_model.pth --classes class1 class2 ...
    python predict_gesture.py --video path/to/video.npz --model path/to/best_model.pth --classes_file classes.txt
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import warnings

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# Model (same architecture as training)
# ─────────────────────────────────────────────
class VideoMAEClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3,
                 model_name="MCG-NJU/videomae-small-finetuned-kinetics"):
        super().__init__()
        self.videomae = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        hidden_size = self.videomae.config.hidden_size
        self.videomae.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, num_classes)
        )

    def forward(self, pixel_values):
        return self.videomae(pixel_values).logits


# ─────────────────────────────────────────────
# Video Loaders
# ─────────────────────────────────────────────
def load_npz_video(path: Path, num_frames: int) -> np.ndarray:
    """Load frames from an NPZ file."""
    data = np.load(path)
    if 'frames' in data:
        frames = data['frames']
    elif 'video' in data:
        frames = data['video']
    else:
        frames = data[data.files[0]]

    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8) if frames.max() <= 1.0 \
                 else np.clip(frames, 0, 255).astype(np.uint8)

    return frames


def load_mp4_video(path: Path, num_frames: int) -> np.ndarray:
    """Load frames from an MP4/AVI video file using OpenCV."""
    try:
        import cv2
    except ImportError:
        print("❌  OpenCV is required for MP4/AVI files: pip install opencv-python")
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        raw_frames.append(frame_resized)
    cap.release()

    if len(raw_frames) == 0:
        print("❌  No frames could be read from the video.")
        sys.exit(1)

    return np.array(raw_frames, dtype=np.uint8)


def temporal_sample(frames: np.ndarray, num_frames: int) -> np.ndarray:
    """Uniformly sample `num_frames` from the frame array."""
    n = len(frames)
    if n == 0:
        return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
    if n <= num_frames:
        indices = list(range(n)) + [n - 1] * (num_frames - n)
    else:
        indices = np.linspace(0, n - 1, num_frames, dtype=int)
    return frames[indices]


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def predict(video_path: str,
            model_path: str,
            class_names: list[str],
            num_frames: int = 16,
            top_k: int = 5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    print("  VideoMAE — Single-Video Gesture Classifier")
    print(f"{'='*55}")
    print(f"  Video  : {video_path.name}")
    print(f"  Model  : {model_path.name}")
    print(f"  Device : {device}")
    print(f"  Classes: {len(class_names)}")
    print(f"{'='*55}\n")

    # ── Load checkpoint ───────────────────────
    print("⏳ Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            cfg = checkpoint.get('config', {})
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            cfg = checkpoint.get('config', {})
        else:
            state_dict = checkpoint
            cfg = {}
    else:
        state_dict = checkpoint
        cfg = {}

    model_name = cfg.get('model_name', 'MCG-NJU/videomae-small-finetuned-kinetics')
    dropout_rate = cfg.get('dropout_rate', 0.3)
    num_frames = cfg.get('num_frames', num_frames)

    model = VideoMAEClassifier(
        num_classes=len(class_names),
        dropout_rate=dropout_rate,
        model_name=model_name
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded  ({sum(p.numel() for p in model.parameters()):,} parameters)\n")

    # ── Load & preprocess video ───────────────
    print("⏳ Loading video frames...")
    ext = video_path.suffix.lower()
    if ext == '.npz':
        frames = load_npz_video(video_path, num_frames)
    elif ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
        frames = load_mp4_video(video_path, num_frames)
    else:
        print(f"❌  Unsupported format '{ext}'. Supported: .npz, .mp4, .avi, .mov, .mkv")
        sys.exit(1)

    print(f"   Raw frames loaded : {len(frames)}")
    frames = temporal_sample(frames, num_frames)
    print(f"   Frames after sampling: {len(frames)}\n")

    # ── Preprocess with VideoMAE processor ────
    print("⏳ Running inference...")
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    inputs = processor(list(frames), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)   # (1, T, C, H, W)

    # ── Forward pass ──────────────────────────
    with torch.no_grad():
        logits = model(pixel_values)                   # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)[0]       # (num_classes,)

    probs_np = probs.cpu().numpy()
    pred_idx  = int(np.argmax(probs_np))
    pred_label = class_names[pred_idx]
    confidence = probs_np[pred_idx] * 100

    # ── Print results ─────────────────────────
    print(f"\n{'='*55}")
    print("  PREDICTION RESULT")
    print(f"{'='*55}")
    print(f"  🏆 Gesture  : {pred_label}")
    print(f"  📊 Confidence: {confidence:.2f}%")
    print(f"{'='*55}")

    # Top-K
    top_k = min(top_k, len(class_names))
    top_indices = np.argsort(probs_np)[::-1][:top_k]

    print(f"\n  Top-{top_k} Predictions:")
    print(f"  {'Rank':<6} {'Class':<35} {'Confidence':>10}")
    print(f"  {'-'*53}")
    for rank, idx in enumerate(top_indices, 1):
        marker = " ◀" if rank == 1 else ""
        print(f"  {rank:<6} {class_names[idx]:<35} {probs_np[idx]*100:>9.2f}%{marker}")

    print(f"\n{'='*55}\n")

    return {
        'predicted_class': pred_label,
        'predicted_index': pred_idx,
        'confidence': confidence,
        'all_probabilities': {class_names[i]: float(probs_np[i]) for i in range(len(class_names))}
    }


# ─────────────────────────────────────────────
# ✏️  CONFIGURE YOUR PATHS HERE
# ─────────────────────────────────────────────

VIDEO_PATH   = './../../videos_directory/npz_preprocessed_videos_splitted_dataset/test/BA/BA_1_aug_06_crop.npz'
# VIDEO_PATH   = './../../videos_directory/npz_preprocessed_videos_splitted_dataset/test/M_SHA/M_SHA_1_aug_18_crop.npz'
# VIDEO_PATH   = './../../videos_directory/npz_preprocessed_videos_splitted_dataset/test/RA/RA_1_aug_06_crop.npz'
MODEL_PATH   = './../../kaggle/training_output_hybrid/best_model.pth'

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
    "YAN"
]

# Option B — load from a text file (one class per line).
# Set to None to use CLASS_NAMES above, or set a path to use the file instead:
CLASSES_FILE = None   # e.g. "classes.txt"

NUM_FRAMES   = 16     # overridden automatically if checkpoint stores config
TOP_K        = 5      # how many top predictions to display

# ─────────────────────────────────────────────


def main():
    # Resolve class names (file takes priority over inline list)
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
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        class_names=class_names,
        num_frames=NUM_FRAMES,
        top_k=TOP_K,
    )


if __name__ == "__main__":
    main()