#!/usr/bin/env python3
"""
ConstrainedViViT Single Video Gesture Classifier
Provide a video file (MP4/AVI/NPZ), get the predicted gesture with confidence scores.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# ✏️  CONFIGURE YOUR PATHS HERE
# ─────────────────────────────────────────────

VIDEO_PATH = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test/BA/BA_1_aug_04.npz'
# VIDEO_PATH = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test/M_SHA/M_SHA_1_aug_06.npz'
# VIDEO_PATH   = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test/RA/RA_4_aug_19.npz'
MODEL_PATH = './../../kaggle/training_output_vivit/best_model.pth'

# Option A — list class names directly (must match training order):
CLASS_NAMES  = [
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

NUM_FRAMES   = 16     # overridden automatically from checkpoint shapes
TOP_K        = 5      # how many top predictions to display

# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# Model — exact copy of training architecture
# ─────────────────────────────────────────────
class SimplePatchEmbed(nn.Module):
    def __init__(self, img_size=112, patch_size=14, embed_dim=164):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class SimpleAttention(nn.Module):
    def __init__(self, dim=164, heads=4, dropout=0.16):
        super().__init__()
        self.heads   = heads
        self.scale   = (dim // heads) ** -0.5
        self.qkv     = nn.Linear(dim, dim * 3, bias=True)
        self.proj    = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads,
                                   C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.dropout((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        return self.dropout(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))


class SimpleBlock(nn.Module):
    """MLP as nn.Sequential so state-dict keys are mlp.0.* and mlp.3.*"""
    def __init__(self, dim=164, heads=4, mlp_ratio=2.0, dropout=0.16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = SimpleAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConstrainedViViT(nn.Module):
    """Identical to training code — architecture + weight initialisation."""
    def __init__(self, num_classes=36, num_frames=16, img_size=112,
                 patch_size=14, embed_dim=164, mlp_ratio=2.0,
                 spatial_depth=2, temporal_depth=2, heads=4, dropout=0.16):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim  = embed_dim

        self.patch_embed  = SimplePatchEmbed(img_size, patch_size, embed_dim)
        n_patches         = self.patch_embed.n_patches

        self.spatial_cls  = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.temporal_cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.spatial_pos  = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames + 1, embed_dim) * 0.02)

        self.spatial_blocks  = nn.ModuleList(
            [SimpleBlock(embed_dim, heads, mlp_ratio, dropout) for _ in range(spatial_depth)])
        self.temporal_blocks = nn.ModuleList(
            [SimpleBlock(embed_dim, heads, mlp_ratio, dropout) for _ in range(temporal_depth)])

        self.norm = nn.LayerNorm(embed_dim)
        # head.0 = Linear, head.1 = GELU, head.2 = Dropout, head.3 = Linear
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        # Spatial encoding
        x = self.patch_embed(x.view(B * T, C, H, W))
        x = torch.cat([self.spatial_cls.expand(B * T, -1, -1), x], dim=1) + self.spatial_pos
        for blk in self.spatial_blocks:
            x = blk(x)
        x = x[:, 0]
        # Temporal encoding
        x = x.view(B, T, self.embed_dim)
        x = torch.cat([self.temporal_cls.expand(B, -1, -1), x], dim=1) + self.temporal_pos
        for blk in self.temporal_blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))


# ─────────────────────────────────────────────
# Auto-detect config from checkpoint shapes
# (no manual editing needed — reads everything
#  from the saved tensor dimensions)
# ─────────────────────────────────────────────
def infer_config(state_dict):
    E          = int(state_dict['spatial_cls'].shape[2])
    num_frames = int(state_dict['temporal_pos'].shape[1]) - 1
    patch_size = int(state_dict['patch_embed.proj.weight'].shape[2])
    n_patches  = int(state_dict['spatial_pos'].shape[1]) - 1
    img_size   = int(round((n_patches ** 0.5) * patch_size))
    mlp_ratio  = int(state_dict['spatial_blocks.0.mlp.0.weight'].shape[0]) / E
    s_depth    = sum(1 for k in state_dict if k.startswith('spatial_blocks.')
                     and k.endswith('.norm1.weight'))
    t_depth    = sum(1 for k in state_dict if k.startswith('temporal_blocks.')
                     and k.endswith('.norm1.weight'))
    return dict(
        embed_dim=E, num_frames=num_frames, img_size=img_size,
        patch_size=patch_size, mlp_ratio=mlp_ratio,
        spatial_depth=s_depth, temporal_depth=t_depth,
        num_heads=4, dropout=0.16,
    )


# ─────────────────────────────────────────────
# Video Loaders
# ─────────────────────────────────────────────
def load_npz_video(path: Path) -> np.ndarray:
    """Load frames from an NPZ file."""
    data = np.load(path)
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
    Apply the same transform as NPZDataset (is_training=False):
        Resize → ToTensor → Normalize(ImageNet)
    Returns tensor of shape (1, T, C, H, W).
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tensors = [transform(Image.fromarray(f)) for f in frames]   # each: (C, H, W)
    video   = torch.stack(tensors, dim=0)                        # (T, C, H, W)
    return video.unsqueeze(0)                                    # (1, T, C, H, W)


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def predict(video_path: str,
            model_path: str,
            class_names: list,
            num_frames: int = 16,
            top_k: int = 5):

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    print("  ConstrainedViViT — Single-Video Gesture Classifier")
    print(f"{'='*55}")
    print(f"  Video  : {video_path.name}")
    print(f"  Model  : {model_path.name}")
    print(f"  Device : {device}")
    print(f"  Classes: {len(class_names)}")
    print(f"{'='*55}\n")

    # ── Load checkpoint ───────────────────────
    print("⏳ Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict   = checkpoint['model_state_dict']
            saved_classes = checkpoint.get('class_names')
            print(f"   Epoch   : {checkpoint.get('epoch', 'N/A')}")
            val = checkpoint.get('accuracy')
            if isinstance(val, float):
                print(f"   Val acc : {val * 100:.2f}%")
        elif 'model' in checkpoint:
            state_dict   = checkpoint['model']
            saved_classes = checkpoint.get('class_names')
        elif 'state_dict' in checkpoint:
            state_dict   = checkpoint['state_dict']
            saved_classes = None
        else:
            state_dict   = checkpoint
            saved_classes = None
    else:
        state_dict   = checkpoint
        saved_classes = None

    # Use class names embedded in checkpoint if available
    if saved_classes is not None:
        print(f"   Using {len(saved_classes)} class names from checkpoint.")
        class_names = saved_classes

    # ── Auto-detect architecture ──────────────
    print("⏳ Auto-detecting model config from checkpoint...")
    cfg = infer_config(state_dict)
    num_frames = cfg['num_frames']   # always trust the checkpoint
    img_size   = cfg['img_size']

    print(f"   embed_dim      : {cfg['embed_dim']}")
    print(f"   num_frames     : {cfg['num_frames']}")
    print(f"   img_size       : {cfg['img_size']}")
    print(f"   patch_size     : {cfg['patch_size']}")
    print(f"   spatial_depth  : {cfg['spatial_depth']}")
    print(f"   temporal_depth : {cfg['temporal_depth']}")

    model = ConstrainedViViT(
        num_classes    = len(class_names),
        num_frames     = cfg['num_frames'],
        img_size       = cfg['img_size'],
        patch_size     = cfg['patch_size'],
        embed_dim      = cfg['embed_dim'],
        mlp_ratio      = cfg['mlp_ratio'],
        spatial_depth  = cfg['spatial_depth'],
        temporal_depth = cfg['temporal_depth'],
        heads          = cfg['num_heads'],
        dropout        = cfg['dropout'],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded  ({sum(p.numel() for p in model.parameters()):,} parameters)\n")

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
    pixel_values = preprocess_frames(frames, img_size).to(device)  # (1, T, C, H, W)

    with torch.no_grad():
        logits = model(pixel_values)                # (1, num_classes)
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
        'predicted_class': pred_label,
        'predicted_index': pred_idx,
        'confidence':      confidence,
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
        top_k       = TOP_K,
    )


if __name__ == "__main__":
    main()