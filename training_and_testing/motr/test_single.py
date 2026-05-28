#!/usr/bin/env python3
"""
Motion Transformer (MoTr) Single Video Gesture Classifier
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
VIDEO_PATH = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test/BA/BA_1_aug_04.npz'
# VIDEO_PATH = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test/M_SHA/M_SHA_1_aug_06.npz'
# VIDEO_PATH   = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test/RA/RA_1_aug_06.npz'
MODEL_PATH = './../../kaggle/training_output_motr/best_model.pth'

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

NUM_FRAMES = 12    # overridden automatically from checkpoint config if available
TOP_K      = 5     # how many top predictions to display

# ─────────────────────────────────────────────
# Model — exact copy of training architecture
# ─────────────────────────────────────────────
class MotionExtractor(nn.Module):
    """Extract motion features using temporal differences."""
    def __init__(self, in_chans=3):
        super().__init__()
        self.motion_conv = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        motion = [x[:, :, t+1] - x[:, :, t] for t in range(T - 1)]
        motion = torch.stack(motion, dim=2)
        motion_features = [self.motion_conv(motion[:, :, t]) for t in range(T - 1)]
        return torch.stack(motion_features, dim=2)


class SpatialPatchEmbed(nn.Module):
    """Spatial patch embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.grid_size  = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TemporalAttention(nn.Module):
    """Multi-head attention (used for both temporal and spatial streams)."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.attn_drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    """Feed-forward MLP block."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MotionTransformerBlock(nn.Module):
    """Motion Transformer block with temporal + spatial attention."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.spatial_norm  = nn.LayerNorm(dim)
        self.spatial_attn  = TemporalAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.mlp_norm      = nn.LayerNorm(dim)
        self.mlp           = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.temporal_attn(self.temporal_norm(x))
        x = x + self.spatial_attn(self.spatial_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MotionTransformer(nn.Module):
    """Motion Transformer for video classification."""
    def __init__(self, img_size=224, patch_size=16, num_frames=16, in_chans=3,
                 num_classes=1000, embed_dim=768, num_heads=12, num_layers=12,
                 mlp_ratio=4., qkv_bias=True, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.num_frames  = num_frames

        self.motion_extractor = MotionExtractor(in_chans=in_chans)

        reduced_size     = img_size // 16
        self.num_patches = reduced_size ** 2

        self.spatial_embed = SpatialPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.motion_proj   = nn.Linear(256 * (reduced_size ** 2), embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_frames * self.spatial_embed.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            MotionTransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout, attention_dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 4, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, T, H, W = x.shape

        motion_feat = self.motion_extractor(x)

        # Spatial tokens from each frame
        appearance_tokens = torch.cat(
            [self.spatial_embed(x[:, :, t]) for t in range(T)], dim=1
        )

        # Motion tokens projected to embed_dim, padded to T frames
        B_m, C_m, T_m, H_m, W_m = motion_feat.shape
        motion_tokens = self.motion_proj(motion_feat.view(B, T_m, -1))
        motion_pad    = torch.zeros(B, 1, self.embed_dim, device=x.device)
        motion_tokens = torch.cat([motion_tokens, motion_pad], dim=1)

        # Expand motion tokens to match spatial token layout
        motion_expanded = (
            motion_tokens.unsqueeze(2)
            .repeat(1, 1, self.spatial_embed.num_patches, 1)
            .view(B, T * self.spatial_embed.num_patches, self.embed_dim)
        )

        tokens = appearance_tokens + motion_expanded
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)

        for block in self.blocks:
            tokens = block(tokens)

        return self.head(self.norm(tokens[:, 0]))


# ─────────────────────────────────────────────
# Auto-detect config from checkpoint
# ─────────────────────────────────────────────
def infer_config(checkpoint: dict) -> dict:
    """
    Pull architecture hyper-parameters from the checkpoint.
    Prefers an explicit 'config' dict saved during training;
    falls back to reading tensor shapes from the state dict.
    """
    # Use saved config if available
    saved_cfg = checkpoint.get('config', {})

    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Derive from state-dict shapes when not in saved config
    embed_dim   = saved_cfg.get('embed_dim',
                  int(state_dict['cls_token'].shape[2]))
    num_frames  = saved_cfg.get('num_frames',
                  int(state_dict['pos_embed'].shape[1] - 1)
                  // int(state_dict['spatial_embed.proj.weight'].shape[0] // embed_dim
                         if False else 1))   # fallback handled below
    patch_size  = saved_cfg.get('patch_size',
                  int(state_dict['spatial_embed.proj.weight'].shape[2]))
    num_patches = int(state_dict['spatial_embed.proj.weight'].shape[0])   # embed_dim — not patches
    # Correct num_patches: spatial_embed.proj.weight shape is (embed_dim, in_chans, P, P)
    # pos_embed shape: (1, num_frames * num_spatial_patches + 1, embed_dim)
    pos_len     = int(state_dict['pos_embed'].shape[1]) - 1   # strip cls

    num_layers  = saved_cfg.get('num_layers',
                  sum(1 for k in state_dict if k.startswith('blocks.')
                      and k.endswith('.temporal_norm.weight')))
    num_heads   = saved_cfg.get('num_heads', 6)
    mlp_ratio   = saved_cfg.get('mlp_ratio', 2.0)
    num_classes = int(state_dict['head.7.weight'].shape[0])   # last Linear in head
    img_size    = saved_cfg.get('img_size', 112)
    dropout     = saved_cfg.get('dropout_rate', 0.3)
    attn_drop   = saved_cfg.get('attention_dropout', 0.2)
    qkv_bias    = saved_cfg.get('qkv_bias', True)

    # num_frames: re-derive from pos_embed and img_size / patch_size
    n_spatial   = (img_size // patch_size) ** 2
    if n_spatial > 0 and pos_len % n_spatial == 0:
        num_frames = pos_len // n_spatial
    else:
        num_frames = saved_cfg.get('num_frames', 12)

    return dict(
        img_size          = img_size,
        patch_size        = patch_size,
        num_frames        = num_frames,
        embed_dim         = embed_dim,
        num_layers        = num_layers,
        num_heads         = num_heads,
        mlp_ratio         = mlp_ratio,
        num_classes       = num_classes,
        dropout           = dropout,
        attention_dropout = attn_drop,
        qkv_bias          = qkv_bias,
    )


# ─────────────────────────────────────────────
# Video Loaders
# ─────────────────────────────────────────────
def load_npz_video(path: Path) -> np.ndarray:
    """Load frames from an NPZ file."""
    data   = np.load(path)
    frames = data['frames'] if 'frames' in data else data[data.files[0]]
    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8) if frames.max() <= 1.0 \
                 else np.clip(frames, 0, 255).astype(np.uint8)
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
    """Uniformly sample `num_frames` — identical to training."""
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
    Apply the same preprocessing as NPZDirectoryDataset:
        resize → float32/255 → Normalize(ImageNet mean/std)
    Returns tensor of shape (1, C, T, H, W)  — MoTr expects (B, C, T, H, W).
    """
    try:
        import cv2
        resized = np.array([cv2.resize(f, (img_size, img_size)) for f in frames])
    except ImportError:
        from PIL import Image
        resized = np.array([
            np.array(Image.fromarray(f).resize((img_size, img_size))) for f in frames
        ])

    resized = resized.astype(np.float32) / 255.0
    mean    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    resized = (resized - mean) / std                            # (T, H, W, C)

    tensor = torch.from_numpy(resized).permute(3, 0, 1, 2)     # (C, T, H, W)
    return tensor.unsqueeze(0)                                  # (1, C, T, H, W)


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def predict(video_path: str,
            model_path: str,
            class_names: list,
            num_frames:  int = 12,
            top_k:       int = 5):

    # MPS available on Apple Silicon but use CPU as safe default;
    # MoTr's ops are all MPS-compatible, so we can try MPS here.
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

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
    print("  Motion Transformer — Single-Video Gesture Classifier")
    print(f"{'='*55}")
    print(f"  Video  : {video_path.name}")
    print(f"  Model  : {model_path.name}")
    print(f"  Device : {device}")
    print(f"  Classes: {len(class_names)}")
    print(f"{'='*55}\n")

    # ── Load checkpoint ───────────────────────
    print("⏳ Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    ckpt_epoch   = 'N/A'
    ckpt_val_acc = 'N/A'

    if isinstance(checkpoint, dict):
        ckpt_epoch   = checkpoint.get('epoch',        'N/A')
        ckpt_val_acc = checkpoint.get('val_accuracy', 'N/A')
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    if isinstance(ckpt_epoch, int):
        print(f"   Epoch   : {ckpt_epoch}")
    if isinstance(ckpt_val_acc, float):
        print(f"   Val acc : {ckpt_val_acc * 100:.2f}%")

    # ── Auto-detect config ────────────────────
    print("⏳ Auto-detecting model config from checkpoint...")
    cfg = infer_config(checkpoint)

    print(f"   embed_dim         : {cfg['embed_dim']}")
    print(f"   num_frames        : {cfg['num_frames']}")
    print(f"   img_size          : {cfg['img_size']}")
    print(f"   patch_size        : {cfg['patch_size']}")
    print(f"   num_layers        : {cfg['num_layers']}")
    print(f"   num_heads         : {cfg['num_heads']}")
    print(f"   num_classes       : {cfg['num_classes']}")

    num_frames = cfg['num_frames']   # always trust the checkpoint
    img_size   = cfg['img_size']

    # Sanity-check class list length
    if cfg['num_classes'] != len(class_names):
        print(f"\n⚠️  WARNING: checkpoint has {cfg['num_classes']} output classes "
              f"but {len(class_names)} class names were provided.")
        print(f"   Truncating / padding class list to match checkpoint.\n")
        if len(class_names) > cfg['num_classes']:
            class_names = class_names[:cfg['num_classes']]
        else:
            class_names += [f"class_{i}" for i in range(len(class_names), cfg['num_classes'])]

    # ── Build & load model ────────────────────
    model = MotionTransformer(
        img_size          = cfg['img_size'],
        patch_size        = cfg['patch_size'],
        num_frames        = cfg['num_frames'],
        in_chans          = 3,
        num_classes       = cfg['num_classes'],
        embed_dim         = cfg['embed_dim'],
        num_heads         = cfg['num_heads'],
        num_layers        = cfg['num_layers'],
        mlp_ratio         = cfg['mlp_ratio'],
        qkv_bias          = cfg['qkv_bias'],
        dropout           = cfg['dropout'],
        attention_dropout = cfg['attention_dropout'],
    )

    state_dict = checkpoint.get('model_state_dict', checkpoint)
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
        top_k       = TOP_K,
    )


if __name__ == "__main__":
    main()