#!/usr/bin/env python3
"""
test_vivit_mac.py
=================
Comprehensive test script for ConstrainedViViT trained in vivit_plateau_85.py.

Architecture is a byte-for-byte copy of the training file so checkpoint
weights always load cleanly. Config (embed_dim, num_frames, mlp_ratio, etc.)
is auto-detected from checkpoint tensor shapes — no manual editing needed.

Checkpoint format (from training code):
    {
        'model_state_dict': OrderedDict,
        'accuracy':         float,
        'epoch':            int,
        'class_names':      list[str],
    }

Outputs
───────
<OUT_DIR>/
  plots/
    confusion_matrix_counts.png
    confusion_matrix_normalised.png
    per_class_metrics.png
    class_accuracy_sorted.png
    top_k_accuracy.png
    pr_curves_combined.png
    roc_curves_combined.png
    pr_curves_individual/pr_<CLASS>.png    (P/R curve + P/R/F1 vs threshold)
    roc_curves_individual/roc_<CLASS>.png  (ROC curve + TPR/FPR/Youden's J)
    calibration_and_confidence.png
    summary_report.png
  metrics/
    metrics.json
    classification_report.txt
    confusion_matrix.csv
    average_precision_scores.csv
    auc_scores.csv
    detailed_predictions.csv
    misclassified_samples.csv
"""

import json, warnings, logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score,
    roc_curve, auc,
    top_k_accuracy_score,
    balanced_accuracy_score,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_vivit_mac.log', mode='a'),
    ]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# PATHS — update before running
# ═══════════════════════════════════════════════════════════════
MODEL_PATH = './../../kaggle/training_output_vivit/new_output_final/best_model.pth'
TEST_DIR   = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test'
OUT_DIR    = './../../output/vivit/new'
# ═══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
# Dataset  (matches training code — no augmentation)
# ─────────────────────────────────────────────────────────────
class NPZDataset(Dataset):
    def __init__(self, root_dir, class_names=None, num_frames=16, img_size=112):
        self.root_dir   = Path(root_dir)
        self.num_frames = num_frames
        self.img_size   = img_size
        self.classes    = class_names if class_names is not None else \
                          sorted(d.name for d in self.root_dir.iterdir() if d.is_dir())
        self.label2id   = {c: i for i, c in enumerate(self.classes)}
        self.samples    = []
        self._scan()
        # Identical to training is_training=False branch
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _scan(self):
        for c in self.classes:
            d = self.root_dir / c
            if not d.exists():
                continue
            files = sorted(d.glob('*.npz'))
            for f in files:
                self.samples.append((f, self.label2id[c], c))
            log.info(f"  {c}: {len(files)} videos")
        log.info(f"Dataset : {self.root_dir}")
        log.info(f"Classes : {len(self.classes)},  Samples : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_video(self, path):
        try:
            data   = np.load(path)
            frames = data['frames'] if 'frames' in data else data[data.files[0]]
            if frames.dtype != np.uint8:
                frames = (frames * 255).astype(np.uint8) if frames.max() <= 1.0 \
                         else np.clip(frames, 0, 255).astype(np.uint8)
            return frames
        except Exception as e:
            log.error(f"Load error {path}: {e}")
            return np.zeros((self.num_frames, self.img_size, self.img_size, 3), np.uint8)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        frames = self._load_video(path)
        n      = len(frames)
        # Frame sampling — identical to training code
        if n == 0:
            frames = np.zeros((self.num_frames, self.img_size, self.img_size, 3), np.uint8)
        elif n <= self.num_frames:
            indices = list(range(n)) + [n - 1] * (self.num_frames - n)
            frames  = frames[indices]
        else:
            indices = np.linspace(0, n - 1, self.num_frames, dtype=int)
            frames  = frames[indices]
        video = torch.stack([self.transform(Image.fromarray(f)) for f in frames], dim=0)
        return video, label, path.name


# ─────────────────────────────────────────────────────────────
# Model — exact copy of training code
# ─────────────────────────────────────────────────────────────
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

        self.patch_embed = SimplePatchEmbed(img_size, patch_size, embed_dim)
        n_patches        = self.patch_embed.n_patches

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
        self.apply(self._init_weights)
        log.info(f"ConstrainedViViT — {sum(p.numel() for p in self.parameters()):,} params")
        log.info(f"  embed={embed_dim}, spatial={spatial_depth}, temporal={temporal_depth}, "
                 f"heads={heads}, mlp_ratio={mlp_ratio}, dropout={dropout}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # Spatial
        x = self.patch_embed(x.view(B * T, C, H, W))
        x = torch.cat([self.spatial_cls.expand(B * T, -1, -1), x], dim=1) + self.spatial_pos
        for blk in self.spatial_blocks: x = blk(x)
        x = x[:, 0]
        # Temporal
        x = x.view(B, T, self.embed_dim)
        x = torch.cat([self.temporal_cls.expand(B, -1, -1), x], dim=1) + self.temporal_pos
        for blk in self.temporal_blocks: x = blk(x)
        return self.head(self.norm(x[:, 0]))


# ─────────────────────────────────────────────────────────────
# Auto-detect config from checkpoint shapes
# ─────────────────────────────────────────────────────────────
def infer_config(sd):
    """
    Reads every architecture dimension from checkpoint tensor shapes.
    Works for ANY ConstrainedViViT checkpoint — no manual config editing.

    Derivation
    ───────────
    spatial_cls           [1, 1, E]   → embed_dim  = E
    temporal_pos          [1, T+1, E] → num_frames = T
    patch_embed.proj.w    [E, 3, p, p]→ patch_size = p
    spatial_pos           [1, P+1, E] → img_size   = sqrt(P) * p
    spatial_blocks.0.mlp.0.w [H, E]  → mlp_ratio  = H / E
    count of *.norm1.weight keys      → depth
    """
    E          = int(sd['spatial_cls'].shape[2])
    num_frames = int(sd['temporal_pos'].shape[1]) - 1
    patch_size = int(sd['patch_embed.proj.weight'].shape[2])
    n_patches  = int(sd['spatial_pos'].shape[1]) - 1
    img_size   = int(round((n_patches ** 0.5) * patch_size))
    mlp_ratio  = int(sd['spatial_blocks.0.mlp.0.weight'].shape[0]) / E

    s_depth = sum(1 for k in sd if k.startswith('spatial_blocks.')  and k.endswith('.norm1.weight'))
    t_depth = sum(1 for k in sd if k.startswith('temporal_blocks.') and k.endswith('.norm1.weight'))

    cfg = dict(embed_dim=E, num_frames=num_frames, img_size=img_size,
               patch_size=patch_size, mlp_ratio=mlp_ratio,
               spatial_depth=s_depth, temporal_depth=t_depth,
               num_heads=4, dropout=0.16, batch_size=24)

    log.info("Config auto-detected from checkpoint:")
    for k, v in cfg.items():
        log.info(f"  {k:<16}: {v}")
    return cfg


# ─────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────
def load_checkpoint(path, device):
    raw = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict, got {type(raw)}")
    # Training code saves under 'model_state_dict'
    if 'model_state_dict' in raw:
        sd, class_names = raw['model_state_dict'], raw.get('class_names')
        meta = {'epoch': raw.get('epoch', 'N/A'), 'val_acc': raw.get('accuracy')}
    elif 'model' in raw:
        sd, class_names = raw['model'], raw.get('class_names')
        meta = {'epoch': raw.get('epoch', 'N/A'), 'val_acc': raw.get('vl_acc')}
    elif 'state_dict' in raw:
        sd, class_names = raw['state_dict'], None
        meta = {'epoch': raw.get('epoch', 'N/A'), 'val_acc': None}
    else:
        sd, class_names, meta = raw, None, {'epoch': 'N/A', 'val_acc': None}

    log.info(f"Checkpoint loaded : {path}")
    log.info(f"  epoch           : {meta['epoch']}")
    if isinstance(meta.get('val_acc'), float):
        log.info(f"  val accuracy    : {meta['val_acc']*100:.2f}%")
    if class_names:
        log.info(f"  class_names     : {len(class_names)} classes")
    return sd, class_names, meta


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    preds, labels, probs, files = [], [], [], []
    for videos, lbls, fnames in tqdm(loader, desc='  Inference', ncols=90):
        videos = videos.to(device)
        logits = model(videos)
        p      = torch.softmax(logits, dim=1)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(lbls.numpy())
        probs.extend(p.cpu().numpy())
        files.extend(fnames)
    return np.array(preds), np.array(labels), np.array(probs), files


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_probs, cn):
    report = classification_report(y_true, y_pred, target_names=cn, output_dict=True)
    f1_per = [report[c]['f1-score'] for c in cn]
    top_k  = {k: float(top_k_accuracy_score(y_true, y_probs, k=k))
               for k in [1, 3, 5] if k <= len(cn)}
    per_ap  = {c: float(average_precision_score((y_true==i).astype(int), y_probs[:,i]))
               for i, c in enumerate(cn)}
    per_auc = {}
    for i, c in enumerate(cn):
        fpr, tpr, _ = roc_curve((y_true==i).astype(int), y_probs[:,i])
        per_auc[c]  = float(auc(fpr, tpr))
    return {
        'accuracy':          float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_precision':   float(precision_score(y_true, y_pred, average='macro')),
        'macro_recall':      float(recall_score(y_true, y_pred, average='macro')),
        'macro_f1':          float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1':       float(f1_score(y_true, y_pred, average='weighted')),
        'mean_ap':           float(np.mean(list(per_ap.values()))),
        'mean_auc':          float(np.mean(list(per_auc.values()))),
        'top_k': top_k, 'per_class_ap': per_ap, 'per_class_auc': per_auc,
        'class_report': report,
        'best_class':    cn[int(np.argmax(f1_per))],
        'worst_class':   cn[int(np.argmin(f1_per))],
        'best_class_f1':  float(max(f1_per)),
        'worst_class_f1': float(min(f1_per)),
        'num_classes':   len(cn), 'total_samples': int(len(y_true)),
    }


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────
CB = '#2E86AB'; CR = '#E84855'; CG = '#44BBA4'; CP = '#7B2D8B'

def _save(fig, path, dpi=400):
    fig.savefig(path, dpi=dpi, bbox_inches='tight'); plt.close(fig)
    log.info(f"  ✓  {Path(path).name}")


def plot_cm_counts(y_true, y_pred, cn, out):
    fs = max(16, len(cn) * 0.65)
    fig, ax = plt.subplots(figsize=(fs, fs * 0.9))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=cn, yticklabels=cn, linewidths=0.4, linecolor='#ccc',
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_title('Confusion Matrix — Counts', fontsize=17, fontweight='bold', pad=18)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('True',      fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout(); _save(fig, out / 'confusion_matrix_counts.png')


def plot_cm_normalised(y_true, y_pred, cn, out):
    cm  = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fs  = max(16, len(cn) * 0.65)
    fig, ax = plt.subplots(figsize=(fs, fs * 0.9))
    sns.heatmap(cmn, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=cn, yticklabels=cn, vmin=0, vmax=1,
                linewidths=0.4, linecolor='#ccc',
                cbar_kws={'label': 'Recall per class'}, ax=ax)
    ax.set_title('Confusion Matrix — Normalised', fontsize=17, fontweight='bold', pad=18)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('True',      fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout(); _save(fig, out / 'confusion_matrix_normalised.png')


def plot_per_class_metrics(cn, report, out):
    x = np.arange(len(cn))
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    for ax, key, label, color in [
        (axes[0,0], 'precision', 'Precision', CB),
        (axes[0,1], 'recall',    'Recall',    CG),
        (axes[1,0], 'f1-score',  'F1-Score',  CR),
        (axes[1,1], 'support',   'Support',   CP),
    ]:
        vals = [report[c][key] for c in cn]
        ax.bar(x, vals, 0.65, color=color, alpha=0.82, edgecolor='white')
        ax.set_title(f'{label} per Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(cn, rotation=45, ha='right', fontsize=9)
        ax.grid(True, axis='y', alpha=0.25, ls='--')
        if key != 'support': ax.set_ylim(0, 1.08)
        m = np.mean(vals)
        ax.axhline(m, color='black', ls='--', lw=1.5, label=f'Mean: {m:.3f}')
        ax.legend(fontsize=10)
    fig.suptitle('Per-Class Metrics — ConstrainedViViT', fontsize=17, fontweight='bold', y=1.01)
    fig.tight_layout(); _save(fig, out / 'per_class_metrics.png')


def plot_class_accuracy_sorted(y_true, y_pred, cn, out):
    accs = [float((y_pred[y_true==i]==i).sum()/(y_true==i).sum()) if (y_true==i).sum() else 0.0
            for i in range(len(cn))]
    idx = np.argsort(accs)[::-1]
    ns, vs = [cn[i] for i in idx], [accs[i] for i in idx]
    fig, ax = plt.subplots(figsize=(14, max(8, len(cn) * 0.38)))
    ax.barh(range(len(ns)), vs, color=plt.cm.RdYlGn(np.array(vs)), alpha=0.86, edgecolor='white')
    ax.set_yticks(range(len(ns))); ax.set_yticklabels(ns, fontsize=10)
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy (sorted)', fontsize=15, fontweight='bold')
    ax.set_xlim(0, 1.12); ax.grid(True, axis='x', alpha=0.25, ls='--')
    ax.axvline(np.mean(vs), color='black', ls='--', lw=1.5, label=f'Mean: {np.mean(vs):.3f}')
    ax.legend()
    for i, v in enumerate(vs): ax.text(v+0.01, i, f'{v:.3f}', va='center', fontsize=8.5)
    fig.tight_layout(); _save(fig, out / 'class_accuracy_sorted.png')


def plot_top_k(top_k_dict, out):
    ks, vs = sorted(top_k_dict), [top_k_dict[k]*100 for k in sorted(top_k_dict)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, vs, 'o-', lw=2.5, ms=9, color=CB)
    ax.fill_between(ks, vs, alpha=0.12, color=CB)
    for k, v in zip(ks, vs): ax.text(k, v+1.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('K', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Top-K Accuracy', fontsize=15, fontweight='bold')
    ax.set_xticks(ks); ax.set_ylim(0, 110); ax.grid(True, alpha=0.25, ls='--')
    fig.tight_layout(); _save(fig, out / 'top_k_accuracy.png')


def plot_pr_combined(y_true, y_probs, cn, out):
    colors = plt.cm.tab20(np.linspace(0, 1, len(cn)))
    fig, ax = plt.subplots(figsize=(15, 10))
    ap_vals = []
    for i, (c, col) in enumerate(zip(cn, colors)):
        yb = (y_true==i).astype(int)
        p, r, _ = precision_recall_curve(yb, y_probs[:,i])
        ap = average_precision_score(yb, y_probs[:,i]); ap_vals.append(ap)
        ax.plot(r, p, color=col, lw=1.5, alpha=0.85, label=f'{c}  AP={ap:.3f}')
    ax.axhline(1/len(cn), color='gray', ls=':', lw=1.2, label=f'Random={1/len(cn):.3f}')
    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title(f'PR Curves — All Classes  (mAP={np.mean(ap_vals):.4f})',
                 fontsize=15, fontweight='bold')
    ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.grid(True, alpha=0.2, ls='--')
    ax.legend(loc='lower center', ncol=3, fontsize=7.5, framealpha=0.9)
    fig.tight_layout(); _save(fig, out / 'pr_curves_combined.png')
    log.info(f"    mAP = {np.mean(ap_vals):.4f}")


def plot_pr_individual(y_true, y_probs, cn, ind_dir):
    Path(ind_dir).mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(tqdm(cn, desc='  PR individual', ncols=80)):
        yb = (y_true==i).astype(int)
        p, r, thresh = precision_recall_curve(yb, y_probs[:,i])
        ap = average_precision_score(yb, y_probs[:,i])
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax.plot(r, p, color=CB, lw=2.8, label=f'AP={ap:.4f}')
        ax.fill_between(r, p, alpha=0.18, color=CB)
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(f'PR Curve: {c}', fontsize=14, fontweight='bold')
        ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.grid(True, alpha=0.25, ls='--'); ax.legend()
        ax.text(0.04, 0.08, f'Samples: {int(yb.sum())}\nAP: {ap:.4f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.55))
        if len(thresh) > 0:
            f1_t = np.where((p[:-1]+r[:-1])>0, 2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-9), 0)
            ax2.plot(thresh, p[:-1], color=CB, lw=2, label='Precision')
            ax2.plot(thresh, r[:-1], color=CG,  lw=2, label='Recall')
            ax2.plot(thresh, f1_t,   color=CR,  lw=2, label='F1')
            ax2.axvline(thresh[np.argmax(f1_t)], color='black', ls='--', lw=1.2,
                        label=f'Best F1@{thresh[np.argmax(f1_t)]:.2f}')
        ax2.set_xlabel('Threshold'); ax2.set_ylabel('Score')
        ax2.set_title(f'P/R/F1 vs Threshold: {c}', fontsize=14, fontweight='bold')
        ax2.set_xlim(0,1); ax2.set_ylim(0,1.05); ax2.grid(True, alpha=0.25, ls='--'); ax2.legend()
        fig.tight_layout()
        _save(fig, Path(ind_dir)/f'pr_{c.replace("/","_")}.png', dpi=300)
    log.info(f"  ✓  {len(cn)} individual PR curves")


def plot_roc_combined(y_true, y_probs, cn, out):
    colors = plt.cm.tab20(np.linspace(0, 1, len(cn)))
    fig, ax = plt.subplots(figsize=(15, 10)); auc_vals = []
    for i, (c, col) in enumerate(zip(cn, colors)):
        yb = (y_true==i).astype(int)
        fpr, tpr, _ = roc_curve(yb, y_probs[:,i])
        a = auc(fpr, tpr); auc_vals.append(a)
        ax.plot(fpr, tpr, color=col, lw=1.5, alpha=0.85, label=f'{c}  AUC={a:.3f}')
    ax.plot([0,1],[0,1],'k--', lw=1.5, label='Random')
    ax.set_xlabel('FPR', fontsize=13, fontweight='bold')
    ax.set_ylabel('TPR', fontsize=13, fontweight='bold')
    ax.set_title(f'ROC Curves — All Classes  (Mean AUC={np.mean(auc_vals):.4f})',
                 fontsize=15, fontweight='bold')
    ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.grid(True, alpha=0.2, ls='--')
    ax.legend(loc='lower right', ncol=3, fontsize=7.5, framealpha=0.9)
    fig.tight_layout(); _save(fig, out / 'roc_curves_combined.png')
    log.info(f"    Mean AUC = {np.mean(auc_vals):.4f}")


def plot_roc_individual(y_true, y_probs, cn, ind_dir):
    Path(ind_dir).mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(tqdm(cn, desc='  ROC individual', ncols=80)):
        yb = (y_true==i).astype(int)
        fpr, tpr, thresh = roc_curve(yb, y_probs[:,i])
        a = auc(fpr, tpr)
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax.plot(fpr, tpr, color=CB, lw=2.8, label=f'AUC={a:.4f}')
        ax.fill_between(fpr, tpr, alpha=0.18, color=CB)
        ax.plot([0,1],[0,1],'k--', lw=1.3, label='Random')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title(f'ROC Curve: {c}', fontsize=14, fontweight='bold')
        ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.grid(True, alpha=0.25, ls='--'); ax.legend()
        ax.text(0.55, 0.12, f'Samples: {int(yb.sum())}\nAUC: {a:.4f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.55))
        if len(thresh) > 1:
            youden = tpr[:-1] - fpr[:-1]
            ax2.plot(thresh[1:], tpr[1:], color=CG, lw=2, label='TPR (Sensitivity)')
            ax2.plot(thresh[1:], fpr[1:], color=CR,  lw=2, label='FPR (1-Specificity)')
            ax2.plot(thresh[1:], youden,  color=CP,  lw=2, label="Youden's J")
            ax2.axvline(thresh[np.argmax(youden)], color='black', ls='--', lw=1.2,
                        label=f'Best J@{thresh[np.argmax(youden)]:.2f}')
        ax2.set_xlabel('Threshold'); ax2.set_ylabel('Rate')
        ax2.set_title(f'TPR/FPR/Youden: {c}', fontsize=14, fontweight='bold')
        ax2.set_ylim(-0.05, 1.05); ax2.grid(True, alpha=0.25, ls='--'); ax2.legend()
        fig.tight_layout()
        _save(fig, Path(ind_dir)/f'roc_{c.replace("/","_")}.png', dpi=300)
    log.info(f"  ✓  {len(cn)} individual ROC curves")


def plot_calibration(y_true, y_probs, out):
    max_p = y_probs.max(axis=1); is_ok = (y_probs.argmax(axis=1) == y_true)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fp, mp = calibration_curve(is_ok, max_p, n_bins=10)
    ax.plot([0,1],[0,1],'k--', lw=1.5, label='Perfect')
    ax.plot(mp, fp, 'o-', lw=2, ms=7, color=CB, label='Model')
    ax.fill_between(mp, fp, mp, alpha=0.12, color=CR, label='Gap')
    ax.set_xlabel('Mean Predicted Confidence'); ax.set_ylabel('Fraction Correct')
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.legend(); ax.grid(True, alpha=0.25, ls='--')
    bins = np.linspace(0, 1, 26)
    ax2.hist(max_p[is_ok],  bins=bins, color=CG, alpha=0.75, label='Correct',   edgecolor='white')
    ax2.hist(max_p[~is_ok], bins=bins, color=CR, alpha=0.75, label='Incorrect', edgecolor='white')
    ax2.set_xlabel('Max Softmax Confidence'); ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.25, ls='--')
    fig.tight_layout(); _save(fig, out / 'calibration_and_confidence.png')


def plot_summary(metrics, meta, cfg, out):
    fig = plt.figure(figsize=(16, 11)); ax = fig.add_subplot(111); ax.axis('off')
    ep    = meta.get('epoch', 'N/A')
    val   = meta.get('val_acc')
    val_s = f"{val*100:.2f}%" if isinstance(val, float) else str(val)
    text = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║       ConstrainedViViT (85-86% plateau target)  ·  TEST SUMMARY         ║
╚══════════════════════════════════════════════════════════════════════════╝

📊  OVERALL PERFORMANCE
{'═'*74}
  Accuracy                  : {metrics['accuracy']*100:7.3f}%
  Balanced Accuracy         : {metrics['balanced_accuracy']*100:7.3f}%
  Macro Precision           : {metrics['macro_precision']*100:7.3f}%
  Macro Recall              : {metrics['macro_recall']*100:7.3f}%
  Macro F1-Score            : {metrics['macro_f1']*100:7.3f}%
  Weighted F1-Score         : {metrics['weighted_f1']*100:7.3f}%
  Mean Average Precision    : {metrics['mean_ap']*100:7.3f}%
  Mean ROC-AUC              : {metrics['mean_auc']*100:7.3f}%

📈  TOP-K ACCURACY
{'═'*74}
  Top-1  :  {metrics['top_k'].get(1,0)*100:7.3f}%
  Top-3  :  {metrics['top_k'].get(3,0)*100:7.3f}%
  Top-5  :  {metrics['top_k'].get(5,0)*100:7.3f}%

🔍  CLASS EXTREMES
{'═'*74}
  Best  : {metrics['best_class']:<22}  F1 = {metrics['best_class_f1']*100:.2f}%
  Worst : {metrics['worst_class']:<22}  F1 = {metrics['worst_class_f1']*100:.2f}%

⚙️   MODEL CONFIG (auto-detected from checkpoint)
{'═'*74}
  embed_dim      : {cfg.get('embed_dim')}     num_frames : {cfg.get('num_frames')}
  img_size       : {cfg.get('img_size')}     patch_size  : {cfg.get('patch_size')}
  spatial_depth  : {cfg.get('spatial_depth')}       temporal_depth : {cfg.get('temporal_depth')}
  num_heads      : {cfg.get('num_heads')}       mlp_ratio      : {cfg.get('mlp_ratio')}
  dropout        : {cfg.get('dropout')}

🔖  CHECKPOINT
{'═'*74}
  Epoch : {ep}    Val accuracy : {val_s}

📅  Test date : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
"""
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=9.5,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='#deeeff', alpha=0.45))
    fig.tight_layout(); _save(fig, out / 'summary_report.png')


# ─────────────────────────────────────────────────────────────
# Save metrics / CSV
# ─────────────────────────────────────────────────────────────
def save_metrics(metrics, cn, y_true, y_pred, y_probs, filenames, mdir):
    def _f(v):
        if isinstance(v, (np.floating, float)): return float(v)
        if isinstance(v, (np.integer, int)):    return int(v)
        return v

    skip = {'class_report', 'per_class_ap', 'per_class_auc', 'top_k'}
    out  = {k: _f(v) for k, v in metrics.items() if k not in skip}
    out['top_k']         = {str(k): float(v) for k, v in metrics['top_k'].items()}
    out['per_class_ap']  = {k: float(v) for k, v in metrics['per_class_ap'].items()}
    out['per_class_auc'] = {k: float(v) for k, v in metrics['per_class_auc'].items()}
    with open(mdir / 'metrics.json', 'w') as f: json.dump(out, f, indent=2)
    log.info("  ✓  metrics.json")

    rpt = classification_report(y_true, y_pred, target_names=cn)
    with open(mdir / 'classification_report.txt', 'w') as f:
        f.write("ConstrainedViViT — Classification Report\n" + "═"*78 + "\n\n")
        f.write(rpt)
        f.write(f"\nmAP  : {metrics['mean_ap']:.4f}\nmAUC : {metrics['mean_auc']:.4f}\n")
    log.info("  ✓  classification_report.txt")

    pd.DataFrame(confusion_matrix(y_true, y_pred), index=cn, columns=cn
                 ).to_csv(mdir / 'confusion_matrix.csv')
    log.info("  ✓  confusion_matrix.csv")

    (pd.DataFrame(list(metrics['per_class_ap'].items()), columns=['class','ap'])
       .sort_values('ap', ascending=False).to_csv(mdir/'average_precision_scores.csv', index=False))
    (pd.DataFrame(list(metrics['per_class_auc'].items()), columns=['class','auc'])
       .sort_values('auc', ascending=False).to_csv(mdir/'auc_scores.csv', index=False))
    log.info("  ✓  average_precision_scores.csv  /  auc_scores.csv")

    df = pd.DataFrame({
        'filename':        filenames,
        'true_label':      [cn[l] for l in y_true],
        'predicted_label': [cn[p] for p in y_pred],
        'correct':         (y_true == y_pred),
        'confidence':      y_probs.max(axis=1),
    })
    for i, c in enumerate(cn): df[f'prob_{c}'] = y_probs[:, i]
    df.to_csv(mdir / 'detailed_predictions.csv', index=False)
    df[~df['correct']].sort_values('confidence', ascending=False
                                   ).to_csv(mdir / 'misclassified_samples.csv', index=False)
    log.info(f"  ✓  detailed_predictions.csv  /  misclassified_samples.csv "
             f"({(~(y_true==y_pred)).sum()} errors)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    out         = Path(__file__).parent / OUT_DIR    
    plots_dir   = out / 'plots'
    metrics_dir = out / 'metrics'
    for d in [plots_dir, metrics_dir,
              plots_dir/'pr_curves_individual',
              plots_dir/'roc_curves_individual']:
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 78)
    log.info("ConstrainedViViT — Comprehensive Testing")
    log.info("=" * 78)
    log.info(f"Model    : {MODEL_PATH}")
    log.info(f"Test dir : {TEST_DIR}")
    log.info(f"Out dir  : {OUT_DIR}")

    if not Path(MODEL_PATH).exists():
        log.error(f"Model not found: {MODEL_PATH}"); return
    if not Path(TEST_DIR).exists():
        log.error(f"Test dir not found: {TEST_DIR}"); return

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps');  log.info("Device : MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda'); log.info("Device : CUDA")
    else:
        device = torch.device('cpu');  log.info("Device : CPU")

    # Load & auto-detect
    state_dict, saved_classes, meta = load_checkpoint(MODEL_PATH, device)
    config = infer_config(state_dict)

    # Dataset
    dataset = NPZDataset(TEST_DIR, class_names=saved_classes,
                         num_frames=config['num_frames'], img_size=config['img_size'])
    loader  = DataLoader(dataset, batch_size=config['batch_size'],
                         shuffle=False, num_workers=0, pin_memory=False)
    cn = dataset.classes

    # Model
    model = ConstrainedViViT(
        num_classes=len(cn), num_frames=config['num_frames'],
        img_size=config['img_size'], patch_size=config['patch_size'],
        embed_dim=config['embed_dim'], mlp_ratio=config['mlp_ratio'],
        spatial_depth=config['spatial_depth'], temporal_depth=config['temporal_depth'],
        heads=config['num_heads'], dropout=config['dropout'],
    ).to(device)
    model.load_state_dict(state_dict)
    log.info("✓ Weights loaded — zero size mismatches")

    # Run
    log.info("\n── Inference ──────────────────────────────────────────")
    y_pred, y_true, y_probs, filenames = run_inference(model, loader, device)

    log.info("\n── Metrics ────────────────────────────────────────────")
    metrics = compute_metrics(y_true, y_pred, y_probs, cn)

    log.info("\n── Plots ──────────────────────────────────────────────")
    plot_cm_counts(y_true, y_pred, cn, plots_dir)
    plot_cm_normalised(y_true, y_pred, cn, plots_dir)
    plot_per_class_metrics(cn, metrics['class_report'], plots_dir)
    plot_class_accuracy_sorted(y_true, y_pred, cn, plots_dir)
    plot_top_k(metrics['top_k'], plots_dir)
    plot_pr_combined(y_true, y_probs, cn, plots_dir)
    plot_pr_individual(y_true, y_probs, cn, plots_dir/'pr_curves_individual')
    plot_roc_combined(y_true, y_probs, cn, plots_dir)
    plot_roc_individual(y_true, y_probs, cn, plots_dir/'roc_curves_individual')
    plot_calibration(y_true, y_probs, plots_dir)
    plot_summary(metrics, meta, config, plots_dir)

    log.info("\n── Saving metrics ─────────────────────────────────────")
    save_metrics(metrics, cn, y_true, y_pred, y_probs, filenames, metrics_dir)

    # Console summary
    print("\n" + "═"*72)
    print("  RESULTS — ConstrainedViViT")
    print("═"*72)
    print(f"  Accuracy          : {metrics['accuracy']*100:.3f}%")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']*100:.3f}%")
    print(f"  Macro F1          : {metrics['macro_f1']*100:.3f}%")
    print(f"  Weighted F1       : {metrics['weighted_f1']*100:.3f}%")
    print(f"  mAP               : {metrics['mean_ap']*100:.3f}%")
    print(f"  Mean AUC          : {metrics['mean_auc']*100:.3f}%")
    for k, v in sorted(metrics['top_k'].items()):
        print(f"  Top-{k} Accuracy   : {v*100:.3f}%")
    print(f"\n  Best  : {metrics['best_class']}  (F1 {metrics['best_class_f1']*100:.2f}%)")
    print(f"  Worst : {metrics['worst_class']}  (F1 {metrics['worst_class_f1']*100:.2f}%)")
    print(f"\n  Samples : {metrics['total_samples']}  |  Classes : {metrics['num_classes']}")
    print(f"  Output  → {out.resolve()}")
    print("═"*72 + "\n")
    log.info("✅  Done!")


if __name__ == '__main__':
    main()