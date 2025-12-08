#!/usr/bin/env python3
"""
ViViT Configured to Reach 85-90% Validation Accuracy at Epoch 50
WITH COMPREHENSIVE PLOTTING
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import random
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import math

warnings.filterwarnings('ignore')

# Set seaborn style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('training_85_90.log', mode='a')]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(42)

# -------------------------
# Dataset with HEAVY Augmentation
# -------------------------
class NPZDirectoryDataset(Dataset):
    def __init__(self, root_dir, num_frames=8, is_training=True, img_size=96):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.is_training = is_training
        self.img_size = img_size

        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        self._scan_directory()

        logger.info(f"Dataset: {root_dir}")
        logger.info(f"Classes: {len(self.classes)}, Samples: {len(self.samples)}")

    def _scan_directory(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist")
        
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        for cid, d in enumerate(class_dirs):
            self.classes.append(d.name)
            self.class_to_idx[d.name] = cid
            files = list(d.glob("*.npz"))
            for f in files:
                self.samples.append({'path': f, 'class_idx': cid, 'class_name': d.name})
            logger.info(f"  {d.name}: {len(files)} videos")

    def __len__(self):
        return len(self.samples)

    def load_video(self, npz_path):
        try:
            data = np.load(npz_path)
            if 'frames' in data:
                frames = data['frames']
            elif 'video' in data:
                frames = data['video']
            else:
                frames = data[data.files[0]]

            if frames.dtype != np.uint8:
                if frames.max() <= 1.0:
                    frames = (frames * 255).astype(np.uint8)
                else:
                    frames = np.clip(frames, 0, 255).astype(np.uint8)
            return frames
        except Exception as e:
            logger.error(f"Error loading {npz_path}: {e}")
            return np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)

    def temporal_sampling(self, frames):
        n = len(frames)
        if n == 0:
            return np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)
        
        if n <= self.num_frames:
            indices = list(range(n)) + [n-1] * (self.num_frames - n)
        else:
            if self.is_training and random.random() > 0.3:
                start = random.randint(0, max(0, n - self.num_frames))
                indices = list(range(start, min(start + self.num_frames, n)))
                if len(indices) < self.num_frames:
                    indices += [indices[-1]] * (self.num_frames - len(indices))
            else:
                indices = np.linspace(0, n-1, self.num_frames, dtype=int)
        return frames[indices]

    def resize_frames(self, frames):
        resized = []
        for frame in frames:
            resized.append(cv2.resize(frame, (self.img_size, self.img_size)))
        return np.array(resized)

    def apply_heavy_augmentations(self, frames):
        if not self.is_training:
            return frames

        if random.random() > 0.3:
            factor = random.uniform(0.6, 1.4)
            frames = np.clip(frames * factor, 0, 255).astype(np.uint8)

        if random.random() > 0.3:
            factor = random.uniform(0.6, 1.4)
            mean = frames.mean(axis=(1,2,3), keepdims=True)
            frames = np.clip((frames - mean) * factor + mean, 0, 255).astype(np.uint8)

        if random.random() > 0.5:
            for i in range(len(frames)):
                hsv = cv2.cvtColor(frames[i], cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] *= random.uniform(0.6, 1.4)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                frames[i] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if random.random() > 0.5:
            frames = frames[:, :, ::-1, :].copy()

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = frames.shape[1:3]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            rotated = []
            for frame in frames:
                rotated.append(cv2.warpAffine(frame, M, (w, h)))
            frames = np.array(rotated)

        if random.random() > 0.5:
            kernel_size = random.choice([3, 5, 7])
            blurred = []
            for frame in frames:
                blurred.append(cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0))
            frames = np.array(blurred)

        if random.random() > 0.4:
            noise = np.random.normal(0, random.uniform(10, 25), frames.shape)
            frames = np.clip(frames + noise, 0, 255).astype(np.uint8)

        if random.random() > 0.4:
            num_erase = random.randint(1, 3)
            for _ in range(num_erase):
                t_idx = random.randint(0, len(frames) - 1)
                h, w = frames.shape[1:3]
                x = random.randint(0, w - w//3)
                y = random.randint(0, h - h//3)
                w_erase = random.randint(w//6, w//3)
                h_erase = random.randint(h//6, h//3)
                frames[t_idx, y:y+h_erase, x:x+w_erase, :] = random.randint(0, 255)

        if random.random() > 0.6:
            num_drop = random.randint(1, 2)
            drop_indices = random.sample(range(len(frames)), num_drop)
            for idx in drop_indices:
                if idx > 0:
                    frames[idx] = frames[idx-1]

        return frames

    def normalize(self, frames):
        frames = frames.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames = (frames - mean) / std
        return frames

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self.load_video(sample['path'])
        label = sample['class_idx']

        frames = self.temporal_sampling(frames)
        frames = self.resize_frames(frames)
        
        if self.is_training:
            frames = self.apply_heavy_augmentations(frames)

        frames = self.normalize(frames)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        
        return frames, label


# -------------------------
# Model Components
# -------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=96, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        B, E, Tp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SlowLearningViViT(nn.Module):
    def __init__(self, img_size=96, patch_size=16, tubelet_size=2, num_frames=8,
                 in_chans=3, num_classes=1000, embed_dim=256, num_heads=8, num_layers=6,
                 mlp_ratio=4., dropout=0.3, attention_dropout=0.3):
        super().__init__()
        
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, 
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        
        num_patches = ((img_size // patch_size) ** 2) * (num_frames // tubelet_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=dropout,
                attn_drop=attention_dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(embed_dim, num_classes)
        )
        
        self._init_weights()
        
        logger.info(f"SlowLearningViViT: {embed_dim}D, {num_layers} layers")
        logger.info(f"Dropout: {dropout}")
        logger.info(f"Params: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


# -------------------------
# Label Smoothing
# -------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.2):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# -------------------------
# Warmup Scheduler
# -------------------------
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# -------------------------
# Plotting Functions
# -------------------------
class TrainingPlotter:
    def __init__(self, plot_dir):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📊 Plots will be saved to: {self.plot_dir}")
    
    def plot_loss_curves(self, train_losses, val_losses, epoch):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.plot_dir / f'loss_curves_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_accuracy_curves(self, train_accs, val_accs, epoch):
        """Plot training and validation accuracy curves"""
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(train_accs) + 1)
        
        train_accs_pct = [acc * 100 for acc in train_accs]
        val_accs_pct = [acc * 100 for acc in val_accs]
        
        plt.plot(epochs, train_accs_pct, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, val_accs_pct, 'r-', label='Validation Accuracy', linewidth=2)
        
        # Add target zone
        plt.axhspan(85, 90, alpha=0.2, color='green', label='Target Zone (85-90%)')
        
        plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 105])
        plt.tight_layout()
        
        plt.savefig(self.plot_dir / f'accuracy_curves_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_overfitting_gap(self, train_accs, val_accs, epoch):
        """Plot the gap between training and validation accuracy"""
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(train_accs) + 1)
        
        gaps = [(train - val) * 100 for train, val in zip(train_accs, val_accs)]
        
        plt.plot(epochs, gaps, 'purple', linewidth=2, marker='o', markersize=4)
        plt.fill_between(epochs, gaps, 0, alpha=0.3, color='purple')
        
        plt.title('Overfitting Gap (Train Acc - Val Acc)', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Gap (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()
        
        plt.savefig(self.plot_dir / f'overfitting_gap_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_learning_rate(self, lr_history, epoch):
        """Plot learning rate schedule"""
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(lr_history) + 1)
        
        plt.plot(epochs, lr_history, 'green', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.plot_dir / f'learning_rate_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_combined_metrics(self, train_losses, val_losses, train_accs, val_accs, lr_history, epoch):
        """Plot all metrics in a single figure"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        train_accs_pct = [acc * 100 for acc in train_accs]
        val_accs_pct = [acc * 100 for acc in val_accs]
        axes[0, 1].plot(epochs, train_accs_pct, 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, val_accs_pct, 'r-', label='Val', linewidth=2)
        axes[0, 1].axhspan(85, 90, alpha=0.2, color='green', label='Target')
        axes[0, 1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 105])
        
        # Overfitting gap
        gaps = [(train - val) * 100 for train, val in zip(train_accs, val_accs)]
        axes[1, 0].plot(epochs, gaps, 'purple', linewidth=2, marker='o', markersize=3)
        axes[1, 0].fill_between(epochs, gaps, 0, alpha=0.3, color='purple')
        axes[1, 0].set_title('Overfitting Gap', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Gap (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        # Learning rate
        axes[1, 1].plot(epochs, lr_history, 'green', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'combined_metrics_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_table(self, train_losses, val_losses, train_accs, val_accs, epoch):
        """Create a table showing metrics progression"""
        fig, ax = plt.subplots(figsize=(14, max(10, len(train_losses) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        data = []
        for i, (tl, vl, ta, va) in enumerate(zip(train_losses, val_losses, train_accs, val_accs), 1):
            gap = (ta - va) * 100
            data.append([
                i,
                f"{tl:.4f}",
                f"{vl:.4f}",
                f"{ta*100:.2f}%",
                f"{va*100:.2f}%",
                f"{gap:.2f}%"
            ])
        
        # Create table
        table = ax.table(
            cellText=data,
            colLabels=['Epoch', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'Gap'],
            cellLoc='center',
            loc='center',
            colWidths=[0.1, 0.18, 0.18, 0.18, 0.18, 0.18]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(data) + 1):
            for j in range(6):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Training Metrics Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.plot_dir / f'metrics_table_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_final_summary(self, train_losses, val_losses, train_accs, val_accs, lr_history, best_val_acc, config):
        """Create a comprehensive final summary plot"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2 = fig.add_subplot(gs[1, :2])
        train_accs_pct = [acc * 100 for acc in train_accs]
        val_accs_pct = [acc * 100 for acc in val_accs]
        ax2.plot(epochs, train_accs_pct, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accs_pct, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.axhspan(85, 90, alpha=0.2, color='green', label='Target Zone')
        ax2.axhline(y=best_val_acc*100, color='gold', linestyle='--', linewidth=2, label=f'Best: {best_val_acc*100:.2f}%')
        ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        
        # Overfitting gap
        ax3 = fig.add_subplot(gs[2, :2])
        gaps = [(train - val) * 100 for train, val in zip(train_accs, val_accs)]
        ax3.plot(epochs, gaps, 'purple', linewidth=2, marker='o', markersize=3)
        ax3.fill_between(epochs, gaps, 0, alpha=0.3, color='purple')
        ax3.set_title('Overfitting Gap', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Gap (%)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        # Learning rate
        ax4 = fig.add_subplot(gs[0, 2])
        ax4.plot(epochs, lr_history, 'green', linewidth=2)
        ax4.set_title('Learning Rate', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('LR')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Summary statistics
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        summary_text = f"""
TRAINING SUMMARY
{'='*25}

Best Val Acc: {best_val_acc*100:.2f}%
Final Train Acc: {train_accs[-1]*100:.2f}%
Final Val Acc: {val_accs[-1]*100:.2f}%
Final Gap: {gaps[-1]:.2f}%

Min Train Loss: {min(train_losses):.4f}
Min Val Loss: {min(val_losses):.4f}

Total Epochs: {len(train_losses)}
"""
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Configuration
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        config_text = f"""
MODEL CONFIG
{'='*25}

Image Size: {config['img_size']}x{config['img_size']}
Frames: {config['num_frames']}
Embed Dim: {config['embed_dim']}
Layers: {config['num_layers']}
Heads: {config['num_heads']}

Learning Rate: {config['learning_rate']:.0e}
Dropout: {config['dropout_rate']}
Batch Size: {config['batch_size']}
Weight Decay: {config['weight_decay']}
"""
        ax6.text(0.1, 0.9, config_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('ViViT Training - Final Summary', fontsize=18, fontweight='bold', y=0.98)
        plt.savefig(self.plot_dir / 'final_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("✓ Final summary plot saved")
    
    def plot_per_epoch_comparison(self, train_losses, val_losses, train_accs, val_accs, epoch):
        """Bar chart comparing train vs val for current epoch"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Loss comparison
        categories = ['Train', 'Validation']
        losses = [train_losses[-1], val_losses[-1]]
        colors = ['blue', 'red']
        
        bars1 = ax1.bar(categories, losses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_title(f'Loss Comparison - Epoch {epoch}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, loss in zip(bars1, losses):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Accuracy comparison
        accs = [train_accs[-1] * 100, val_accs[-1] * 100]
        bars2 = ax2.bar(categories, accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_title(f'Accuracy Comparison - Epoch {epoch}', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim([0, 105])
        ax2.axhline(y=85, color='green', linestyle='--', alpha=0.5, label='Target Min')
        ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Target Max')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        for bar, acc in zip(bars2, accs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f'epoch_{epoch}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_metrics_csv(self, train_losses, val_losses, train_accs, val_accs, lr_history):
        """Save all metrics to CSV"""
        df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': [acc * 100 for acc in train_accs],
            'val_accuracy': [acc * 100 for acc in val_accs],
            'accuracy_gap': [(ta - va) * 100 for ta, va in zip(train_accs, val_accs)],
            'learning_rate': lr_history
        })
        
        csv_path = self.plot_dir / 'training_metrics.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Metrics saved to: {csv_path}")


# -------------------------
# Trainer with Plotting
# -------------------------
class Trainer:
    def __init__(self, config):
        self.cfg = config
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device = torch.device('cuda')
        self.n_gpus = torch.cuda.device_count()
        
        logger.info(f"Found {self.n_gpus} GPU(s)")

        self.save_dir = Path(self.cfg['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plot directory
        self.plot_dir = self.save_dir / 'plots'
        self.plotter = TrainingPlotter(self.plot_dir)

        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")))

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.lr_history = []
        self.best_val_acc = 0.0
        self.model = None

    def create_data_loaders(self):
        train_ds = NPZDirectoryDataset(
            self.cfg['train_dir'], 
            num_frames=self.cfg['num_frames'], 
            is_training=True,
            img_size=self.cfg['img_size']
        )
        
        val_ds = NPZDirectoryDataset(
            self.cfg['val_dir'], 
            num_frames=self.cfg['num_frames'], 
            is_training=False,
            img_size=self.cfg['img_size']
        )

        self.class_names = train_ds.classes
        self.num_classes = len(self.class_names)
        logger.info(f"Total classes: {self.num_classes}")

        train_loader = DataLoader(
            train_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=True,
            num_workers=self.cfg['num_workers'], 
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg['num_workers'], 
            pin_memory=True
        )
        
        return train_loader, val_loader

    def train_epoch(self, model, loader, optimizer, criterion, epoch):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        skipped = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if not torch.isfinite(loss):
                logger.warning(f"NaN/Inf loss at batch {batch_idx}")
                skipped += 1
                continue
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if not torch.isfinite(grad_norm):
                logger.warning(f"NaN/Inf gradients at batch {batch_idx}")
                optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                skipped += 1
                continue
            
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            if batch_idx % 20 == 0:
                pbar.set_postfix({
                    'loss': f"{total_loss/(batch_idx+1-skipped):.4f}", 
                    'acc': f"{100.*correct/total:.2f}%",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })

        epoch_loss = total_loss / max(1, len(loader) - skipped)
        epoch_acc = correct / total if total > 0 else 0.0

        return epoch_loss, epoch_acc

    def evaluate(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Validation", leave=False):
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        epoch_loss = total_loss / len(loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def train(self):
        train_loader, val_loader = self.create_data_loaders()

        model = SlowLearningViViT(
            img_size=self.cfg['img_size'],
            patch_size=self.cfg['patch_size'],
            tubelet_size=self.cfg['tubelet_size'],
            num_frames=self.cfg['num_frames'],
            in_chans=3,
            num_classes=self.num_classes,
            embed_dim=self.cfg['embed_dim'],
            num_heads=self.cfg['num_heads'],
            num_layers=self.cfg['num_layers'],
            mlp_ratio=self.cfg['mlp_ratio'],
            dropout=self.cfg['dropout_rate'],
            attention_dropout=self.cfg['attention_dropout']
        )

        if self.n_gpus > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        self.model = model

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total params: {total_params:,}")

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.cfg['learning_rate'], 
            weight_decay=self.cfg['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=self.cfg['warmup_epochs'],
            total_epochs=self.cfg['epochs'],
            base_lr=self.cfg['learning_rate'],
            min_lr=self.cfg['min_lr']
        )
        
        criterion = LabelSmoothingCrossEntropy(smoothing=self.cfg['label_smoothing'])

        logger.info("\nStarting training...")
        for epoch in range(1, self.cfg['epochs'] + 1):
            current_lr = scheduler.step()
            self.lr_history.append(current_lr)
            
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            val_loss, val_acc = self.evaluate(model, val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            logger.info(f"\nEpoch {epoch}/{self.cfg['epochs']}")
            logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
            logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")
            logger.info(f"Gap: {(train_acc - val_acc)*100:.2f}%")
            logger.info(f"LR: {current_lr:.6f}")

            self.writer.add_scalar("Train/Loss", train_loss, epoch)
            self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
            self.writer.add_scalar("Val/Loss", val_loss, epoch)
            self.writer.add_scalar("Val/Accuracy", val_acc, epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            # Generate plots every epoch
            logger.info("📊 Generating plots...")
            self.plotter.plot_loss_curves(self.train_losses, self.val_losses, epoch)
            self.plotter.plot_accuracy_curves(self.train_accs, self.val_accs, epoch)
            self.plotter.plot_overfitting_gap(self.train_accs, self.val_accs, epoch)
            self.plotter.plot_learning_rate(self.lr_history, epoch)
            
            # Generate combined plot every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                self.plotter.plot_combined_metrics(
                    self.train_losses, self.val_losses, 
                    self.train_accs, self.val_accs, 
                    self.lr_history, epoch
                )
                self.plotter.plot_per_epoch_comparison(
                    self.train_losses, self.val_losses,
                    self.train_accs, self.val_accs, epoch
                )
                self.plotter.plot_metrics_table(
                    self.train_losses, self.val_losses,
                    self.train_accs, self.val_accs, epoch
                )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'config': self.cfg,
                }, self.save_dir / "best_model.pth")
                logger.info(f"✓ New best! Val Acc: {val_acc*100:.2f}%")
            
            if epoch % 5 == 0:
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'config': self.cfg,
                }, self.save_dir / f"checkpoint_epoch_{epoch}.pth")
                logger.info(f"💾 Checkpoint saved: epoch_{epoch}")

        # Generate final summary plots
        logger.info("\n📊 Generating final summary plots...")
        self.plotter.plot_final_summary(
            self.train_losses, self.val_losses,
            self.train_accs, self.val_accs,
            self.lr_history, self.best_val_acc, self.cfg
        )
        self.plotter.save_metrics_csv(
            self.train_losses, self.val_losses,
            self.train_accs, self.val_accs,
            self.lr_history
        )

        self.writer.close()
        logger.info(f"\n✓ Training Complete! Best Val Acc: {self.best_val_acc*100:.2f}%")
        logger.info(f"✓ All plots saved to: {self.plot_dir}")
        
        return model


# -------------------------
# Main
# -------------------------
def main():
    config = {
        # DATA
        'train_dir': '/kaggle/input/npz-lightweight-videos-without-cropping-dataset/npz_lightweight_videos_without_cropping_splitted_dataset/train',
        'val_dir': '/kaggle/input/npz-lightweight-videos-without-cropping-dataset/npz_lightweight_videos_without_cropping_splitted_dataset/val',
        'save_dir': '/kaggle/working/',

        # MODEL
        'img_size': 112,
        'patch_size': 16,
        'tubelet_size': 2,
        'num_frames': 8,
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'mlp_ratio': 4.0,
        
        # REGULARIZATION
        'dropout_rate': 0.2,
        'attention_dropout': 0.3,
        'label_smoothing': 0.2,

        # TRAINING
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 2e-4,
        'min_lr': 1e-7,
        'warmup_epochs': 10,
        'weight_decay': 0.15,

        # SYSTEM
        'num_workers': 4,
    }

    logger.info("\n" + "="*80)
    logger.info("VIVIT CONFIGURED FOR 85-90% AT EPOCH 50 - WITH PLOTTING")
    logger.info("="*80)
    logger.info("\nPLOTS WILL INCLUDE:")
    logger.info("  ✓ Loss curves (train & val)")
    logger.info("  ✓ Accuracy curves (train & val)")
    logger.info("  ✓ Overfitting gap analysis")
    logger.info("  ✓ Learning rate schedule")
    logger.info("  ✓ Combined metrics dashboard")
    logger.info("  ✓ Per-epoch comparisons")
    logger.info("  ✓ Metrics summary table")
    logger.info("  ✓ Final comprehensive summary")
    logger.info("  ✓ CSV export of all metrics")
    logger.info("="*80 + "\n")

    trainer = Trainer(config)
    
    try:
        model = trainer.train()
        logger.info("\n" + "="*80)
        logger.info("✓ TRAINING COMPLETED!")
        logger.info("="*80)
        logger.info(f"✓ Best validation accuracy: {trainer.best_val_acc*100:.2f}%")
        logger.info(f"✓ Model saved to: {config['save_dir']}/best_model.pth")
        logger.info(f"✓ All plots saved to: {config['save_dir']}/plots/")
        logger.info(f"✓ Metrics CSV saved to: {config['save_dir']}/plots/training_metrics.csv")
        logger.info("")
        
        if 85 <= trainer.best_val_acc * 100 <= 90:
            logger.info("🎯 PERFECT! Achieved 85-90% target range!")
        elif 80 <= trainer.best_val_acc * 100 < 85:
            logger.info("✓ CLOSE! Got 80-85%")
        elif 90 < trainer.best_val_acc * 100 <= 95:
            logger.info("⚠ SLIGHTLY OVER! Got 90-95%")
        elif trainer.best_val_acc * 100 > 95:
            logger.info("❌ TOO HIGH! Got 95%+")
        else:
            logger.info("⚠ BELOW TARGET! Got <80%")
        
        logger.info("="*80)
        return model
        
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()