#!/usr/bin/env python3
"""
ViViT Testing Code for Mac - With PR Curves
Tests the trained model on test/validation set and generates comprehensive metrics including PR curves
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                            f1_score, precision_score, recall_score, 
                            precision_recall_curve, average_precision_score)
import json

warnings.filterwarnings('ignore')

# Set seaborn style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('testing_results.log', mode='a')]
)
logger = logging.getLogger(__name__)


# -------------------------
# Dataset (Same as Training)
# -------------------------
class NPZDirectoryDataset(Dataset):
    def __init__(self, root_dir, num_frames=8, img_size=96):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
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
            indices = np.linspace(0, n-1, self.num_frames, dtype=int)
        return frames[indices]

    def resize_frames(self, frames):
        resized = []
        for frame in frames:
            resized.append(cv2.resize(frame, (self.img_size, self.img_size)))
        return np.array(resized)

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
        frames = self.normalize(frames)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        
        return frames, label, sample['path'].name


# -------------------------
# Model Components (Same as Training)
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
# Testing Plotter (UPDATED WITH PR CURVES)
# -------------------------
class TestingPlotter:
    def __init__(self, plot_dir):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory for individual PR curves
        self.pr_curves_dir = self.plot_dir / 'pr_curves_individual'
        self.pr_curves_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 Test plots will be saved to: {self.plot_dir}")
        logger.info(f"📈 Individual PR curves will be saved to: {self.pr_curves_dir}")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix with counts only - matching 3D CNN style"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure with appropriate size
        fig_size = max(16, len(class_names) * 0.6)
        plt.figure(figsize=(fig_size, fig_size * 0.9))
        
        # Plot with counts - matching 3D CNN style
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
        
        plt.title('Confusion Matrix (Counts)', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'confusion_matrix.png', dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Confusion matrix saved (400 DPI)")
    
    def plot_normalized_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot normalized confusion matrix (percentages)"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig_size = max(16, len(class_names) * 0.6)
        plt.figure(figsize=(fig_size, fig_size * 0.9))
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
                   xticklabels=class_names, yticklabels=class_names,
                   vmin=0, vmax=1, cbar_kws={'label': 'Percentage'},
                   linewidths=0.5, linecolor='gray')
        
        plt.title('Normalized Confusion Matrix (%)', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'confusion_matrix_normalized.png', dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Normalized confusion matrix saved (400 DPI)")
    
    def plot_pr_curves_combined(self, y_true, y_probs, class_names):
        """Plot all PR curves on one graph with AP scores"""
        plt.figure(figsize=(14, 10))
        
        # Generate colors for each class
        colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
        
        # Store AP scores for sorting
        ap_scores = []
        
        # Plot PR curve for each class
        for i, class_name in enumerate(class_names):
            # Convert to binary classification (one vs rest)
            y_true_binary = (y_true == i).astype(int)
            y_scores = y_probs[:, i]
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
            ap = average_precision_score(y_true_binary, y_scores)
            ap_scores.append((class_name, ap))
            
            # Plot with class name and AP score in legend
            plt.plot(recall, precision, color=colors[i], lw=1.5, alpha=0.8,
                    label=f'{class_name} (AP={ap:.3f})')
        
        plt.xlabel('Recall', fontsize=13, fontweight='bold')
        plt.ylabel('Precision', fontsize=13, fontweight='bold')
        plt.title('Precision-Recall Curves - All Classes', fontsize=16, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        # Place legend in bottom center inside the plot
        plt.legend(loc='lower center', fontsize=8, framealpha=0.9, ncol=2)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'pr_curves_combined.png', dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Combined PR curves saved (400 DPI)")
        
        # Calculate and log mean AP
        mean_ap = np.mean([ap for _, ap in ap_scores])
        logger.info(f"  Mean Average Precision (mAP): {mean_ap:.4f}")
        
        return ap_scores, mean_ap
    
    def plot_pr_curves_individual(self, y_true, y_probs, class_names):
        """Plot individual PR curves for each class in separate files"""
        logger.info("Generating individual PR curves...")
        
        ap_scores = []
        
        for i, class_name in enumerate(class_names):
            # Convert to binary classification
            y_true_binary = (y_true == i).astype(int)
            y_scores = y_probs[:, i]
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
            ap = average_precision_score(y_true_binary, y_scores)
            ap_scores.append(ap)
            
            # Create individual plot
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='#2E86AB', lw=3, label=f'AP = {ap:.3f}')
            plt.fill_between(recall, precision, alpha=0.3, color='#A23B72')
            
            plt.xlabel('Recall', fontsize=13, fontweight='bold')
            plt.ylabel('Precision', fontsize=13, fontweight='bold')
            plt.title(f'Precision-Recall Curve: {class_name}', fontsize=15, fontweight='bold', pad=15)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.legend(loc='best', fontsize=12, framealpha=0.9)
            
            # Add text box with statistics
            num_samples = np.sum(y_true_binary)
            text_str = f'Samples: {num_samples}\nAP: {ap:.4f}'
            plt.text(0.05, 0.05, text_str, transform=plt.gca().transAxes,
                    fontsize=11, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save with sanitized filename
            safe_filename = class_name.replace('/', '_').replace('\\', '_')
            plt.savefig(self.pr_curves_dir / f'pr_curve_{safe_filename}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"✓ {len(class_names)} individual PR curves saved (300 DPI)")
        return ap_scores
    
    def plot_per_class_metrics(self, class_names, precisions, recalls, f1_scores, supports):
        """Plot per-class performance metrics - improved style"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(len(class_names))
        width = 0.6
        
        # Precision
        bars1 = axes[0, 0].bar(x, precisions, width, color='#3498db', alpha=0.85)
        axes[0, 0].set_ylabel('Precision', fontsize=13, fontweight='bold')
        axes[0, 0].set_title('Precision per Class', fontsize=15, fontweight='bold', pad=15)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0, 0].set_ylim([0, 1.05])
        axes[0, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0, 0].axhline(y=np.mean(precisions), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(precisions):.3f}')
        axes[0, 0].legend(fontsize=11, framealpha=0.9)
        
        # Recall
        bars2 = axes[0, 1].bar(x, recalls, width, color='#2ecc71', alpha=0.85)
        axes[0, 1].set_ylabel('Recall', fontsize=13, fontweight='bold')
        axes[0, 1].set_title('Recall per Class', fontsize=15, fontweight='bold', pad=15)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0, 1].set_ylim([0, 1.05])
        axes[0, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0, 1].axhline(y=np.mean(recalls), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(recalls):.3f}')
        axes[0, 1].legend(fontsize=11, framealpha=0.9)
        
        # F1-Score
        bars3 = axes[1, 0].bar(x, f1_scores, width, color='#e74c3c', alpha=0.85)
        axes[1, 0].set_ylabel('F1-Score', fontsize=13, fontweight='bold')
        axes[1, 0].set_title('F1-Score per Class', fontsize=15, fontweight='bold', pad=15)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1, 0].set_ylim([0, 1.05])
        axes[1, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1, 0].axhline(y=np.mean(f1_scores), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(f1_scores):.3f}')
        axes[1, 0].legend(fontsize=11, framealpha=0.9)
        
        # Support (number of samples)
        bars4 = axes[1, 1].bar(x, supports, width, color='#9b59b6', alpha=0.85)
        axes[1, 1].set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
        axes[1, 1].set_title('Support per Class', fontsize=15, fontweight='bold', pad=15)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1, 1].axhline(y=np.mean(supports), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(supports):.1f}')
        axes[1, 1].legend(fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'per_class_metrics.png', dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Per-class metrics plot saved (400 DPI)")
    
    def plot_top_k_accuracy(self, top_k_accs):
        """Plot Top-K accuracy"""
        plt.figure(figsize=(10, 6))
        
        k_values = list(top_k_accs.keys())
        accuracies = [top_k_accs[k] * 100 for k in k_values]
        
        plt.plot(k_values, accuracies, 'o-', linewidth=2.5, markersize=10, color='#3498db')
        plt.title('Top-K Accuracy', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('K', fontsize=13, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim([0, 105])
        
        for k, acc in zip(k_values, accuracies):
            plt.text(k, acc + 2, f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'top_k_accuracy.png', dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Top-K accuracy plot saved (400 DPI)")
    
    def create_summary_report(self, metrics, config, save_path):
        """Create a comprehensive summary report"""
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        summary_text = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                     MODEL TESTING SUMMARY REPORT                      ║
╚═══════════════════════════════════════════════════════════════════════╝

📊 OVERALL PERFORMANCE
{'='*75}
  • Overall Accuracy:        {metrics['accuracy']*100:6.2f}%
  • Macro Avg Precision:     {metrics['macro_precision']*100:6.2f}%
  • Macro Avg Recall:        {metrics['macro_recall']*100:6.2f}%
  • Macro Avg F1-Score:      {metrics['macro_f1']*100:6.2f}%
  • Weighted Avg F1-Score:   {metrics['weighted_f1']*100:6.2f}%
  • Mean Average Precision:  {metrics.get('mean_ap', 0)*100:6.2f}%

📈 TOP-K ACCURACY
{'='*75}
  • Top-1 Accuracy:          {metrics['top_1_acc']*100:6.2f}%
  • Top-3 Accuracy:          {metrics['top_3_acc']*100:6.2f}%
  • Top-5 Accuracy:          {metrics['top_5_acc']*100:6.2f}%

🎯 DATASET INFORMATION
{'='*75}
  • Number of Classes:       {metrics['num_classes']}
  • Total Test Samples:      {metrics['total_samples']}
  • Samples per Class:       {metrics['samples_per_class']:.1f} (avg)

⚙️  MODEL CONFIGURATION
{'='*75}
  • Model Architecture:      ViViT (Video Vision Transformer)
  • Embedding Dimension:     {config.get('embed_dim', 'N/A')}
  • Number of Layers:        {config.get('num_layers', 'N/A')}
  • Number of Heads:         {config.get('num_heads', 'N/A')}
  • Image Size:              {config.get('img_size', 'N/A')}x{config.get('img_size', 'N/A')}
  • Number of Frames:        {config.get('num_frames', 'N/A')}
  • Dropout Rate:            {config.get('dropout_rate', 'N/A')}

🔍 BEST & WORST PERFORMING CLASSES
{'='*75}
  Best Class:  {metrics['best_class']['name']} (F1: {metrics['best_class']['f1']*100:.2f}%)
  Worst Class: {metrics['worst_class']['name']} (F1: {metrics['worst_class']['f1']*100:.2f}%)

📅 TEST DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*75}

"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Summary report saved (400 DPI)")


# -------------------------
# Model Tester
# -------------------------
class ModelTester:
    def __init__(self, model_path, test_dir, config=None):
        self.model_path = Path(model_path)
        self.test_dir = Path(test_dir)
        
        # Use CPU for Mac (MPS support is optional)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
            logger.info("Using MPS (Apple Silicon GPU)")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Create results directory
        self.results_dir = Path('./../../output/vivit')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_dir = self.results_dir / 'plots'
        self.plotter = TestingPlotter(self.plot_dir)
        
        # Load model and config
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.config = config if config else self.checkpoint.get('config', {})
        
        logger.info(f"Loaded checkpoint from: {model_path}")
        logger.info(f"Checkpoint epoch: {self.checkpoint.get('epoch', 'N/A')}")
        logger.info(f"Checkpoint val accuracy: {self.checkpoint.get('val_accuracy', 'N/A')}")
    
    def load_model(self, num_classes):
        """Load the trained model"""
        model = SlowLearningViViT(
            img_size=self.config.get('img_size', 112),
            patch_size=self.config.get('patch_size', 16),
            tubelet_size=self.config.get('tubelet_size', 2),
            num_frames=self.config.get('num_frames', 8),
            in_chans=3,
            num_classes=num_classes,
            embed_dim=self.config.get('embed_dim', 256),
            num_heads=self.config.get('num_heads', 8),
            num_layers=self.config.get('num_layers', 6),
            mlp_ratio=self.config.get('mlp_ratio', 4.0),
            dropout=self.config.get('dropout_rate', 0.2),
            attention_dropout=self.config.get('attention_dropout', 0.3)
        )
        
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {total_params:,} parameters")
        
        return model
    
    def test(self):
        """Run testing on test dataset"""
        logger.info("\n" + "="*80)
        logger.info("STARTING MODEL TESTING")
        logger.info("="*80)
        
        # Load test dataset
        test_dataset = NPZDirectoryDataset(
            self.test_dir,
            num_frames=self.config.get('num_frames', 8),
            img_size=self.config.get('img_size', 112)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        class_names = test_dataset.classes
        num_classes = len(class_names)
        
        # Load model
        model = self.load_model(num_classes)
        
        # Testing
        all_preds = []
        all_labels = []
        all_probs = []
        all_filenames = []
        
        logger.info("\nRunning inference...")
        with torch.no_grad():
            for inputs, labels, filenames in tqdm(test_loader, desc="Testing"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_filenames.extend(filenames)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        logger.info("\nCalculating metrics...")
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs, class_names)
        
        # Generate plots
        logger.info("\nGenerating plots...")
        self.plotter.plot_confusion_matrix(all_labels, all_preds, class_names)
        self.plotter.plot_normalized_confusion_matrix(all_labels, all_preds, class_names)
        
        # Generate PR curves (NEW)
        logger.info("\nGenerating PR curves...")
        ap_scores, mean_ap = self.plotter.plot_pr_curves_combined(all_labels, all_probs, class_names)
        individual_ap_scores = self.plotter.plot_pr_curves_individual(all_labels, all_probs, class_names)
        
        # Add mAP to metrics
        metrics['mean_ap'] = mean_ap
        metrics['per_class_ap'] = {class_names[i]: ap_scores[i][1] for i in range(len(class_names))}
        
        # Get per-class metrics from classification report
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        precisions = [report[cn]['precision'] for cn in class_names]
        recalls = [report[cn]['recall'] for cn in class_names]
        f1_scores = [report[cn]['f1-score'] for cn in class_names]
        supports = [report[cn]['support'] for cn in class_names]
        
        self.plotter.plot_per_class_metrics(class_names, precisions, recalls, f1_scores, supports)
        
        # Top-K accuracy
        top_k_accs = {}
        for k in [1, 3, 5]:
            if k <= num_classes:
                top_k_accs[k] = self.calculate_top_k_accuracy(all_probs, all_labels, k)
        
        if top_k_accs:
            self.plotter.plot_top_k_accuracy(top_k_accs)
        
        # Add top-k to metrics
        metrics.update({
            'top_1_acc': top_k_accs.get(1, 0),
            'top_3_acc': top_k_accs.get(3, 0),
            'top_5_acc': top_k_accs.get(5, 0)
        })
        
        # Create summary report
        self.plotter.create_summary_report(
            metrics, 
            self.config, 
            self.plot_dir / 'summary_report.png'
        )
        
        # Save detailed results
        self.save_results(metrics, class_names, all_labels, all_preds, all_probs, all_filenames)
        
        # Print summary
        self.print_summary(metrics)
        
        logger.info("\n" + "="*80)
        logger.info("TESTING COMPLETED!")
        logger.info("="*80)
        logger.info(f"All results saved to: {self.results_dir}")
        
        return metrics
    
    def calculate_metrics(self, y_true, y_pred, y_probs, class_names):
        """Calculate comprehensive metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class F1 scores
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        class_f1_scores = {cn: report[cn]['f1-score'] for cn in class_names}
        
        best_class = max(class_f1_scores.items(), key=lambda x: x[1])
        worst_class = min(class_f1_scores.items(), key=lambda x: x[1])
        
        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'num_classes': len(class_names),
            'total_samples': len(y_true),
            'samples_per_class': len(y_true) / len(class_names),
            'best_class': {'name': best_class[0], 'f1': best_class[1]},
            'worst_class': {'name': worst_class[0], 'f1': worst_class[1]},
            'class_f1_scores': class_f1_scores
        }
        
        return metrics
    
    def calculate_top_k_accuracy(self, y_probs, y_true, k):
        """Calculate Top-K accuracy"""
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        correct = 0
        for i, label in enumerate(y_true):
            if label in top_k_preds[i]:
                correct += 1
        return correct / len(y_true)
    
    def save_results(self, metrics, class_names, y_true, y_pred, y_probs, filenames):
        """Save detailed results to files"""
        # Save metrics as JSON
        metrics_serializable = {
            k: float(v) if isinstance(v, (np.floating, float)) else v 
            for k, v in metrics.items() 
            if k not in ['best_class', 'worst_class', 'class_f1_scores', 'per_class_ap']
        }
        metrics_serializable['best_class'] = {
            'name': metrics['best_class']['name'],
            'f1': float(metrics['best_class']['f1'])
        }
        metrics_serializable['worst_class'] = {
            'name': metrics['worst_class']['name'],
            'f1': float(metrics['worst_class']['f1'])
        }
        metrics_serializable['class_f1_scores'] = {
            k: float(v) for k, v in metrics['class_f1_scores'].items()
        }
        if 'per_class_ap' in metrics:
            metrics_serializable['per_class_ap'] = {
                k: float(v) for k, v in metrics['per_class_ap'].items()
            }
        
        with open(self.results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_serializable, f, indent=4)
        logger.info("✓ Metrics saved to metrics.json")
        
        # Save classification report
        report = classification_report(y_true, y_pred, target_names=class_names)
        with open(self.results_dir / 'classification_report.txt', 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
            if 'mean_ap' in metrics:
                f.write(f"\n\nMean Average Precision (mAP): {metrics['mean_ap']:.4f}\n")
        logger.info("✓ Classification report saved")
        
        # Save detailed predictions
        results_df = pd.DataFrame({
            'filename': filenames,
            'true_label': [class_names[label] for label in y_true],
            'predicted_label': [class_names[pred] for pred in y_pred],
            'correct': y_true == y_pred,
            'confidence': np.max(y_probs, axis=1)
        })
        
        # Add probability columns for each class
        for i, class_name in enumerate(class_names):
            results_df[f'prob_{class_name}'] = y_probs[:, i]
        
        results_df.to_csv(self.results_dir / 'detailed_predictions.csv', index=False)
        logger.info("✓ Detailed predictions saved to CSV")
        
        # Save misclassified samples
        misclassified_df = results_df[~results_df['correct']].copy()
        misclassified_df = misclassified_df.sort_values('confidence', ascending=False)
        misclassified_df.to_csv(self.results_dir / 'misclassified_samples.csv', index=False)
        logger.info(f"✓ Found {len(misclassified_df)} misclassified samples")
        
        # Save confusion matrix as CSV
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(self.results_dir / 'confusion_matrix.csv')
        logger.info("✓ Confusion matrix saved to CSV")
        
        # Save AP scores as CSV
        if 'per_class_ap' in metrics:
            ap_df = pd.DataFrame([
                {'class': class_name, 'average_precision': ap}
                for class_name, ap in metrics['per_class_ap'].items()
            ])
            ap_df = ap_df.sort_values('average_precision', ascending=False)
            ap_df.to_csv(self.results_dir / 'average_precision_scores.csv', index=False)
            logger.info("✓ Average Precision scores saved to CSV")
    
    def print_summary(self, metrics):
        """Print summary to console"""
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        print(f"\n📊 Overall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"📈 Macro F1-Score:   {metrics['macro_f1']*100:.2f}%")
        print(f"📉 Weighted F1:      {metrics['weighted_f1']*100:.2f}%")
        if 'mean_ap' in metrics:
            print(f"🎯 Mean AP (mAP):    {metrics['mean_ap']*100:.2f}%")
        print(f"\n✅ Best Class:  {metrics['best_class']['name']} (F1: {metrics['best_class']['f1']*100:.2f}%)")
        print(f"❌ Worst Class: {metrics['worst_class']['name']} (F1: {metrics['worst_class']['f1']*100:.2f}%)")
        print(f"\n📁 Total Samples: {metrics['total_samples']}")
        print(f"🏷️  Number of Classes: {metrics['num_classes']}")
        
        if 'top_1_acc' in metrics:
            print(f"\n🎯 Top-1 Accuracy: {metrics['top_1_acc']*100:.2f}%")
            if 'top_3_acc' in metrics and metrics['top_3_acc'] > 0:
                print(f"🎯 Top-3 Accuracy: {metrics['top_3_acc']*100:.2f}%")
            if 'top_5_acc' in metrics and metrics['top_5_acc'] > 0:
                print(f"🎯 Top-5 Accuracy: {metrics['top_5_acc']*100:.2f}%")
        
        print("="*80 + "\n")


# -------------------------
# Main Testing Function
# -------------------------
def main():
    """
    Main testing function
    
    Usage:
        python test_vivit_mac.py
    
    Make sure to update the paths below:
        - model_path: Path to your trained model checkpoint
        - test_dir: Path to your test/validation dataset
    """
    
    # ===== CONFIGURE THESE PATHS =====
    model_path = './../../kaggle/training_output_vivit/best_model.pth'
    test_dir = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test'
    # ==================================
    
    logger.info("\n" + "="*80)
    logger.info("ViViT MODEL TESTING FOR MAC - WITH PR CURVES")
    logger.info("="*80)
    logger.info(f"\nModel: {model_path}")
    logger.info(f"Test Data: {test_dir}")
    logger.info("="*80 + "\n")
    
    # Check if paths exist
    if not Path(model_path).exists():
        logger.error(f"❌ Model file not found: {model_path}")
        logger.info("\nPlease update the 'model_path' variable with the correct path to your model.")
        return
    
    if not Path(test_dir).exists():
        logger.error(f"❌ Test directory not found: {test_dir}")
        logger.info("\nPlease update the 'test_dir' variable with the correct path to your test data.")
        return
    
    try:
        # Initialize tester
        tester = ModelTester(model_path, test_dir)
        
        # Run testing
        metrics = tester.test()
        
        # Success message
        logger.info("\n✅ Testing completed successfully!")
        logger.info(f"📁 Results saved to: {tester.results_dir}")
        logger.info("\n📊 Generated files:")
        logger.info("   • metrics.json - Numerical metrics (including mAP)")
        logger.info("   • classification_report.txt - Detailed per-class metrics")
        logger.info("   • detailed_predictions.csv - Per-sample predictions")
        logger.info("   • misclassified_samples.csv - Incorrectly classified samples")
        logger.info("   • confusion_matrix.csv - Confusion matrix data")
        logger.info("   • average_precision_scores.csv - AP scores per class")
        logger.info("   • plots/confusion_matrix.png (counts only, 400 DPI)")
        logger.info("   • plots/confusion_matrix_normalized.png (percentages, 400 DPI)")
        logger.info("   • plots/per_class_metrics.png (400 DPI)")
        logger.info("   • plots/top_k_accuracy.png (400 DPI)")
        logger.info("   • plots/pr_curves_combined.png (all classes, 400 DPI)")
        logger.info("   • plots/pr_curves_individual/*.png (36 individual curves, 300 DPI)")
        logger.info("   • plots/summary_report.png (400 DPI)")
        
    except Exception as e:
        logger.error(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()