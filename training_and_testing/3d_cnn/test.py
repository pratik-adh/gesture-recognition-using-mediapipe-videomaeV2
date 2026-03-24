#!/usr/bin/env python3
"""
3D CNN Testing Code for Mac - Fixed Version
Fixes:
  1. Checkpoint key: weights are stored under 'model' key (not 'model_state_dict')
  2. Model architecture: uses Working3DCNN (64→128→256→512) matching training code
     instead of Simple3DCNN (64→128→256→256) which caused all key mismatches
All original testing features (PR curves, ROC curves, per-class metrics, etc.) preserved.
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
                             precision_recall_curve, average_precision_score,
                             roc_curve, auc)
import json

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('testing_results.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# Dataset  (unchanged from original)
# ============================================================
class NPZDirectoryDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, img_size=112):
        self.root_dir  = Path(root_dir)
        self.num_frames = num_frames
        self.img_size   = img_size
        self.samples      = []
        self.class_to_idx = {}
        self.classes      = []
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
            indices = list(range(n)) + [n - 1] * (self.num_frames - n)
        else:
            indices = np.linspace(0, n - 1, self.num_frames, dtype=int)
        return frames[indices]

    def resize_frames(self, frames):
        return np.array([cv2.resize(f, (self.img_size, self.img_size)) for f in frames])

    def normalize(self, frames):
        frames = frames.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        return (frames - mean) / std

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self.load_video(sample['path'])
        label  = sample['class_idx']
        frames = self.temporal_sampling(frames)
        frames = self.resize_frames(frames)
        frames = self.normalize(frames)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()   # (C,T,H,W)
        return frames, label, sample['path'].name


# ============================================================
# Model — MUST match the training architecture exactly
#
# FIX 2: The training code (Working3DCNN) uses:
#   features: 64 → 128 → 256 → 512  (AdaptiveAvgPool → 512-dim vector)
#   classifier: Dropout → Linear(512,256) → ReLU → Dropout → Linear(256,C)
#
# The original test code used Simple3DCNN:
#   features: 64 → 128 → 256 → 256
#   classifier: Dropout → Linear(256,C)
#
# Those architectures produce completely different state_dict keys,
# which is why every single key was reported as missing/unexpected.
# ============================================================
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================
# Testing Plotter  (unchanged from original)
# ============================================================
class TestingPlotter:
    def __init__(self, plot_dir):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.pr_curves_dir  = self.plot_dir / 'pr_curves_individual'
        self.roc_curves_dir = self.plot_dir / 'roc_curves_individual'
        self.pr_curves_dir.mkdir(parents=True, exist_ok=True)
        self.roc_curves_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📊 Test plots will be saved to: {self.plot_dir}")
        logger.info(f"📈 Individual PR curves will be saved to: {self.pr_curves_dir}")
        logger.info(f"📉 Individual ROC curves will be saved to: {self.roc_curves_dir}")

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        cm       = confusion_matrix(y_true, y_pred)
        fig_size = max(16, len(class_names) * 0.6)
        plt.figure(figsize=(fig_size, fig_size * 0.9))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
        plt.title('Confusion Matrix (Counts)', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label',      fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'confusion_matrix.png', dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Confusion matrix saved (400 DPI)")

    def plot_normalized_confusion_matrix(self, y_true, y_pred, class_names):
        cm            = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig_size      = max(16, len(class_names) * 0.6)
        plt.figure(figsize=(fig_size, fig_size * 0.9))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, cbar_kws={'label': 'Percentage'},
                    linewidths=0.5, linecolor='gray')
        plt.title('Normalized Confusion Matrix (%)', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label',      fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'confusion_matrix_normalized.png', dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Normalized confusion matrix saved (400 DPI)")

    def plot_pr_curves_combined(self, y_true, y_probs, class_names):
        plt.figure(figsize=(14, 10))
        colors    = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
        ap_scores = []
        for i, class_name in enumerate(class_names):
            y_bin      = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_bin, y_probs[:, i])
            ap         = average_precision_score(y_bin, y_probs[:, i])
            ap_scores.append((class_name, ap))
            plt.plot(recall, precision, color=colors[i], lw=1.5, alpha=0.8,
                     label=f'{class_name} (AP={ap:.3f})')
        plt.xlabel('Recall',    fontsize=13, fontweight='bold')
        plt.ylabel('Precision', fontsize=13, fontweight='bold')
        plt.title('Precision-Recall Curves — All Classes', fontsize=16, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower center', fontsize=8, framealpha=0.9, ncol=2)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'pr_curves_combined.png', dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Combined PR curves saved (400 DPI)")
        mean_ap = np.mean([ap for _, ap in ap_scores])
        logger.info(f"  Mean Average Precision (mAP): {mean_ap:.4f}")
        return ap_scores, mean_ap

    def plot_pr_curves_individual(self, y_true, y_probs, class_names):
        logger.info("Generating individual PR curves...")
        ap_scores = []
        for i, class_name in enumerate(class_names):
            y_bin      = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_bin, y_probs[:, i])
            ap         = average_precision_score(y_bin, y_probs[:, i])
            ap_scores.append(ap)
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='#2E86AB', lw=3, label=f'AP = {ap:.3f}')
            plt.fill_between(recall, precision, alpha=0.3, color='#A23B72')
            plt.xlabel('Recall',    fontsize=13, fontweight='bold')
            plt.ylabel('Precision', fontsize=13, fontweight='bold')
            plt.title(f'Precision-Recall Curve: {class_name}', fontsize=15, fontweight='bold', pad=15)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.legend(loc='best', fontsize=12, framealpha=0.9)
            n_samples = int(np.sum(y_bin))
            plt.text(0.05, 0.05, f'Samples: {n_samples}\nAP: {ap:.4f}',
                     transform=plt.gca().transAxes, fontsize=11, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout()
            safe = class_name.replace('/', '_').replace('\\', '_')
            plt.savefig(self.pr_curves_dir / f'pr_curve_{safe}.png', dpi=300, bbox_inches='tight')
            plt.close()
        logger.info(f"✓ {len(class_names)} individual PR curves saved (300 DPI)")
        return ap_scores

    def plot_roc_curves_combined(self, y_true, y_probs, class_names):
        plt.figure(figsize=(14, 10))
        colors     = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
        auc_scores = []
        for i, class_name in enumerate(class_names):
            y_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
            roc_auc     = auc(fpr, tpr)
            auc_scores.append((class_name, roc_auc))
            plt.plot(fpr, tpr, color=colors[i], lw=1.5, alpha=0.8,
                     label=f'{class_name} (AUC={roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate',  fontsize=13, fontweight='bold')
        plt.title('ROC Curves — All Classes', fontsize=16, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower right', fontsize=8, framealpha=0.9, ncol=2)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'roc_curves_combined.png', dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Combined ROC curves saved (400 DPI)")
        mean_auc = np.mean([a for _, a in auc_scores])
        logger.info(f"  Mean AUC: {mean_auc:.4f}")
        return auc_scores, mean_auc

    def plot_roc_curves_individual(self, y_true, y_probs, class_names):
        logger.info("Generating individual ROC curves...")
        auc_scores = []
        for i, class_name in enumerate(class_names):
            y_bin       = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
            roc_auc     = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='#2E86AB', lw=3, label=f'AUC = {roc_auc:.3f}')
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            plt.fill_between(fpr, tpr, alpha=0.3, color='#A23B72')
            plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
            plt.ylabel('True Positive Rate',  fontsize=13, fontweight='bold')
            plt.title(f'ROC Curve: {class_name}', fontsize=15, fontweight='bold', pad=15)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.legend(loc='best', fontsize=12, framealpha=0.9)
            n_samples = int(np.sum(y_bin))
            plt.text(0.95, 0.05, f'Samples: {n_samples}\nAUC: {roc_auc:.4f}',
                     transform=plt.gca().transAxes, fontsize=11,
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout()
            safe = class_name.replace('/', '_').replace('\\', '_')
            plt.savefig(self.roc_curves_dir / f'roc_curve_{safe}.png', dpi=300, bbox_inches='tight')
            plt.close()
        logger.info(f"✓ {len(class_names)} individual ROC curves saved (300 DPI)")
        return auc_scores

    def plot_per_class_metrics(self, class_names, precisions, recalls, f1_scores, supports):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        x     = np.arange(len(class_names))
        width = 0.6

        for ax, values, label, color, title in [
            (axes[0, 0], precisions, 'Precision',           '#3498db', 'Precision per Class'),
            (axes[0, 1], recalls,    'Recall',               '#2ecc71', 'Recall per Class'),
            (axes[1, 0], f1_scores,  'F1-Score',             '#e74c3c', 'F1-Score per Class'),
            (axes[1, 1], supports,   'Number of Samples',    '#9b59b6', 'Support per Class'),
        ]:
            ax.bar(x, values, width, color=color, alpha=0.85)
            ax.set_ylabel(label, fontsize=13, fontweight='bold')
            ax.set_title(title,  fontsize=15, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            if label != 'Number of Samples':
                ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.axhline(y=np.mean(values), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(values):.3f}')
            ax.legend(fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(self.plot_dir / 'per_class_metrics.png', dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Per-class metrics plot saved (400 DPI)")

    def plot_top_k_accuracy(self, top_k_accs):
        plt.figure(figsize=(10, 6))
        k_values   = list(top_k_accs.keys())
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

    def plot_class_accuracy_comparison(self, y_true, y_pred, class_names):
        class_accuracies = []
        for i in range(len(class_names)):
            mask = (y_true == i)
            class_accuracies.append(
                float((y_pred[mask] == y_true[mask]).sum() / mask.sum()) if mask.sum() > 0 else 0.0
            )
        sorted_idx   = np.argsort(class_accuracies)[::-1]
        sorted_names = [class_names[i] for i in sorted_idx]
        sorted_accs  = [class_accuracies[i] for i in sorted_idx]
        plt.figure(figsize=(14, 10))
        colors = plt.cm.RdYlGn(np.array(sorted_accs))
        bars   = plt.barh(range(len(sorted_names)), sorted_accs, color=colors, alpha=0.85)
        plt.yticks(range(len(sorted_names)), sorted_names, fontsize=10)
        plt.xlabel('Accuracy', fontsize=13, fontweight='bold')
        plt.title('Per-Class Accuracy (Sorted)', fontsize=16, fontweight='bold', pad=15)
        plt.xlim([0, 1.05])
        plt.grid(True, alpha=0.3, axis='x', linestyle='--')
        for i, (bar, acc) in enumerate(zip(bars, sorted_accs)):
            plt.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'class_accuracy_comparison.png', dpi=400, bbox_inches='tight')
        plt.close()
        logger.info("✓ Class accuracy comparison saved (400 DPI)")

    def create_summary_report(self, metrics, config, save_path):
        fig = plt.figure(figsize=(16, 10))
        ax  = fig.add_subplot(111)
        ax.axis('off')
        summary_text = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                   3D CNN TESTING SUMMARY REPORT                       ║
╚═══════════════════════════════════════════════════════════════════════╝

📊 OVERALL PERFORMANCE
{'='*75}
  • Overall Accuracy:        {metrics['accuracy']*100:6.2f}%
  • Macro Avg Precision:     {metrics['macro_precision']*100:6.2f}%
  • Macro Avg Recall:        {metrics['macro_recall']*100:6.2f}%
  • Macro Avg F1-Score:      {metrics['macro_f1']*100:6.2f}%
  • Weighted Avg F1-Score:   {metrics['weighted_f1']*100:6.2f}%
  • Mean Average Precision:  {metrics.get('mean_ap', 0)*100:6.2f}%
  • Mean AUC (ROC):          {metrics.get('mean_auc', 0)*100:6.2f}%

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
  • Model Architecture:      Working3DCNN  (64→128→256→512)
  • Image Size:              {config.get('img_size', 112)}x{config.get('img_size', 112)}
  • Number of Frames:        {config.get('num_frames', 16)}
  • Dropout Rate:            {config.get('dropout', 0.35)}

🔍 BEST & WORST PERFORMING CLASSES
{'='*75}
  Best Class:  {metrics['best_class']['name']} (F1: {metrics['best_class']['f1']*100:.2f}%)
  Worst Class: {metrics['worst_class']['name']} (F1: {metrics['worst_class']['f1']*100:.2f}%)

  Checkpoint Epoch:   {metrics.get('ckpt_epoch', 'N/A')}
  Checkpoint Val Acc: {metrics.get('ckpt_val_acc', 'N/A')}

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


# ============================================================
# Model Tester  — key fix is in __init__ and load_model
# ============================================================
class ModelTester:
    def __init__(self, model_path, test_dir, config=None):
        self.model_path = Path(model_path)
        self.test_dir   = Path(test_dir)

        # Use CPU on Mac (MPS doesn't support 3D pooling operations)
        self.device = torch.device('cpu')
        logger.info("Using CPU (MPS doesn't support 3D pooling operations)")

        self.results_dir = Path('./../../output/3DCNN/3dcnn_final')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.results_dir / 'plots'
        self.plotter  = TestingPlotter(self.plot_dir)

        # ── FIX 1: Correct checkpoint extraction ─────────────────────────
        # The training code (v3 full logging) saves checkpoints as:
        #   torch.save({'epoch':..., 'model': state_dict, 'optimizer':...,
        #               'lr':..., 'tr_loss':..., 'tr_acc':...,
        #               'vl_loss':..., 'vl_acc':...}, path)
        #
        # So model weights live under the 'model' key.
        # The old test code checked for 'model_state_dict' / 'state_dict',
        # neither of which exists, so it fell through to treating the entire
        # checkpoint dict AS the state dict — causing the key mismatch error.
        # ──────────────────────────────────────────────────────────────────
        raw = torch.load(model_path, map_location=self.device)

        # Prefer user-supplied config; extract from checkpoint as fallback
        self.config = config if config else {}

        if isinstance(raw, dict):
            # ── Case A: our training checkpoint format ────────────────────
            if 'model' in raw:
                self.state_dict  = raw['model']
                self.ckpt_epoch  = raw.get('epoch',   'N/A')
                self.ckpt_val_acc = raw.get('vl_acc', 'N/A')
                # Merge any config keys saved inside checkpoint
                for k in ('dropout', 'num_frames', 'img_size', 'batch_size'):
                    if k in raw and k not in self.config:
                        self.config[k] = raw[k]
                logger.info(f"Loaded checkpoint from  : {model_path}")
                logger.info(f"Checkpoint epoch        : {self.ckpt_epoch}")
                logger.info(f"Checkpoint val accuracy : "
                            f"{self.ckpt_val_acc*100:.2f}%"
                            if isinstance(self.ckpt_val_acc, float)
                            else f"Checkpoint val accuracy : {self.ckpt_val_acc}")

            # ── Case B: other common formats ─────────────────────────────
            elif 'model_state_dict' in raw:
                self.state_dict   = raw['model_state_dict']
                self.ckpt_epoch   = raw.get('epoch',    'N/A')
                self.ckpt_val_acc = raw.get('val_accuracy', 'N/A')
                logger.info(f"Loaded checkpoint (model_state_dict key): {model_path}")
            elif 'state_dict' in raw:
                self.state_dict   = raw['state_dict']
                self.ckpt_epoch   = raw.get('epoch', 'N/A')
                self.ckpt_val_acc = 'N/A'
                logger.info(f"Loaded checkpoint (state_dict key): {model_path}")

            # ── Case C: the dict itself is already a state dict ───────────
            else:
                # Verify by checking for typical weight key patterns
                sample_keys = list(raw.keys())[:5]
                has_weight_keys = any('weight' in k or 'bias' in k for k in sample_keys)
                if has_weight_keys:
                    self.state_dict   = raw
                    self.ckpt_epoch   = 'N/A'
                    self.ckpt_val_acc = 'N/A'
                    logger.info(f"Loaded raw state dict from: {model_path}")
                else:
                    # Print available keys to help debug future issues
                    logger.warning(f"Unexpected checkpoint keys: {list(raw.keys())}")
                    logger.warning("Attempting to use checkpoint dict directly as state dict.")
                    self.state_dict   = raw
                    self.ckpt_epoch   = 'N/A'
                    self.ckpt_val_acc = 'N/A'
        else:
            # Rare: checkpoint is a bare tensor (shouldn't happen but handle gracefully)
            raise TypeError(f"Unexpected checkpoint type: {type(raw)}. "
                            f"Expected a dict.")

    def load_model(self, num_classes):
        """
        Load the trained model.
        FIX 2: Uses Working3DCNN to match the training architecture.
        """
        model = Working3DCNN(
            num_classes=num_classes,
            dropout=self.config.get('dropout', 0.35)
        )
        model.load_state_dict(self.state_dict)
        model = model.to(self.device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: Working3DCNN  ({total_params:,} parameters)")
        return model

    def test(self):
        logger.info("\n" + "=" * 80)
        logger.info("STARTING MODEL TESTING")
        logger.info("=" * 80)

        test_dataset = NPZDirectoryDataset(
            self.test_dir,
            num_frames=self.config.get('num_frames', 16),
            img_size=self.config.get('img_size', 112)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        class_names = test_dataset.classes
        num_classes = len(class_names)
        model       = self.load_model(num_classes)

        all_preds, all_labels, all_probs, all_filenames = [], [], [], []

        logger.info("\nRunning inference...")
        with torch.no_grad():
            for inputs, labels, filenames in tqdm(test_loader, desc="Testing"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                probs   = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_filenames.extend(filenames)

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)

        logger.info("\nCalculating metrics...")
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs, class_names)

        # Inject checkpoint metadata for summary report
        metrics['ckpt_epoch']   = self.ckpt_epoch
        metrics['ckpt_val_acc'] = (
            f"{self.ckpt_val_acc*100:.2f}%" if isinstance(self.ckpt_val_acc, float)
            else str(self.ckpt_val_acc)
        )

        logger.info("\nGenerating plots...")
        self.plotter.plot_confusion_matrix(all_labels, all_preds, class_names)
        self.plotter.plot_normalized_confusion_matrix(all_labels, all_preds, class_names)

        logger.info("\nGenerating PR curves...")
        ap_scores, mean_ap   = self.plotter.plot_pr_curves_combined(all_labels, all_probs, class_names)
        _                    = self.plotter.plot_pr_curves_individual(all_labels, all_probs, class_names)

        logger.info("\nGenerating ROC curves...")
        auc_scores, mean_auc = self.plotter.plot_roc_curves_combined(all_labels, all_probs, class_names)
        _                    = self.plotter.plot_roc_curves_individual(all_labels, all_probs, class_names)

        metrics['mean_ap']       = mean_ap
        metrics['mean_auc']      = mean_auc
        metrics['per_class_ap']  = {class_names[i]: ap_scores[i][1]  for i in range(len(class_names))}
        metrics['per_class_auc'] = {class_names[i]: auc_scores[i][1] for i in range(len(class_names))}

        report     = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        precisions = [report[cn]['precision'] for cn in class_names]
        recalls    = [report[cn]['recall']    for cn in class_names]
        f1_scores  = [report[cn]['f1-score']  for cn in class_names]
        supports   = [report[cn]['support']   for cn in class_names]
        self.plotter.plot_per_class_metrics(class_names, precisions, recalls, f1_scores, supports)
        self.plotter.plot_class_accuracy_comparison(all_labels, all_preds, class_names)

        top_k_accs = {}
        for k in [1, 3, 5]:
            if k <= num_classes:
                top_k_accs[k] = self.calculate_top_k_accuracy(all_probs, all_labels, k)
        if top_k_accs:
            self.plotter.plot_top_k_accuracy(top_k_accs)

        metrics.update({
            'top_1_acc': top_k_accs.get(1, 0),
            'top_3_acc': top_k_accs.get(3, 0),
            'top_5_acc': top_k_accs.get(5, 0)
        })

        self.plotter.create_summary_report(
            metrics, self.config, self.plot_dir / 'summary_report.png')

        self.save_results(metrics, class_names, all_labels, all_preds, all_probs, all_filenames)
        self.print_summary(metrics)

        logger.info("\n" + "=" * 80)
        logger.info("TESTING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"All results saved to: {self.results_dir}")
        return metrics

    def calculate_metrics(self, y_true, y_pred, y_probs, class_names):
        report         = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        class_f1_scores = {cn: report[cn]['f1-score'] for cn in class_names}
        best_class     = max(class_f1_scores.items(), key=lambda x: x[1])
        worst_class    = min(class_f1_scores.items(), key=lambda x: x[1])
        return {
            'accuracy':         accuracy_score(y_true, y_pred),
            'macro_precision':  precision_score(y_true, y_pred, average='macro'),
            'macro_recall':     recall_score(y_true, y_pred, average='macro'),
            'macro_f1':         f1_score(y_true, y_pred, average='macro'),
            'weighted_f1':      f1_score(y_true, y_pred, average='weighted'),
            'num_classes':      len(class_names),
            'total_samples':    len(y_true),
            'samples_per_class': len(y_true) / len(class_names),
            'best_class':       {'name': best_class[0],  'f1': best_class[1]},
            'worst_class':      {'name': worst_class[0], 'f1': worst_class[1]},
            'class_f1_scores':  class_f1_scores
        }

    def calculate_top_k_accuracy(self, y_probs, y_true, k):
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        correct = sum(1 for i, label in enumerate(y_true) if label in top_k_preds[i])
        return correct / len(y_true)

    def save_results(self, metrics, class_names, y_true, y_pred, y_probs, filenames):
        # JSON metrics
        def _serial(v):
            if isinstance(v, (np.floating, float)): return float(v)
            if isinstance(v, (np.integer, int)):    return int(v)
            return v

        skip = {'best_class', 'worst_class', 'class_f1_scores', 'per_class_ap', 'per_class_auc'}
        m_out = {k: _serial(v) for k, v in metrics.items() if k not in skip}
        m_out['best_class']       = {'name': metrics['best_class']['name'],
                                      'f1':   float(metrics['best_class']['f1'])}
        m_out['worst_class']      = {'name': metrics['worst_class']['name'],
                                      'f1':   float(metrics['worst_class']['f1'])}
        m_out['class_f1_scores']  = {k: float(v) for k, v in metrics['class_f1_scores'].items()}
        if 'per_class_ap'  in metrics:
            m_out['per_class_ap']  = {k: float(v) for k, v in metrics['per_class_ap'].items()}
        if 'per_class_auc' in metrics:
            m_out['per_class_auc'] = {k: float(v) for k, v in metrics['per_class_auc'].items()}

        with open(self.results_dir / 'metrics.json', 'w') as f:
            json.dump(m_out, f, indent=4)
        logger.info("✓ Metrics saved to metrics.json")

        # Classification report
        report_str = classification_report(y_true, y_pred, target_names=class_names)
        with open(self.results_dir / 'classification_report.txt', 'w') as f:
            f.write("CLASSIFICATION REPORT — Working3DCNN\n" + "=" * 80 + "\n\n")
            f.write(report_str)
            if 'mean_ap'  in metrics: f.write(f"\n\nMean AP  (mAP): {metrics['mean_ap']:.4f}\n")
            if 'mean_auc' in metrics: f.write(f"Mean AUC (ROC): {metrics['mean_auc']:.4f}\n")
        logger.info("✓ Classification report saved")

        # Detailed predictions CSV
        results_df = pd.DataFrame({
            'filename':        filenames,
            'true_label':      [class_names[l] for l in y_true],
            'predicted_label': [class_names[p] for p in y_pred],
            'correct':         y_true == y_pred,
            'confidence':      np.max(y_probs, axis=1)
        })
        for i, cn in enumerate(class_names):
            results_df[f'prob_{cn}'] = y_probs[:, i]
        results_df.to_csv(self.results_dir / 'detailed_predictions.csv', index=False)
        logger.info("✓ Detailed predictions saved to CSV")

        # Misclassified samples
        misclf = results_df[~results_df['correct']].sort_values('confidence', ascending=False)
        misclf.to_csv(self.results_dir / 'misclassified_samples.csv', index=False)
        logger.info(f"✓ Found {len(misclf)} misclassified samples")

        # Confusion matrix CSV
        cm    = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(self.results_dir / 'confusion_matrix.csv')
        logger.info("✓ Confusion matrix saved to CSV")

        # AP scores CSV
        if 'per_class_ap' in metrics:
            ap_df = pd.DataFrame([{'class': cn, 'average_precision': ap}
                                   for cn, ap in metrics['per_class_ap'].items()])
            ap_df.sort_values('average_precision', ascending=False).to_csv(
                self.results_dir / 'average_precision_scores.csv', index=False)
            logger.info("✓ Average Precision scores saved to CSV")

        # AUC scores CSV
        if 'per_class_auc' in metrics:
            auc_df = pd.DataFrame([{'class': cn, 'auc': av}
                                    for cn, av in metrics['per_class_auc'].items()])
            auc_df.sort_values('auc', ascending=False).to_csv(
                self.results_dir / 'auc_scores.csv', index=False)
            logger.info("✓ AUC scores saved to CSV")

    def print_summary(self, metrics):
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY — Working3DCNN")
        print("=" * 80)
        print(f"\n📊 Overall Accuracy : {metrics['accuracy']*100:.2f}%")
        print(f"📈 Macro F1-Score   : {metrics['macro_f1']*100:.2f}%")
        print(f"📉 Weighted F1      : {metrics['weighted_f1']*100:.2f}%")
        if 'mean_ap'  in metrics: print(f"🎯 Mean AP (mAP)   : {metrics['mean_ap']*100:.2f}%")
        if 'mean_auc' in metrics: print(f"📐 Mean AUC (ROC)  : {metrics['mean_auc']*100:.2f}%")
        print(f"\n✅ Best Class  : {metrics['best_class']['name']}  "
              f"(F1: {metrics['best_class']['f1']*100:.2f}%)")
        print(f"❌ Worst Class : {metrics['worst_class']['name']}  "
              f"(F1: {metrics['worst_class']['f1']*100:.2f}%)")
        print(f"\n📁 Total Samples   : {metrics['total_samples']}")
        print(f"🏷️  Classes          : {metrics['num_classes']}")
        if 'top_1_acc' in metrics:
            print(f"\n🎯 Top-1 Accuracy  : {metrics['top_1_acc']*100:.2f}%")
            if metrics.get('top_3_acc', 0): print(f"🎯 Top-3 Accuracy  : {metrics['top_3_acc']*100:.2f}%")
            if metrics.get('top_5_acc', 0): print(f"🎯 Top-5 Accuracy  : {metrics['top_5_acc']*100:.2f}%")
        print(f"\n🔖 Checkpoint epoch    : {metrics.get('ckpt_epoch', 'N/A')}")
        print(f"🔖 Checkpoint val acc  : {metrics.get('ckpt_val_acc', 'N/A')}")
        print("=" * 80 + "\n")


# ============================================================
# Main
# ============================================================
def main():
    # ===== UPDATE THESE PATHS =====
    model_path = './../../kaggle/training_output_3dcnn/final_new/checkpoint_ep20.pth'
    test_dir   = './../../videos_directory/npz_lightweight_videos_without_cropping_splitted_dataset/test'
    # ==============================

    logger.info("\n" + "=" * 80)
    logger.info("3D CNN MODEL TESTING FOR MAC")
    logger.info("=" * 80)
    logger.info(f"\nModel    : {model_path}")
    logger.info(f"Test Data: {test_dir}")
    logger.info("=" * 80 + "\n")

    if not Path(model_path).exists():
        logger.error(f"❌ Model file not found: {model_path}")
        return
    if not Path(test_dir).exists():
        logger.error(f"❌ Test directory not found: {test_dir}")
        return

    try:
        tester  = ModelTester(model_path, test_dir)
        metrics = tester.test()

        logger.info("\n✅ Testing completed successfully!")
        logger.info(f"📁 Results saved to: {tester.results_dir}")
        logger.info("\n📊 Generated files:")
        for f in [
            "metrics.json",
            "classification_report.txt",
            "detailed_predictions.csv",
            "misclassified_samples.csv",
            "confusion_matrix.csv",
            "average_precision_scores.csv",
            "auc_scores.csv",
            "plots/confusion_matrix.png (400 DPI)",
            "plots/confusion_matrix_normalized.png (400 DPI)",
            "plots/per_class_metrics.png (400 DPI)",
            "plots/top_k_accuracy.png (400 DPI)",
            "plots/pr_curves_combined.png (400 DPI)",
            "plots/pr_curves_individual/*.png (300 DPI)",
            "plots/roc_curves_combined.png (400 DPI)",
            "plots/roc_curves_individual/*.png (300 DPI)",
            "plots/class_accuracy_comparison.png (400 DPI)",
            "plots/summary_report.png (400 DPI)",
        ]:
            logger.info(f"   • {f}")

    except Exception as e:
        logger.error(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()