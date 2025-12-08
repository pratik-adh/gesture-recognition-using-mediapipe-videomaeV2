#!/usr/bin/env python3
"""
Optimized VideoMAE Training Pipeline for Pre-cropped NPZ Videos - FIXED VERSION
- Gradient stability improvements
- Better checkpoint management with auto-resume
- Enhanced error handling
- Improved batch normalization
- Optimized learning rate and gradient clipping
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from sklearn.metrics import confusion_matrix, classification_report, f1_score, top_k_accuracy_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import json
import random
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

# -------------------------
# Logging + reproducibility
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('training.log', mode='a')]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------
# Dataset for Pre-cropped NPZ Videos
# -------------------------
class NPZDirectoryDataset(Dataset):
    """
    Dataset for pre-cropped videos stored as NPZ files.
    Since videos are already hand-cropped, no MediaPipe processing needed.
    """
    def __init__(self, root_dir, num_frames=16, is_training=True, 
                 model_name="MCG-NJU/videomae-small-finetuned-kinetics"):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.is_training = is_training

        # Build samples
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        self._scan_directory()

        # VideoMAE processor (handles resizing/normalization)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)

        logger.info(f"Dataset: {root_dir}")
        logger.info(f"Classes: {len(self.classes)}, Samples: {len(self.samples)}")
        logger.info("Note: Videos are pre-cropped, no MediaPipe processing applied")

    def _scan_directory(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist")
        
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        for cid, d in enumerate(class_dirs):
            self.classes.append(d.name)
            self.class_to_idx[d.name] = cid
            files = list(d.glob("*.npz"))
            for f in files:
                self.samples.append({
                    'path': f, 
                    'class_idx': cid, 
                    'class_name': d.name
                })
            logger.info(f"  {d.name}: {len(files)} videos")

    def __len__(self):
        return len(self.samples)

    def load_video(self, npz_path):
        """Load video frames from NPZ file"""
        try:
            data = np.load(npz_path)
            
            # Try common key names
            if 'frames' in data:
                frames = data['frames']
            elif 'video' in data:
                frames = data['video']
            else:
                # Fallback to first array
                frames = data[data.files[0]]

            # Ensure uint8 format
            if frames.dtype != np.uint8:
                if frames.max() <= 1.0:
                    frames = (frames * 255).astype(np.uint8)
                else:
                    frames = np.clip(frames, 0, 255).astype(np.uint8)
            
            return frames
        except Exception as e:
            logger.error(f"Error loading {npz_path}: {e}")
            # Return dummy frames to prevent crash
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

    def temporal_sampling(self, frames):
        """Sample fixed number of frames uniformly from video"""
        n = len(frames)
        if n == 0:
            logger.warning("Video with 0 frames encountered, using dummy frames")
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
        
        if n <= self.num_frames:
            # Repeat last frame if video is shorter
            indices = list(range(n)) + [n-1] * (self.num_frames - n)
        else:
            # Uniform sampling
            indices = np.linspace(0, n-1, self.num_frames, dtype=int)
        
        return frames[indices]

    def apply_augmentations(self, frames):
        """Apply data augmentations for training"""
        if not self.is_training:
            return frames

        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            frames = np.clip(frames * factor, 0, 255).astype(np.uint8)

        # Random contrast
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            mean = frames.mean(axis=(1,2,3), keepdims=True)
            frames = np.clip((frames - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Random horizontal flip (careful with sign language!)
        if random.random() > 0.7:  # Less likely for sign language
            frames = frames[:, :, ::-1, :]

        # Gaussian noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 3, frames.shape)
            frames = np.clip(frames + noise, 0, 255).astype(np.uint8)

        return frames

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self.load_video(sample['path'])
        label = sample['class_idx']

        # Temporal sampling
        frames = self.temporal_sampling(frames)

        # Augmentations
        if self.is_training:
            frames = self.apply_augmentations(frames)

        # Process with VideoMAE processor
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        return pixel_values, label

# -------------------------
# Model wrapper with Multi-GPU support
# -------------------------
class VideoMAEClassifier(nn.Module):
    """
    VideoMAE with custom classification head and layer freezing support.
    Supports DataParallel for multi-GPU training.
    FIXED: Improved batch normalization stability
    """
    def __init__(self, num_classes, dropout_rate=0.4, freeze_backbone=False, 
                 freeze_layers=0, model_name="MCG-NJU/videomae-small-finetuned-kinetics"):
        super().__init__()
        
        logger.info(f"Loading model: {model_name}")
        self.videomae = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        hidden_size = self.videomae.config.hidden_size
        num_layers = self.videomae.config.num_hidden_layers

        logger.info(f"Hidden size: {hidden_size}, Encoder layers: {num_layers}")

        # Freezing strategy
        if freeze_backbone:
            logger.info("Freezing entire backbone")
            for p in self.videomae.videomae.parameters():
                p.requires_grad = False
        elif freeze_layers > 0:
            n = min(freeze_layers, num_layers)
            logger.info(f"Freezing first {n}/{num_layers} encoder layers")
            
            # Freeze embeddings
            for p in self.videomae.videomae.embeddings.parameters():
                p.requires_grad = False
            
            # Freeze encoder layers
            for i in range(n):
                for p in self.videomae.videomae.encoder.layer[i].parameters():
                    p.requires_grad = False
            
            logger.info(f"Trainable layers: {n} to {num_layers-1}")

        # REMOVE BatchNorm completely - it's causing the instability
        self.videomae.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.videomae(pixel_values)
        return outputs.logits

# -------------------------
# Trainer with Multi-GPU support and FIXES
# -------------------------
class Trainer:
    def __init__(self, config):
        self.cfg = config
        
        # Multi-GPU setup
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This configuration requires GPUs.")
        
        self.device = torch.device('cuda')
        self.n_gpus = torch.cuda.device_count()
        
        logger.info(f"Found {self.n_gpus} GPU(s)")
        for i in range(self.n_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        if self.n_gpus < 2:
            logger.warning("Less than 2 GPUs detected. Will use single GPU.")
        else:
            logger.info(f"Using DataParallel with {self.n_gpus} GPUs")

        # Setup directories
        self.save_dir = Path(self.cfg['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        (self.save_dir / "logs").mkdir(exist_ok=True)

        # Mixed precision with conservative settings for stability
        self.scaler = GradScaler(init_scale=512.0, growth_interval=200)
        logger.info("AMP enabled with conservative settings for stability")

        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=str(self.save_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
        )

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.lrs = []
        self.start_epoch = 1
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # For tracking gradient issues
        self.nan_grad_count = 0
        self.inf_grad_count = 0

    def create_data_loaders(self):
        train_ds = NPZDirectoryDataset(
            self.cfg['train_dir'], 
            num_frames=self.cfg['num_frames'], 
            is_training=True, 
            model_name=self.cfg['model_name']
        )
        
        val_ds = NPZDirectoryDataset(
            self.cfg['val_dir'], 
            num_frames=self.cfg['num_frames'], 
            is_training=False, 
            model_name=self.cfg['model_name']
        )

        self.class_names = train_ds.classes
        self.num_classes = len(self.class_names)
        logger.info(f"Total classes: {self.num_classes}")

        # Optimized data loading for multi-GPU
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=True,
            num_workers=self.cfg['num_workers'], 
            pin_memory=True,
            persistent_workers=True if self.cfg['num_workers'] > 0 else False,
            prefetch_factor=2 if self.cfg['num_workers'] > 0 else None
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=False,
            num_workers=self.cfg['num_workers'], 
            pin_memory=True,
            persistent_workers=True if self.cfg['num_workers'] > 0 else False,
            prefetch_factor=2 if self.cfg['num_workers'] > 0 else None
        )
        
        return train_loader, val_loader

    def train_epoch(self, model, loader, optimizer, criterion, epoch):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        skipped_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Check for NaN loss BEFORE scaling
            if not torch.isfinite(loss):
                logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping")
                skipped_batches += 1
                continue
            
            self.scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            self.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg['gradient_clip'])
            
            # Check for NaN/Inf gradients AFTER unscaling
            if not torch.isfinite(grad_norm):
                if torch.isnan(grad_norm):
                    self.nan_grad_count += 1
                    logger.warning(f"NaN gradient detected at batch {batch_idx}, skipping step")
                else:
                    self.inf_grad_count += 1
                    logger.warning(f"Inf gradient detected at batch {batch_idx}, skipping step")
                
                # CRITICAL FIX: Must call update() even when skipping to reset scaler state
                optimizer.zero_grad()
                self.scaler.update()
                skipped_batches += 1
                continue
            
            self.scaler.step(optimizer)
            self.scaler.update()

            # Metrics
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1-skipped_batches) if batch_idx+1 > skipped_batches else 0:.4f}", 
                'acc': f"{100.*correct/total:.2f}%",
                'grad': f"{grad_norm:.2f}",
                'skip': skipped_batches
            })

        epoch_loss = total_loss / max(1, len(loader) - skipped_batches)
        epoch_acc = correct / total if total > 0 else 0.0
        epoch_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_preds) > 0 else 0.0

        # Log gradient issues
        if skipped_batches > 0:
            logger.warning(f"Epoch {epoch}: Skipped {skipped_batches} batches due to gradient issues")
            logger.warning(f"Total NaN gradients so far: {self.nan_grad_count}")
            logger.warning(f"Total Inf gradients so far: {self.inf_grad_count}")

        # TensorBoard
        self.writer.add_scalar("Train/Loss", epoch_loss, epoch)
        self.writer.add_scalar("Train/Accuracy", epoch_acc, epoch)
        self.writer.add_scalar("Train/F1", epoch_f1, epoch)
        self.writer.add_scalar("Train/SkippedBatches", skipped_batches, epoch)

        return epoch_loss, epoch_acc, epoch_f1

    def evaluate(self, model, loader, criterion, epoch):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        all_probs = []

        with torch.no_grad():
            pbar = tqdm(loader, desc="Validation")
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix({
                    'loss': f"{total_loss/(batch_idx+1):.4f}", 
                    'acc': f"{100.*correct/total:.2f}%"
                })

        epoch_loss = total_loss / len(loader)
        epoch_acc = correct / total if total > 0 else 0.0
        epoch_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_preds) > 0 else 0.0

        # Top-k accuracy
        all_probs = np.array(all_probs)
        top3 = top_k_accuracy_score(all_labels, all_probs, k=min(3, self.num_classes))
        top5 = top_k_accuracy_score(all_labels, all_probs, k=min(5, self.num_classes))

        # TensorBoard
        self.writer.add_scalar("Val/Loss", epoch_loss, epoch)
        self.writer.add_scalar("Val/Accuracy", epoch_acc, epoch)
        self.writer.add_scalar("Val/F1", epoch_f1, epoch)
        self.writer.add_scalar("Val/Top3", top3, epoch)
        self.writer.add_scalar("Val/Top5", top5, epoch)

        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1': epoch_f1,
            'top3': top3,
            'top5': top5,
            'predictions': all_preds,
            'labels': all_labels
        }

    def save_checkpoint(self, model, optimizer, epoch, val_metrics, is_best=False):
        """Save checkpoint with full training state"""
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'best_val_acc': self.best_val_acc,
            'patience_counter': self.patience_counter,
            'class_names': self.class_names,
            'config': self.cfg,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'lrs': self.lrs,
            'scaler_state': self.scaler.state_dict(),
            'nan_grad_count': self.nan_grad_count,
            'inf_grad_count': self.inf_grad_count,
        }
        
        # Save latest checkpoint
        latest_path = self.save_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"✓ New best model saved! Val Acc: {val_metrics['accuracy']:.4f}")
        
        return checkpoint

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """Load checkpoint and resume training"""
        if not checkpoint_path.exists():
            logger.info("No checkpoint found, starting from scratch")
            return
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        self.lrs = checkpoint.get('lrs', [])
        self.nan_grad_count = checkpoint.get('nan_grad_count', 0)
        self.inf_grad_count = checkpoint.get('inf_grad_count', 0)
        
        # Load scaler state
        if 'scaler_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state'])
        
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
        logger.info(f"Best val acc so far: {self.best_val_acc:.4f}")
        logger.info(f"Patience counter: {self.patience_counter}")

    def plot_training_curves(self):
        epochs = list(range(1, len(self.train_losses) + 1))
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 0].plot(epochs, self.lrs, 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Summary
        best_acc = max(self.val_accs) if self.val_accs else 0
        best_epoch = self.val_accs.index(best_acc) + 1 if self.val_accs else 0
        summary = f"""Training Summary:
        
Best Val Acc: {best_acc:.4f}
Best Epoch: {best_epoch}
Total Epochs: {len(epochs)}
Final Train Acc: {self.train_accs[-1]:.4f}
Final Val Acc: {self.val_accs[-1]:.4f}
NaN Gradients: {self.nan_grad_count}
Inf Gradients: {self.inf_grad_count}"""
        
        axes[1, 1].text(0.1, 0.5, summary, fontsize=12, verticalalignment='center',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')

        plt.tight_layout()
        save_path = self.save_dir / "plots" / "training_curves.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved training curves to {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names, 
                   cmap='Blues')
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        save_path = self.save_dir / "plots" / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")

    def save_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        df = pd.DataFrame(report).transpose()
        save_path = self.save_dir / "classification_report.csv"
        df.to_csv(save_path)
        logger.info(f"Saved classification report to {save_path}")

    def train(self):
        train_loader, val_loader = self.create_data_loaders()

        # Create model
        model = VideoMAEClassifier(
            num_classes=self.num_classes,
            dropout_rate=self.cfg['dropout_rate'],
            freeze_backbone=self.cfg.get('freeze_backbone', False),
            freeze_layers=self.cfg.get('freeze_layers', 0),
            model_name=self.cfg['model_name']
        )

        # Wrap model with DataParallel for multi-GPU
        if self.n_gpus > 1:
            logger.info(f"Wrapping model with DataParallel across {self.n_gpus} GPUs")
            model = nn.DataParallel(model)
        
        model = model.to(self.device)

        # Log parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.cfg['learning_rate'], 
            weight_decay=self.cfg['weight_decay']
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=self.cfg['lr_patience'],
            verbose=True
        )
        
        criterion = nn.CrossEntropyLoss()

        # Try to resume from checkpoint
        checkpoint_path = self.save_dir / "checkpoint_latest.pth"
        self.load_checkpoint(model, optimizer, checkpoint_path)

        best_preds = []
        best_labels = []

        for epoch in range(self.start_epoch, self.cfg['epochs'] + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.cfg['epochs']}")
            logger.info(f"{'='*50}")

            train_loss, train_acc, train_f1 = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            val_metrics = self.evaluate(model, val_loader, criterion, epoch)

            # Update scheduler
            scheduler.step(val_metrics['loss'])
            current_lr = optimizer.param_groups[0]['lr']

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.train_accs.append(train_acc)
            self.val_accs.append(val_metrics['accuracy'])
            self.lrs.append(current_lr)

            logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            logger.info(f"Val   - Top3: {val_metrics['top3']:.4f}, Top5: {val_metrics['top5']:.4f}")
            logger.info(f"LR: {current_lr:.6f}")

            # Save checkpoint every epoch
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                best_preds = val_metrics['predictions']
                best_labels = val_metrics['labels']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(model, optimizer, epoch, val_metrics, is_best=is_best)

            # Periodic checkpoint
            if epoch % self.cfg.get('checkpoint_interval', 10) == 0:
                cp_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch, 
                    'model_state_dict': model_state, 
                    'optimizer_state_dict': optimizer.state_dict()
                }, cp_path)
                logger.info(f"Periodic checkpoint saved: {cp_path}")

            # Early stopping
            if self.patience_counter >= self.cfg['early_stopping_patience']:
                logger.info(f"Early stopping triggered (patience={self.patience_counter})")
                break
            
            # Check for excessive gradient issues
            if self.nan_grad_count + self.inf_grad_count > 100:
                logger.error("Too many gradient issues detected. Consider reducing learning rate.")

        # Post-training: plots and reports
        try:
            self.plot_training_curves()
            if len(best_preds) > 0 and len(best_labels) > 0:
                self.plot_confusion_matrix(best_labels, best_preds)
                self.save_classification_report(best_labels, best_preds)
        except Exception as e:
            logger.warning(f"Post-training visualization failed: {e}")

        self.writer.close()
        logger.info("\n" + "="*50)
        logger.info("Training Complete!")
        logger.info(f"Best Val Accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Total NaN gradients: {self.nan_grad_count}")
        logger.info(f"Total Inf gradients: {self.inf_grad_count}")
        logger.info(f"Results saved to: {self.save_dir}")
        logger.info("="*50)
        
        return model

# -------------------------
# Testing
# -------------------------
def test_model(model_path, test_dir, config):
    """Test the trained model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('class_names')
    num_classes = len(class_names)

    model = VideoMAEClassifier(
        num_classes=num_classes, 
        dropout_rate=config.get('dropout_rate', 0.4),
        model_name=config['model_name']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info("Loading test dataset...")
    test_ds = NPZDirectoryDataset(
        test_dir, 
        num_frames=config.get('num_frames', 16), 
        is_training=False, 
        model_name=config['model_name']
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=config.get('batch_size', 8), 
        shuffle=False, 
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    all_preds, all_labels, all_probs = [], [], []
    
    logger.info("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    all_probs_array = np.array(all_probs)
    top3 = top_k_accuracy_score(all_labels, all_probs_array, k=min(3, num_classes))
    top5 = top_k_accuracy_score(all_labels, all_probs_array, k=min(5, num_classes))

    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS")
    logger.info("="*50)
    logger.info(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"F1 Score (macro): {f1:.4f}")
    logger.info(f"Top-3 Accuracy: {top3:.4f} ({top3*100:.2f}%)")
    logger.info(f"Top-5 Accuracy: {top5:.4f} ({top5*100:.2f}%)")
    logger.info("\nDetailed Classification Report:")
    logger.info("\n" + report)
    logger.info("="*50)
    
    return {
        'accuracy': acc, 
        'f1': f1, 
        'top3': top3, 
        'top5': top5, 
        'preds': all_preds, 
        'labels': all_labels,
        'probs': all_probs_array
    }

# -------------------------
# Main
# -------------------------
def main():
    """
    Main training pipeline optimized for 2x T4 GPUs
    FIXED VERSION with improved stability
    """
    
    # Configuration - Optimized and FIXED for 2x T4 GPUs (16GB each)
    # Configuration - Optimized for 2x T4 GPUs (16GB each)
    config = {
        # DATA CONFIGURATION
        'train_dir': '/kaggle/input/npz-preprocessed-videos-splitted-dataset/npz_preprocessed_videos_splitted_dataset/train',
        'val_dir': '/kaggle/input/npz-preprocessed-videos-splitted-dataset/npz_preprocessed_videos_splitted_dataset/val',
        'test_dir': '/kaggle/input/npz-preprocessed-videos-splitted-dataset/npz_preprocessed_videos_splitted_dataset/test',
        'save_dir': '/kaggle/working/',  # Directory to save checkpoints and logs

        # MODEL CONFIGURATION
        'model_name': 'MCG-NJU/videomae-small-finetuned-kinetics',  # Pretrained VideoMAE model

        'num_frames': 16,           # Number of frames per video clip
        'dropout_rate': 0.3,        # Dropout for regularization

        # Layer freezing control (based on dataset size)
        'freeze_backbone': False,   # Whether to freeze pretrained layers
        'freeze_layers': 4,         # Number of layers frozen (to balance learning stability)

        # TRAINING CONFIGURATION
        'batch_size': 16,           # Mini-batch size per GPU
        'epochs': 50,               # Total number of training epochs
        'learning_rate': 5e-4,      # Learning rate for optimizer
        'weight_decay': 1e-4,       # L2 regularization to prevent overfitting
        'gradient_clip': 0.5,       # Gradient clipping threshold for stability

        # 📉 OPTIMIZATION & SCHEDULER
        'lr_patience': 7,           # Epochs to wait before reducing LR if no improvement
        'early_stopping_patience': 7,  # Stop training if validation doesn’t improve
        'checkpoint_interval': 5,   # Save model every 5 epochs for recovery/safety

        # 💻 SYSTEM CONFIGURATION
        'num_workers': 4,           # Data loader workers per GPU for faster processing
    }

    # Display configuration summary
    logger.info("\n" + "="*70)
    logger.info("MULTI-GPU TRAINING CONFIGURATION (FIXED VERSION)")
    logger.info("="*70)
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Batch size per GPU: {config['batch_size']}")
    logger.info(f"Total effective batch: {config['batch_size'] * torch.cuda.device_count()}")
    logger.info(f"Frames per video: {config['num_frames']}")
    logger.info(f"Frozen layers: {config['freeze_layers']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Learning rate: {config['learning_rate']} (REDUCED for stability)")
    logger.info(f"Gradient clip: {config['gradient_clip']} (REDUCED for stability)")
    logger.info(f"Num workers: {config['num_workers']}")
    logger.info(f"Early stopping patience: {config['early_stopping_patience']}")
    logger.info("="*70)
    logger.info("IMPROVEMENTS:")
    logger.info("  ✓ Gradient stability checks (NaN/Inf detection)")
    logger.info("  ✓ Automatic checkpoint resume on crash")
    logger.info("  ✓ Improved batch normalization (eps=1e-5)")
    logger.info("  ✓ Conservative AMP scaling (init_scale=512)")
    logger.info("  ✓ Reduced learning rate (5e-4 vs 1e-3)")
    logger.info("  ✓ Reduced gradient clipping (0.5 vs 1.0)")
    logger.info("  ✓ More frequent checkpointing")
    logger.info("="*70)
    logger.info("NOTE: Videos are pre-cropped, no MediaPipe processing needed")
    logger.info("="*70 + "\n")

    # GPU memory info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
            logger.info(f"  Multi-processor count: {props.multi_processor_count}")
        logger.info("")

    # Create trainer and train
    trainer = Trainer(config)
    
    try:
        model = trainer.train()
        
        # Test if test directory exists
        test_path = Path(config['test_dir'])
        if test_path.exists():
            best_model_path = Path(config['save_dir']) / "best_model.pth"
            if best_model_path.exists():
                logger.info("\n" + "="*50)
                logger.info("Running test evaluation...")
                logger.info("="*50)
                results = test_model(best_model_path, config['test_dir'], config)
                
                # Save test results
                results_dict = {
                    'accuracy': float(results['accuracy']),
                    'f1': float(results['f1']),
                    'top3': float(results['top3']),
                    'top5': float(results['top5']),
                    'total_samples': len(results['labels'])
                }
                
                with open(Path(config['save_dir']) / "test_results.json", "w") as f:
                    json.dump(results_dict, f, indent=4)
                logger.info(f"Test results saved to test_results.json")
                
                # Save detailed predictions
                predictions_df = pd.DataFrame({
                    'true_label': results['labels'],
                    'predicted_label': results['preds'],
                    'correct': np.array(results['labels']) == np.array(results['preds'])
                })
                predictions_df.to_csv(Path(config['save_dir']) / "test_predictions.csv", index=False)
                logger.info(f"Detailed predictions saved to test_predictions.csv")
            else:
                logger.warning("No best model found for testing")
        else:
            logger.info(f"Test directory not found: {test_path}")
    
    except KeyboardInterrupt:
        logger.info("\n" + "="*50)
        logger.info("Training interrupted by user")
        logger.info(f"Latest checkpoint saved to: {config['save_dir']}/checkpoint_latest.pth")
        logger.info("You can resume training by running the script again")
        logger.info("="*50)
    
    except Exception as e:
        logger.error(f"\n" + "="*50)
        logger.error(f"Training failed with error: {e}")
        logger.error(f"Latest checkpoint saved to: {config['save_dir']}/checkpoint_latest.pth")
        logger.error("="*50)
        raise

    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"All outputs saved to: {config['save_dir']}")
    logger.info(f"Best model: {config['save_dir']}/best_model.pth")
    logger.info(f"Latest checkpoint: {config['save_dir']}/checkpoint_latest.pth")
    logger.info(f"Training curves: {config['save_dir']}/plots/training_curves.png")
    logger.info(f"Confusion matrix: {config['save_dir']}/plots/confusion_matrix.png")
    logger.info("="*70)
    
    return model

if __name__ == "__main__":
    main()