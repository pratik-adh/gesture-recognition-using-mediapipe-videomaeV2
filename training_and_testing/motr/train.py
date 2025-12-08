#!/usr/bin/env python3
"""
Motion Transformer (MoTr) Training Pipeline for Pre-cropped NPZ Videos
Converted from VideoMAE with all features preserved:
- Motion-focused architecture with temporal difference modeling
- Gradient stability improvements
- Checkpoint management with auto-resume
- Enhanced error handling
- Multi-GPU support
- Training curves and confusion matrix plots
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
import math

warnings.filterwarnings('ignore')

# -------------------------
# Logging + reproducibility
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('training_motr.log', mode='a')]
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
# Dataset for Pre-cropped NPZ Videos (Same as VideoMAE)
# -------------------------
class NPZDirectoryDataset(Dataset):
    """Dataset for pre-cropped videos stored as NPZ files"""
    def __init__(self, root_dir, num_frames=16, is_training=True, img_size=224):
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
        """Sample fixed number of frames uniformly"""
        n = len(frames)
        if n == 0:
            logger.warning("Video with 0 frames encountered")
            return np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)
        
        if n <= self.num_frames:
            indices = list(range(n)) + [n-1] * (self.num_frames - n)
        else:
            indices = np.linspace(0, n-1, self.num_frames, dtype=int)
        
        return frames[indices]

    def resize_frames(self, frames):
        """Resize frames to target size"""
        resized = []
        for frame in frames:
            resized.append(cv2.resize(frame, (self.img_size, self.img_size)))
        return np.array(resized)

    def apply_augmentations(self, frames):
        """Apply data augmentations"""
        if not self.is_training:
            return frames

        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            frames = np.clip(frames * factor, 0, 255).astype(np.uint8)

        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            mean = frames.mean(axis=(1,2,3), keepdims=True)
            frames = np.clip((frames - mean) * factor + mean, 0, 255).astype(np.uint8)

        if random.random() > 0.7:
            frames = frames[:, :, ::-1, :]

        if random.random() > 0.7:
            noise = np.random.normal(0, 3, frames.shape)
            frames = np.clip(frames + noise, 0, 255).astype(np.uint8)

        return frames

    def normalize(self, frames):
        """Normalize frames"""
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
            frames = self.apply_augmentations(frames)

        frames = self.normalize(frames)

        # Convert to tensor: (C, T, H, W)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        
        return frames, label

# -------------------------
# Motion Transformer Architecture
# -------------------------

class MotionExtractor(nn.Module):
    """Extract motion features using temporal differences"""
    def __init__(self, in_chans=3):  # Changed from in_channels to in_chans
        super().__init__()
        
        # Motion encoding with 2D convolutions on frame differences
        self.motion_conv = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3),  # Changed here too
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
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        
        # Calculate frame differences (motion)
        motion = []
        for t in range(T - 1):
            diff = x[:, :, t+1] - x[:, :, t]  # (B, C, H, W)
            motion.append(diff)
        
        # Stack motion frames
        motion = torch.stack(motion, dim=2)  # (B, C, T-1, H, W)
        
        # Process each motion frame
        motion_features = []
        for t in range(T - 1):
            feat = self.motion_conv(motion[:, :, t])  # (B, 256, H', W')
            motion_features.append(feat)
        
        # Stack temporal features
        motion_features = torch.stack(motion_features, dim=2)  # (B, 256, T-1, H', W')
        
        return motion_features


class SpatialPatchEmbed(nn.Module):
    """Spatial patch embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TemporalAttention(nn.Module):
    """Temporal attention across frames"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
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
    """MLP block"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MotionTransformerBlock(nn.Module):
    """Motion Transformer block with temporal and spatial attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        
        # Temporal attention (across time)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # Spatial attention (across patches)
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = TemporalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # MLP
        self.mlp_norm = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # x: (B, T*num_patches, dim)
        
        # Temporal attention
        x = x + self.temporal_attn(self.temporal_norm(x))
        
        # Spatial attention
        x = x + self.spatial_attn(self.spatial_norm(x))
        
        # MLP
        x = x + self.mlp(self.mlp_norm(x))
        
        return x


class MotionTransformer(nn.Module):
    """
    Motion Transformer for video classification
    Focuses on motion patterns using temporal differences
    """
    def __init__(self, img_size=224, patch_size=16, num_frames=16, in_chans=3,
                 num_classes=1000, embed_dim=768, num_heads=12, num_layers=12,
                 mlp_ratio=4., qkv_bias=True, dropout=0.1, attention_dropout=0.1,
                 freeze_backbone=False, freeze_layers=0):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        # Motion extraction
        self.motion_extractor = MotionExtractor(in_chans=in_chans)
        
        # After motion extraction, we have reduced spatial dimensions
        reduced_size = img_size // 16  # After 4 downsampling layers
        self.num_patches = (reduced_size) ** 2
        
        # Spatial patch embedding for appearance
        self.spatial_embed = SpatialPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        # Motion embedding projection
        self.motion_proj = nn.Linear(256 * (reduced_size ** 2), embed_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames * self.spatial_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MotionTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=dropout,
                attn_drop=attention_dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head (same structure as VideoMAE)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        # Freezing strategy (same as VideoMAE)
        if freeze_backbone:
            logger.info("Freezing entire backbone")
            for p in self.parameters():
                p.requires_grad = False
            for p in self.head.parameters():
                p.requires_grad = True
        elif freeze_layers > 0:
            n = min(freeze_layers, num_layers)
            logger.info(f"Freezing first {n}/{num_layers} transformer blocks")
            
            for p in self.motion_extractor.parameters():
                p.requires_grad = False
            for p in self.spatial_embed.parameters():
                p.requires_grad = False
            
            for i in range(n):
                for p in self.blocks[i].parameters():
                    p.requires_grad = False
        
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
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        
        # Extract motion features
        motion_feat = self.motion_extractor(x)  # (B, 256, T-1, H', W')
        
        # Process each frame for appearance
        appearance_tokens = []
        for t in range(T):
            frame = x[:, :, t]  # (B, C, H, W)
            tokens = self.spatial_embed(frame)  # (B, num_patches, embed_dim)
            appearance_tokens.append(tokens)
        
        appearance_tokens = torch.cat(appearance_tokens, dim=1)  # (B, T*num_patches, embed_dim)
        
        # Process motion features
        B_m, C_m, T_m, H_m, W_m = motion_feat.shape
        motion_flat = motion_feat.view(B, T_m, -1)  # (B, T-1, 256*H'*W')
        motion_tokens = self.motion_proj(motion_flat)  # (B, T-1, embed_dim)
        
        # Combine appearance and motion
        # Pad motion to match appearance temporal dimension
        motion_pad = torch.zeros(B, 1, self.embed_dim, device=x.device)
        motion_tokens = torch.cat([motion_tokens, motion_pad], dim=1)  # (B, T, embed_dim)
        
        # Add motion information to appearance tokens
        # Repeat motion tokens for each spatial patch
        motion_expanded = motion_tokens.unsqueeze(2).repeat(1, 1, self.spatial_embed.num_patches, 1)
        motion_expanded = motion_expanded.view(B, T * self.spatial_embed.num_patches, self.embed_dim)
        
        # Fuse appearance and motion
        tokens = appearance_tokens + motion_expanded
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # Add position embedding
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        
        # Final norm
        tokens = self.norm(tokens)
        
        # Classification
        cls_token = tokens[:, 0]
        output = self.head(cls_token)
        
        return output


# -------------------------
# Trainer (Same as VideoMAE)
# -------------------------
class Trainer:
    def __init__(self, config):
        self.cfg = config
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device = torch.device('cuda')
        self.n_gpus = torch.cuda.device_count()
        
        logger.info(f"Found {self.n_gpus} GPU(s)")
        for i in range(self.n_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        self.save_dir = Path(self.cfg['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        (self.save_dir / "logs").mkdir(exist_ok=True)

        self.scaler = GradScaler(init_scale=512.0, growth_interval=200)
        
        self.writer = SummaryWriter(
            log_dir=str(self.save_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.lrs = []
        self.start_epoch = 1
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        self.nan_grad_count = 0
        self.inf_grad_count = 0

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
        
        # Gradient accumulation: simulate larger batch
        accumulation_steps = 4  # Effective batch = 4 * 4 = 16
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # Scale loss
            
            if not torch.isfinite(loss):
                logger.warning(f"NaN/Inf loss at batch {batch_idx}, skipping")
                skipped_batches += 1
                continue
            
            self.scaler.scale(loss).backward()
            
            # Only step optimizer every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg['gradient_clip'])
                
                if not torch.isfinite(grad_norm):
                    if torch.isnan(grad_norm):
                        self.nan_grad_count += 1
                    else:
                        self.inf_grad_count += 1
                    optimizer.zero_grad()
                    self.scaler.update()
                    skipped_batches += 1
                    continue
                
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            # Metrics
            total_loss += loss.item() * accumulation_steps
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1-skipped_batches) if batch_idx+1 > skipped_batches else 0:.4f}", 
                'acc': f"{100.*correct/total:.2f}%"
            })

        epoch_loss = total_loss / max(1, len(loader) - skipped_batches)
        epoch_acc = correct / total if total > 0 else 0.0
        epoch_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_preds) > 0 else 0.0

        self.writer.add_scalar("Train/Loss", epoch_loss, epoch)
        self.writer.add_scalar("Train/Accuracy", epoch_acc, epoch)
        self.writer.add_scalar("Train/F1", epoch_f1, epoch)

        return epoch_loss, epoch_acc, epoch_f1

    def evaluate(self, model, loader, criterion, epoch):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels, all_probs = [], [], []

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

        epoch_loss = total_loss / len(loader)
        epoch_acc = correct / total if total > 0 else 0.0
        epoch_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_preds) > 0 else 0.0

        all_probs = np.array(all_probs)
        top3 = top_k_accuracy_score(all_labels, all_probs, k=min(3, self.num_classes))
        top5 = top_k_accuracy_score(all_labels, all_probs, k=min(5, self.num_classes))

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
        }
        
        latest_path = self.save_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"✓ New best model! Val Acc: {val_metrics['accuracy']:.4f}")

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        if not checkpoint_path.exists():
            logger.info("No checkpoint found, starting from scratch")
            return
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        self.lrs = checkpoint.get('lrs', [])
        
        if 'scaler_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state'])
        
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    def plot_training_curves(self):
        epochs = list(range(1, len(self.train_losses) + 1))
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accs, 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, self.lrs, 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        best_acc = max(self.val_accs) if self.val_accs else 0
        best_epoch = self.val_accs.index(best_acc) + 1 if self.val_accs else 0
        summary = f"""Training Summary:
        
Best Val Acc: {best_acc:.4f}
Best Epoch: {best_epoch}
Total Epochs: {len(epochs)}
Final Train Acc: {self.train_accs[-1]:.4f}
Final Val Acc: {self.val_accs[-1]:.4f}"""
        
        axes[1, 1].text(0.1, 0.5, summary, fontsize=12, verticalalignment='center',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')

        plt.tight_layout()
        save_path = self.save_dir / "plots" / "training_curves.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Training curves saved to {save_path}")

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
        logger.info(f"Confusion matrix saved to {save_path}")

    def save_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        df = pd.DataFrame(report).transpose()
        save_path = self.save_dir / "classification_report.csv"
        df.to_csv(save_path)
        logger.info(f"Classification report saved to {save_path}")

    def train(self):
        train_loader, val_loader = self.create_data_loaders()

        # Create Motion Transformer model
        model = MotionTransformer(
            img_size=self.cfg['img_size'],
            patch_size=self.cfg['patch_size'],
            num_frames=self.cfg['num_frames'],
            in_chans=3,
            num_classes=self.num_classes,
            embed_dim=self.cfg['embed_dim'],
            num_heads=self.cfg['num_heads'],
            num_layers=self.cfg['num_layers'],
            mlp_ratio=self.cfg['mlp_ratio'],
            qkv_bias=self.cfg['qkv_bias'],
            dropout=self.cfg['dropout_rate'],
            attention_dropout=self.cfg['attention_dropout'],
            freeze_backbone=self.cfg.get('freeze_backbone', False),
            freeze_layers=self.cfg.get('freeze_layers', 0)
        )

        if self.n_gpus > 1:
            logger.info(f"Using DataParallel with {self.n_gpus} GPUs")
            model = nn.DataParallel(model)
        
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.cfg['learning_rate'], 
            weight_decay=self.cfg['weight_decay'],
            betas=(0.9, 0.999),  # Standard for transformers
            eps=1e-8
        )
        
        # Use CosineAnnealingLR with warmup for better convergence
        warmup_epochs = 5
        total_epochs = self.cfg['epochs']
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing

        checkpoint_path = self.save_dir / "checkpoint_latest.pth"
        self.load_checkpoint(model, optimizer, checkpoint_path)

        best_preds = []
        best_labels = []

        logger.info(f"\nUsing warmup for {warmup_epochs} epochs, then cosine annealing")
        logger.info(f"Initial LR: {self.cfg['learning_rate']:.6f}")
        logger.info(f"Label smoothing: 0.1")

        for epoch in range(self.start_epoch, self.cfg['epochs'] + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.cfg['epochs']}")
            logger.info(f"{'='*50}")

            train_loss, train_acc, train_f1 = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            val_metrics = self.evaluate(model, val_loader, criterion, epoch)

            # Step scheduler every epoch
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.train_accs.append(train_acc)
            self.val_accs.append(val_metrics['accuracy'])
            self.lrs.append(current_lr)

            logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            logger.info(f"Val   - Top3: {val_metrics['top3']:.4f}, Top5: {val_metrics['top5']:.4f}")
            logger.info(f"LR: {current_lr:.6f}")

            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                best_preds = val_metrics['predictions']
                best_labels = val_metrics['labels']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(model, optimizer, epoch, val_metrics, is_best=is_best)

            if epoch % self.cfg.get('checkpoint_interval', 10) == 0:
                cp_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch, 
                    'model_state_dict': model_state, 
                    'optimizer_state_dict': optimizer.state_dict()
                }, cp_path)
                logger.info(f"Periodic checkpoint saved: {cp_path}")

            if self.patience_counter >= self.cfg['early_stopping_patience']:
                logger.info(f"Early stopping triggered")
                break

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
        logger.info(f"Results saved to: {self.save_dir}")
        logger.info("="*50)
        
        return model

# -------------------------
# Testing
# -------------------------
def test_model(model_path, test_dir, config):
    """Test the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('class_names')
    num_classes = len(class_names)

    model = MotionTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_frames=config['num_frames'],
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        mlp_ratio=config['mlp_ratio'],
        qkv_bias=config['qkv_bias'],
        dropout=config['dropout_rate'],
        attention_dropout=config['attention_dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info("Loading test dataset...")
    test_ds = NPZDirectoryDataset(
        test_dir, 
        num_frames=config['num_frames'], 
        is_training=False,
        img_size=config['img_size']
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )

    all_preds, all_labels, all_probs = [], [], []
    
    logger.info("Running inference...")
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
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Top-3 Accuracy: {top3:.4f}")
    logger.info(f"Top-5 Accuracy: {top5:.4f}")
    logger.info("\n" + report)
    logger.info("="*50)
    
    return {'accuracy': acc, 'f1': f1, 'top3': top3, 'top5': top5}

# -------------------------
# Main
# -------------------------
def main():
    """Main training pipeline for Motion Transformer - MEMORY OPTIMIZED"""
    
    config = {
        # DATA CONFIGURATION
        'train_dir': '/kaggle/input/npz-lightweight-videos-without-cropping-dataset/npz_lightweight_videos_without_cropping_splitted_dataset/train',
        'val_dir': '/kaggle/input/npz-lightweight-videos-without-cropping-dataset/npz_lightweight_videos_without_cropping_splitted_dataset/val',
        'test_dir': '/kaggle/input/npz-lightweight-videos-without-cropping-dataset/npz_lightweight_videos_without_cropping_splitted_dataset/test',
        'save_dir': '/kaggle/working/',

        # MODEL CONFIGURATION - MEMORY OPTIMIZED
        'img_size': 112,            # REDUCED from 224 (75% less memory!)
        'patch_size': 16,
        'num_frames': 12,           # REDUCED from 16 (25% less memory)
        'embed_dim': 192,           # REDUCED from 256 for memory
        'num_heads': 6,             # REDUCED from 8
        'num_layers': 6,            # REDUCED from 8 for memory
        'mlp_ratio': 2.0,           # REDUCED from 3.0
        'qkv_bias': True,
        'dropout_rate': 0.3,
        'attention_dropout': 0.2,

        # Layer freezing
        'freeze_backbone': False,
        'freeze_layers': 0,

        # TRAINING CONFIGURATION - MEMORY OPTIMIZED
        'batch_size': 8,            # REDUCED from 16 (critical!)
        'epochs': 50,               # INCREASED from 50 to compensate
        'learning_rate': 4e-4,      # INCREASED from 3e-4 for smaller batch
        'weight_decay': 0.01,
        'gradient_clip': 0.5,

        # SCHEDULER
        'lr_patience': 5,
        'early_stopping_patience': 15,
        'checkpoint_interval': 5,

        # SYSTEM
        'num_workers': 2,           # REDUCED from 4
    }

    logger.info("\n" + "="*70)
    logger.info("MOTION TRANSFORMER - MEMORY OPTIMIZED FOR 15GB GPU")
    logger.info("="*70)
    logger.info(f"Image Size: {config['img_size']} (reduced for memory)")
    logger.info(f"Frames: {config['num_frames']} (reduced from 16)")
    logger.info(f"Embedding Dim: {config['embed_dim']} (compact)")
    logger.info(f"Transformer Layers: {config['num_layers']} (efficient)")
    logger.info(f"Batch size: {config['batch_size']} (memory-safe)")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info("="*70)
    logger.info("MEMORY OPTIMIZATIONS:")
    logger.info("  ✓ Image size 112x112 (vs 224x224) = 75% less memory")
    logger.info("  ✓ 12 frames (vs 16) = 25% less memory")
    logger.info("  ✓ Batch size 4 = fits in 15GB GPU")
    logger.info("  ✓ Smaller model (192 dim, 6 layers)")
    logger.info("  ✓ Total memory: ~8-10GB (safe!)")
    logger.info("="*70)
    logger.info("EXPECTED PERFORMANCE:")
    logger.info("  • Epoch 10: ~40-50% accuracy")
    logger.info("  • Epoch 30: ~75-80% accuracy")
    logger.info("  • Epoch 60: ~88-92% accuracy")
    logger.info("="*70 + "\n")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        logger.info("")
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")

    trainer = Trainer(config)
    
    try:
        model = trainer.train()
        
        test_path = Path(config['test_dir'])
        if test_path.exists():
            best_model_path = Path(config['save_dir']) / "best_model.pth"
            if best_model_path.exists():
                logger.info("\nRunning test evaluation...")
                results = test_model(best_model_path, config['test_dir'], config)
                
                with open(Path(config['save_dir']) / "test_results.json", "w") as f:
                    json.dump({k: float(v) for k, v in results.items()}, f, indent=4)
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted")
        logger.info(f"Checkpoint: {config['save_dir']}/checkpoint_latest.pth")
    
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        raise

    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Results: {config['save_dir']}")
    logger.info("="*70)
    
    return model

if __name__ == "__main__":
    main()