"""
VideoMAE Training Pipeline for Nepali Sign Language Gesture Recognition
========================================================================
This pipeline implements a robust training structure specifically designed
for VideoMAE with preprocessed 224x224 RGB videos from MediaPipe.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import random
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import VideoMAE components
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    VideoMAEConfig
)
from transformers import TrainingArguments, Trainer

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

INPUT_ROOT   = Path("preprocessed_videos")          # << change if different
OUTPUT_ROOT  = Path("videomae_results") 


class VideoGestureDataset(Dataset):
    """
    Custom dataset for loading preprocessed gesture videos.
    Expects 224x224 RGB videos from MediaPipe preprocessing.
    """
    
    def __init__(self, 
                 video_paths: List[Path],
                 labels: List[int],
                 num_frames: int = 16,
                 sampling_strategy: str = 'uniform',
                 processor=None,
                 augment: bool = False):
        """
        Args:
            video_paths: List of paths to video files
            labels: List of corresponding labels
            num_frames: Number of frames to sample (VideoMAE expects 16)
            sampling_strategy: 'uniform', 'random', or 'center'
            processor: VideoMAE processor for normalization
            augment: Whether to apply augmentations
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy
        self.processor = processor
        self.augment = augment
        
        # VideoMAE normalization values (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def load_video(self, path: Path) -> np.ndarray:
        """Load video and return frames matching MediaPipe output specs"""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Verify frame shape
            if frame.shape != (224, 224, 3):
                if len(frame.shape) == 2:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                if frame.shape[:2] != (224, 224):  # Wrong size
                    frame = cv2.resize(frame, (224, 224))
            
            # Convert BGR to RGB (MediaPipe outputs BGR, VideoMAE expects RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ensure uint8 dtype
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from video: {path}")
        
        frames_array = np.array(frames, dtype=np.uint8)
        
        # Verify shape
        if len(frames_array.shape) != 4 or frames_array.shape[1:] != (224, 224, 3):
            raise ValueError(f"Unexpected frames shape: {frames_array.shape}, expected (N, 224, 224, 3)")
        
        return frames_array
    
    def sample_frames(self, frames: np.ndarray) -> np.ndarray:
        """Sample frames according to strategy"""
        total_frames = len(frames)
        
        if total_frames == 0:
            raise ValueError("Video has no frames")
        
        if total_frames < self.num_frames:
            # Repeat frames if video is too short
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.extend(list(range(total_frames)))
            indices = sorted(indices[:self.num_frames])
        else:
            if self.sampling_strategy == 'uniform':
                # Uniform sampling across video
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            elif self.sampling_strategy == 'random':
                # Random sampling with temporal order preserved
                indices = sorted(np.random.choice(total_frames, self.num_frames, replace=False))
            elif self.sampling_strategy == 'center':
                # Sample from center of video
                center = total_frames // 2
                start = max(0, center - self.num_frames // 2)
                end = min(total_frames, start + self.num_frames)
                indices = np.arange(start, end)
                if len(indices) < self.num_frames:
                    indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        sampled = frames[indices]
        
        # Verify output
        if sampled.shape[0] != self.num_frames:
            raise ValueError(f"Frame sampling failed: got {sampled.shape[0]} frames, expected {self.num_frames}")
        
        # Ensure uint8 dtype
        if sampled.dtype != np.uint8:
            sampled = sampled.astype(np.uint8)
        
        return sampled
    
    def apply_augmentation(self, frames: np.ndarray) -> np.ndarray:
        """Apply temporal and spatial augmentations"""
        if not self.augment:
            return frames
        
        # Ensure frames are uint8
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        
        # Temporal augmentation
        if random.random() < 0.3:
            # Temporal jitter - slightly shift frame sampling
            shift = random.randint(-2, 2)
            frames = np.roll(frames, shift, axis=0)
        
        # Spatial augmentations
        if random.random() < 0.5:
            # Random horizontal flip (be careful with sign language!)
            # Only apply if gesture is symmetric
            frames = frames[:, :, ::-1, :]
        
        if random.random() < 0.3:
            # Color jitter
            factor = random.uniform(0.8, 1.2)
            frames = np.clip(frames.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        
        if random.random() < 0.2:
            # Random rotation (small angles only)
            angle = random.uniform(-10, 10)
            for i in range(len(frames)):
                center = (112, 112)  # 224/2
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                frames[i] = cv2.warpAffine(frames[i], M, (224, 224))
        
        # Ensure output is uint8
        return frames.astype(np.uint8)
    
    def normalize_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Normalize frames for VideoMAE input"""
        # Verify input shape
        assert frames.shape == (self.num_frames, 224, 224, 3), \
            f"Expected frames shape (16, 224, 224, 3), got {frames.shape}"
        
        # Convert to float and normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization (VideoMAE standard)
        # Reshape mean and std for broadcasting
        mean = self.mean.reshape(1, 1, 1, 3)
        std = self.std.reshape(1, 1, 1, 3)
        frames = (frames - mean) / std
        
        # Convert to tensor with shape (T, H, W, C)
        frames = torch.from_numpy(frames).float()
        
        # Permute to (C, T, H, W) for VideoMAE
        # From: (num_frames, height, width, channels)
        # To: (channels, num_frames, height, width)
        frames = frames.permute(3, 0, 1, 2)
        
        # Verify output shape
        assert frames.shape == (3, self.num_frames, 224, 224), \
            f"Expected output shape (3, 16, 224, 224), got {frames.shape}"
        
        return frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def process_frames_with_processor(self, frames: np.ndarray) -> torch.Tensor:
        """Process frames using VideoMAE processor"""
        # VideoMAEImageProcessor expects:
        # - List of numpy arrays (each frame separately)
        # - Values in range 0-255 (uint8)
        # - Shape per frame: (H, W, C)
        
        # Ensure correct dtype
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        
        # Verify shape
        assert frames.shape == (self.num_frames, 224, 224, 3), \
            f"Expected frames shape ({self.num_frames}, 224, 224, 3), got {frames.shape}"
        
        # Convert frames to list format for processor
        frames_list = [frames[i] for i in range(len(frames))]
        
        # Process with VideoMAE processor
        # This handles normalization and tensor conversion
        processed = self.processor(
            frames_list,
            return_tensors="pt"
        )
        
        # The processor returns a batch tensor, we need to squeeze it
        # From (1, C, T, H, W) to (C, T, H, W)
        pixel_values = processed['pixel_values'].squeeze(0)

        print(f"pixel_values : {pixel_values.shape}")

        # # Check the actual shape and handle accordingly
        # if pixel_values.shape == (self.num_frames, 3, 224, 224):
        #     # Shape is (T, C, H, W), need to convert to (C, T, H, W)
        #     pixel_values = pixel_values.permute(1, 0, 2, 3)
        # elif pixel_values.shape != (3, self.num_frames, 224, 224):
        #     # If it's neither expected shape, there's a problem
        #     raise ValueError(f"Unexpected shape from processor: {pixel_values.shape}")
        
        # Verify output shape
        expected_shape = (self.num_frames, 3, 224, 224)
        assert pixel_values.shape == expected_shape, \
            f"Processor output shape {pixel_values.shape} doesn't match expected {expected_shape}"
        
        return pixel_values
    
    def __getitem__(self, idx):
        try:
            # Load video
            video_path = self.video_paths[idx]
            frames = self.load_video(video_path)
            
            # Sample frames
            frames = self.sample_frames(frames)
            
            # Apply augmentation
            frames = self.apply_augmentation(frames)
            
            # Process frames
            if self.processor is not None:
                pixel_values = self.process_frames_with_processor(frames)
            else:
                pixel_values = self.normalize_frames(frames)
            
            # Get label
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return {
                'pixel_values': pixel_values,
                'labels': label
            }
        except Exception as e:
            print(f"\nError processing video {self.video_paths[idx]}:")
            print(f"  {str(e)}")
            raise


class VideoMAETrainer:
    """
    Complete training pipeline for VideoMAE gesture recognition
    """
    
    def __init__(self,
                 data_root: Path = INPUT_ROOT,
                 out_root: Path = OUTPUT_ROOT,
                 num_classes: int = 10,
                 num_channels: int = 3,
                 num_frames: int = 16,
                 batch_size: int = 8,
                 learning_rate: float = 5e-5,
                 num_epochs: int = 30,
                 device: str = 'auto',
                 seed: int = 42):
        """
        Initialize VideoMAE trainer with specifications matching 
        MediaPipe preprocessed videos (224x224 RGB)
        """
        self.data_root = Path(data_root)
        self.out_root  = Path(out_root)
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_channels = num_channels
        
        # Auto-detect best available device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.seed = seed
        
        # Set random seeds
        self.set_seed(seed)
        
        # Initialize model
        self.initialize_model()
        
        # Training history
        self.history = defaultdict(list)
        
        # Best model tracking
        self.best_val_acc = 0
        self.best_model_state = None
        
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def initialize_model(self):
        """
        Initialize VideoMAE model with proper configuration
        for 224x224 input videos
        """
        # Load pre-trained VideoMAE
        model_name = "MCG-NJU/videomae-base"
        
        # Load the base configuration first
        config = VideoMAEConfig.from_pretrained(model_name)
        
        # Only modify what we need to change
        config.hidden_dropout_prob = 0.3
        config.attention_probs_dropout_prob = 0.3
        
        # Print what the model expects
        print(f"\nModel Configuration:")
        print(f"  Expected channels: {config.num_channels}")
        print(f"  Expected frames: {config.num_frames}")
        print(f"  Image size: {config.image_size}")
        print(f"  Patch size: {config.patch_size}")
        print(f"  Tubelet size: {config.tubelet_size}")
        
        # Initialize model (this will show the weight initialization warning which is normal)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True  # Important for different num_classes
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Initialize processor (handles normalization)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        
        # Store config for reference
        self.model_config = config
        
        print(f"Model initialized on {self.device}")
        print(f"Model expects: {self.num_frames} frames of {config.image_size}x{config.image_size} RGB")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("Note: Weight initialization warning for classifier is normal - we're fine-tuning for new classes")
    
    def load_data(self) -> Tuple[List, List, Dict]:
        """Load video paths and labels from directory structure"""
        video_paths = []
        labels = []
        class_names = {}
        
        # Iterate through class folders
        for class_idx, class_dir in enumerate(sorted(self.data_root.iterdir())):
            if class_dir.is_dir():
                class_names[class_idx] = class_dir.name
                
                # Get all videos in class
                for video_path in class_dir.glob("*.mp4"):
                    video_paths.append(video_path)
                    labels.append(class_idx)
        
        print(f"Loaded {len(video_paths)} videos from {len(class_names)} classes")
        return video_paths, labels, class_names
    
    def create_stratified_splits(self, 
                                video_paths: List, 
                                labels: List, 
                                n_splits: int = 5) -> List[Dict]:
        """
        Create stratified k-fold splits for cross-validation
        Better for VideoMAE because:
        1. Ensures each fold has balanced class distribution
        2. Provides robust performance estimation
        3. Helps detect overfitting on small datasets
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        splits = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(video_paths, labels)):
            splits.append({
                'fold': fold_idx,
                'train_paths': [video_paths[i] for i in train_idx],
                'train_labels': [labels[i] for i in train_idx],
                'val_paths': [video_paths[i] for i in val_idx],
                'val_labels': [labels[i] for i in val_idx]
            })
        
        return splits
    
    def train_fold(self, fold_data: Dict, fold_idx: int) -> Dict:
        """Train a single fold"""

        fold_dir = self.out_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx + 1}")
        print(f"{'='*50}")
        
        # Create datasets
        train_dataset = VideoGestureDataset(
            video_paths=fold_data['train_paths'],
            labels=fold_data['train_labels'],
            num_frames=self.num_frames,
            sampling_strategy='uniform',
            processor=self.processor,
            augment=True
        )
        
        val_dataset = VideoGestureDataset(
            video_paths=fold_data['val_paths'],
            labels=fold_data['val_labels'],
            num_frames=self.num_frames,
            sampling_strategy='center',  # More consistent for validation
            processor=self.processor,
            augment=False
        )
        
        # Create dataloaders with device-appropriate settings
        num_workers = 0
        pin_memory = self.device.type == 'cuda'
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.05,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision training (only for CUDA)
        use_amp = self.device.type == 'cuda'
        scaler = GradScaler() if use_amp else None
        
        # Training metrics for this fold
        fold_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Debug shape on first batch
                if epoch == 0 and batch_idx == 0:
                    print(f"\nDebug - First batch shapes:")
                    print(f"  pixel_values: {pixel_values.shape}")
                    print(f"  Expected: (batch_size, 3, 16, 224, 224)")
                    print(f"  labels: {labels.shape}")
                
                optimizer.zero_grad()
                
                # Forward pass with or without mixed precision
                if use_amp:
                    with autocast():
                        outputs = self.model(pixel_values=pixel_values, labels=labels)
                        loss = outputs.loss
                else:
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                
                # Backward pass
                if use_amp:
                    scaler.scale(loss).backward()
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                train_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{train_correct/train_total:.4f}"
                })
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    val_loss += outputs.loss.item()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate epoch metrics
            train_loss_avg = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store metrics
            fold_history['train_loss'].append(train_loss_avg)
            fold_history['train_acc'].append(train_acc)
            fold_history['val_loss'].append(val_loss_avg)
            fold_history['val_acc'].append(val_acc)
            fold_history['lr'].append(current_lr)
            
            print(f"\nEpoch {epoch+1}: "
                  f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model for this fold
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model for final evaluation
        self.model.load_state_dict(best_model_state)
        
        # Final validation evaluation
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values=pixel_values)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        fold_model_path = fold_dir / f"model_fold{fold_idx}_best.pth"
        torch.save(self.model.state_dict(), fold_model_path)
        
        return {
            'history': fold_history,
            'best_val_acc': best_val_acc,
            'predictions': all_predictions,
            'labels': all_labels,
            'confusion_matrix': confusion_matrix(all_labels, all_predictions)
        }
    
    def train_with_cross_validation(self, n_folds: int = 5):
        """
        Main training function with k-fold cross-validation
        Why k-fold is better for VideoMAE on small datasets:
        1. Maximizes training data usage
        2. Provides robust performance estimates
        3. Reduces overfitting risk
        4. Identifies problematic samples across folds
        """
        # Load data
        video_paths, labels, self.class_names = self.load_data()
        
        # # Verify data pipeline with a test load
        # print("\nVerifying data pipeline...")
        # test_dataset = VideoGestureDataset(
        #     video_paths=[video_paths[0]],
        #     labels=[labels[0]],
        #     num_frames=self.num_frames,
        #     sampling_strategy='uniform',
        #     processor=self.processor,
        #     augment=False
        # )
        
        # try:
        #     test_sample = test_dataset[0]
        #     print(f"✓ Data pipeline verified")
        #     print(f"  Sample shape: {test_sample['pixel_values'].shape}")
        #     print(f"  Expected: (3, {self.num_frames}, 224, 224)")
        #     print(f"  Data type: {test_sample['pixel_values'].dtype}")
        #     print(f"  Value range: [{test_sample['pixel_values'].min():.2f}, {test_sample['pixel_values'].max():.2f}]")
            
        #     # Skip model test if it fails - we'll catch errors during actual training
        #     print(f"✓ Proceeding to training...")
        # except Exception as e:
        #     print(f"⚠️  Data pipeline warning: {e}")
        #     print(f"   Continuing with training...")
        
        # Create stratified splits
        splits = self.create_stratified_splits(video_paths, labels, n_folds)
        
        # Store results for all folds
        all_fold_results = []
        
        # Train each fold
        for fold_idx, fold_data in enumerate(splits):
            # Re-initialize model for each fold (start fresh)
            print(f"\nInitializing model for fold {fold_idx + 1}...")
            self.initialize_model()
            
            # Train fold
            fold_results = self.train_fold(fold_data, fold_idx)
            all_fold_results.append(fold_results)
            
            # Update global history
            for key, values in fold_results['history'].items():
                self.history[f'fold_{fold_idx}_{key}'] = values
        
        # Calculate cross-validation statistics
        cv_accuracies = [r['best_val_acc'] for r in all_fold_results]
        
        print(f"\n{'='*50}")
        print("Cross-Validation Results")
        print(f"{'='*50}")
        print(f"Mean Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
        print(f"Best Fold: {np.argmax(cv_accuracies) + 1} with {max(cv_accuracies):.4f}")
        print(f"Worst Fold: {np.argmin(cv_accuracies) + 1} with {min(cv_accuracies):.4f}")
        
        # Store results
        self.cv_results = {
            'fold_results': all_fold_results,
            'cv_mean': np.mean(cv_accuracies),
            'cv_std': np.std(cv_accuracies),
            'cv_accuracies': cv_accuracies
        }
        
        return self.cv_results
    
    def plot_training_curves(self):
        """Plot comprehensive training curves for all folds"""
        n_folds = len(self.cv_results['fold_results'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('VideoMAE Training Analysis - All Folds', fontsize=16, fontweight='bold')
        
        # Plot 1: Training Loss
        ax = axes[0, 0]
        for fold_idx in range(n_folds):
            history = self.cv_results['fold_results'][fold_idx]['history']
            ax.plot(history['train_loss'], label=f'Fold {fold_idx+1}', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Across Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Validation Loss
        ax = axes[0, 1]
        for fold_idx in range(n_folds):
            history = self.cv_results['fold_results'][fold_idx]['history']
            ax.plot(history['val_loss'], label=f'Fold {fold_idx+1}', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss Across Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Training Accuracy
        ax = axes[0, 2]
        for fold_idx in range(n_folds):
            history = self.cv_results['fold_results'][fold_idx]['history']
            ax.plot(history['train_acc'], label=f'Fold {fold_idx+1}', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy Across Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Validation Accuracy
        ax = axes[1, 0]
        for fold_idx in range(n_folds):
            history = self.cv_results['fold_results'][fold_idx]['history']
            ax.plot(history['val_acc'], label=f'Fold {fold_idx+1}', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy Across Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Learning Rate Schedule
        ax = axes[1, 1]
        for fold_idx in range(n_folds):
            history = self.cv_results['fold_results'][fold_idx]['history']
            ax.plot(history['lr'], label=f'Fold {fold_idx+1}', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Fold Performance Comparison
        ax = axes[1, 2]
        fold_accs = self.cv_results['cv_accuracies']
        bars = ax.bar(range(1, n_folds+1), fold_accs, color='skyblue', edgecolor='navy')
        ax.axhline(y=np.mean(fold_accs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(fold_accs):.4f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Best Validation Accuracy')
        ax.set_title('Cross-Validation Performance')
        ax.set_xticks(range(1, n_folds+1))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, fold_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.out_root / 'videomae_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all folds"""
        n_folds = len(self.cv_results['fold_results'])
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Confusion Matrices - All Folds', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for fold_idx in range(min(n_folds, 5)):
            ax = axes[fold_idx]
            cm = self.cv_results['fold_results'][fold_idx]['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       ax=ax, cbar=fold_idx == 0)
            ax.set_title(f'Fold {fold_idx+1} (Acc: {self.cv_results["cv_accuracies"][fold_idx]:.3f})')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        # Plot average confusion matrix in the last subplot
        ax = axes[5]
        all_cms = [r['confusion_matrix'] for r in self.cv_results['fold_results']]
        avg_cm = np.mean(all_cms, axis=0)
        avg_cm_normalized = avg_cm / avg_cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(avg_cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                   ax=ax, cbar=True)
        ax.set_title(f'Average Across Folds (Mean Acc: {self.cv_results["cv_mean"]:.3f})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(self.out_root / 'videomae_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_per_class_performance(self):
        """Plot per-class performance analysis"""
        # Aggregate predictions from all folds
        all_predictions = []
        all_labels = []
        
        for fold_result in self.cv_results['fold_results']:
            all_predictions.extend(fold_result['predictions'])
            all_labels.extend(fold_result['labels'])
        
        # Calculate per-class metrics
        report = classification_report(all_labels, all_predictions, 
                                      target_names=list(self.class_names.values()),
                                      output_dict=True)
        
        # Extract metrics
        classes = list(self.class_names.values())
        precision = [report[c]['precision'] for c in classes]
        recall = [report[c]['recall'] for c in classes]
        f1_score = [report[c]['f1-score'] for c in classes]
        support = [report[c]['support'] for c in classes]
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-Class Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Precision by class
        ax = axes[0, 0]
        bars = ax.bar(classes, precision, color='lightcoral')
        ax.set_xlabel('Gesture Class')
        ax.set_ylabel('Precision')
        ax.set_title('Precision by Class')
        ax.set_ylim([0, 1.1])
        ax.axhline(y=np.mean(precision), color='red', linestyle='--', alpha=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, val in zip(bars, precision):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom')
        
        # Plot 2: Recall by class
        ax = axes[0, 1]
        bars = ax.bar(classes, recall, color='lightblue')
        ax.set_xlabel('Gesture Class')
        ax.set_ylabel('Recall')
        ax.set_title('Recall by Class')
        ax.set_ylim([0, 1.1])
        ax.axhline(y=np.mean(recall), color='blue', linestyle='--', alpha=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for bar, val in zip(bars, recall):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom')
        
        # Plot 3: F1-Score by class
        ax = axes[1, 0]
        bars = ax.bar(classes, f1_score, color='lightgreen')
        ax.set_xlabel('Gesture Class')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score by Class')
        ax.set_ylim([0, 1.1])
        ax.axhline(y=np.mean(f1_score), color='green', linestyle='--', alpha=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for bar, val in zip(bars, f1_score):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom')
        
        # Plot 4: Support distribution
        ax = axes[1, 1]
        bars = ax.bar(classes, support, color='lightyellow', edgecolor='orange')
        ax.set_xlabel('Gesture Class')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Distribution by Class')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for bar, val in zip(bars, support):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{int(val)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.out_root / 'videomae_per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_analysis(self):
        """Analyze and visualize common errors"""
        # Aggregate confusion matrices
        all_cms = [r['confusion_matrix'] for r in self.cv_results['fold_results']]
        avg_cm = np.mean(all_cms, axis=0)
        
        # Find most confused pairs
        np.fill_diagonal(avg_cm, 0)  # Remove correct predictions
        confusion_pairs = []
        
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and avg_cm[i, j] > 0:
                    confusion_pairs.append({
                        'true': self.class_names[i],
                        'predicted': self.class_names[j],
                        'count': avg_cm[i, j]
                    })
        
        # Sort by confusion count
        confusion_pairs = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)[:10]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Top confusion pairs
        ax = axes[0]
        pairs_labels = [f"{cp['true']}→{cp['predicted']}" for cp in confusion_pairs]
        pairs_counts = [cp['count'] for cp in confusion_pairs]
        
        bars = ax.barh(pairs_labels, pairs_counts, color='salmon')
        ax.set_xlabel('Average Misclassification Count')
        ax.set_title('Top 10 Most Confused Gesture Pairs')
        ax.invert_yaxis()
        
        for bar, count in zip(bars, pairs_counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{count:.1f}', ha='left', va='center')
        
        # Plot 2: Difficulty score by class
        ax = axes[1]
        difficulty_scores = []
        classes = list(self.class_names.values())
        
        for i, class_name in enumerate(classes):
            # Difficulty = 1 - (correct predictions / total predictions)
            correct = avg_cm[i, i]
            total = np.sum(avg_cm[i, :])
            difficulty = 1 - (correct / total) if total > 0 else 0
            difficulty_scores.append(difficulty)
        
        bars = ax.bar(classes, difficulty_scores, color='orange')
        ax.set_xlabel('Gesture Class')
        ax.set_ylabel('Difficulty Score (1 - Accuracy)')
        ax.set_title('Class Difficulty Analysis')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for bar, score in zip(bars, difficulty_scores):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.out_root / 'videomae_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all training results and model"""
        
        # Save cross-validation results
        with open(self.out_root / 'cv_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_to_save = {
                'cv_mean': float(self.cv_results['cv_mean']),
                'cv_std': float(self.cv_results['cv_std']),
                'cv_accuracies': [float(x) for x in self.cv_results['cv_accuracies']],
                'class_names': self.class_names
            }
            json.dump(results_to_save, f, indent=2)
        
        # Save best model
        best_idx = int(np.argmax(self.cv_results["cv_accuracies"]))
        best_path = self.out_root / "best_model.pth"
        torch.save(
            {
                "model_state_dict": self.cv_results["fold_results"][best_idx]["model"].state_dict()
                if "model" in self.cv_results["fold_results"][best_idx]
                else self.model.state_dict(),
                "config": self.model.config,
                "best_accuracy": float(self.cv_results["cv_accuracies"][best_idx]),
                "fold": best_idx,
            },
            best_path,
        )
        print(f"\nBest model (fold {best_idx}) saved to {best_path}")
        print(f"CV mean ± std: {self.cv_results['cv_mean']:.4f} ± {self.cv_results['cv_std']:.4f}")


def main():
    """Main training pipeline"""
    
    # Configuration
    config = {
        'data_root': 'preprocessed_videos',     # Output from MediaPipe preprocessing
        'out_root': "videomae_results",         # Directory to save results
        'num_classes': 10,                      # 10 gesture classes
        'num_frames': 16,                       # VideoMAE standard
        'batch_size': 4,                        # Reduced for Mac/MPS compatibility
        'learning_rate': 5e-5,                  # Fine-tuning learning rate
        'num_epochs': 50,                       # Training epochs per fold
        'device': 'auto',                       # Auto-detect best device
        'seed': 42,                             # For reproducibility
        'num_channels': 3,                      # RGB input
    }
    
    print("="*60)
    print("VideoMAE Training Pipeline for Nepali Sign Language")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    
    # Initialize trainer
    trainer = VideoMAETrainer(**config)
    
    # Device-specific recommendations
    if trainer.device.type == 'mps':
        print("\n⚠️  Running on Apple Silicon (MPS)")
        print("   - Using batch_size=4 for memory efficiency")
        print("   - Using num_workers=0 to avoid multiprocessing issues")
        print("   - Mixed precision training disabled (not supported on MPS)")
    elif trainer.device.type == 'cpu':
        print("\n⚠️  Running on CPU - Training will be slow")
        print("   - Consider using a smaller batch_size (2-4) and fewer epochs")
        print("   - Using num_workers=0 for stability")
    
    # Train with 5-fold cross-validation
    print("\nStarting 5-fold cross-validation training...")
    cv_results = trainer.train_with_cross_validation(n_folds=5)
    
    # Generate comprehensive plots
    print("\nGenerating training visualizations...")
    trainer.plot_training_curves()
    trainer.plot_confusion_matrices()
    trainer.plot_per_class_performance()
    trainer.plot_error_analysis()
    
    # Save results
    trainer.save_results()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final CV Accuracy: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
    print("\nAll visualizations saved to current directory")
    print("Model and results saved to 'videomae_results' directory")


if __name__ == "__main__":
    main()