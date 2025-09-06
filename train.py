import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import random
from typing import Tuple, List, Dict, Optional
from transformers import VideoMAEForVideoClassification, VideoMAEConfig, VideoMAEImageProcessor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import json
import time
import gc
import warnings
import yaml
import logging
from dataclasses import dataclass
warnings.filterwarnings('ignore')

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150


# =================== CONFIGURATION MANAGEMENT ===================

@dataclass
class TrainingConfig:
    """Configuration class optimized for small datasets with VideoMAE-Small-Kinetics"""
    # Data settings
    data_dir: str = "tensor_preprocessed"
    save_dir: str = "videomae_small_kinetics_output"
    num_labels: int = 36  # Updated for your dataset
    
    # Model settings - Optimized for small kinetics model and small dataset
    model_name: str = "MCG-NJU/videomae-small-finetuned-kinetics"
    freeze_backbone: bool = True  # Critical for small datasets
    freeze_layers_from: int = -1  # Only unfreeze last layer if not fully frozen
    dropout_rate: float = 0.5  # High dropout for regularization
    use_feature_extractor_mode: bool = True  # Additional conservative option
    
    # Training settings - Conservative for small dataset (~145 samples/class)
    batch_size: int = 8  # Increased slightly due to smaller model
    epochs: int = 30  # More epochs due to conservative learning
    learning_rate: float = 1e-4  # Higher LR possible due to kinetics fine-tuning
    weight_decay: float = 0.01  # Moderate weight decay
    warmup_steps: int = 100
    
    # Regularization settings - Enhanced for overfitting prevention
    label_smoothing: float = 0.1  # Add label smoothing
    mixup_alpha: float = 0.2  # Add mixup augmentation
    cutmix_alpha: float = 1.0  # Add cutmix augmentation
    augmentation_prob: float = 0.8  # Probability of applying augmentations
    
    # Validation settings
    n_folds: int = 5
    val_frequency: int = 1
    early_stopping_patience: int = 10  # Increased patience for small dataset
    min_delta: float = 0.001  # Minimum improvement threshold
    
    # Optimization settings
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # Hardware settings
    num_workers: int = 2
    pin_memory: bool = True
    
    # Advanced training strategies
    progressive_unfreezing: bool = False  # Option for staged unfreezing
    use_cosine_annealing: bool = True
    use_warmup: bool = True
    save_best_model_only: bool = True
    
    # Misc
    seed: int = 42
    save_plots_every_fold: bool = True
    log_level: str = "INFO"
    
    def validate_config(self):
        """Validate configuration for small dataset"""
        warnings = []
        
        if not self.freeze_backbone and self.num_labels * 145 < 10000:  # Assuming ~145 samples/class
            warnings.append("WARNING: Unfrozen backbone with small dataset may cause overfitting")
        
        if self.batch_size > 16:
            warnings.append("WARNING: Large batch size may reduce regularization effect")
        
        if self.learning_rate > 5e-4 and not self.freeze_backbone:
            warnings.append("WARNING: High learning rate with unfrozen backbone may cause instability")
            
        return warnings


def setup_logging(log_level: str = "INFO", save_dir: str = "output"):
    """Setup logging configuration"""
    os.makedirs(save_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{save_dir}/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_config(config: TrainingConfig, filepath: str):
    """Save configuration to YAML file"""
    with open(filepath, 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)


def load_config(filepath: str) -> TrainingConfig:
    """Load configuration from YAML file"""
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)


# =================== MPS OPTIMIZATIONS ===================

def optimize_for_device(device_type: str) -> bool:
    """Apply device-specific optimizations"""
    if device_type == "mps":
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_PROFILE'] = '0'
        return True
    elif device_type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return True
    return False


# =================== AUGMENTATION UTILITIES ===================

def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation adapted for video data"""
    if alpha <= 0:
        return x, y, y, 1
    
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # For video data: (B, C, T, H, W)
    _, _, T, H, W = x.shape
    
    # Generate random box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, :, bby1:bby2, bbx1:bbx2] = x[index, :, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


# =================== IMPROVED DATASET CLASS ===================

class ImprovedVideoDataset(Dataset):
    """Enhanced dataset with caching, memory management, and statistics"""
    
    def __init__(self, data_paths: List[Tuple[Path, int]], 
                 cache_strategy: str = "selective",
                 cache_threshold: int = 1000,
                 apply_augmentation: bool = False,
                 augmentation_config: dict = None,
                 logger: logging.Logger = None):
        
        self.data_paths = data_paths
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.apply_augmentation = apply_augmentation
        self.augmentation_config = augmentation_config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Analyze dataset
        self._analyze_dataset()
        
        # Setup caching strategy
        if cache_strategy == "all" or (cache_strategy == "selective" and len(data_paths) < cache_threshold):
            self._preload_all()
        elif cache_strategy == "selective":
            self._selective_cache(min(200, len(data_paths) // 5))  # Cache more for small datasets
    
    def _analyze_dataset(self):
        """Analyze dataset composition"""
        label_counts = {}
        for _, label in self.data_paths:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        self.logger.info(f"Dataset Analysis:")
        self.logger.info(f"  Total samples: {len(self.data_paths)}")
        self.logger.info(f"  Classes: {len(label_counts)}")
        self.logger.info(f"  Samples per class: {list(label_counts.values())}")
        
        # Check balance
        min_samples = min(label_counts.values())
        max_samples = max(label_counts.values())
        balance_ratio = min_samples / max_samples if max_samples > 0 else 0
        
        if balance_ratio < 0.8:
            self.logger.warning(f"Dataset imbalance detected! Ratio: {balance_ratio:.2f}")
        else:
            self.logger.info(f"Dataset is well balanced. Ratio: {balance_ratio:.2f}")
        
        # Small dataset warning
        avg_samples = len(self.data_paths) / len(label_counts)
        if avg_samples < 200:
            self.logger.warning(f"Small dataset detected: {avg_samples:.1f} avg samples/class")
            self.logger.info("  Recommendation: Use frozen backbone and high regularization")
    
    def _preload_all(self):
        """Preload all tensors with progress tracking"""
        self.logger.info(f"Preloading all {len(self.data_paths)} tensors...")
        
        for idx, (path, label) in enumerate(tqdm(self.data_paths, desc="Preloading")):
            try:
                tensor = torch.load(path, map_location='cpu')
                # Convert to half precision for memory efficiency
                if tensor.dtype == torch.float32:
                    tensor = tensor.half()
                self.cache[idx] = (tensor, label)
            except Exception as e:
                self.logger.error(f"Failed to load {path}: {e}")
                raise
        
        self.logger.info(f"Preloaded {len(self.cache)} tensors")
    
    def _selective_cache(self, sample_size: int):
        """Cache a subset of data for performance"""
        indices = random.sample(range(len(self.data_paths)), sample_size)
        
        self.logger.info(f"Caching {sample_size} sample tensors...")
        for idx in tqdm(indices, desc="Selective caching"):
            path, label = self.data_paths[idx]
            try:
                tensor = torch.load(path, map_location='cpu')
                if tensor.dtype == torch.float32:
                    tensor = tensor.half()
                self.cache[idx] = (tensor, label)
            except Exception as e:
                self.logger.error(f"Failed to load {path}: {e}")
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            self.cache_hits += 1
            tensor, label = self.cache[idx]
            return tensor.float(), label
        
        self.cache_misses += 1
        path, label = self.data_paths[idx]
        
        try:
            tensor = torch.load(path, map_location='cpu')
            tensor = tensor.permute(1, 0, 2, 3)     # (T,C,H,W) -> (C,T,H,W)
            return tensor.float(), label
        except Exception as e:
            self.logger.error(f"Failed to load {path}: {e}")
            # Return a zero tensor with correct shape for VideoMAE-Small
            return torch.zeros((3, 16, 224, 224)), label
    
    def get_cache_stats(self):
        """Get caching statistics"""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cached_items': len(self.cache)
        }


# =================== ENHANCED MODEL CLASS FOR VIDEOMAE-SMALL-KINETICS ===================

class EnhancedVideoMAESmallKinetics(nn.Module):
    """Enhanced VideoMAE-Small-Kinetics optimized for small datasets"""
    
    def __init__(self, config: TrainingConfig, num_classes: int, logger: logging.Logger = None):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.logger = logger or logging.getLogger(__name__)
        
        # Load the pre-trained Kinetics model
        self.logger.info(f"Loading pre-trained model: {config.model_name}")
        try:
            self.model = VideoMAEForVideoClassification.from_pretrained(
                config.model_name,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.float32  # Ensure float32 for stability
            )
            self.logger.info("Successfully loaded VideoMAE-Small-Finetuned-Kinetics")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
        # Get model configuration for architecture details
        self.model_config = self.model.config
        hidden_size = self.model_config.hidden_size  # Should be 384 for small model
        
        self.logger.info(f"Model Configuration:")
        self.logger.info(f"  Hidden size: {hidden_size}")
        self.logger.info(f"  Num attention heads: {self.model_config.num_attention_heads}")
        self.logger.info(f"  Num hidden layers: {self.model_config.num_hidden_layers}")
        self.logger.info(f"  Original num labels: {self.model_config.num_labels}")
        
        # Replace classifier for your number of classes
        self._build_classifier(hidden_size, num_classes)
        
        # Apply freezing strategy
        self._apply_freezing_strategy()
        
        # Add label smoothing loss
        if config.label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
            self.logger.info(f"Using label smoothing: {config.label_smoothing}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Log model statistics
        self._log_model_stats()
    
    def _build_classifier(self, hidden_size: int, num_classes: int):
        """Build appropriate classifier based on configuration"""
        if self.config.use_feature_extractor_mode:
            # Simple linear classifier for maximum stability
            self.model.classifier = nn.Sequential(
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(hidden_size, num_classes)
            )
            self.logger.info("Using simple feature extractor classifier")
            
        else:
            # Two-layer classifier with more capacity
            self.model.classifier = nn.Sequential(
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout_rate / 2),
                nn.Linear(hidden_size // 2, num_classes)
            )
            self.logger.info("Using two-layer classifier")
    
    def _apply_freezing_strategy(self):
        """Apply freezing strategy optimized for small datasets"""
        
        if self.config.freeze_backbone:
            self.logger.info("Freezing backbone - training only classifier")
            
            # Freeze all parameters except classifier
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            
            # Progressive unfreezing option
            if self.config.progressive_unfreezing and hasattr(self.config, 'freeze_layers_from'):
                if self.config.freeze_layers_from < 0:
                    layers_to_unfreeze = abs(self.config.freeze_layers_from)
                    self.logger.info(f"Unfreezing last {layers_to_unfreeze} encoder layers")
                    
                    # Access encoder layers through VideoMAE structure
                    if hasattr(self.model, 'videomae') and hasattr(self.model.videomae, 'encoder'):
                        encoder_layers = self.model.videomae.encoder.layer
                        for layer in encoder_layers[-layers_to_unfreeze:]:
                            for param in layer.parameters():
                                param.requires_grad = True
                                
        else:
            self.logger.info("Training all parameters (full fine-tuning)")
            self.logger.warning("Full fine-tuning with small dataset - monitor for overfitting!")
    
    def _log_model_stats(self):
        """Log detailed model parameter statistics"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = total_params - trainable_params
        
        self.logger.info(f"Model Statistics:")
        self.logger.info(f"  Model: {self.config.model_name}")
        self.logger.info(f"  Architecture: VideoMAE-Small (384 hidden)")
        self.logger.info(f"  Output classes: {self.num_classes}")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        self.logger.info(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        
        # Calculate parameter-to-sample ratio
        estimated_samples = self.num_classes * 145  # Assuming ~145 samples per class
        param_sample_ratio = trainable_params / estimated_samples
        self.logger.info(f"  Parameter-to-sample ratio: {param_sample_ratio:.1f}:1")
        
        if param_sample_ratio > 10:
            self.logger.warning("  High parameter-to-sample ratio - overfitting risk!")
        elif param_sample_ratio < 0.1:
            self.logger.info("  Low parameter-to-sample ratio - good for generalization")
        
        # Memory estimation
        param_size_gb = total_params * 4 / (1024**3)
        self.logger.info(f"  Estimated parameter memory: {param_size_gb:.2f} GB")
        
        # Strategy summary
        if self.config.freeze_backbone:
            if self.config.use_feature_extractor_mode:
                strategy = "Feature extraction (minimal fine-tuning)"
            else:
                strategy = "Frozen backbone with classifier training"
        else:
            strategy = "Full fine-tuning"
        
        self.logger.info(f"  Training strategy: {strategy}")
    
    def forward(self, pixel_values, labels=None, apply_augmentation=False, config=None):
        """Enhanced forward pass with augmentation support"""
        
        # Apply augmentations during training
        if apply_augmentation and config and self.training:
            if np.random.random() < config.augmentation_prob:
                # Randomly choose between mixup and cutmix
                if np.random.random() < 0.5 and config.mixup_alpha > 0:
                    pixel_values, labels_a, labels_b, lam = mixup_data(
                        pixel_values, labels, config.mixup_alpha
                    )
                    outputs = self.model(pixel_values=pixel_values)
                    
                    # Mixed loss for mixup
                    loss_a = self.criterion(outputs.logits, labels_a)
                    loss_b = self.criterion(outputs.logits, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                    
                    # Create custom output
                    class CustomOutput:
                        def __init__(self, logits, loss):
                            self.logits = logits
                            self.loss = loss
                    
                    return CustomOutput(outputs.logits, loss)
                
                elif config.cutmix_alpha > 0:
                    pixel_values, labels_a, labels_b, lam = cutmix_data(
                        pixel_values, labels, config.cutmix_alpha
                    )
                    outputs = self.model(pixel_values=pixel_values)
                    
                    # Mixed loss for cutmix
                    loss_a = self.criterion(outputs.logits, labels_a)
                    loss_b = self.criterion(outputs.logits, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                    
                    # Create custom output
                    class CustomOutput:
                        def __init__(self, logits, loss):
                            self.logits = logits
                            self.loss = loss
                    
                    return CustomOutput(outputs.logits, loss)
        
        # Standard forward pass
        if labels is not None:
            outputs = self.model(pixel_values=pixel_values)
            # Use custom loss function if specified
            if hasattr(self, 'criterion'):
                loss = self.criterion(outputs.logits, labels)
                # Create custom output with our loss
                class CustomOutput:
                    def __init__(self, logits, loss):
                        self.logits = logits
                        self.loss = loss
                return CustomOutput(outputs.logits, loss)
            else:
                return self.model(pixel_values=pixel_values, labels=labels)
        else:
            return self.model(pixel_values=pixel_values)


# =================== ENHANCED TRAINING FUNCTIONS ===================

class EarlyStopping:
    """Enhanced early stopping utility"""
    def __init__(self, patience: int = 5, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module = None):
        if self.best_score is None:
            self.best_score = score
            if model and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if model and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
    
    def restore_best_model(self, model: nn.Module):
        """Restore the best model weights"""
        if self.best_weights:
            model.load_state_dict(self.best_weights)


def create_optimizer_and_scheduler(model, config: TrainingConfig, num_training_steps: int):
    """Create optimizer and learning rate scheduler optimized for VideoMAE-Small"""
    
    # Different learning rates for different components
    if config.freeze_backbone:
        # Only classifier parameters are trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
    else:
        # Differentiated learning rates for full fine-tuning
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "classifier" in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': config.learning_rate * 0.1},  # Lower LR for pre-trained backbone
            {'params': classifier_params, 'lr': config.learning_rate}        # Higher LR for new classifier
        ], weight_decay=config.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    
    # Create scheduler
    if config.use_cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01
        )
    else:
        # StepLR as alternative
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.epochs // 3,
            gamma=0.1
        )
    
    # Add warmup scheduler if requested
    if config.use_warmup and config.warmup_steps > 0:
        from torch.optim.lr_scheduler import LambdaLR
        
        def warmup_lambda(step):
            if step < config.warmup_steps:
                return step / config.warmup_steps
            return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        return optimizer, scheduler, warmup_scheduler
    
    return optimizer, scheduler


def train_epoch_enhanced(model, dataloader, optimizer, scheduler, config: TrainingConfig, 
                        device, epoch: int, logger: logging.Logger, warmup_scheduler=None):
    """Enhanced training loop with advanced features"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(dataloader)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and device.type == "cuda" else None
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    for batch_idx, (videos, labels) in enumerate(pbar):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply warmup scheduler
        if warmup_scheduler and epoch == 0:
            warmup_scheduler.step()
        
        # Mixed precision forward pass with augmentation
        if config.mixed_precision and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(
                    pixel_values=videos, 
                    labels=labels,
                    apply_augmentation=True,
                    config=config
                )
                loss = outputs.loss / config.gradient_accumulation_steps
        else:
            outputs = model(
                pixel_values=videos, 
                labels=labels,
                apply_augmentation=True,
                config=config
            )
            loss = outputs.loss / config.gradient_accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation and clipping
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Statistics
        total_loss += outputs.loss.item()
        with torch.no_grad():
            predicted = outputs.logits.argmax(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        current_acc = 100. * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{total_loss/(batch_idx+1):.3f}',
            'Acc': f'{current_acc:.1f}%',
            'LR': f'{current_lr:.2e}'
        })
        
        # Memory cleanup
        if batch_idx % 50 == 0 and device.type == "mps":
            torch.mps.synchronize()
    
    # Handle remaining gradients
    if (len(dataloader)) % config.gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    # Step scheduler (not warmup)
    if warmup_scheduler is None or epoch > 0:
        scheduler.step()
    
    avg_loss = total_loss / num_batches
    accuracy = 100. * correct / total
    
    logger.info(f"Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def validate_enhanced(model, dataloader, device, epoch: int, logger: logging.Logger, 
                     return_predictions: bool = False):
    """Enhanced validation with detailed metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Validation")
        for videos, labels in pbar:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(pixel_values=videos, labels=labels, apply_augmentation=False)
            total_loss += outputs.loss.item()
            
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_acc = accuracy_score(all_labels, all_preds) * 100
            pbar.set_postfix({
                'Loss': f'{total_loss/len(all_labels)*len(dataloader.dataset):.3f}',
                'Acc': f'{current_acc:.1f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    if return_predictions:
        return avg_loss, accuracy, all_preds, all_labels
    return avg_loss, accuracy


# =================== ENHANCED PLOTTING FUNCTIONS ===================

def plot_fold_training_curves(fold_idx, history, config: TrainingConfig, save_dir: str):
    """Plot comprehensive training curves for each fold"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    val_epochs = range(config.val_frequency, len(history['train_loss'])+1, config.val_frequency)
    val_epochs = list(val_epochs[:len(history['val_loss'])])
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    if history['val_loss']:
        ax1.plot(val_epochs, history['val_loss'], 'r-o', linewidth=2, label='Validation Loss', 
                markersize=4, alpha=0.8)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Fold {fold_idx+1}: Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Acc', alpha=0.8)
    if history['val_acc']:
        ax2.plot(val_epochs, history['val_acc'], 'r-o', linewidth=2, label='Validation Acc',
                markersize=4, alpha=0.8)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Fold {fold_idx+1}: Accuracy Curves')
    ax2.set_ylim([0, 100])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate schedule
    ax3 = axes[1, 0]
    if history['learning_rates']:
        ax3.plot(epochs, history['learning_rates'], 'g-', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title(f'Fold {fold_idx+1}: Learning Rate Schedule')
        ax3.set_yscale('log')  # Log scale for better visualization
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, f'Learning Rate: {config.learning_rate}', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title(f'Fold {fold_idx+1}: Learning Rate')
    
    # Plot 4: Training statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
    overfit_gap = final_train_acc - final_val_acc if history['val_acc'] else 0
    
    stats_text = f"""
    Fold {fold_idx+1} Statistics:
    
    Final Training Accuracy: {final_train_acc:.2f}%
    Final Validation Accuracy: {final_val_acc:.2f}%
    Best Validation Accuracy: {best_val_acc:.2f}%
    
    Overfitting Gap: {overfit_gap:.2f}%
    
    Final Training Loss: {history['train_loss'][-1]:.4f}
    Final Validation Loss: {history['val_loss'][-1]:.4f if history['val_loss'] else 'N/A'}
    
    Total Epochs: {len(history['train_loss'])}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plot_path = f"{save_dir}/fold_{fold_idx+1}_training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_confusion_matrix(y_true, y_pred, class_names, fold_idx, save_dir: str):
    """Plot confusion matrix for the fold"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(max(8, len(class_names)), max(8, len(class_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if len(class_names) <= 20 else False, 
                yticklabels=class_names if len(class_names) <= 20 else False)
    plt.title(f'Fold {fold_idx+1}: Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if len(class_names) <= 20:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    plt.tight_layout()
    
    plot_path = f"{save_dir}/fold_{fold_idx+1}_confusion_matrix.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_all_folds_summary(all_histories, fold_accuracies, config: TrainingConfig, 
                          total_time: float, save_dir: str):
    """Create comprehensive summary plots for all folds"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All training curves
    ax1 = axes[0, 0]
    for i, hist in enumerate(all_histories):
        epochs = range(1, len(hist['train_loss']) + 1)
        ax1.plot(epochs, hist['train_loss'], alpha=0.6, linewidth=1.5, label=f'Fold {i+1}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss - All Folds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: All validation curves
    ax2 = axes[0, 1]
    for i, hist in enumerate(all_histories):
        if hist['val_loss']:
            val_epochs = range(config.val_frequency, len(hist['train_loss'])+1, config.val_frequency)
            val_epochs = list(val_epochs[:len(hist['val_loss'])])
            ax2.plot(val_epochs, hist['val_loss'], 'o-', alpha=0.7, linewidth=1.5, 
                    markersize=3, label=f'Fold {i+1}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss - All Folds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Fold performance bar chart
    ax3 = axes[1, 0]
    bars = ax3.bar(range(1, len(fold_accuracies)+1), fold_accuracies, 
                   color='skyblue', edgecolor='navy', alpha=0.7)
    mean_acc = np.mean(fold_accuracies)
    ax3.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_acc:.2f}%')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, fold_accuracies)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Best Validation Accuracy (%)')
    ax3.set_title('Fold Performance Comparison')
    ax3.set_xticks(range(1, len(fold_accuracies)+1))
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate comprehensive statistics
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    min_acc = np.min(fold_accuracies)
    max_acc = np.max(fold_accuracies)
    
    final_train_accs = [hist['train_acc'][-1] for hist in all_histories]
    final_val_accs = [hist['val_acc'][-1] if hist['val_acc'] else 0 for hist in all_histories]
    mean_train_acc = np.mean(final_train_accs)
    mean_overfit = np.mean([t - v for t, v in zip(final_train_accs, final_val_accs)])
    
    total_epochs = sum(len(h['train_acc']) for h in all_histories)
    avg_time_per_epoch = total_time / total_epochs if total_epochs > 0 else 0
    
    stats_text = f"""
VIDEOMAE-SMALL-KINETICS SUMMARY

PERFORMANCE METRICS
Validation Accuracy:
  Mean: {mean_acc:.2f}% ± {std_acc:.2f}%
  Range: {min_acc:.2f}% - {max_acc:.2f}%
  Best Fold: {np.argmax(fold_accuracies)+1} ({max_acc:.2f}%)

Training vs Validation:
  Final Train Acc: {mean_train_acc:.2f}%
  Overfitting Gap: {mean_overfit:.2f}%

TIMING STATISTICS
Total Time: {total_time:.1f} minutes
Time per Fold: {total_time/len(all_histories):.1f} min
Time per Epoch: {avg_time_per_epoch:.2f} min

MODEL CONFIGURATION
Model: VideoMAE-Small-Kinetics
Backbone: {'Frozen' if config.freeze_backbone else 'Trainable'}
Batch Size: {config.batch_size}
Learning Rate: {config.learning_rate}
Epochs: {config.epochs}
Augmentation: {'Yes' if config.mixup_alpha > 0 or config.cutmix_alpha > 0 else 'No'}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plot_path = f"{save_dir}/all_folds_comprehensive_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


# =================== MAIN TRAINING FUNCTION ===================

def train_fold_enhanced(fold_idx: int, train_paths: List[Tuple[Path, int]], 
                       val_paths: List[Tuple[Path, int]], config: TrainingConfig,
                       class_names: List[str], device: torch.device, logger: logging.Logger):
    """Enhanced fold training optimized for VideoMAE-Small-Kinetics"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING FOLD {fold_idx + 1}/{config.n_folds}")
    logger.info(f"Train samples: {len(train_paths)} | Validation samples: {len(val_paths)}")
    logger.info(f"{'='*70}")
    
    # Create datasets with intelligent caching (more aggressive for small datasets)
    train_dataset = ImprovedVideoDataset(
        train_paths, 
        cache_strategy="all" if len(train_paths) < 3000 else "selective",
        cache_threshold=3000,
        apply_augmentation=True,
        augmentation_config=config.__dict__,
        logger=logger
    )
    
    val_dataset = ImprovedVideoDataset(
        val_paths,
        cache_strategy="all",
        apply_augmentation=False,
        logger=logger
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0
    )
    
    # Create model
    model = EnhancedVideoMAESmallKinetics(config, len(class_names), logger)
    model = model.to(device)
    
    # Create optimizer and scheduler
    num_training_steps = len(train_loader) * config.epochs
    if config.use_warmup:
        optimizer, scheduler, warmup_scheduler = create_optimizer_and_scheduler(
            model, config, num_training_steps
        )
    else:
        optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)
        warmup_scheduler = None
    
    # Initialize tracking
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0
    best_model_state = None
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.min_delta,
        restore_best_weights=True
    )
    
    logger.info(f"Starting training for {config.epochs} epochs...")
    fold_start_time = time.time()
    
    # Training loop
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        
        # Training phase
        train_loss, train_acc = train_epoch_enhanced(
            model, train_loader, optimizer, scheduler, config, 
            device, epoch, logger, warmup_scheduler
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        if epoch % config.val_frequency == 0 or epoch == config.epochs - 1:
            val_loss, val_acc, val_preds, val_labels = validate_enhanced(
                model, val_loader, device, epoch, logger, return_predictions=True
            )
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_predictions': val_preds,
                    'val_labels': val_labels
                }
                logger.info(f"New best validation accuracy: {val_acc:.2f}%")
            
            # Early stopping check
            early_stopping(val_acc, model)
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                logger.info(f"Restoring best model weights (acc: {early_stopping.best_score:.2f}%)")
                early_stopping.restore_best_model(model)
                break
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
        
        # Memory cleanup
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()
    
    fold_time = (time.time() - fold_start_time) / 60
    logger.info(f"Fold {fold_idx+1} completed in {fold_time:.1f} minutes")
    
    # Generate fold plots
    if config.save_plots_every_fold:
        plot_path = plot_fold_training_curves(fold_idx, history, config, config.save_dir)
        logger.info(f"Training curves saved: {plot_path}")
        
        # Plot confusion matrix for best model
        if best_model_state and 'val_predictions' in best_model_state:
            cm_path = plot_confusion_matrix(
                best_model_state['val_labels'], 
                best_model_state['val_predictions'],
                class_names, fold_idx, config.save_dir
            )
            logger.info(f"Confusion matrix saved: {cm_path}")
    
    # Log dataset cache statistics
    train_cache_stats = train_dataset.get_cache_stats()
    val_cache_stats = val_dataset.get_cache_stats()
    logger.info(f"Cache Stats - Train: {train_cache_stats['hit_rate']:.1%} hit rate, "
               f"Val: {val_cache_stats['hit_rate']:.1%} hit rate")
    
    # Final validation with best model
    if early_stopping.best_score:
        final_val_acc = early_stopping.best_score
    else:
        final_val_acc = best_val_acc
    
    # Cleanup
    del model, optimizer, scheduler, train_dataset, val_dataset
    if warmup_scheduler:
        del warmup_scheduler
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    return final_val_acc, history, best_model_state


def main():
    """Main function optimized for VideoMAE-Small-Kinetics with small datasets"""
    
    # Load or create configuration
    config = TrainingConfig()
    
    # Validate configuration
    config_warnings = config.validate_config()
    
    # Setup logging and directories
    os.makedirs(config.save_dir, exist_ok=True)
    logger = setup_logging(config.log_level, config.save_dir)
    
    # Log configuration warnings
    for warning in config_warnings:
        logger.warning(warning)
    
    # Save configuration
    save_config(config, f"{config.save_dir}/config.yaml")
    
    logger.info("="*80)
    logger.info("VIDEOMAE-SMALL-KINETICS TRAINING FOR SMALL DATASETS")
    logger.info("="*80)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Strategy: {'Feature Extraction' if config.freeze_backbone else 'Fine-tuning'}")
    logger.info(f"Dataset size optimizations: {'Enabled' if config.freeze_backbone else 'Disabled'}")
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        optimize_for_device("mps")
        logger.info("Using MPS (Apple Silicon) with optimizations")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        optimize_for_device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU - training will be slow")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    if device.type == "cuda":
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    
    # Load dataset
    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {config.data_dir}")
    
    data_paths = []
    class_names = []
    
    logger.info("Loading dataset...")
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            class_idx = len(class_names)
            class_names.append(class_dir.name)
            
            pt_files = list(class_dir.glob("*.pt"))
            if not pt_files:
                logger.warning(f"No .pt files found in {class_dir}")
                continue
                
            for pt_file in pt_files:
                data_paths.append((pt_file, class_idx))
            
            logger.info(f"  Class {class_idx}: {class_dir.name} - {len(pt_files)} samples")
    
    if not data_paths:
        raise ValueError(f"No .pt files found in {config.data_dir}")
    
    # Update config with actual number of classes
    config.num_labels = len(class_names)
    
    logger.info(f"\nDataset Summary:")
    logger.info(f"  Total classes: {len(class_names)}")
    logger.info(f"  Total samples: {len(data_paths)}")
    logger.info(f"  Average samples per class: {len(data_paths)/len(class_names):.1f}")
    
    # Check if dataset is suitable for current configuration
    avg_samples_per_class = len(data_paths) / len(class_names)
    if avg_samples_per_class < 50:
        logger.error("Dataset too small! Need at least 50 samples per class")
        return
    elif avg_samples_per_class < 100:
        logger.warning("Very small dataset - ensure frozen backbone is used")
        config.freeze_backbone = True
        config.use_feature_extractor_mode = True
    
    # Test data loading
    logger.info("Testing data loading speed...")
    test_start = time.time()
    try:
        test_tensor = torch.load(data_paths[0][0], map_location='cpu')
        load_time = (time.time() - test_start) * 1000
        logger.info(f"  Single tensor load time: {load_time:.1f}ms")
        logger.info(f"  Tensor shape: {test_tensor.shape}")
        logger.info(f"  Estimated epoch time: {len(data_paths) * load_time / 60000:.1f} minutes")
        
        # Verify tensor format matches VideoMAE expectations
        expected_shape = (3, 16, 224, 224)  # (C, T, H, W)
        if test_tensor.shape != expected_shape:
            logger.warning(f"Tensor shape {test_tensor.shape} doesn't match expected {expected_shape}")
            logger.warning("This may cause issues with VideoMAE-Small model")
            
    except Exception as e:
        logger.error(f"Failed to load test tensor: {e}")
        raise
    
    # Setup K-fold cross-validation
    labels = [label for _, label in data_paths]
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    
    logger.info(f"\nStarting {config.n_folds}-fold cross-validation...")
    
    # Training tracking
    fold_accuracies = []
    all_histories = []
    all_best_models = []
    
    total_start_time = time.time()
    
    # Train each fold
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(data_paths)), labels)):
        train_paths = [data_paths[i] for i in train_idx]
        val_paths = [data_paths[i] for i in val_idx]
        
        # Verify fold balance
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        
        train_class_counts = np.bincount(train_labels)
        val_class_counts = np.bincount(val_labels)
        
        logger.info(f"\nFold {fold_idx+1} class distribution:")
        logger.info(f"  Train: min={train_class_counts.min()}, max={train_class_counts.max()}")
        logger.info(f"  Val: min={val_class_counts.min()}, max={val_class_counts.max()}")
        
        # Check if fold has sufficient samples
        if train_class_counts.min() < 3:
            logger.warning(f"Fold {fold_idx+1} has classes with < 3 training samples")
        if val_class_counts.min() < 1:
            logger.warning(f"Fold {fold_idx+1} has classes with no validation samples")
        
        # Train fold
        try:
            fold_acc, history, best_model = train_fold_enhanced(
                fold_idx, train_paths, val_paths, config, class_names, device, logger
            )
            
            fold_accuracies.append(fold_acc)
            all_histories.append(history)
            all_best_models.append(best_model)
            
            logger.info(f"Fold {fold_idx+1} completed - Best accuracy: {fold_acc:.2f}%")
            
        except Exception as e:
            logger.error(f"Fold {fold_idx+1} failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Inter-fold cleanup
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Calculate final results
    total_time = (time.time() - total_start_time) / 60
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS - VIDEOMAE-SMALL-KINETICS")
    logger.info("="*80)
    
    for i, acc in enumerate(fold_accuracies):
        logger.info(f"Fold {i+1}: {acc:.2f}%")
    
    logger.info(f"\nCross-Validation Results:")
    logger.info(f"  Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logger.info(f"  Best Fold: {np.argmax(fold_accuracies)+1} ({np.max(fold_accuracies):.2f}%)")
    logger.info(f"  Worst Fold: {np.argmin(fold_accuracies)+1} ({np.min(fold_accuracies):.2f}%)")
    logger.info(f"  Range: {np.max(fold_accuracies) - np.min(fold_accuracies):.2f}%")
    
    # Performance analysis
    if std_acc > 5:
        logger.warning(f"High variance detected ({std_acc:.2f}%) - consider more regularization")
    elif std_acc < 2:
        logger.info(f"Low variance ({std_acc:.2f}%) - consistent performance across folds")
    
    logger.info(f"\nTraining Time:")
    logger.info(f"  Total time: {total_time:.1f} minutes")
    logger.info(f"  Average time per fold: {total_time/config.n_folds:.1f} minutes")
    
    # Compare with expected performance for small datasets
    if mean_acc > 80:
        logger.info("Excellent performance for small dataset!")
    elif mean_acc > 70:
        logger.info("Good performance for small dataset")
    elif mean_acc > 60:
        logger.info("Reasonable performance - consider hyperparameter tuning")
    else:
        logger.warning("Low performance - check data quality or increase regularization")
    
    # Generate comprehensive summary plots
    summary_plot_path = plot_all_folds_summary(
        all_histories, fold_accuracies, config, total_time, config.save_dir
    )
    logger.info(f"Comprehensive summary plot saved: {summary_plot_path}")
    
    # Save detailed results
    results = {
        'model_info': {
            'model_name': config.model_name,
            'architecture': 'VideoMAE-Small-Kinetics',
            'hidden_size': 384,
            'strategy': 'Feature Extraction' if config.freeze_backbone else 'Fine-tuning',
            'total_parameters': '~22M',
            'trainable_parameters': '~28K' if config.freeze_backbone else '~22M'
        },
        'dataset_info': {
            'num_classes': len(class_names),
            'num_samples': len(data_paths),
            'avg_samples_per_class': len(data_paths) / len(class_names),
            'class_names': class_names,
            'dataset_balance': 'good' if (max([labels.count(i) for i in range(len(class_names))]) / 
                                        min([labels.count(i) for i in range(len(class_names))])) < 2 else 'imbalanced'
        },
        'cross_validation_results': {
            'fold_accuracies': fold_accuracies,
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'min_accuracy': float(np.min(fold_accuracies)),
            'max_accuracy': float(np.max(fold_accuracies)),
            'best_fold': int(np.argmax(fold_accuracies)) + 1,
            'worst_fold': int(np.argmin(fold_accuracies)) + 1,
            'performance_consistency': 'high' if std_acc < 3 else 'medium' if std_acc < 6 else 'low'
        },
        'training_info': {
            'total_time_minutes': total_time,
            'time_per_fold_minutes': total_time / config.n_folds,
            'total_epochs': sum(len(h['train_acc']) for h in all_histories),
            'avg_epochs_per_fold': sum(len(h['train_acc']) for h in all_histories) / config.n_folds,
            'device_used': str(device),
            'early_stopping_triggered': any(len(h['train_acc']) < config.epochs for h in all_histories),
            'config': config.__dict__
        },
        'optimization_info': {
            'augmentation_used': config.mixup_alpha > 0 or config.cutmix_alpha > 0,
            'label_smoothing': config.label_smoothing,
            'regularization_dropout': config.dropout_rate,
            'learning_rate_strategy': 'cosine_annealing' if config.use_cosine_annealing else 'step',
            'warmup_used': config.use_warmup,
            'mixed_precision': config.mixed_precision
        }
    }
    
    # Save results to JSON
    results_path = f"{config.save_dir}/detailed_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Detailed results saved: {results_path}")
    
    # Create performance comparison with expected baselines
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE ANALYSIS")
    logger.info("="*60)
    
    # Expected performance baselines for different scenarios
    if config.freeze_backbone:
        if avg_samples_per_class < 100:
            expected_range = (60, 80)
            baseline_desc = "frozen backbone, very small dataset"
        else:
            expected_range = (70, 85)
            baseline_desc = "frozen backbone, small dataset"
    else:
        expected_range = (50, 70)
        baseline_desc = "full fine-tuning, high overfitting risk"
    
    logger.info(f"Expected performance range for {baseline_desc}: {expected_range[0]}-{expected_range[1]}%")
    logger.info(f"Achieved performance: {mean_acc:.2f}%")
    
    if mean_acc >= expected_range[1]:
        logger.info("EXCELLENT: Performance exceeds expectations!")
    elif mean_acc >= expected_range[0]:
        logger.info("GOOD: Performance within expected range")
    else:
        logger.warning("BELOW EXPECTED: Consider hyperparameter tuning or data quality check")
    
    # Create simple summary plot
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Fold comparison
    plt.subplot(1, 3, 1)
    bars = plt.bar(range(1, len(fold_accuracies)+1), fold_accuracies, 
                   color='lightcoral', edgecolor='darkred', alpha=0.8)
    plt.axhline(y=mean_acc, color='navy', linestyle='--', linewidth=2,
                label=f'Mean: {mean_acc:.1f}%')
    
    for i, (bar, val) in enumerate(zip(bars, fold_accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xlabel('Fold')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('VideoMAE-Small-Kinetics\nK-Fold Cross-Validation Results')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 100)
    
    # Plot 2: Learning curves (average)
    plt.subplot(1, 3, 2)
    
    # Calculate average learning curves
    max_epochs = max(len(h['val_acc']) for h in all_histories if h['val_acc'])
    avg_val_acc = []
    
    for epoch in range(max_epochs):
        epoch_accs = []
        for hist in all_histories:
            if hist['val_acc'] and epoch < len(hist['val_acc']):
                epoch_accs.append(hist['val_acc'][epoch])
        if epoch_accs:
            avg_val_acc.append(np.mean(epoch_accs))
    
    if avg_val_acc:
        epochs_range = range(1, len(avg_val_acc) + 1)
        plt.plot(epochs_range, avg_val_acc, 'o-', color='darkblue', 
                linewidth=2, markersize=4, alpha=0.8)
        plt.axhline(y=mean_acc, color='red', linestyle='--', alpha=0.7,
                   label=f'Final Mean: {mean_acc:.1f}%')
        
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('Average Learning Curve\n(Across All Folds)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
    
    # Plot 3: Model comparison info
    plt.subplot(1, 3, 3)
    plt.axis('off')
    
    comparison_text = f"""
    MODEL COMPARISON
    
    VideoMAE-Small-Kinetics:
    • Architecture: 384 hidden dim
    • Parameters: ~22M total
    • Trainable: {'~28K' if config.freeze_backbone else '~22M'}
    • Strategy: {'Frozen' if config.freeze_backbone else 'Fine-tuned'}
    
    DATASET CHARACTERISTICS:
    • Classes: {len(class_names)}
    • Total samples: {len(data_paths)}
    • Avg per class: {avg_samples_per_class:.0f}
    • Size category: {'Very Small' if avg_samples_per_class < 100 else 'Small'}
    
    RESULTS:
    • Mean CV Accuracy: {mean_acc:.1f}%
    • Standard Deviation: {std_acc:.1f}%
    • Best Fold: {np.max(fold_accuracies):.1f}%
    • Performance: {'Excellent' if mean_acc > 80 else 'Good' if mean_acc > 70 else 'Fair'}
    
    RECOMMENDATION:
    {'✓ Current setup optimal' if mean_acc > 75 and std_acc < 5 else 
     '→ Consider more regularization' if std_acc > 5 else
     '→ Try different augmentation' if mean_acc < 70 else
     '→ Setup looks good'}
    """
    
    plt.text(0.05, 0.95, comparison_text, transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    simple_plot_path = f"{config.save_dir}/videomae_small_summary_results.png"
    plt.savefig(simple_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Summary plot saved: {simple_plot_path}")
    
    # Generate recommendations for improvement
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATIONS FOR OPTIMIZATION")
    logger.info("="*60)
    
    if std_acc > 5:
        logger.info("• High variance detected:")
        logger.info("  - Increase regularization (dropout, weight decay)")
        logger.info("  - Use more aggressive data augmentation")
        logger.info("  - Consider ensemble methods")
    
    if mean_acc < 70:
        logger.info("• Performance improvement suggestions:")
        logger.info("  - Try different augmentation strategies")
        logger.info("  - Experiment with learning rates")
        logger.info("  - Consider progressive unfreezing")
        logger.info("  - Check data quality and labeling")
    
    if not config.freeze_backbone and avg_samples_per_class < 200:
        logger.info("• For small datasets:")
        logger.info("  - STRONGLY RECOMMEND: Use frozen backbone")
        logger.info("  - Current full fine-tuning likely causing overfitting")
    
    if config.freeze_backbone and mean_acc > 85:
        logger.info("• Excellent performance achieved:")
        logger.info("  - Current configuration is optimal")
        logger.info("  - Consider testing on additional validation data")
    
    # Save training script configuration for reproduction
    final_config_path = f"{config.save_dir}/final_optimal_config.yaml"
    save_config(config, final_config_path)
    
    logger.info(f"\nTraining completed successfully!")
    logger.info(f"All results saved to: {config.save_dir}/")
    logger.info(f"Optimal configuration saved to: {final_config_path}")
    
    # Create a simple readme file with results
    readme_content = f"""# VideoMAE-Small-Kinetics Training Results

## Model Configuration
- **Model**: {config.model_name}
- **Architecture**: VideoMAE-Small (384 hidden dimensions)
- **Training Strategy**: {'Feature Extraction (Frozen Backbone)' if config.freeze_backbone else 'Full Fine-tuning'}
- **Trainable Parameters**: {'~28K' if config.freeze_backbone else '~22M'}

## Dataset Information
- **Classes**: {len(class_names)}
- **Total Samples**: {len(data_paths)}
- **Average Samples per Class**: {avg_samples_per_class:.0f}
- **Dataset Size Category**: {'Very Small' if avg_samples_per_class < 100 else 'Small'}

## Cross-Validation Results
- **Mean Accuracy**: {mean_acc:.2f}% ± {std_acc:.2f}%
- **Best Fold**: Fold {np.argmax(fold_accuracies)+1} ({np.max(fold_accuracies):.2f}%)
- **Performance Range**: {np.min(fold_accuracies):.2f}% - {np.max(fold_accuracies):.2f}%
- **Total Training Time**: {total_time:.1f} minutes

## Training Configuration
- **Epochs**: {config.epochs}
- **Batch Size**: {config.batch_size}
- **Learning Rate**: {config.learning_rate}
- **Regularization**: Dropout {config.dropout_rate}, Weight Decay {config.weight_decay}
- **Augmentation**: {'Yes' if config.mixup_alpha > 0 or config.cutmix_alpha > 0 else 'No'}

## Files Generated
- `detailed_results.json`: Complete training metrics
- `config.yaml`: Training configuration used
- `final_optimal_config.yaml`: Optimized configuration for reproduction
- `fold_*_training_curves.png`: Individual fold training curves
- `fold_*_confusion_matrix.png`: Confusion matrices per fold
- `all_folds_comprehensive_summary.png`: Complete analysis
- `videomae_small_summary_results.png`: Quick summary
- `training.log`: Detailed training logs

## Performance Assessment
**Result**: {'Excellent' if mean_acc > 80 else 'Good' if mean_acc > 70 else 'Fair' if mean_acc > 60 else 'Needs Improvement'}

**Recommendation**: {'Current setup is optimal for your dataset size and composition.' if mean_acc > 75 and std_acc < 5 else 
'Consider increasing regularization to reduce variance.' if std_acc > 5 else
'Try hyperparameter tuning to improve performance.' if mean_acc < 70 else
'Setup performs well for the given constraints.'}
"""
    
    readme_path = f"{config.save_dir}/README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Results summary saved to: {readme_path}")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Mean CV Accuracy: {results['cross_validation_results']['mean_accuracy']:.2f}%")
        print(f"Results saved to: {results['training_info']['config']['save_dir']}")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise