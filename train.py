
#!/usr/bin/env python3
"""
Hybrid NSL Pipeline: Combines Enhanced Preprocessing with VideoMAE Training
This integrates the sophisticated preprocessing from your code with the VideoMAE training pipeline
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from transformers import VideoMAEForVideoClassification
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your enhanced preprocessor (assuming it's in a separate file)
# If not, include the entire NSLEnhancedProcessor class here
from preprocessing_videos import NSLEnhancedProcessor, preprocess_nsl_dataset

# ====================== DATASET CLASS FOR ENHANCED PREPROCESSED DATA ======================

class NSLEnhancedDataset(Dataset):
    """Dataset class for NSL videos preprocessed with enhanced pipeline"""
    
    def __init__(self, data_dir, transform=None, train=True, load_detection_info=True):
        """
        Initialize dataset for enhanced preprocessed data
        
        Args:
            data_dir: Path to preprocessed data directory
            transform: Optional transforms to apply
            train: Whether this is training data
            load_detection_info: Whether to load detection metadata
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.train = train
        self.load_detection_info = load_detection_info
        
        # Load preprocessing summary if available
        summary_path = self.data_dir / 'nsl_preprocessing_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self.preprocessing_summary = json.load(f)
                self.class_names = sorted(self.preprocessing_summary['dataset_info']['original_class_distribution'].keys())
        else:
            # Fallback: discover classes from directory structure
            self.class_names = sorted([d.name for d in self.data_dir.iterdir() 
                                     if d.is_dir() and not d.name.startswith('.')])
            self.preprocessing_summary = None
        
        # Collect all preprocessed video files
        self.samples = []
        self.labels = []
        self.metadata = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
                
            # Support both .npz and .npy formats
            npz_files = list(class_dir.glob('*.npz'))
            npy_files = list(class_dir.glob('*.npy'))
            
            for video_file in npz_files + npy_files:
                self.samples.append(video_file)
                self.labels.append(class_idx)
                
                # Store filename for tracking
                self.metadata.append({
                    'class': class_name,
                    'filename': video_file.name,
                    'class_idx': class_idx
                })
        
        self.num_classes = len(self.class_names)
        
        # Log dataset statistics
        logging.info(f"Loaded NSL dataset: {len(self.samples)} videos, {self.num_classes} classes")
        if self.preprocessing_summary:
            detection_rate = self.preprocessing_summary['detection_performance']['average_enhanced_rate']
            logging.info(f"Average detection rate: {detection_rate:.1%}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            # Load preprocessed frames based on file type
            if video_path.suffix == '.npz':
                data = np.load(video_path, allow_pickle=True)
                frames = data['frames']
                
                # Optionally load detection info for analysis
                if self.load_detection_info and 'detection_info' in data:
                    detection_info = data['detection_info']
                else:
                    detection_info = None
                    
                if 'metadata' in data:
                    video_metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
                else:
                    video_metadata = {}
            else:
                # .npy format (simpler)
                frames = np.load(video_path)
                detection_info = None
                video_metadata = {}
            
            # Ensure frames are in correct format
            frames = frames.astype(np.float32)
            
            # Normalize to [0, 1] if not already
            if frames.max() > 1.0:
                frames = frames / 255.0
            
            # Convert to tensor [T, C, H, W]
            if len(frames.shape) == 4:  # [T, H, W, C]
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
            elif len(frames.shape) == 3:  # Grayscale [T, H, W]
                frames = torch.from_numpy(frames).unsqueeze(1).repeat(1, 3, 1, 1)
            
            # Apply additional transforms if specified
            if self.transform:
                frames = self.transform(frames)
            
            # Add data augmentation for training
            if self.train:
                frames = self.apply_training_augmentation(frames)
            
            return frames, label
            
        except Exception as e:
            logging.error(f"Error loading {video_path}: {e}")
            # Return black frames as fallback
            frames = torch.zeros((16, 3, 224, 224), dtype=torch.float32)
            return frames, label
    
    def apply_training_augmentation(self, frames):
        """Apply augmentation during training for better generalization"""
        # Random temporal jittering
        if torch.rand(1) < 0.2:
            # Randomly drop and repeat frames
            indices = torch.randperm(frames.shape[0])[:14]
            indices = torch.sort(indices)[0]
            # Repeat some frames to maintain length
            frames = frames[indices]
            frames = torch.cat([frames, frames[-2:].repeat(1, 1, 1, 1)], dim=0)
        
        # Random brightness adjustment
        if torch.rand(1) < 0.3:
            brightness_factor = 0.8 + torch.rand(1) * 0.4  # 0.8 to 1.2
            frames = frames * brightness_factor
            frames = torch.clamp(frames, 0, 1)
        
        # Random contrast adjustment
        if torch.rand(1) < 0.3:
            contrast_factor = 0.8 + torch.rand(1) * 0.4
            mean = frames.mean(dim=[2, 3], keepdim=True)
            frames = (frames - mean) * contrast_factor + mean
            frames = torch.clamp(frames, 0, 1)
        
        return frames
    
    def get_class_weights(self):
        """Calculate class weights for balanced training"""
        from collections import Counter
        label_counts = Counter(self.labels)
        total = len(self.labels)
        
        weights = []
        for i in range(self.num_classes):
            count = label_counts.get(i, 1)
            weight = total / (self.num_classes * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)

# ====================== ENHANCED VIDEOMAE MODEL ======================

class EnhancedVideoMAEClassifier(nn.Module):
    """Enhanced VideoMAE with stronger regularization for NSL"""
    
    def __init__(self, num_classes, dropout_rate=0.5, hidden_dim=512):
        super().__init__()
        
        # Load pretrained VideoMAE
        self.videomae = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Enhanced classification head with more regularization
        base_hidden = self.videomae.config.hidden_size
        
        self.videomae.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.videomae.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, pixel_values, return_features=False):
        """Forward pass with optional feature extraction"""
        outputs = self.videomae.videomae(pixel_values=pixel_values)
        sequence_output = outputs[0]
        
        # Global average pooling
        features = sequence_output.mean(dim=1)
        
        if return_features:
            return features
        
        logits = self.videomae.classifier(features)
        return logits

# ====================== ENHANCED TRAINER ======================

class NSLEnhancedTrainer:
    """Enhanced trainer with better monitoring and early stopping"""
    
    def __init__(self, data_dir, output_dir, num_folds=5, device=None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_folds = num_folds
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Setup logging
        log_file = self.output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"Using device: {self.device}")
        
        # Load dataset to get basic info
        self.dataset = NSLEnhancedDataset(self.data_dir)
        self.num_classes = self.dataset.num_classes
        self.class_names = self.dataset.class_names
        
        logging.info(f"Dataset: {len(self.dataset)} samples, {self.num_classes} classes")
    
    def create_data_loaders(self, train_indices, val_indices, batch_size=8):
        """Create balanced data loaders"""
        # Create subsets
        train_subset = torch.utils.data.Subset(self.dataset, train_indices)
        val_subset = torch.utils.data.Subset(self.dataset, val_indices)
        
        # Calculate class weights for balanced training
        train_labels = [self.dataset.labels[i] for i in train_indices]
        class_counts = np.bincount(train_labels, minlength=self.num_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        
        # Create weighted sampler for balanced training
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_indices),
            replacement=True
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, torch.tensor(class_weights, dtype=torch.float32)
    
    def train_epoch(self, model, loader, criterion, optimizer, grad_clip=1.0):
        """Train for one epoch with gradient accumulation"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for batch_idx, (frames, labels) in enumerate(pbar):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
        
        return total_loss / len(loader), 100. * correct / total
    
    def validate(self, model, loader, criterion):
        """Validate with per-class metrics"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        class_correct = np.zeros(self.num_classes)
        class_total = np.zeros(self.num_classes)
        
        with torch.no_grad():
            for frames, labels in tqdm(loader, desc="Validation", leave=False):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
        
        # Calculate per-class accuracy
        per_class_acc = []
        for i in range(self.num_classes):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0.0)
        
        avg_acc = 100. * correct / total
        mean_class_acc = np.mean(per_class_acc)
        
        return total_loss / len(loader), avg_acc, mean_class_acc, per_class_acc
    
    def plot_training_curves(self, history, fold):
        """Generate comprehensive training plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title(f'Training and Validation Loss - Fold {fold+1}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].plot(history['val_mean_class_acc'], label='Val Mean Class Acc', linewidth=2, linestyle='--')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title(f'Training and Validation Accuracy - Fold {fold+1}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(history['lr'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title(f'Learning Rate Schedule - Fold {fold+1}')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Per-class accuracy heatmap (final epoch)
        if 'per_class_acc' in history and len(history['per_class_acc']) > 0:
            final_per_class = history['per_class_acc'][-1]
            im = axes[1, 1].imshow([final_per_class], cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
            axes[1, 1].set_yticks([0])
            axes[1, 1].set_yticklabels(['Final'])
            axes[1, 1].set_xticks(range(len(self.class_names)))
            axes[1, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
            axes[1, 1].set_title(f'Per-Class Accuracy (%) - Fold {fold+1}')
            
            # Add text annotations
            for i, acc in enumerate(final_per_class):
                axes[1, 1].text(i, 0, f'{acc:.0f}', ha='center', va='center')
            
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_curves_fold_{fold+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def train_fold(self, fold, train_indices, val_indices, config):
        """Train single fold with enhanced monitoring"""
        logging.info(f"\n{'='*60}")
        logging.info(f"Training Fold {fold+1}/{self.num_folds}")
        logging.info(f"Train: {len(train_indices)} samples, Val: {len(val_indices)} samples")
        logging.info(f"{'='*60}")
        
        # Create data loaders
        train_loader, val_loader, class_weights = self.create_data_loaders(
            train_indices, val_indices, config['batch_size']
        )
        
        # Initialize model
        model = EnhancedVideoMAEClassifier(
            num_classes=self.num_classes,
            dropout_rate=config['dropout_rate'],
            hidden_dim=config['hidden_dim']
        ).to(self.device)
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=config['learning_rate'] * 0.01
        )
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_mean_class_acc': [], 'per_class_acc': [],
            'lr': []
        }
        
        best_val_acc = 0
        best_mean_class_acc = 0
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, config['grad_clip']
            )
            
            # Validate
            val_loss, val_acc, mean_class_acc, per_class_acc = self.validate(
                model, val_loader, criterion
            )
            
            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_mean_class_acc'].append(mean_class_acc)
            history['per_class_acc'].append(per_class_acc)
            history['lr'].append(current_lr)
            
            # Log epoch results
            logging.info(f"Epoch {epoch+1}/{config['epochs']}")
            logging.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            logging.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Mean Class Acc: {mean_class_acc:.2f}%")
            logging.info(f"  LR: {current_lr:.6f}")
            
            # Log worst performing classes
            worst_classes = np.argsort(per_class_acc)[:3]
            for idx in worst_classes:
                if per_class_acc[idx] < 70:
                    logging.warning(f"  Low accuracy for {self.class_names[idx]}: {per_class_acc[idx]:.1f}%")
            
            # Save best model
            if mean_class_acc > best_mean_class_acc:
                best_mean_class_acc = mean_class_acc
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'mean_class_acc': mean_class_acc,
                    'per_class_acc': per_class_acc,
                    'config': config,
                    'class_names': self.class_names
                }
                torch.save(checkpoint, self.output_dir / f'best_model_fold_{fold+1}.pth')
                logging.info(f"  ✓ Saved best model (Mean Class Acc: {mean_class_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Plot training curves
        self.plot_training_curves(history, fold)
        
        # Save training history
        history_file = self.output_dir / f'history_fold_{fold+1}.json'
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        return best_val_acc, best_mean_class_acc, history
    
    def train(self, config=None):
        """Main training loop with K-Fold validation"""
        if config is None:
            config = {
                'batch_size': 8,
                'epochs': 50,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'dropout_rate': 0.5,
                'hidden_dim': 512,
                'grad_clip': 1.0,
                'patience': 15
            }
        
        # Log configuration
        logging.info("Training Configuration:")
        for key, value in config.items():
            logging.info(f"  {key}: {value}")
        
        # Get labels for stratification
        labels = np.array(self.dataset.labels)
        
        # Initialize K-Fold
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        
        # Results storage
        fold_results = []
        all_histories = []
        
        # Train each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(labels)), labels)):
            val_acc, mean_class_acc, history = self.train_fold(
                fold, train_idx, val_idx, config
            )
            
            fold_results.append({
                'fold': fold + 1,
                'val_acc': val_acc,
                'mean_class_acc': mean_class_acc,
                'best_epoch': history['val_mean_class_acc'].index(max(history['val_mean_class_acc'])) + 1
            })
            all_histories.append(history)
            
            logging.info(f"\nFold {fold+1} Results:")
            logging.info(f"  Best Val Acc: {val_acc:.2f}%")
            logging.info(f"  Best Mean Class Acc: {mean_class_acc:.2f}%")
        
        # Calculate final statistics
        val_accs = [r['val_acc'] for r in fold_results]
        mean_class_accs = [r['mean_class_acc'] for r in fold_results]
        
        # Print final results
        logging.info(f"\n{'='*60}")
        logging.info("K-FOLD CROSS-VALIDATION RESULTS")
        logging.info(f"{'='*60}")
        
        for result in fold_results:
            logging.info(f"Fold {result['fold']}: Val Acc={result['val_acc']:.2f}%, "
                        f"Mean Class Acc={result['mean_class_acc']:.2f}% "
                        f"(Best @ Epoch {result['best_epoch']})")
        
        logging.info(f"\nOverall Statistics:")
        logging.info(f"  Val Acc: {np.mean(val_accs):.2f}% ± {np.std(val_accs):.2f}%")
        logging.info(f"  Mean Class Acc: {np.mean(mean_class_accs):.2f}% ± {np.std(mean_class_accs):.2f}%")
        
        # Save comprehensive results
        results = {
            'fold_results': fold_results,
            'overall_val_acc': {
                'mean': float(np.mean(val_accs)),
                'std': float(np.std(val_accs)),
                'min': float(np.min(val_accs)),
                'max': float(np.max(val_accs))
            },
            'overall_mean_class_acc': {
                'mean': float(np.mean(mean_class_accs)),
                'std': float(np.std(mean_class_accs)),
                'min': float(np.min(mean_class_accs)),
                'max': float(np.max(mean_class_accs))
            },
            'config': config,
            'num_folds': self.num_folds,
            'dataset_info': {
                'num_samples': len(self.dataset),
                'num_classes': self.num_classes,
                'class_names': self.class_names
            },
            'training_date': datetime.now().isoformat()
        }
        
        results_file = self.output_dir / 'final_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"\nResults saved to: {results_file}")
        
        return results

# ====================== MAIN EXECUTION ======================

def main():
    """Main execution function"""
    
    # Configuration
    PREPROCESSING_CONFIG = {
        # Paths
        'input_dir': 'main_dataset',  # Your raw video dataset
        'output_dir': 'mediapipe_preprocessed',  # Preprocessed output
        
        # Preprocessing parameters (matching your enhanced preprocessor)
        'target_size': (224, 224),
        'num_frames': 16,
        'num_workers': 4,  # Multiprocessing workers
        'confidence': 0.3,
        'hand_padding_ratio': 0.35,
        'use_pose_fallback': True,
        'stabilize_crops': True,
        'min_hand_size_ratio': 0.05,
        'prioritize_signing_space': True,
        'signing_space_bounds': (0.1, 0.7),
        'apply_preprocessing_augmentation': True,
        'augmentation_probability': 0.3,
    }
    
    TRAINING_CONFIG = {
        'batch_size': 8,
        'epochs': 60,
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'dropout_rate': 0.5,
        'hidden_dim': 512,
        'grad_clip': 1.0,
        'patience': 15
    }
    
    print("="*70)
    print("HYBRID NSL PIPELINE: Enhanced Preprocessing + VideoMAE Training")
    print("="*70)
    
    # Step 1: Run enhanced preprocessing (if not already done)
    preprocessed_dir = Path(PREPROCESSING_CONFIG['output_dir'])
    
    if not preprocessed_dir.exists() or len(list(preprocessed_dir.glob('*/*.npz'))) == 0:
        print("\n[Step 1] Running Enhanced Preprocessing...")
        print("-"*50)
        
        # Note: Import and run your enhanced preprocessor here
        # from nsl_enhanced_preprocessor import preprocess_nsl_dataset
        preprocess_nsl_dataset(
            PREPROCESSING_CONFIG['input_dir'],
            PREPROCESSING_CONFIG['output_dir'],
            PREPROCESSING_CONFIG
        )
        
        print("⚠️  Please run the enhanced preprocessing first using your code!")
        print("Once preprocessing is complete, run this script again.")
        return
    else:
        print("\n✓ Preprocessed data found at:", preprocessed_dir)
    
    # Step 2: Train VideoMAE model with enhanced data
    print("\n[Step 2] Training VideoMAE with Enhanced Preprocessed Data...")
    print("-"*50)
    
    output_dir = Path("training_output_enhanced")
    trainer = NSLEnhancedTrainer(
        data_dir=preprocessed_dir,
        output_dir=output_dir,
        num_folds=5
    )
    
    results = trainer.train(TRAINING_CONFIG)
    
    # Step 3: Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Preprocessed data: {preprocessed_dir}")
    print(f"Training outputs: {output_dir}")
    print(f"Final accuracy: {results['overall_mean_class_acc']['mean']:.2f}% ± "
          f"{results['overall_mean_class_acc']['std']:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()