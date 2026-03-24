#!/usr/bin/env python3
"""
Optimized VideoMAE Training Pipeline with Optuna Hyperparameter Optimization
"""

import os
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
import random
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
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

class NPZDirectoryDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, is_training=True, 
                 model_name="MCG-NJU/videomae-small-finetuned-kinetics"):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.is_training = is_training
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        self._scan_directory()
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)

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
        except Exception:
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

    def temporal_sampling(self, frames):
        n = len(frames)
        if n == 0:
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
        if n <= self.num_frames:
            indices = list(range(n)) + [n-1] * (self.num_frames - n)
        else:
            indices = np.linspace(0, n-1, self.num_frames, dtype=int)
        return frames[indices]

    def apply_augmentations(self, frames):
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

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self.load_video(sample['path'])
        label = sample['class_idx']
        frames = self.temporal_sampling(frames)
        if self.is_training:
            frames = self.apply_augmentations(frames)
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, label

class VideoMAEClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4, freeze_backbone=False, 
                 freeze_layers=0, model_name="MCG-NJU/videomae-small-finetuned-kinetics"):
        super().__init__()
        self.videomae = VideoMAEForVideoClassification.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        hidden_size = self.videomae.config.hidden_size
        num_layers = self.videomae.config.num_hidden_layers

        if freeze_backbone:
            for p in self.videomae.videomae.parameters():
                p.requires_grad = False
        elif freeze_layers > 0:
            n = min(freeze_layers, num_layers)
            for p in self.videomae.videomae.embeddings.parameters():
                p.requires_grad = False
            for i in range(n):
                for p in self.videomae.videomae.encoder.layer[i].parameters():
                    p.requires_grad = False

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

class Trainer:
    def __init__(self, config, trial=None):
        self.cfg = config
        self.trial = trial
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.device = torch.device('cuda')
        self.n_gpus = torch.cuda.device_count()
        
        self.save_dir = Path(self.cfg['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        (self.save_dir / "logs").mkdir(exist_ok=True)

        self.scaler = GradScaler(init_scale=512.0, growth_interval=200)

        if trial is None:
            self.writer = SummaryWriter(
                log_dir=str(self.save_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
            )
        else:
            self.writer = None

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
            self.cfg['train_dir'], num_frames=self.cfg['num_frames'], 
            is_training=True, model_name=self.cfg['model_name']
        )
        val_ds = NPZDirectoryDataset(
            self.cfg['val_dir'], num_frames=self.cfg['num_frames'], 
            is_training=False, model_name=self.cfg['model_name']
        )

        self.class_names = train_ds.classes
        self.num_classes = len(self.class_names)

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg['batch_size'], shuffle=True,
            num_workers=self.cfg['num_workers'], pin_memory=True,
            persistent_workers=True if self.cfg['num_workers'] > 0 else False,
            prefetch_factor=2 if self.cfg['num_workers'] > 0 else None
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg['batch_size'], shuffle=False,
            num_workers=self.cfg['num_workers'], pin_memory=True,
            persistent_workers=True if self.cfg['num_workers'] > 0 else False,
            prefetch_factor=2 if self.cfg['num_workers'] > 0 else None
        )
        return train_loader, val_loader

    def train_epoch(self, model, loader, optimizer, criterion, epoch):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        skipped_batches = 0

        for batch_idx, (inputs, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue
            
            self.scaler.scale(loss).backward()
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

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / max(1, len(loader) - skipped_batches)
        epoch_acc = correct / total if total > 0 else 0.0

        return epoch_loss, epoch_acc

    def evaluate(self, model, loader, criterion, epoch):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Validation"):
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
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')

        all_probs = np.array(all_probs)
        top3 = top_k_accuracy_score(all_labels, all_probs, k=min(3, self.num_classes))
        top5 = top_k_accuracy_score(all_labels, all_probs, k=min(5, self.num_classes))

        return {
            'loss': epoch_loss, 'accuracy': epoch_acc, 'f1': epoch_f1,
            'top3': top3, 'top5': top5, 'predictions': all_preds, 'labels': all_labels
        }

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
        plt.savefig(self.save_dir / "plots" / "training_curves.png", dpi=300)
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=self.class_names, yticklabels=self.class_names, cmap='Blues')
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.save_dir / "plots" / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(self.save_dir / "classification_report.csv")

    def train(self):
        train_loader, val_loader = self.create_data_loaders()

        model = VideoMAEClassifier(
            num_classes=self.num_classes, dropout_rate=self.cfg['dropout_rate'],
            freeze_backbone=self.cfg.get('freeze_backbone', False),
            freeze_layers=self.cfg.get('freeze_layers', 0),
            model_name=self.cfg['model_name']
        )

        if self.n_gpus > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.cfg['learning_rate'], weight_decay=self.cfg['weight_decay']
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                     patience=self.cfg['lr_patience'], verbose=True)
        criterion = nn.CrossEntropyLoss()

        best_preds, best_labels = [], []

        for epoch in range(self.start_epoch, self.cfg['epochs'] + 1):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            val_metrics = self.evaluate(model, val_loader, criterion, epoch)

            scheduler.step(val_metrics['loss'])
            current_lr = optimizer.param_groups[0]['lr']

            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.train_accs.append(train_acc)
            self.val_accs.append(val_metrics['accuracy'])
            self.lrs.append(current_lr)

            logger.info(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_metrics['accuracy']:.4f}, LR={current_lr:.6f}")

            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                best_preds = val_metrics['predictions']
                best_labels = val_metrics['labels']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.trial is not None:
                self.trial.report(val_metrics['accuracy'], epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

            if self.patience_counter >= self.cfg['early_stopping_patience']:
                break

        if self.trial is None:
            self.plot_training_curves()
            if len(best_preds) > 0:
                self.plot_confusion_matrix(best_labels, best_preds)
                self.save_classification_report(best_labels, best_preds)
            if self.writer:
                self.writer.close()

        return self.best_val_acc

def objective(trial, base_config):
    config = base_config.copy()
    
    config['num_frames'] = trial.suggest_categorical('num_frames', [8, 12, 16, 20, 24])
    config['batch_size'] = trial.suggest_categorical('batch_size', [4, 8, 16, 24, 32])
    config['learning_rate'] = trial.suggest_categorical('learning_rate', [3e-4, 4e-4, 5e-4, 6e-4])
    config['weight_decay'] = trial.suggest_categorical('weight_decay', [1e-4, 2e-4, 3e-4, 4e-4])
    config['dropout_rate'] = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3, 0.4, 0.5])
    config['gradient_clip'] = trial.suggest_categorical('gradient_clip', [0.3, 0.5, 1.0, 2.0])
    config['num_workers'] = trial.suggest_categorical('num_workers', [2, 4, 6, 8])
    
    trainer = Trainer(config, trial=trial)
    best_val_acc = trainer.train()
    
    return best_val_acc

def main():
    base_config = {
        'train_dir': '/kaggle/input/npz-preprocessed-videos-splitted-dataset/npz_preprocessed_videos_splitted_dataset/train',
        'val_dir': '/kaggle/input/npz-preprocessed-videos-splitted-dataset/npz_preprocessed_videos_splitted_dataset/val',
        'test_dir': '/kaggle/input/npz-preprocessed-videos-splitted-dataset/npz_preprocessed_videos_splitted_dataset/test',
        'save_dir': '/kaggle/working/',
        'model_name': 'MCG-NJU/videomae-small-finetuned-kinetics',
        'freeze_backbone': False,
        'freeze_layers': 2,
        'epochs': 50,
        'lr_patience': 7,
        'early_stopping_patience': 7,
        'checkpoint_interval': 5,
    }


    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    study.optimize(lambda trial: objective(trial, base_config), n_trials=30)
    
    results_df = study.trials_dataframe()
    results_df.to_csv(Path(base_config['save_dir']) / "optuna_results.csv", index=False)
    logger.info(f"\nResults saved to optuna_results.csv")

    logger.info("\n" + "="*70)
    logger.info("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    logger.info("="*70)
    
    final_config = base_config.copy()
    final_config.update(study.best_params)
    
    trainer = Trainer(final_config, trial=None)
    trainer.train()

if __name__ == "__main__":
    main()