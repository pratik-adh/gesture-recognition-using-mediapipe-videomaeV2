#!/usr/bin/env python3
"""
Video Vision Transformer (ViViT) with Optuna Bayesian Optimization
Hyperparameter tuning for Nepali Sign Language Recognition
"""

import os, numpy as np, torch, torch.nn as nn, torch.optim as optim, random, json, logging, warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from PIL import Image
from collections import Counter
import optuna
from optuna.trial import TrialState

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed(42)

# ===== DATASET =====
class NPZDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, img_size=112, is_training=True):
        self.root_dir, self.num_frames, self.img_size, self.is_training = Path(root_dir), num_frames, img_size, is_training
        self.samples, self.classes = [], []
        self._scan_directory()
        
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def _scan_directory(self):
        for cid, d in enumerate(sorted([d for d in self.root_dir.iterdir() if d.is_dir()])):
            self.classes.append(d.name)
            files = list(d.glob("*.npz"))
            for f in files:
                self.samples.append((f, cid))
        logger.info(f"Loaded {len(self.classes)} classes, {len(self.samples)} samples")
    
    def load_video(self, path):
        try:
            data = np.load(path)
            frames = data['frames'] if 'frames' in data else data[data.files[0]]
            if frames.dtype != np.uint8:
                frames = (frames * 255).astype(np.uint8) if frames.max() <= 1.0 else np.clip(frames, 0, 255).astype(np.uint8)
            return frames
        except:
            return np.zeros((self.num_frames, 112, 112, 3), dtype=np.uint8)
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self.load_video(path)
        n = len(frames)
        
        # Sample frames
        if n == 0:
            frames = np.zeros((self.num_frames, 112, 112, 3), dtype=np.uint8)
        elif n <= self.num_frames:
            indices = list(range(n)) + [n-1] * (self.num_frames - n)
            frames = frames[indices]
        else:
            indices = np.linspace(0, n-1, self.num_frames, dtype=int)
            frames = frames[indices]
        
        video = torch.stack([self.transform(Image.fromarray(f)) for f in frames], dim=0)
        return video, label

# ===== ViViT MODEL =====

class SimplePatchEmbed(nn.Module):
    """Patch embedding"""
    def __init__(self, img_size=112, patch_size=14, embed_dim=160):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SimpleAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim=160, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class SimpleBlock(nn.Module):
    """Transformer block"""
    def __init__(self, dim=160, heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimpleAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViViT(nn.Module):
    """Video Vision Transformer"""
    def __init__(self, num_classes=36, num_frames=16, img_size=112, patch_size=14,
                 embed_dim=160, spatial_depth=2, temporal_depth=2, heads=4, dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = SimplePatchEmbed(img_size, patch_size, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable embeddings
        self.spatial_cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.temporal_cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, num_frames + 1, embed_dim) * 0.02)
        
        # Transformers
        self.spatial_blocks = nn.ModuleList([
            SimpleBlock(embed_dim, heads, 2.0, dropout) for _ in range(spatial_depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            SimpleBlock(embed_dim, heads, 2.0, dropout) for _ in range(temporal_depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Spatial processing
        x = x.view(B * T, C, H, W)
        x = self.patch_embed(x)
        
        cls_tokens = self.spatial_cls.expand(B * T, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.spatial_pos
        
        for block in self.spatial_blocks:
            x = block(x)
        
        x = x[:, 0]
        
        # Temporal processing
        x = x.view(B, T, self.embed_dim)
        
        temporal_cls = self.temporal_cls.expand(B, -1, -1)
        x = torch.cat([temporal_cls, x], dim=1)
        x = x + self.temporal_pos
        
        for block in self.temporal_blocks:
            x = block(x)
        
        # Classification
        x = x[:, 0]
        x = self.norm(x)
        x = self.head(x)
        
        return x

# ===== TRAINING FUNCTIONS =====

def train_epoch(model, loader, optimizer, criterion, device, gradient_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return total_loss / len(loader), acc, f1

# ===== OPTUNA OBJECTIVE =====

def objective(trial, train_dir, val_dir, n_trials_epochs=15):
    """
    Optuna objective function for hyperparameter optimization
    Uses reduced epochs for faster trials
    """
    
    # Suggest hyperparameters
    num_frames = trial.suggest_categorical('num_frames', [10, 12, 14, 16, 20])
    patch_size = trial.suggest_categorical('patch_size', [8, 14, 16])
    embed_dim = trial.suggest_categorical('embed_dim', [128, 160, 192, 256])
    spatial_depth = trial.suggest_int('spatial_depth', 1, 4)
    temporal_depth = trial.suggest_int('temporal_depth', 1, 4)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 6, 8])
    dropout = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.3, 0.4])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 24, 32])
    lr = trial.suggest_categorical('learning_rate', [1e-3, 3e-3, 5e-3])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3])
    gradient_clip = trial.suggest_categorical('gradient_clip', [0.5, 1.0, 2.0])
    num_workers = trial.suggest_categorical('num_workers', [2, 4, 6, 8])
    
    # Check embed_dim divisibility by num_heads
    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create datasets
        train_ds = NPZDataset(train_dir, num_frames, 112, True)
        val_ds = NPZDataset(val_dir, num_frames, 112, False)
        
        num_classes = len(train_ds.classes)
        
        train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
        
        # Create model
        model = ViViT(
            num_classes=num_classes,
            num_frames=num_frames,
            img_size=112,
            patch_size=patch_size,
            embed_dim=embed_dim,
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
            heads=num_heads,
            dropout=dropout
        )
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        
        # Optimizer and criterion
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, 
                             weight_decay=weight_decay, nesterov=True)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        # Training loop (reduced epochs for trials)
        best_val_acc = 0.0
        patience_counter = 0
        patience = 5
        
        for epoch in range(n_trials_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, 
                                               criterion, device, gradient_clip)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            
            logger.info(f"Trial {trial.number} | Epoch {epoch+1}/{n_trials_epochs} | "
                       f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping for trial
            if patience_counter >= patience:
                logger.info(f"Trial {trial.number} early stopped at epoch {epoch+1}")
                break
            
            # Report intermediate value for pruning
            trial.report(val_acc, epoch)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_acc
    
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.TrialPruned()

# ===== FINAL TRAINING WITH BEST PARAMS =====

def final_training(best_params, train_dir, val_dir, save_dir, full_epochs=50):
    """Train final model with optimized hyperparameters"""
    
    logger.info("\n" + "="*70)
    logger.info("FINAL TRAINING WITH OPTIMIZED PARAMETERS")
    logger.info("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    (save_dir / "plots").mkdir(exist_ok=True)
    
    # Create datasets
    train_ds = NPZDataset(train_dir, best_params['num_frames'], 112, True)
    val_ds = NPZDataset(val_dir, best_params['num_frames'], 112, False)
    num_classes = len(train_ds.classes)
    
    train_loader = DataLoader(train_ds, best_params['batch_size'], shuffle=True,
                              num_workers=best_params['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, best_params['batch_size'], shuffle=False,
                           num_workers=best_params['num_workers'], pin_memory=True)
    
    # Create model
    model = ViViT(
        num_classes=num_classes,
        num_frames=best_params['num_frames'],
        img_size=112,
        patch_size=best_params['patch_size'],
        embed_dim=best_params['embed_dim'],
        spatial_depth=best_params['spatial_depth'],
        temporal_depth=best_params['temporal_depth'],
        heads=best_params['num_heads'],
        dropout=best_params['dropout']
    )
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=best_params['learning_rate'],
                         momentum=0.9, weight_decay=best_params['weight_decay'], nesterov=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    patience = 20
    
    # Training loop
    for epoch in range(full_epochs):
        logger.info(f"\nEpoch {epoch+1}/{full_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, 
                                           criterion, device, best_params['gradient_clip'])
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'model_state_dict': state,
                'accuracy': best_val_acc,
                'epoch': epoch,
                'hyperparameters': best_params
            }, save_dir / "best_model.pth")
            
            logger.info(f"✓ Best model saved! Acc: {best_val_acc*100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = list(range(1, len(train_losses) + 1))
    
    axes[0].plot(epochs, [a*100 for a in train_accs], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, [a*100 for a in val_accs], 'r-', label='Val', linewidth=2)
    axes[0].set_title('Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
    axes[1].set_title('Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "plots" / "training_curves.png", dpi=150)
    plt.close()
    
    logger.info("\n" + "="*70)
    logger.info(f"FINAL TRAINING COMPLETE - Best Val Acc: {best_val_acc*100:.2f}%")
    logger.info("="*70)
    
    return best_val_acc

# ===== MAIN =====

def main():
    # Configuration
    TRAIN_DIR = '/kaggle/input/npz-lightweight-videos-without-cropping-dataset/npz_lightweight_videos_without_cropping_splitted_dataset/train'
    VAL_DIR = '/kaggle/input/npz-lightweight-videos-without-cropping-dataset/npz_lightweight_videos_without_cropping_splitted_dataset/val'
    SAVE_DIR = '/kaggle/working/'
    
    N_TRIALS = 50  # Number of Optuna trials
    N_TRIALS_EPOCHS = 15  # Epochs per trial (reduced for faster optimization)
    FULL_EPOCHS = 50  # Full training epochs with best params
    
    logger.info("\n" + "="*70)
    logger.info("ViViT HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    logger.info("="*70)
    logger.info(f"Trials: {N_TRIALS} | Epochs per trial: {N_TRIALS_EPOCHS}")
    logger.info(f"Full training epochs: {FULL_EPOCHS}")
    logger.info("="*70 + "\n")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Optimize
    logger.info("Starting Optuna optimization...")
    study.optimize(
        lambda trial: objective(trial, TRAIN_DIR, VAL_DIR, N_TRIALS_EPOCHS),
        n_trials=N_TRIALS,
        timeout=None,
        show_progress_bar=True
    )
    
    # Get best parameters
    logger.info("\n" + "="*70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best validation accuracy: {study.best_value*100:.2f}%")
    logger.info("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Save optimization results
    save_dir = Path(SAVE_DIR)
    with open(save_dir / 'optuna_study.json', 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials)
        }, f, indent=4)
    
    # Plot optimization history
    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(save_dir / 'plots' / 'optuna_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(save_dir / 'plots' / 'optuna_importances.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate Optuna plots: {str(e)}")
    
    # Final training with best parameters
    logger.info("\n" + "="*70)
    logger.info("STARTING FINAL TRAINING")
    logger.info("="*70)
    
    final_acc = final_training(study.best_params, TRAIN_DIR, VAL_DIR, SAVE_DIR, FULL_EPOCHS)
    
    logger.info("\n" + "="*70)
    logger.info(f"ALL DONE! Final Best Accuracy: {final_acc*100:.2f}%")
    logger.info("="*70)

if __name__ == "__main__":
    main()