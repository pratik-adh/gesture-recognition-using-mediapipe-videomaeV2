# #!/usr/bin/env python3
# """
# Fixed VideoMAE Training Pipeline for Nepali Sign Language Recognition
# macOS compatible - Resolves pickle/multiprocessing issues
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from transformers import (
#     VideoMAEForVideoClassification, 
#     VideoMAEImageProcessor,
#     get_cosine_schedule_with_warmup
# )
# import numpy as np
# from pathlib import Path
# import json
# import logging
# from tqdm import tqdm
# from datetime import datetime
# import wandb
# from sklearn.model_selection import StratifiedKFold, GroupKFold
# from sklearn.metrics import f1_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict, Counter
# import random
# import cv2
# from PIL import Image
# import warnings
# import gc
# import traceback
# import os

# warnings.filterwarnings('ignore')

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO, 
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# def verify_preprocessed_data(data_path):
#     """Verify the format of preprocessed data"""
#     try:
#         data = np.load(data_path, allow_pickle=True)
#         frames = data['frames']
        
#         # Check expected shape
#         if frames.ndim != 4:
#             logger.error(f"Wrong frame dimensions: {frames.shape}")
#             return False
        
#         # Check frame count
#         if frames.shape[0] != 16:
#             logger.warning(f"Unexpected frame count: {frames.shape[0]}")
        
#         # Check frame size
#         if frames.shape[1:3] != (224, 224):
#             logger.warning(f"Unexpected frame size: {frames.shape[1:3]}")
        
#         # Check channels
#         if frames.shape[3] != 3:
#             logger.error(f"Wrong channel count: {frames.shape[3]}")
#             return False
        
#         # Check data type and range
#         if frames.dtype != np.uint8:
#             logger.warning(f"Frames not uint8: {frames.dtype}")
        
#         if frames.min() < 0 or frames.max() > 255:
#             logger.error(f"Invalid pixel range: [{frames.min()}, {frames.max()}]")
#             return False
        
#         return True
#     except Exception as e:
#         logger.error(f"Error verifying {data_path}: {e}")
#         return False


# # Global cache for data loading (to avoid pickle issues)
# _data_cache = {}

# def load_cached_data(path):
#     """Load data with simple caching (pickle-safe)"""
#     path_str = str(path)
#     if path_str not in _data_cache:
#         _data_cache[path_str] = np.load(path_str, allow_pickle=True)
#     return _data_cache[path_str]


# class FixedNSLDataset(Dataset):
#     """
#     Fixed dataset that properly handles MediaPipe preprocessed data
#     macOS compatible version without LRU cache
#     """
    
#     def __init__(self, 
#                  data_paths, 
#                  labels, 
#                  processor,
#                  augment=True,
#                  mix_up_alpha=0.2,
#                  cutmix_prob=0.5,
#                  use_detection_weighting=True,
#                  debug_mode=False):
#         """
#         Args:
#             data_paths: List of .npz file paths
#             labels: List of labels
#             processor: VideoMAE processor
#             augment: Whether to apply augmentations
#             mix_up_alpha: MixUp augmentation parameter
#             cutmix_prob: CutMix probability
#             use_detection_weighting: Weight samples by detection quality
#             debug_mode: Enable debug logging
#         """
#         self.data_paths = [Path(p) for p in data_paths]
#         self.labels = labels
#         self.processor = processor
#         self.augment = augment
#         self.mix_up_alpha = mix_up_alpha
#         self.cutmix_prob = cutmix_prob
#         self.use_detection_weighting = use_detection_weighting
#         self.debug_mode = debug_mode
        
#         # Verify data paths exist
#         missing_files = [p for p in self.data_paths if not p.exists()]
#         if missing_files:
#             logger.error(f"Missing files: {missing_files[:5]}")
#             raise FileNotFoundError(f"{len(missing_files)} files not found")
        
#         # Load and verify metadata
#         self.detection_rates = []
#         self.enhanced_rates = []
#         self.metadata_stats = defaultdict(list)
#         self.valid_indices = []
        
#         logger.info("Loading and verifying dataset...")
#         for idx, path in enumerate(tqdm(self.data_paths, desc="Verifying data")):
#             try:
#                 # Load with allow_pickle for metadata
#                 data = load_cached_data(str(path))
                
#                 # Verify frames exist and have correct shape
#                 if 'frames' not in data:
#                     logger.warning(f"No frames in {path}")
#                     continue
                
#                 frames = data['frames']
#                 if frames.shape != (16, 224, 224, 3):
#                     logger.warning(f"Unexpected shape {frames.shape} in {path}")
#                     # Don't skip, we'll handle in __getitem__
                
#                 # Extract metadata safely
#                 metadata = {}
#                 if 'metadata' in data:
#                     try:
#                         metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
#                         if not isinstance(metadata, dict):
#                             metadata = {}
#                     except:
#                         metadata = {}
                
#                 # Store detection rates
#                 detection_rate = metadata.get('detection_rate', 1.0)
#                 enhanced_rate = metadata.get('enhanced_rate', detection_rate)
                
#                 self.detection_rates.append(detection_rate)
#                 self.enhanced_rates.append(enhanced_rate)
#                 self.valid_indices.append(idx)
                
#                 # Collect metadata statistics
#                 for key, value in metadata.items():
#                     if isinstance(value, (int, float)):
#                         self.metadata_stats[key].append(value)
                
#             except Exception as e:
#                 logger.error(f"Error loading {path}: {e}")
#                 # Use default values
#                 self.detection_rates.append(0.5)
#                 self.enhanced_rates.append(0.5)
#                 self.valid_indices.append(idx)
        
#         if not self.valid_indices:
#             raise ValueError("No valid data files found!")
        
#         # Calculate sample weights
#         if use_detection_weighting and self.enhanced_rates:
#             # Combine detection quality with class balancing
#             combined_rates = np.array(self.enhanced_rates)
#             # Avoid zero weights
#             combined_rates = np.clip(combined_rates, 0.1, 1.0)
#             self.sample_weights = combined_rates / combined_rates.sum()
#         else:
#             self.sample_weights = np.ones(len(self.data_paths)) / len(self.data_paths)
        
#         # Class distribution
#         self.class_counts = Counter(labels)
#         self.num_classes = len(set(labels))
        
#         # Calculate class weights for loss weighting
#         total_samples = len(labels)
#         self.class_weights = torch.tensor([
#             total_samples / (self.num_classes * max(self.class_counts[i], 1)) 
#             for i in range(self.num_classes)
#         ], dtype=torch.float32)
        
#         # Log statistics
#         logger.info(f"Dataset initialized: {len(self.valid_indices)}/{len(data_paths)} valid samples")
#         logger.info(f"Detection rate: mean={np.mean(self.detection_rates):.2f}, "
#                    f"std={np.std(self.detection_rates):.2f}")
#         logger.info(f"Class distribution: {dict(self.class_counts)}")
        
#         # Debug: verify first sample
#         if self.debug_mode and len(self.valid_indices) > 0:
#             self._debug_first_sample()
    
#     def _debug_first_sample(self):
#         """Debug first sample to verify data loading"""
#         try:
#             logger.info("Debug: Testing first sample...")
#             sample = self.__getitem__(0)
#             logger.info(f"Sample keys: {sample.keys()}")
#             logger.info(f"Pixel values shape: {sample['pixel_values'].shape}")
#             logger.info(f"Label: {sample['labels']}")
#             logger.info("First sample loaded successfully!")
#         except Exception as e:
#             logger.error(f"Error loading first sample: {e}")
#             traceback.print_exc()
    
#     def __len__(self):
#         return len(self.data_paths)
    
#     def __getitem__(self, idx):
#         try:
#             # Load preprocessed data
#             data = load_cached_data(str(self.data_paths[idx]))
            
#             # Extract frames - CRITICAL: ensure correct format
#             frames = data['frames']
            
#             # Ensure frames are in correct format (16, 224, 224, 3)
#             if frames.shape != (16, 224, 224, 3):
#                 logger.warning(f"Reshaping frames from {frames.shape} to (16, 224, 224, 3)")
#                 # Handle common issues
#                 if frames.shape[0] < 16:
#                     # Pad with last frame
#                     padding = np.repeat(frames[-1:], 16 - frames.shape[0], axis=0)
#                     frames = np.concatenate([frames, padding], axis=0)
#                 elif frames.shape[0] > 16:
#                     # Take first 16
#                     frames = frames[:16]
                
#                 # Resize if needed
#                 if frames.shape[1:3] != (224, 224):
#                     resized = []
#                     for frame in frames:
#                         resized.append(cv2.resize(frame, (224, 224)))
#                     frames = np.array(resized)
            
#             # Ensure uint8 type
#             if frames.dtype != np.uint8:
#                 frames = np.clip(frames, 0, 255).astype(np.uint8)
            
#             # Make a copy to avoid modifying cached data
#             frames = frames.copy()
            
#             # Get label
#             label = self.labels[idx]
            
#             # Extract metadata
#             metadata = {}
#             if 'metadata' in data:
#                 try:
#                     metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
#                     if not isinstance(metadata, dict):
#                         metadata = {}
#                 except:
#                     metadata = {}
            
#             # Extract detection info
#             detection_info = []
#             if 'detection_info' in data:
#                 try:
#                     detection_info = data['detection_info']
#                     if isinstance(detection_info, np.ndarray):
#                         detection_info = detection_info.tolist() if detection_info.size > 0 else []
#                 except:
#                     detection_info = []
            
#             # Apply augmentations
#             if self.augment:
#                 frames = self.apply_augmentations(frames, metadata, detection_info)
                
#                 # MixUp or CutMix with lower probability for small dataset
#                 if random.random() < 0.2:  # Reduced from 0.3
#                     frames, label = self.apply_mixing_augmentation(frames, label, idx)
            
#             # CRITICAL: Convert frames to PIL Images for VideoMAE processor
#             # VideoMAE expects PIL Images, not numpy arrays directly
#             pil_frames = []
#             for frame in frames:
#                 # Ensure frame is uint8 and in [0, 255] range
#                 frame = np.clip(frame, 0, 255).astype(np.uint8)
#                 pil_frames.append(Image.fromarray(frame))
            
#             # Process with VideoMAE processor
#             inputs = self.processor(
#                 pil_frames, 
#                 return_tensors="pt",
#                 do_resize=False,  # We already resized
#                 do_center_crop=False  # We already cropped
#             )
            
#             # Extract pixel values and squeeze batch dimension
#             pixel_values = inputs['pixel_values'].squeeze(0)
            
#             # Verify output shape
#             if self.debug_mode and idx == 0:
#                 logger.info(f"Processed pixel_values shape: {pixel_values.shape}")
#                 logger.info(f"Pixel values range: [{pixel_values.min():.2f}, {pixel_values.max():.2f}]")
            
#             # Return with proper label format
#             if isinstance(label, tuple):
#                 # Mixed label from augmentation
#                 return {
#                     'pixel_values': pixel_values,
#                     'labels': label,
#                     'detection_rate': metadata.get('enhanced_rate', metadata.get('detection_rate', 1.0)),
#                     'original_idx': idx
#                 }
#             else:
#                 # Regular label
#                 return {
#                     'pixel_values': pixel_values,
#                     'labels': torch.tensor(label, dtype=torch.long),
#                     'detection_rate': metadata.get('enhanced_rate', metadata.get('detection_rate', 1.0)),
#                     'original_idx': idx
#                 }
                
#         except Exception as e:
#             logger.error(f"Error loading sample {idx} from {self.data_paths[idx]}: {e}")
#             traceback.print_exc()
            
#             # Return a dummy sample to prevent training crash
#             dummy_frames = torch.zeros((3, 16, 224, 224))
#             return {
#                 'pixel_values': dummy_frames,
#                 'labels': torch.tensor(0, dtype=torch.long),
#                 'detection_rate': 0.0,
#                 'original_idx': idx
#             }
    
#     def apply_augmentations(self, frames, metadata, detection_info):
#         """Apply augmentations with adjusted probabilities for small dataset"""
#         detection_rate = metadata.get('enhanced_rate', metadata.get('detection_rate', 1.0))
        
#         # Reduce augmentation intensity for small dataset
#         if detection_rate > 0.7:
#             # Light augmentations only
#             if random.random() < 0.2:  # Reduced probability
#                 frames = self.temporal_shift(frames, max_shift=1)
            
#             if random.random() < 0.3:
#                 frames = self.color_jitter(frames, intensity=0.1)
            
#             if random.random() < 0.2:
#                 frames = self.brightness_adjustment(frames, factor_range=(0.9, 1.1))
        
#         elif detection_rate > 0.4:
#             # Very light augmentations
#             if random.random() < 0.15:
#                 frames = self.brightness_adjustment(frames, factor_range=(0.95, 1.05))
        
#         return frames
    
#     def temporal_shift(self, frames, max_shift=1):
#         """Shift frames temporally"""
#         if len(frames) <= max_shift * 2:
#             return frames
        
#         shift = random.randint(-max_shift, max_shift)
#         if shift == 0:
#             return frames
        
#         shifted = np.zeros_like(frames)
#         if shift > 0:
#             shifted[shift:] = frames[:-shift]
#             shifted[:shift] = frames[0]
#         else:
#             shifted[:shift] = frames[-shift:]
#             shifted[shift:] = frames[-1]
        
#         return shifted
    
#     def color_jitter(self, frames, intensity=0.1):
#         """Apply mild color jittering"""
#         # Brightness
#         brightness = random.uniform(1-intensity, 1+intensity)
#         frames = np.clip(frames * brightness, 0, 255)
        
#         # Contrast (mild)
#         if random.random() < 0.3:
#             contrast = random.uniform(1-intensity/2, 1+intensity/2)
#             frames = np.clip((frames - 128) * contrast + 128, 0, 255)
        
#         return frames.astype(np.uint8)
    
#     def brightness_adjustment(self, frames, factor_range=(0.9, 1.1)):
#         """Simple brightness adjustment"""
#         factor = random.uniform(*factor_range)
#         return np.clip(frames * factor, 0, 255).astype(np.uint8)
    
#     def apply_mixing_augmentation(self, frames, label, idx):
#         """Apply MixUp or CutMix augmentation (with reduced intensity)"""
#         # Get a random sample
#         mix_idx = random.randint(0, len(self.data_paths) - 1)
#         if mix_idx == idx:
#             mix_idx = (idx + 1) % len(self.data_paths)
        
#         # Load mixing sample
#         mix_data = load_cached_data(str(self.data_paths[mix_idx]))
#         mix_frames = mix_data['frames'].copy()
        
#         # Ensure same shape
#         if mix_frames.shape != frames.shape:
#             if mix_frames.shape[0] < 16:
#                 padding = np.repeat(mix_frames[-1:], 16 - mix_frames.shape[0], axis=0)
#                 mix_frames = np.concatenate([mix_frames, padding], axis=0)
#             elif mix_frames.shape[0] > 16:
#                 mix_frames = mix_frames[:16]
        
#         mix_label = self.labels[mix_idx]
        
#         # Use only MixUp for small dataset (CutMix can be too aggressive)
#         lam = np.random.beta(self.mix_up_alpha, self.mix_up_alpha)
#         lam = max(0.7, lam)  # Ensure dominant sample
        
#         frames = (lam * frames + (1 - lam) * mix_frames).astype(np.uint8)
        
#         return frames, (torch.tensor(label, dtype=torch.long), 
#                        torch.tensor(mix_label, dtype=torch.long), 
#                        lam)


# def custom_collate_fn(batch):
#     """Custom collate function to handle mixed labels and filter invalid samples"""
#     # Filter out invalid samples
#     valid_batch = []
#     for item in batch:
#         if item['pixel_values'].shape[0] > 0:  # Check if not dummy sample
#             valid_batch.append(item)
    
#     if not valid_batch:
#         # Return a minimal valid batch if all samples are invalid
#         return {
#             'pixel_values': torch.zeros((1, 3, 16, 224, 224)),
#             'labels': torch.tensor([0]),
#             'detection_rate': torch.tensor([0.0]),
#             'original_idx': torch.tensor([0])
#         }
    
#     # Separate the batch components
#     pixel_values = []
#     labels = []
#     detection_rates = []
#     original_idxs = []
    
#     for item in valid_batch:
#         pixel_values.append(item['pixel_values'])
#         labels.append(item['labels'])
#         detection_rates.append(item['detection_rate'])
#         original_idxs.append(item['original_idx'])
    
#     # Stack pixel values
#     pixel_values = torch.stack(pixel_values)
    
#     # Handle labels
#     has_mixed = any(isinstance(l, tuple) for l in labels)
    
#     if has_mixed:
#         processed_labels = labels
#     else:
#         processed_labels = torch.stack(labels)
    
#     return {
#         'pixel_values': pixel_values,
#         'labels': processed_labels,
#         'detection_rate': torch.tensor(detection_rates),
#         'original_idx': torch.tensor(original_idxs)
#     }


# class EnhancedNSLTrainer:
#     """
#     Production-ready trainer with advanced techniques
#     """
    
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device(
#             'cuda' if torch.cuda.is_available() 
#             else 'mps' if torch.backends.mps.is_available() 
#             else 'cpu'
#         )
#         logger.info(f"Using device: {self.device}")
        
#         # Initialize processor
#         self.processor = VideoMAEImageProcessor.from_pretrained(
#             config['model_name']
#         )
        
#         # Setup experiment tracking
#         if config.get('use_wandb', False):
#             wandb.init(
#                 project="nsl-recognition",
#                 config=config,
#                 name=f"videomae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#             )
        
#         # Results storage
#         self.fold_results = []
#         self.best_models = []
        
#         # Training history for analysis
#         self.training_history = defaultdict(list)
    
#     def create_model(self):
#         """Create and initialize model"""
#         model = VideoMAEForVideoClassification.from_pretrained(
#             self.config['model_name'],
#             num_labels=self.config['num_classes'],
#             ignore_mismatched_sizes=True
#         )
        
#         # Freeze early layers for faster training (optional)
#         if self.config.get('freeze_backbone_layers', 0) > 0:
#             for i, layer in enumerate(model.videomae.encoder.layer):
#                 if i < self.config['freeze_backbone_layers']:
#                     for param in layer.parameters():
#                         param.requires_grad = False
#             logger.info(f"Froze first {self.config['freeze_backbone_layers']} backbone layers")
        
#         return model.to(self.device)
    
#     def create_optimizer_scheduler(self, model, num_training_steps):
#         """Create optimizer and scheduler"""
#         # Separate parameters for different learning rates
#         backbone_params = []
#         head_params = []
        
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 if 'classifier' in name or 'fc' in name:
#                     head_params.append(param)
#                 else:
#                     backbone_params.append(param)
        
#         # Different learning rates for backbone and head
#         optimizer = torch.optim.AdamW([
#             {'params': backbone_params, 'lr': self.config['learning_rate']},
#             {'params': head_params, 'lr': self.config['learning_rate'] * 10}
#         ], weight_decay=self.config['weight_decay'])
        
#         # Cosine scheduler with warmup
#         scheduler = get_cosine_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=int(num_training_steps * 0.1),
#             num_training_steps=num_training_steps
#         )
        
#         return optimizer, scheduler
    
#     def compute_loss(self, outputs, labels, class_weights=None):
#         """Compute loss with label smoothing or mixing and class weighting"""
#         if isinstance(labels, list):
#             # Mixed batch - some labels might be tuples (mixed), some might be regular
#             total_loss = 0
#             num_samples = 0
            
#             for i, label in enumerate(labels):
#                 if isinstance(label, tuple):
#                     # Mixed label (label_a, label_b, lambda)
#                     labels_a, labels_b, lam = label
#                     # Get the logit for this sample
#                     logit = outputs.logits[i:i+1]  # Keep batch dimension
                    
#                     # Compute mixed loss with class weights
#                     if class_weights is not None:
#                         weight_a = class_weights[labels_a.item()]
#                         weight_b = class_weights[labels_b.item()]
#                         loss_a = F.cross_entropy(logit, labels_a.unsqueeze(0).to(self.device), 
#                                                 reduction='none') * weight_a
#                         loss_b = F.cross_entropy(logit, labels_b.unsqueeze(0).to(self.device), 
#                                                 reduction='none') * weight_b
#                     else:
#                         loss_a = F.cross_entropy(logit, labels_a.unsqueeze(0).to(self.device))
#                         loss_b = F.cross_entropy(logit, labels_b.unsqueeze(0).to(self.device))
                    
#                     sample_loss = lam * loss_a + (1 - lam) * loss_b
#                     total_loss += sample_loss.mean()
#                 else:
#                     # Regular label
#                     logit = outputs.logits[i:i+1]
#                     if class_weights is not None:
#                         weight = class_weights[label.item()]
#                         sample_loss = F.cross_entropy(logit, label.unsqueeze(0).to(self.device), 
#                                                      reduction='none') * weight
#                     else:
#                         sample_loss = F.cross_entropy(logit, label.unsqueeze(0).to(self.device))
#                     total_loss += sample_loss.mean()
                
#                 num_samples += 1
            
#             # Average loss over batch
#             loss = total_loss / num_samples if num_samples > 0 else total_loss
#         elif isinstance(labels, tuple):
#             # All labels are mixed
#             labels_a, labels_b, lam = labels
#             loss_a = F.cross_entropy(outputs.logits, labels_a)
#             loss_b = F.cross_entropy(outputs.logits, labels_b)
#             loss = lam * loss_a + (1 - lam) * loss_b
#         else:
#             # Regular labels with optional label smoothing
#             if self.config.get('label_smoothing', 0) > 0:
#                 loss = F.cross_entropy(
#                     outputs.logits, 
#                     labels, 
#                     label_smoothing=self.config['label_smoothing'],
#                     weight=class_weights.to(self.device) if class_weights is not None else None
#                 )
#             else:
#                 loss = F.cross_entropy(
#                     outputs.logits, 
#                     labels,
#                     weight=class_weights.to(self.device) if class_weights is not None else None
#                 )
        
#         return loss
    
#     def train_fold(self, fold_idx, train_paths, train_labels, val_paths, val_labels):
#         """Train on a single fold with advanced techniques"""
#         logger.info(f"\n{'='*50}")
#         logger.info(f"Training Fold {fold_idx + 1}/{self.config['num_folds']}")
#         logger.info(f"{'='*50}")
        
#         # Create model
#         model = self.create_model()
        
#         # Create datasets with debug mode for first fold
#         train_dataset = FixedNSLDataset(
#             train_paths, 
#             train_labels, 
#             self.processor,
#             augment=True,
#             mix_up_alpha=self.config.get('mixup_alpha', 0.1),
#             cutmix_prob=self.config.get('cutmix_prob', 0.0),
#             use_detection_weighting=True,
#             debug_mode=(fold_idx == 0)  # Debug first fold
#         )
        
#         val_dataset = FixedNSLDataset(
#             val_paths, 
#             val_labels, 
#             self.processor,
#             augment=False,
#             use_detection_weighting=False,
#             debug_mode=False
#         )
        
#         # Get class weights from train dataset
#         class_weights = train_dataset.class_weights
        
#         # Create weighted sampler for imbalanced classes
#         class_counts = Counter(train_labels)
#         class_sample_weights = {cls: 1.0 / max(count, 1) for cls, count in class_counts.items()}
#         sample_weights = [class_sample_weights[label] for label in train_labels]
        
#         # Combine with detection quality weights
#         if train_dataset.use_detection_weighting:
#             detection_weights = train_dataset.sample_weights
#             combined_weights = np.array(sample_weights) * detection_weights
#             combined_weights = combined_weights / combined_weights.sum()
#         else:
#             combined_weights = sample_weights
        
#         sampler = WeightedRandomSampler(
#             combined_weights, 
#             len(combined_weights),
#             replacement=True
#         )
        
#         # CRITICAL FOR MACOS: Set num_workers=0 to avoid multiprocessing issues
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=self.config['batch_size'],
#             sampler=sampler,
#             num_workers=0,  # Set to 0 for macOS compatibility
#             pin_memory=(self.device.type == 'cuda'),
#             drop_last=True,
#             collate_fn=custom_collate_fn
#         )
        
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=self.config['batch_size'],
#             shuffle=False,
#             num_workers=0,  # Set to 0 for macOS compatibility
#             pin_memory=(self.device.type == 'cuda'),
#             collate_fn=custom_collate_fn
#         )
        
#         # Create optimizer and scheduler
#         num_training_steps = len(train_loader) * self.config['epochs']
#         optimizer, scheduler = self.create_optimizer_scheduler(model, num_training_steps)
        
#         # Mixed precision training
#         scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
#         # Training metrics
#         best_val_acc = 0
#         best_val_f1 = 0
#         patience_counter = 0
#         train_losses = []
#         val_losses = []
#         val_accs = []
#         val_f1s = []
        
#         # Training loop
#         for epoch in range(self.config['epochs']):
#             # Train
#             model.train()
#             train_loss = 0
#             train_correct = 0
#             train_total = 0
            
#             pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
#             for batch_idx, batch in enumerate(pbar):
#                 pixel_values = batch['pixel_values'].to(self.device)
#                 labels = batch['labels']
#                 detection_rates = batch['detection_rate']
                
#                 # Handle mixed labels
#                 if isinstance(labels, list):
#                     processed_labels = []
#                     for l in labels:
#                         if isinstance(l, tuple):
#                             processed_labels.append(l)
#                         else:
#                             processed_labels.append(l.to(self.device))
#                     labels = processed_labels
#                 else:
#                     labels = labels.to(self.device)
                
#                 optimizer.zero_grad()
                
#                 # Mixed precision training
#                 if scaler:
#                     with torch.cuda.amp.autocast():
#                         outputs = model(pixel_values=pixel_values)
#                         loss = self.compute_loss(outputs, labels, class_weights)
                    
#                     scaler.scale(loss).backward()
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(
#                         model.parameters(), 
#                         self.config.get('grad_clip', 1.0)
#                     )
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     outputs = model(pixel_values=pixel_values)
#                     loss = self.compute_loss(outputs, labels, class_weights)
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(
#                         model.parameters(), 
#                         self.config.get('grad_clip', 1.0)
#                     )
#                     optimizer.step()
                
#                 scheduler.step()
                
#                 # Update metrics
#                 train_loss += loss.item()
                
#                 # For accuracy, only count non-mixed samples
#                 if not isinstance(labels, list):
#                     _, predicted = outputs.logits.max(1)
#                     train_total += labels.size(0)
#                     train_correct += predicted.eq(labels).sum().item()
                
#                 # Update progress bar
#                 current_acc = train_correct / train_total if train_total > 0 else 0
#                 avg_detection = detection_rates.mean().item()
#                 pbar.set_postfix({
#                     'loss': f'{loss.item():.4f}',
#                     'acc': f'{current_acc:.3f}',
#                     'lr': f'{scheduler.get_last_lr()[0]:.2e}',
#                     'det_rate': f'{avg_detection:.2f}'
#                 })
                
#                 # Clear cache periodically to prevent memory issues
#                 if batch_idx % 50 == 0 and self.device.type == 'cuda':
#                     torch.cuda.empty_cache()
            
#             # Validation
#             model.eval()
#             val_loss = 0
#             val_correct = 0
#             val_total = 0
#             all_preds = []
#             all_labels = []
#             all_probs = []
#             all_detection_rates = []
            
#             with torch.no_grad():
#                 for batch in tqdm(val_loader, desc='Validation'):
#                     pixel_values = batch['pixel_values'].to(self.device)
#                     labels = batch['labels'].to(self.device)
#                     detection_rates = batch['detection_rate']
                    
#                     outputs = model(pixel_values=pixel_values, labels=labels)
                    
#                     val_loss += outputs.loss.item()
                    
#                     # Get predictions
#                     probs = F.softmax(outputs.logits, dim=-1)
#                     _, predicted = outputs.logits.max(1)
                    
#                     val_total += labels.size(0)
#                     val_correct += predicted.eq(labels).sum().item()
                    
#                     all_preds.extend(predicted.cpu().numpy())
#                     all_labels.extend(labels.cpu().numpy())
#                     all_probs.extend(probs.cpu().numpy())
#                     all_detection_rates.extend(detection_rates.numpy())
            
#             # Calculate metrics
#             train_loss = train_loss / len(train_loader)
#             val_loss = val_loss / len(val_loader)
#             train_acc = train_correct / train_total if train_total > 0 else 0
#             val_acc = val_correct / val_total if val_total > 0 else 0
#             val_f1 = f1_score(all_labels, all_preds, average='weighted')
            
#             # Calculate per-class accuracy
#             per_class_acc = self.calculate_per_class_accuracy(all_labels, all_preds, self.config['num_classes'])
            
#             # Store metrics
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             val_accs.append(val_acc)
#             val_f1s.append(val_f1)
            
#             # Store in training history
#             self.training_history[f'fold_{fold_idx}_train_loss'].append(train_loss)
#             self.training_history[f'fold_{fold_idx}_val_acc'].append(val_acc)
            
#             # Log metrics
#             logger.info(
#                 f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
#                 f'Train Acc: {train_acc:.3f}'
#             )
#             logger.info(
#                 f'          Val Loss: {val_loss:.4f}, '
#                 f'Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}'
#             )
#             logger.info(f'          Per-class Acc: {[f"{acc:.2f}" for acc in per_class_acc]}')
            
#             if self.config.get('use_wandb', False):
#                 wandb.log({
#                     f'fold_{fold_idx}/train_loss': train_loss,
#                     f'fold_{fold_idx}/train_acc': train_acc,
#                     f'fold_{fold_idx}/val_loss': val_loss,
#                     f'fold_{fold_idx}/val_acc': val_acc,
#                     f'fold_{fold_idx}/val_f1': val_f1,
#                     f'fold_{fold_idx}/lr': scheduler.get_last_lr()[0],
#                     f'fold_{fold_idx}/avg_detection_rate': np.mean(all_detection_rates),
#                     'epoch': epoch
#                 })
            
#             # Save best model
#             if val_acc > best_val_acc:
#                 best_val_acc = val_acc
#                 best_val_f1 = val_f1
#                 patience_counter = 0
                
#                 # Save model
#                 save_path = Path(self.config['save_dir']) / f'best_model_fold_{fold_idx}.pth'
#                 save_path.parent.mkdir(exist_ok=True, parents=True)
#                 torch.save({
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'scheduler_state_dict': scheduler.state_dict(),
#                     'epoch': epoch,
#                     'val_acc': val_acc,
#                     'val_f1': val_f1,
#                     'per_class_acc': per_class_acc,
#                     'config': self.config
#                 }, save_path)
#                 logger.info(f'Saved best model with val_acc: {val_acc:.3f}')
                
#                 # Store confusion matrix for best epoch
#                 best_confusion_matrix = confusion_matrix(all_labels, all_preds)
#                 best_classification_report = classification_report(
#                     all_labels, all_preds, 
#                     target_names=[f'Class_{i}' for i in range(self.config['num_classes'])],
#                     output_dict=True
#                 )
                
#                 # Store best model to list
#                 self.best_models.append({
#                     'fold': fold_idx,
#                     'path': save_path,
#                     'accuracy': val_acc,
#                     'f1': val_f1
#                 })
#             else:
#                 patience_counter += 1
#                 if patience_counter >= self.config['patience']:
#                     logger.info(f"Early stopping at epoch {epoch+1}")
#                     break
            
#             # Clear cache after each epoch
#             if self.device.type == 'cuda':
#                 torch.cuda.empty_cache()
#             gc.collect()
        
#         # Plot training curves
#         self.plot_training_curves(
#             train_losses, val_losses, val_accs, val_f1s, 
#             fold_idx, save_path.parent
#         )
        
#         # Plot confusion matrix
#         self.plot_confusion_matrix(
#             best_confusion_matrix, 
#             fold_idx, 
#             save_path.parent
#         )
        
#         return {
#             'fold': fold_idx,
#             'best_val_acc': best_val_acc,
#             'best_val_f1': best_val_f1,
#             'final_train_loss': train_losses[-1],
#             'final_val_loss': val_losses[-1],
#             'epochs_trained': len(train_losses),
#             'classification_report': best_classification_report,
#             'confusion_matrix': best_confusion_matrix.tolist(),
#             'per_class_acc': per_class_acc
#         }
    
#     def calculate_per_class_accuracy(self, labels, predictions, num_classes):
#         """Calculate accuracy for each class"""
#         per_class_acc = []
#         for cls in range(num_classes):
#             cls_mask = np.array(labels) == cls
#             if cls_mask.sum() > 0:
#                 cls_acc = (np.array(predictions)[cls_mask] == cls).mean()
#                 per_class_acc.append(float(cls_acc))
#             else:
#                 per_class_acc.append(0.0)
#         return per_class_acc
    
#     def plot_training_curves(self, train_losses, val_losses, val_accs, val_f1s, 
#                            fold_idx, save_dir):
#         """Plot and save training curves"""
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
#         # Loss plot
#         axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=4)
#         axes[0].plot(val_losses, label='Val Loss', marker='s', markersize=4)
#         axes[0].set_xlabel('Epoch')
#         axes[0].set_ylabel('Loss')
#         axes[0].set_title(f'Fold {fold_idx}: Training and Validation Loss')
#         axes[0].legend()
#         axes[0].grid(True, alpha=0.3)
        
#         # Accuracy plot
#         axes[1].plot(val_accs, label='Val Accuracy', color='green', marker='o', markersize=4)
#         axes[1].set_xlabel('Epoch')
#         axes[1].set_ylabel('Accuracy')
#         axes[1].set_title(f'Fold {fold_idx}: Validation Accuracy')
#         axes[1].legend()
#         axes[1].grid(True, alpha=0.3)
#         axes[1].set_ylim([0, 1])
        
#         # F1 Score plot
#         axes[2].plot(val_f1s, label='Val F1 Score', color='orange', marker='o', markersize=4)
#         axes[2].set_xlabel('Epoch')
#         axes[2].set_ylabel('F1 Score')
#         axes[2].set_title(f'Fold {fold_idx}: Validation F1 Score')
#         axes[2].legend()
#         axes[2].grid(True, alpha=0.3)
#         axes[2].set_ylim([0, 1])
        
#         plt.tight_layout()
#         plt.savefig(save_dir / f'training_curves_fold_{fold_idx}.png', dpi=100)
#         plt.close()
    
#     def plot_confusion_matrix(self, cm, fold_idx, save_dir):
#         """Plot and save confusion matrix"""
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
#         plt.title(f'Fold {fold_idx}: Confusion Matrix')
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')
#         plt.tight_layout()
#         plt.savefig(save_dir / f'confusion_matrix_fold_{fold_idx}.png', dpi=100)
#         plt.close()
    
#     def run_cross_validation(self, data_paths, labels, groups=None):
#         """Run complete cross-validation"""
#         logger.info("\n" + "="*60)
#         logger.info("Starting Cross-Validation Training")
#         logger.info("="*60)
        
#         # Create stratified k-fold
#         if groups is not None:
#             # Use GroupKFold to keep augmented versions together
#             gkf = GroupKFold(n_splits=self.config['num_folds'])
#             folds = list(gkf.split(data_paths, labels, groups))
#             logger.info(f"Using GroupKFold with {self.config['num_folds']} folds")
#         else:
#             skf = StratifiedKFold(
#                 n_splits=self.config['num_folds'], 
#                 shuffle=True, 
#                 random_state=self.config.get('seed', 42)
#             )
#             folds = list(skf.split(data_paths, labels))
#             logger.info(f"Using StratifiedKFold with {self.config['num_folds']} folds")
        
#         # Train each fold
#         for fold_idx, (train_idx, val_idx) in enumerate(folds):
#             train_paths = data_paths[train_idx]
#             train_labels = labels[train_idx]
#             val_paths = data_paths[val_idx]
#             val_labels = labels[val_idx]
            
#             # Log fold statistics
#             train_class_dist = Counter(train_labels)
#             val_class_dist = Counter(val_labels)
#             logger.info(f"\nFold {fold_idx + 1} Statistics:")
#             logger.info(f"Train samples: {len(train_labels)}, Val samples: {len(val_labels)}")
#             logger.info(f"Train distribution: {dict(train_class_dist)}")
#             logger.info(f"Val distribution: {dict(val_class_dist)}")
            
#             # Train fold
#             fold_result = self.train_fold(
#                 fold_idx, 
#                 train_paths, 
#                 train_labels, 
#                 val_paths, 
#                 val_labels
#             )
            
#             self.fold_results.append(fold_result)
        
#         # Calculate and log final results
#         self.log_final_results()
        
#         return self.fold_results
    
#     def log_final_results(self):
#         """Log and save final cross-validation results"""
#         # Calculate average metrics
#         avg_acc = np.mean([r['best_val_acc'] for r in self.fold_results])
#         std_acc = np.std([r['best_val_acc'] for r in self.fold_results])
#         avg_f1 = np.mean([r['best_val_f1'] for r in self.fold_results])
#         std_f1 = np.std([r['best_val_f1'] for r in self.fold_results])
        
#         # Log results
#         logger.info("\n" + "="*60)
#         logger.info("CROSS-VALIDATION RESULTS")
#         logger.info("="*60)
        
#         for result in self.fold_results:
#             logger.info(f"Fold {result['fold'] + 1}: "
#                         f"Acc={result['best_val_acc']:.3f}, "
#                         f"F1={result['best_val_f1']:.3f}")
        
#         logger.info("-"*60)
#         logger.info(f"Average Accuracy: {avg_acc:.3f} ± {std_acc:.3f}")
#         logger.info(f"Average F1 Score: {avg_f1:.3f} ± {std_f1:.3f}")
#         logger.info("="*60)
        
#         # Save results to JSON
#         results_dict = {
#             'config': self.config,
#             'fold_results': self.fold_results,
#             'summary': {
#                 'avg_accuracy': float(avg_acc),
#                 'std_accuracy': float(std_acc),
#                 'avg_f1': float(avg_f1),
#                 'std_f1': float(std_f1)
#             },
#             'timestamp': datetime.now().isoformat()
#         }
        
#         save_path = Path(self.config['save_dir']) / 'cv_results.json'
#         with open(save_path, 'w') as f:
#             json.dump(results_dict, f, indent=2)
        
#         logger.info(f"Results saved to {save_path}")
        
#         if self.config.get('use_wandb', False):
#             wandb.summary.update({
#                 'cv_avg_accuracy': avg_acc,
#                 'cv_std_accuracy': std_acc,
#                 'cv_avg_f1': avg_f1,
#                 'cv_std_f1': std_f1
#             })


# class TestEvaluator:
#     """Evaluate on test set using ensemble of fold models"""

#     def __init__(self, config, model_paths):
#         self.config = config
#         self.model_paths = model_paths
#         self.device = torch.device(
#             'cuda' if torch.cuda.is_available() 
#             else 'mps' if torch.backends.mps.is_available() 
#             else 'cpu'
#         )
#         self.processor = VideoMAEImageProcessor.from_pretrained(config['model_name'])
        
#     def load_models(self):
#         """Load all fold models for ensemble"""
#         models = []
#         for path in self.model_paths:
#             if not Path(path).exists():
#                 logger.warning(f"Model file not found: {path}")
#                 continue
                
#             model = VideoMAEForVideoClassification.from_pretrained(
#                 self.config['model_name'],
#                 num_labels=self.config['num_classes'],
#                 ignore_mismatched_sizes=True
#             )
#             checkpoint = torch.load(path, map_location=self.device, weights_only=False)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             model.to(self.device)
#             model.eval()
#             models.append(model)
#         return models
    
#     def evaluate_ensemble(self, test_paths, test_labels):
#         """Evaluate using ensemble prediction"""
#         models = self.load_models()
        
#         if not models:
#             logger.error("No models loaded for ensemble evaluation")
#             return None
        
#         # Create test dataset
#         test_dataset = FixedNSLDataset(
#             test_paths,
#             test_labels,
#             self.processor,
#             augment=False,
#             use_detection_weighting=False,
#             debug_mode=False
#         )
        
#         # CRITICAL FOR MACOS: Set num_workers=0
#         test_loader = DataLoader(
#             test_dataset,
#             batch_size=self.config['batch_size'],
#             shuffle=False,
#             num_workers=0,  # Set to 0 for macOS compatibility
#             pin_memory=(self.device.type == 'cuda'),
#             collate_fn=custom_collate_fn
#         )
        
#         all_preds = []
#         all_labels = []
#         all_probs = []
        
#         with torch.no_grad():
#             for batch in tqdm(test_loader, desc='Test Evaluation'):
#                 pixel_values = batch['pixel_values'].to(self.device)
#                 labels = batch['labels'].cpu().numpy()
                
#                 # Ensemble prediction
#                 ensemble_probs = []
#                 for model in models:
#                     outputs = model(pixel_values=pixel_values)
#                     probs = F.softmax(outputs.logits, dim=-1)
#                     ensemble_probs.append(probs.cpu().numpy())
                
#                 # Average probabilities
#                 avg_probs = np.mean(ensemble_probs, axis=0)
#                 preds = np.argmax(avg_probs, axis=1)
                
#                 all_preds.extend(preds)
#                 all_labels.extend(labels)
#                 all_probs.extend(avg_probs)
        
#         # Calculate metrics
#         accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
#         f1 = f1_score(all_labels, all_preds, average='weighted')
        
#         # Confusion matrix
#         cm = confusion_matrix(all_labels, all_preds)
        
#         # Classification report
#         report = classification_report(
#             all_labels, all_preds,
#             target_names=[f'Class_{i}' for i in range(self.config['num_classes'])],
#             output_dict=True
#         )
        
#         # Log results
#         logger.info("\n" + "="*60)
#         logger.info("TEST SET RESULTS (Ensemble)")
#         logger.info("="*60)
#         logger.info(f"Test Accuracy: {accuracy:.3f}")
#         logger.info(f"Test F1 Score: {f1:.3f}")
#         logger.info("\nClassification Report:")
#         logger.info(classification_report(
#             all_labels, all_preds,
#             target_names=[f'Class_{i}' for i in range(self.config['num_classes'])]
#         ))
        
#         # Plot confusion matrix
#         self.plot_test_confusion_matrix(cm)
        
#         return {
#             'accuracy': float(accuracy),
#             'f1_score': float(f1),
#             'confusion_matrix': cm.tolist(),
#             'classification_report': report,
#             'predictions': all_preds,
#             'probabilities': all_probs
#         }
    
#     def plot_test_confusion_matrix(self, cm):
#         """Plot test set confusion matrix"""
#         plt.figure(figsize=(12, 10))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.title('Test Set Confusion Matrix (Ensemble)')
#         plt.xlabel('Predicted')
#         plt.ylabel('Actual')
#         plt.tight_layout()
#         save_path = Path(self.config['save_dir']) / 'test_confusion_matrix.png'
#         plt.savefig(save_path, dpi=100)
#         plt.close()
#         logger.info(f"Test confusion matrix saved to {save_path}")


# def prepare_data_from_mediapipe(preprocessed_dir, test_ratio=0.15, validation_samples=5):
#     """
#     Prepare data from MediaPipe preprocessed files with validation
#     """
#     preprocessed_path = Path(preprocessed_dir)
    
#     # Load preprocessing summary if available
#     summary_path = preprocessed_path / 'preprocessing_summary.json'
#     if summary_path.exists():
#         with open(summary_path, 'r') as f:
#             summary = json.load(f)
#             logger.info(f"Loaded preprocessing summary: {summary.get('successful', 'N/A')} successful videos")
    
#     # Collect all preprocessed files
#     all_files = []
#     all_labels = []
#     all_groups = []
#     class_to_idx = {}
    
#     for class_dir in sorted(preprocessed_path.iterdir()):
#         if not class_dir.is_dir() or class_dir.name.startswith('.'):
#             continue
        
#         class_name = class_dir.name
#         if class_name not in class_to_idx:
#             class_to_idx[class_name] = len(class_to_idx)
        
#         class_idx = class_to_idx[class_name]
        
#         # Get all .npz files
#         npz_files = list(class_dir.glob('*_processed.npz'))
        
#         # Validate a few samples
#         valid_files = []
#         for npz_file in npz_files:
#             # Quick validation of file
#             try:
#                 data = np.load(npz_file, allow_pickle=True)
#                 if 'frames' in data and data['frames'].shape[0] > 0:
#                     valid_files.append(npz_file)
#                 else:
#                     logger.warning(f"Skipping invalid file: {npz_file}")
#             except Exception as e:
#                 logger.warning(f"Skipping corrupted file {npz_file}: {e}")
        
#         logger.info(f"Class {class_name}: {len(valid_files)}/{len(npz_files)} valid files")
        
#         for npz_file in valid_files:
#             # Extract original video name (for grouping augmentations)
#             original_name = npz_file.stem.replace('_processed', '')
#             if '_aug' in original_name:
#                 original_name = original_name.split('_aug')[0]
            
#             all_files.append(npz_file)
#             all_labels.append(class_idx)
#             all_groups.append(f"{class_name}_{original_name}")
    
#     if not all_files:
#         raise ValueError("No valid preprocessed files found!")
    
#     # Convert to numpy arrays
#     all_files = np.array(all_files)
#     all_labels = np.array(all_labels)
#     all_groups = np.array(all_groups)
    
#     # Validate sample data
#     logger.info(f"\nValidating {validation_samples} random samples...")
#     sample_indices = np.random.choice(len(all_files), min(validation_samples, len(all_files)), replace=False)
    
#     for idx in sample_indices:
#         if verify_preprocessed_data(all_files[idx]):
#             logger.info(f"✓ {all_files[idx].name} validated successfully")
#         else:
#             logger.error(f"✗ {all_files[idx].name} validation failed")
    
#     # Split into train+val and test
#     unique_groups = np.unique(all_groups)
#     n_test_groups = max(1, int(len(unique_groups) * test_ratio))
    
#     # Randomly select test groups
#     np.random.seed(42)
#     test_groups = np.random.choice(unique_groups, n_test_groups, replace=False)
#     test_mask = np.isin(all_groups, test_groups)
    
#     # Split data
#     X_trainval = all_files[~test_mask]
#     y_trainval = all_labels[~test_mask]
#     groups_trainval = all_groups[~test_mask]
    
#     X_test = all_files[test_mask]
#     y_test = all_labels[test_mask]
    
#     # Log statistics
#     logger.info(f"\nData Split Statistics:")
#     logger.info(f"Total samples: {len(all_files)}")
#     logger.info(f"Train+Val samples: {len(X_trainval)}")
#     logger.info(f"Test samples: {len(X_test)}")
#     logger.info(f"Number of classes: {len(class_to_idx)}")
#     logger.info(f"Class mapping: {class_to_idx}")
    
#     # Class distribution
#     for split_name, split_labels in [("Train+Val", y_trainval), ("Test", y_test)]:
#         logger.info(f"\n{split_name} class distribution:")
#         for class_name, class_idx in class_to_idx.items():
#             count = np.sum(split_labels == class_idx)
#             logger.info(f"  {class_name}: {count} samples")
    
#     return X_trainval, y_trainval, X_test, y_test, groups_trainval, class_to_idx


# def main():
#     """Main training pipeline"""
    
#     # Set environment variable for macOS
#     os.environ['OMP_NUM_THREADS'] = '1'  # Avoid OpenMP issues on macOS
    
#     # Configuration optimized for small dataset and macOS
#     config = {
#         # Model settings
#         'model_name': 'MCG-NJU/videomae-base',
#         'num_classes': 10,
        
#         # Training settings - ADJUSTED FOR SMALL DATASET
#         'batch_size': 2,  # Small batch size
#         'learning_rate': 1.5e-5,  # Reduced learning rate
#         'weight_decay': 0.001,  # Reduced weight decay
#         'epochs': 50,  # More epochs
#         'patience': 15,  # Higher patience
#         'grad_clip': 1.0,
#         # 'label_smoothing': 0.05,  # Light label smoothing
#         'label_smoothing': 0.1,  # Light label smoothing
        
#         # Augmentation settings - REDUCED
#         # 'mixup_alpha': 0.1,  # Light MixUp
#         'mixup_alpha': 0.05,  # Light MixUp

#         'cutmix_prob': 0.0,  # Disabled CutMix
        
#         # Cross-validation
#         'num_folds': 3,  # 3-fold CV for more training data per fold
        
#         # System settings - CRITICAL FOR MACOS
#         'num_workers': 0,  # MUST BE 0 for macOS to avoid pickle errors
#         'seed': 42,
#         'save_dir': 'models/videomae_nsl_fixed',
        
#         # Optional
#         'use_wandb': False,
#         # 'freeze_backbone_layers': 8,  # Freeze more layers for small dataset
#         'freeze_backbone_layers': 10,  # Freeze more layers for small dataset
#         'dropout': 0.3 
#     }
    
#     # Set random seeds
#     torch.manual_seed(config['seed'])
#     np.random.seed(config['seed'])
#     random.seed(config['seed'])
    
#     # Clear cache before starting
#     global _data_cache
#     _data_cache.clear()
    
#     # Prepare data
#     logger.info("Loading preprocessed data...")
#     X_trainval, y_trainval, X_test, y_test, groups, class_to_idx = prepare_data_from_mediapipe(
#         'mediapipe_preprocessed',  # Your preprocessed directory
#         test_ratio=0.15,
#         validation_samples=5
#     )
    
#     # Update config with actual number of classes
#     config['num_classes'] = len(class_to_idx)
    
#     # Save class mapping
#     save_dir = Path(config['save_dir'])
#     save_dir.mkdir(exist_ok=True, parents=True)
#     with open(save_dir / 'class_mapping.json', 'w') as f:
#         json.dump(class_to_idx, f, indent=2)
    
#     # Initialize trainer
#     trainer = EnhancedNSLTrainer(config)
    
#     # Run cross-validation
#     fold_results = trainer.run_cross_validation(
#         X_trainval, 
#         y_trainval, 
#         groups=groups
#     )
    
#     # Evaluate on test set with ensemble
#     logger.info("\nEvaluating on test set...")
#     model_paths = [save_dir / f'best_model_fold_{i}.pth' for i in range(config['num_folds'])]
#     evaluator = TestEvaluator(config, model_paths)
#     test_results = evaluator.evaluate_ensemble(X_test, y_test)
    
#     # Save test results
#     if test_results:
#         with open(save_dir / 'test_results.json', 'w') as f:
#             json.dump(test_results, f, indent=2)
        
#         logger.info("\n" + "="*60)
#         logger.info("TRAINING COMPLETE!")
#         logger.info(f"Models saved to: {save_dir}")
#         logger.info(f"Final Test Accuracy: {test_results['accuracy']:.3f}")
#         logger.info(f"Final Test F1 Score: {test_results['f1_score']:.3f}")
#         logger.info("="*60)
#     else:
#         logger.error("Test evaluation failed!")
    
#     # Clear cache at the end
#     _data_cache.clear()
#     gc.collect()


# if __name__ == "__main__":
#     main()




























# #!/usr/bin/env python3
# """
# VideoMAE Training Pipeline for Nepali Sign Language Recognition
# Training only - testing functionality moved to separate script
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from transformers import (
#     VideoMAEForVideoClassification, 
#     VideoMAEImageProcessor,
#     get_cosine_schedule_with_warmup
# )
# import numpy as np
# from pathlib import Path
# import json
# import logging
# from tqdm import tqdm
# from datetime import datetime
# from sklearn.model_selection import StratifiedKFold, GroupKFold
# from sklearn.metrics import f1_score, confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict, Counter
# import random
# import cv2
# from PIL import Image
# import warnings
# import gc
# import traceback
# import os

# warnings.filterwarnings('ignore')

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO, 
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Global cache for data loading
# _data_cache = {}

# def load_cached_data(path):
#     """Load data with simple caching (pickle-safe)"""
#     path_str = str(path)
#     if path_str not in _data_cache:
#         _data_cache[path_str] = np.load(path_str, allow_pickle=True)
#     return _data_cache[path_str]

# class ImprovedNSLDataset(Dataset):
#     """Dataset with improved augmentation and detection-aware sampling"""
    
#     def __init__(self, 
#                  data_paths, 
#                  labels, 
#                  processor,
#                  augment=True,
#                  mix_up_alpha=0.05,
#                  cutmix_prob=0.0,
#                  use_detection_weighting=True,
#                  debug_mode=False,
#                  class_weights=None):
        
#         self.data_paths = [Path(p) for p in data_paths]
#         self.labels = labels
#         self.processor = processor
#         self.augment = augment
#         self.mix_up_alpha = mix_up_alpha
#         self.cutmix_prob = cutmix_prob
#         self.use_detection_weighting = use_detection_weighting
#         self.debug_mode = debug_mode
#         self.class_weights_dict = class_weights or {}
        
#         # Initialize class counts first
#         self.class_counts = Counter(labels)
#         self.num_classes = len(set(labels))
        
#         # Verify data paths exist
#         missing_files = [p for p in self.data_paths if not p.exists()]
#         if missing_files:
#             logger.error(f"Missing files: {missing_files[:5]}")
#             raise FileNotFoundError(f"{len(missing_files)} files not found")
        
#         # Load and verify metadata
#         self.detection_rates = []
#         self.enhanced_rates = []
#         self.metadata_stats = defaultdict(list)
#         self.valid_indices = []
#         self.class_detection_rates = defaultdict(list)
        
#         logger.info("Loading and verifying dataset...")
#         for idx, path in enumerate(tqdm(self.data_paths, desc="Verifying data")):
#             try:
#                 data = load_cached_data(str(path))
                
#                 if 'frames' not in data:
#                     logger.warning(f"No frames in {path}")
#                     continue
                
#                 frames = data['frames']
#                 if frames.shape != (16, 224, 224, 3):
#                     logger.warning(f"Unexpected shape {frames.shape} in {path}")
                
#                 # Extract metadata
#                 metadata = {}
#                 if 'metadata' in data:
#                     try:
#                         metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
#                         if not isinstance(metadata, dict):
#                             metadata = {}
#                     except:
#                         metadata = {}
                
#                 # Store detection rates
#                 detection_rate = metadata.get('detection_rate', 1.0)
#                 enhanced_rate = metadata.get('enhanced_rate', detection_rate)
                
#                 self.detection_rates.append(detection_rate)
#                 self.enhanced_rates.append(enhanced_rate)
#                 self.valid_indices.append(idx)
                
#                 # Track per-class detection rates
#                 label = self.labels[idx]
#                 self.class_detection_rates[label].append(enhanced_rate)
                
#             except Exception as e:
#                 logger.error(f"Error loading {path}: {e}")
#                 self.detection_rates.append(0.5)
#                 self.enhanced_rates.append(0.5)
#                 self.valid_indices.append(idx)
        
#         if not self.valid_indices:
#             raise ValueError("No valid data files found!")
        
#         # Calculate weights
#         self.sample_weights = self._calculate_adaptive_weights()
#         self.class_weights = self._calculate_class_weights()
        
#         # Log statistics
#         logger.info(f"Dataset initialized: {len(self.valid_indices)}/{len(data_paths)} valid samples")
#         logger.info(f"Detection rate: mean={np.mean(self.detection_rates):.2f}, "
#                    f"std={np.std(self.detection_rates):.2f}")
#         logger.info(f"Class distribution: {dict(self.class_counts)}")
    
#     def _calculate_adaptive_weights(self):
#         """Calculate sample weights based on detection quality and class balance"""
#         weights = np.ones(len(self.data_paths))
        
#         for idx in range(len(self.data_paths)):
#             label = self.labels[idx]
#             class_weight = 1.0 / max(self.class_counts[label], 1)
            
#             if idx < len(self.enhanced_rates):
#                 detection_weight = np.clip(self.enhanced_rates[idx], 0.2, 1.0)
#             else:
#                 detection_weight = 0.5
            
#             config_weight = self.class_weights_dict.get(int(label), 1.0)
#             weights[idx] = class_weight * detection_weight * config_weight
        
#         return weights / weights.sum()
    
#     def _calculate_class_weights(self):
#         """Calculate class weights for loss function"""
#         total_samples = len(self.labels)
#         weights = []
        
#         for i in range(self.num_classes):
#             class_count = max(self.class_counts[i], 1)
#             base_weight = total_samples / (self.num_classes * class_count)
            
#             if i in self.class_detection_rates and self.class_detection_rates[i]:
#                 avg_detection = np.mean(self.class_detection_rates[i])
#                 detection_factor = 2.0 - avg_detection
#             else:
#                 detection_factor = 1.0
            
#             weights.append(base_weight * detection_factor)
        
#         weights = torch.tensor(weights, dtype=torch.float32)
#         return weights / weights.mean()
    
#     def __len__(self):
#         return len(self.data_paths)
    
#     def __getitem__(self, idx):
#         try:
#             data = load_cached_data(str(self.data_paths[idx]))
#             frames = data['frames']
            
#             # Ensure correct format
#             if frames.shape != (16, 224, 224, 3):
#                 if frames.shape[0] < 16:
#                     padding = np.repeat(frames[-1:], 16 - frames.shape[0], axis=0)
#                     frames = np.concatenate([frames, padding], axis=0)
#                 elif frames.shape[0] > 16:
#                     frames = frames[:16]
                
#                 if frames.shape[1:3] != (224, 224):
#                     resized = []
#                     for frame in frames:
#                         resized.append(cv2.resize(frame, (224, 224)))
#                     frames = np.array(resized)
            
#             if frames.dtype != np.uint8:
#                 frames = np.clip(frames, 0, 255).astype(np.uint8)
            
#             frames = frames.copy()
#             label = self.labels[idx]
            
#             # Extract metadata
#             metadata = {}
#             if 'metadata' in data:
#                 try:
#                     metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
#                     if not isinstance(metadata, dict):
#                         metadata = {}
#                 except:
#                     metadata = {}
            
#             detection_rate = metadata.get('enhanced_rate', metadata.get('detection_rate', 1.0))
            
#             # Apply augmentations
#             if self.augment:
#                 frames = self.apply_adaptive_augmentations(frames, detection_rate)
                
#                 if random.random() < 0.1 and self.mix_up_alpha > 0:
#                     frames, label = self.apply_mixup(frames, label, idx)
            
#             # Convert to PIL Images
#             pil_frames = [Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)) for frame in frames]
            
#             # Process with VideoMAE
#             inputs = self.processor(
#                 pil_frames, 
#                 return_tensors="pt",
#                 do_resize=False,
#                 do_center_crop=False
#             )
            
#             pixel_values = inputs['pixel_values'].squeeze(0)
            
#             if isinstance(label, tuple):
#                 return {
#                     'pixel_values': pixel_values,
#                     'labels': label,
#                     'detection_rate': detection_rate,
#                     'original_idx': idx
#                 }
#             else:
#                 return {
#                     'pixel_values': pixel_values,
#                     'labels': torch.tensor(label, dtype=torch.long),
#                     'detection_rate': detection_rate,
#                     'original_idx': idx
#                 }
                
#         except Exception as e:
#             logger.error(f"Error loading sample {idx}: {e}")
#             traceback.print_exc()
            
#             dummy_frames = torch.zeros((3, 16, 224, 224))
#             return {
#                 'pixel_values': dummy_frames,
#                 'labels': torch.tensor(0, dtype=torch.long),
#                 'detection_rate': 0.0,
#                 'original_idx': idx
#             }
    
#     def apply_adaptive_augmentations(self, frames, detection_rate):
#         """Apply augmentations based on detection quality"""
#         if detection_rate > 0.8:
#             if random.random() < 0.15:
#                 frames = self.temporal_shift(frames, max_shift=1)
#             if random.random() < 0.2:
#                 frames = self.brightness_adjustment(frames, factor_range=(0.95, 1.05))
#         elif detection_rate > 0.5:
#             if random.random() < 0.25:
#                 frames = self.temporal_shift(frames, max_shift=2)
#             if random.random() < 0.3:
#                 frames = self.color_jitter(frames, intensity=0.1)
#             if random.random() < 0.25:
#                 frames = self.brightness_adjustment(frames, factor_range=(0.9, 1.1))
#         else:
#             if random.random() < 0.3:
#                 frames = self.temporal_shift(frames, max_shift=2)
#             if random.random() < 0.4:
#                 frames = self.color_jitter(frames, intensity=0.15)
#             if random.random() < 0.3:
#                 frames = self.brightness_adjustment(frames, factor_range=(0.85, 1.15))
#             if random.random() < 0.2:
#                 frames = self.add_gaussian_noise(frames, intensity=0.05)
        
#         return frames
    
#     def temporal_shift(self, frames, max_shift=2):
#         """Shift frames temporally"""
#         if len(frames) <= max_shift * 2:
#             return frames
        
#         shift = random.randint(-max_shift, max_shift)
#         if shift == 0:
#             return frames
        
#         shifted = np.zeros_like(frames)
#         if shift > 0:
#             shifted[shift:] = frames[:-shift]
#             shifted[:shift] = frames[0]
#         else:
#             shifted[:shift] = frames[-shift:]
#             shifted[shift:] = frames[-1]
        
#         return shifted
    
#     def color_jitter(self, frames, intensity=0.1):
#         """Apply color jittering"""
#         brightness = random.uniform(1-intensity, 1+intensity)
#         frames = np.clip(frames * brightness, 0, 255)
        
#         if random.random() < 0.5:
#             contrast = random.uniform(1-intensity/2, 1+intensity/2)
#             frames = np.clip((frames - 128) * contrast + 128, 0, 255)
        
#         return frames.astype(np.uint8)
    
#     def brightness_adjustment(self, frames, factor_range=(0.9, 1.1)):
#         """Adjust brightness"""
#         factor = random.uniform(*factor_range)
#         return np.clip(frames * factor, 0, 255).astype(np.uint8)
    
#     def add_gaussian_noise(self, frames, intensity=0.05):
#         """Add Gaussian noise"""
#         noise = np.random.randn(*frames.shape) * 255 * intensity
#         return np.clip(frames + noise, 0, 255).astype(np.uint8)
    
#     def apply_mixup(self, frames, label, idx):
#         """Apply MixUp augmentation"""
#         mix_idx = random.randint(0, len(self.data_paths) - 1)
#         if mix_idx == idx:
#             mix_idx = (idx + 1) % len(self.data_paths)
        
#         mix_data = load_cached_data(str(self.data_paths[mix_idx]))
#         mix_frames = mix_data['frames'].copy()
        
#         if mix_frames.shape != frames.shape:
#             if mix_frames.shape[0] < 16:
#                 padding = np.repeat(mix_frames[-1:], 16 - mix_frames.shape[0], axis=0)
#                 mix_frames = np.concatenate([mix_frames, padding], axis=0)
#             elif mix_frames.shape[0] > 16:
#                 mix_frames = mix_frames[:16]
        
#         mix_label = self.labels[mix_idx]
        
#         lam = np.random.beta(self.mix_up_alpha, self.mix_up_alpha)
#         lam = max(0.8, lam)
        
#         frames = (lam * frames + (1 - lam) * mix_frames).astype(np.uint8)
        
#         return frames, (torch.tensor(label, dtype=torch.long), 
#                        torch.tensor(mix_label, dtype=torch.long), 
#                        lam)

# def custom_collate_fn(batch):
#     """Custom collate function to handle mixed labels"""
#     valid_batch = []
#     for item in batch:
#         if item['pixel_values'].shape[0] > 0:
#             valid_batch.append(item)
    
#     if not valid_batch:
#         return {
#             'pixel_values': torch.zeros((1, 3, 16, 224, 224)),
#             'labels': torch.tensor([0]),
#             'detection_rate': torch.tensor([0.0]),
#             'original_idx': torch.tensor([0])
#         }
    
#     pixel_values = []
#     labels = []
#     detection_rates = []
#     original_idxs = []
    
#     for item in valid_batch:
#         pixel_values.append(item['pixel_values'])
#         labels.append(item['labels'])
#         detection_rates.append(item['detection_rate'])
#         original_idxs.append(item['original_idx'])
    
#     pixel_values = torch.stack(pixel_values)
    
#     has_mixed = any(isinstance(l, tuple) for l in labels)
    
#     if has_mixed:
#         processed_labels = labels
#     else:
#         processed_labels = torch.stack(labels)
    
#     return {
#         'pixel_values': pixel_values,
#         'labels': processed_labels,
#         'detection_rate': torch.tensor(detection_rates),
#         'original_idx': torch.tensor(original_idxs)
#     }

# class ImprovedNSLTrainer:
#     """Trainer with cross-validation"""
    
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device(
#             'cuda' if torch.cuda.is_available() 
#             else 'mps' if torch.backends.mps.is_available() 
#             else 'cpu'
#         )
#         logger.info(f"Using device: {self.device}")
        
#         self.processor = VideoMAEImageProcessor.from_pretrained(config['model_name'])
#         self.fold_results = []
#         self.best_models = []
#         self.training_history = defaultdict(list)
    
#     def create_model(self):
#         """Create and initialize model"""
#         model = VideoMAEForVideoClassification.from_pretrained(
#             self.config['model_name'],
#             num_labels=self.config['num_classes'],
#             ignore_mismatched_sizes=True
#         )
        
#         freeze_layers = self.config.get('freeze_backbone_layers', 10)
#         if freeze_layers > 0:
#             for i, layer in enumerate(model.videomae.encoder.layer):
#                 if i < freeze_layers:
#                     for param in layer.parameters():
#                         param.requires_grad = False
#             logger.info(f"Froze first {freeze_layers} backbone layers")
        
#         if hasattr(model, 'classifier'):
#             model.dropout = nn.Dropout(self.config.get('dropout_rate', 0.3))
        
#         return model.to(self.device)
    
#     def create_optimizer_scheduler(self, model, num_training_steps):
#         """Create optimizer and scheduler"""
#         backbone_params = []
#         head_params = []
        
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 if 'classifier' in name or 'fc' in name:
#                     head_params.append(param)
#                 else:
#                     backbone_params.append(param)
        
#         base_lr = self.config.get('learning_rate', 1.5e-5)
        
#         optimizer = torch.optim.AdamW([
#             {'params': backbone_params, 'lr': base_lr},
#             {'params': head_params, 'lr': base_lr * 5}
#         ], weight_decay=self.config.get('weight_decay', 0.01))
        
#         warmup_ratio = self.config.get('warmup_ratio', 0.15)
#         scheduler = get_cosine_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=int(num_training_steps * warmup_ratio),
#             num_training_steps=num_training_steps
#         )
        
#         return optimizer, scheduler
    
#     def compute_loss(self, outputs, labels, class_weights=None, detection_rates=None):
#         """Compute loss with label smoothing and detection weighting"""
#         if isinstance(labels, list):
#             total_loss = 0
#             num_samples = 0
            
#             for i, label in enumerate(labels):
#                 if isinstance(label, tuple):
#                     labels_a, labels_b, lam = label
#                     logit = outputs.logits[i:i+1]
                    
#                     if class_weights is not None:
#                         weight_a = class_weights[labels_a.item()]
#                         weight_b = class_weights[labels_b.item()]
#                         loss_a = F.cross_entropy(logit, labels_a.unsqueeze(0).to(self.device), 
#                                                 reduction='none') * weight_a
#                         loss_b = F.cross_entropy(logit, labels_b.unsqueeze(0).to(self.device), 
#                                                 reduction='none') * weight_b
#                     else:
#                         loss_a = F.cross_entropy(logit, labels_a.unsqueeze(0).to(self.device))
#                         loss_b = F.cross_entropy(logit, labels_b.unsqueeze(0).to(self.device))
                    
#                     sample_loss = lam * loss_a + (1 - lam) * loss_b
                    
#                     if detection_rates is not None:
#                         detection_weight = torch.pow(detection_rates[i], 0.3)
#                         sample_loss = sample_loss * detection_weight
                    
#                     total_loss += sample_loss.mean()
#                 else:
#                     logit = outputs.logits[i:i+1]
                    
#                     if class_weights is not None:
#                         weight = class_weights[label.item()]
#                         sample_loss = F.cross_entropy(logit, label.unsqueeze(0).to(self.device), 
#                                                      reduction='none') * weight
#                     else:
#                         sample_loss = F.cross_entropy(logit, label.unsqueeze(0).to(self.device))
                    
#                     if detection_rates is not None:
#                         detection_weight = torch.pow(detection_rates[i], 0.3)
#                         sample_loss = sample_loss * detection_weight
                    
#                     total_loss += sample_loss.mean()
                
#                 num_samples += 1
            
#             loss = total_loss / num_samples if num_samples > 0 else total_loss
#         else:
#             label_smoothing = self.config.get('label_smoothing', 0.1)
            
#             loss = F.cross_entropy(
#                 outputs.logits, 
#                 labels, 
#                 label_smoothing=label_smoothing,
#                 weight=class_weights.to(self.device) if class_weights is not None else None
#             )
            
#             if detection_rates is not None:
#                 detection_weights = torch.pow(detection_rates, 0.3).to(self.device)
#                 loss = loss * detection_weights.mean()
        
#         return loss
    
#     def train_fold(self, fold_idx, train_paths, train_labels, val_paths, val_labels):
#         """Train on a single fold"""
#         logger.info(f"\n{'='*50}")
#         logger.info(f"Training Fold {fold_idx + 1}/{self.config['num_folds']}")
#         logger.info(f"{'='*50}")
        
#         model = self.create_model()
        
#         train_dataset = ImprovedNSLDataset(
#             train_paths, 
#             train_labels, 
#             self.processor,
#             augment=True,
#             mix_up_alpha=self.config.get('mixup_alpha', 0.05),
#             cutmix_prob=0.0,
#             use_detection_weighting=True,
#             debug_mode=(fold_idx == 0),
#             class_weights={int(k): v for k, v in self.config.get('class_specific_weights', {}).items()}
#         )
        
#         val_dataset = ImprovedNSLDataset(
#             val_paths, 
#             val_labels, 
#             self.processor,
#             augment=False,
#             use_detection_weighting=False,
#             debug_mode=False,
#             class_weights=None
#         )
        
#         class_weights = train_dataset.class_weights
        
#         sampler = WeightedRandomSampler(
#             train_dataset.sample_weights, 
#             len(train_dataset.sample_weights),
#             replacement=True
#         )
        
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=self.config['batch_size'],
#             sampler=sampler,
#             num_workers=0,
#             pin_memory=(self.device.type == 'cuda'),
#             drop_last=True,
#             collate_fn=custom_collate_fn
#         )
        
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=self.config['batch_size'],
#             shuffle=False,
#             num_workers=0,
#             pin_memory=(self.device.type == 'cuda'),
#             collate_fn=custom_collate_fn
#         )
        
#         num_training_steps = len(train_loader) * self.config['epochs']
#         optimizer, scheduler = self.create_optimizer_scheduler(model, num_training_steps)
        
#         scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
#         best_val_acc = 0
#         best_val_f1 = 0
#         patience_counter = 0
#         train_losses = []
#         val_losses = []
#         train_accs = []
#         val_accs = []
#         val_f1s = []
        
#         patience = self.config.get('patience', 15)
        
#         for epoch in range(self.config['epochs']):
#             # Training
#             model.train()
#             train_loss = 0
#             train_correct = 0
#             train_total = 0
            
#             pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
#             for batch_idx, batch in enumerate(pbar):
#                 pixel_values = batch['pixel_values'].to(self.device)
#                 labels = batch['labels']
#                 detection_rates = batch['detection_rate']
                
#                 if isinstance(labels, list):
#                     processed_labels = []
#                     for l in labels:
#                         if isinstance(l, tuple):
#                             processed_labels.append(l)
#                         else:
#                             processed_labels.append(l.to(self.device))
#                     labels = processed_labels
#                 else:
#                     labels = labels.to(self.device)
                
#                 optimizer.zero_grad()
                
#                 if scaler:
#                     with torch.cuda.amp.autocast():
#                         outputs = model(pixel_values=pixel_values)
#                         loss = self.compute_loss(outputs, labels, class_weights, detection_rates)
                    
#                     scaler.scale(loss).backward()
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(
#                         model.parameters(), 
#                         self.config.get('grad_clip', 0.5)
#                     )
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     outputs = model(pixel_values=pixel_values)
#                     loss = self.compute_loss(outputs, labels, class_weights, detection_rates)
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(
#                         model.parameters(), 
#                         self.config.get('grad_clip', 0.5)
#                     )
#                     optimizer.step()
                
#                 scheduler.step()
                
#                 train_loss += loss.item()
                
#                 if not isinstance(labels, list):
#                     _, predicted = outputs.logits.max(1)
#                     train_total += labels.size(0)
#                     train_correct += predicted.eq(labels).sum().item()
                
#                 current_acc = train_correct / train_total if train_total > 0 else 0
#                 avg_detection = detection_rates.mean().item()
#                 pbar.set_postfix({
#                     'loss': f'{loss.item():.4f}',
#                     'acc': f'{current_acc:.3f}',
#                     'lr': f'{scheduler.get_last_lr()[0]:.2e}',
#                     'det_rate': f'{avg_detection:.2f}'
#                 })
                
#                 if batch_idx % 50 == 0:
#                     if self.device.type == 'cuda':
#                         torch.cuda.empty_cache()
#                     elif self.device.type == 'mps':
#                         torch.mps.empty_cache()
            
#             train_loss = train_loss / len(train_loader)
#             train_acc = train_correct / train_total if train_total > 0 else 0
#             train_accs.append(train_acc)
#             train_losses.append(train_loss)
            
#             # Validation
#             model.eval()
#             val_loss = 0
#             val_correct = 0
#             val_total = 0
#             all_preds = []
#             all_labels = []
            
#             with torch.no_grad():
#                 for batch in tqdm(val_loader, desc='Validation'):
#                     pixel_values = batch['pixel_values'].to(self.device)
#                     labels = batch['labels'].to(self.device)
                    
#                     outputs = model(pixel_values=pixel_values, labels=labels)
                    
#                     val_loss += outputs.loss.item()
                    
#                     _, predicted = outputs.logits.max(1)
                    
#                     val_total += labels.size(0)
#                     val_correct += predicted.eq(labels).sum().item()
                    
#                     all_preds.extend(predicted.cpu().numpy())
#                     all_labels.extend(labels.cpu().numpy())
            
#             val_loss = val_loss / len(val_loader)
#             val_acc = val_correct / val_total if val_total > 0 else 0
#             val_f1 = f1_score(all_labels, all_preds, average='weighted')
            
#             val_losses.append(val_loss)
#             val_accs.append(val_acc)
#             val_f1s.append(val_f1)
            
#             logger.info(
#                 f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
#                 f'Train Acc: {train_acc:.3f}'
#             )
#             logger.info(
#                 f'          Val Loss: {val_loss:.4f}, '
#                 f'Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}'
#             )
            
#             # Save best model
#             if val_acc > best_val_acc:
#                 best_val_acc = val_acc
#                 best_val_f1 = val_f1
#                 patience_counter = 0
                
#                 save_path = Path(self.config['save_dir']) / f'best_model_fold_{fold_idx}.pth'
#                 save_path.parent.mkdir(exist_ok=True, parents=True)
#                 torch.save({
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'scheduler_state_dict': scheduler.state_dict(),
#                     'epoch': epoch,
#                     'val_acc': val_acc,
#                     'val_f1': val_f1,
#                     'config': self.config
#                 }, save_path)
#                 logger.info(f'Saved best model with val_acc: {val_acc:.3f}')
                
#                 best_confusion_matrix = confusion_matrix(all_labels, all_preds)
                
#                 self.best_models.append({
#                     'fold': fold_idx,
#                     'path': save_path,
#                     'accuracy': val_acc,
#                     'f1': val_f1
#                 })
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     logger.info(f"Early stopping at epoch {epoch+1}")
#                     break
            
#             if self.device.type == 'cuda':
#                 torch.cuda.empty_cache()
#             elif self.device.type == 'mps':
#                 torch.mps.empty_cache()
        
#         # Store fold results
#         fold_result = {
#             'fold': fold_idx,
#             'best_val_acc': best_val_acc,
#             'best_val_f1': best_val_f1,
#             'train_losses': train_losses,
#             'val_losses': val_losses,
#             'train_accs': train_accs,
#             'val_accs': val_accs,
#             'val_f1s': val_f1s,
#             'confusion_matrix': best_confusion_matrix.tolist() if 'best_confusion_matrix' in locals() else None
#         }
        
#         self.fold_results.append(fold_result)
        
#         # Clean up
#         del model, optimizer, scheduler, train_dataset, val_dataset
#         gc.collect()
#         if self.device.type == 'cuda':
#             torch.cuda.empty_cache()
#         elif self.device.type == 'mps':
#             torch.mps.empty_cache()
        
#         return fold_result
    
#     def train_cross_validation(self, data_paths, labels):
#         """Train with cross-validation"""
#         # Convert to numpy arrays for sklearn
#         data_paths = np.array(data_paths)
#         labels = np.array(labels)
        
#         # Create folds
#         skf = StratifiedKFold(
#             n_splits=self.config['num_folds'], 
#             shuffle=True, 
#             random_state=self.config.get('seed', 42)
#         )
        
#         # Train on each fold
#         for fold_idx, (train_idx, val_idx) in enumerate(skf.split(data_paths, labels)):
#             train_paths = data_paths[train_idx]
#             train_labels = labels[train_idx]
#             val_paths = data_paths[val_idx]
#             val_labels = labels[val_idx]
            
#             # Log fold statistics
#             logger.info(f"\nFold {fold_idx + 1} Statistics:")
#             logger.info(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
            
#             train_dist = Counter(train_labels)
#             val_dist = Counter(val_labels)
#             logger.info(f"Train distribution: {dict(train_dist)}")
#             logger.info(f"Val distribution: {dict(val_dist)}")
            
#             # Train fold
#             fold_result = self.train_fold(
#                 fold_idx, 
#                 train_paths, 
#                 train_labels, 
#                 val_paths, 
#                 val_labels
#             )
        
#         # Calculate average metrics
#         avg_acc = np.mean([r['best_val_acc'] for r in self.fold_results])
#         std_acc = np.std([r['best_val_acc'] for r in self.fold_results])
#         avg_f1 = np.mean([r['best_val_f1'] for r in self.fold_results])
#         std_f1 = np.std([r['best_val_f1'] for r in self.fold_results])
        
#         logger.info("\n" + "="*60)
#         logger.info("CROSS-VALIDATION RESULTS")
#         logger.info("="*60)
#         for result in self.fold_results:
#             logger.info(f"Fold {result['fold'] + 1}: Acc={result['best_val_acc']:.3f}, F1={result['best_val_f1']:.3f}")
#         logger.info("-"*60)
#         logger.info(f"Average Accuracy: {avg_acc:.3f} ± {std_acc:.3f}")
#         logger.info(f"Average F1 Score: {avg_f1:.3f} ± {std_f1:.3f}")
#         logger.info("="*60)
        
#         # Save CV results
#         cv_results = {
#             'fold_results': self.fold_results,
#             'average_accuracy': float(avg_acc),
#             'std_accuracy': float(std_acc),
#             'average_f1': float(avg_f1),
#             'std_f1': float(std_f1),
#             'config': self.config,
#             'timestamp': datetime.now().isoformat()
#         }
        
#         save_path = Path(self.config['save_dir']) / 'cv_results.json'
#         with open(save_path, 'w') as f:
#             json.dump(cv_results, f, indent=2)
        
#         logger.info(f"Results saved to {save_path}")
        
#         return cv_results

# def load_dataset(data_dir, config):
#     """Load preprocessed dataset"""
#     data_dir = Path(data_dir)
    
#     all_data = []
#     all_labels = []
#     class_mapping = {}
    
#     # Load each class
#     for class_idx, class_dir in enumerate(sorted(data_dir.iterdir())):
#         if not class_dir.is_dir() or class_dir.name.startswith('.'):
#             continue
        
#         class_name = class_dir.name
#         class_mapping[class_name] = class_idx
        
#         # Load preprocessed files
#         npz_files = list(class_dir.glob('*.npz'))
        
#         for npz_file in npz_files:
#             all_data.append(npz_file)
#             all_labels.append(class_idx)
    
#     logger.info(f"Loaded {len(all_data)} samples from {len(class_mapping)} classes")
#     logger.info(f"Class mapping: {class_mapping}")
    
#     # Save class mapping
#     save_dir = Path(config['save_dir'])
#     save_dir.mkdir(exist_ok=True, parents=True)
    
#     with open(save_dir / 'class_mapping.json', 'w') as f:
#         json.dump(class_mapping, f, indent=2)
    
#     return all_data, all_labels, class_mapping

# def visualize_training_curves(trainer, save_dir):
#     """Visualize training curves for all folds"""
#     save_dir = Path(save_dir)
#     save_dir.mkdir(exist_ok=True, parents=True)
    
#     n_folds = len(trainer.fold_results)
#     fig, axes = plt.subplots(2, n_folds, figsize=(5*n_folds, 10))
    
#     if n_folds == 1:
#         axes = axes.reshape(-1, 1)
    
#     for fold_idx, result in enumerate(trainer.fold_results):
#         # Loss curves
#         ax = axes[0, fold_idx]
#         ax.plot(result['train_losses'], label='Train Loss', color='blue')
#         ax.plot(result['val_losses'], label='Val Loss', color='red')
#         ax.set_title(f'Fold {fold_idx + 1} - Loss')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Loss')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
        
#         # Accuracy curves
#         ax = axes[1, fold_idx]
#         ax.plot(result['train_accs'], label='Train Acc', color='blue')
#         ax.plot(result['val_accs'], label='Val Acc', color='red')
#         ax.plot(result['val_f1s'], label='Val F1', color='green', linestyle='--')
#         ax.set_title(f'Fold {fold_idx + 1} - Metrics')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Score')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(save_dir / 'training_curves.png', dpi=100)
#     plt.close()
    
#     logger.info(f"Training curves saved to {save_dir / 'training_curves.png'}")

# def main():
#     """Main training function"""
    
#     # Set environment variables for optimization
#     os.environ['OMP_NUM_THREADS'] = '1'
    
#     # Configuration
#     config = {
#         'data_dir': 'mediapipe_preprocessed',
#         'save_dir': 'models/videomae_nsl_improved',
#         'model_name': 'MCG-NJU/videomae-base',
#         'num_classes': 10,
#         'batch_size': 4,
#         'epochs': 50,
#         'learning_rate': 1.5e-5,
#         'weight_decay': 0.01,
#         'warmup_ratio': 0.15,
#         'dropout_rate': 0.3,
#         'label_smoothing': 0.1,
#         'mixup_alpha': 0.05,
#         'grad_clip': 0.5,
#         'patience': 15,
#         'num_folds': 3,
#         'freeze_backbone_layers': 10,
#         'seed': 42,
#         'class_specific_weights': {
#             0: 1.0, 1: 1.2, 2: 1.1, 3: 1.0, 4: 1.0,
#             5: 1.0, 6: 1.1, 7: 1.0, 8: 1.0, 9: 1.0
#         }
#     }
    
#     # Set random seeds
#     random.seed(config['seed'])
#     np.random.seed(config['seed'])
#     torch.manual_seed(config['seed'])
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(config['seed'])
    
#     logger.info("="*60)
#     logger.info("VideoMAE Training for Nepali Sign Language Recognition")
#     logger.info("="*60)
#     logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
#     # Load dataset
#     data_paths, labels, class_mapping = load_dataset(config['data_dir'], config)
#     config['num_classes'] = len(class_mapping)
    
#     # Initialize trainer
#     trainer = ImprovedNSLTrainer(config)
    
#     # Train with cross-validation
#     cv_results = trainer.train_cross_validation(data_paths, labels)
    
#     # Visualize training curves
#     visualize_training_curves(trainer, config['save_dir'])
    
#     logger.info("\n" + "="*60)
#     logger.info("TRAINING COMPLETE!")
#     logger.info("="*60)
#     logger.info(f"Models saved to: {config['save_dir']}")
#     logger.info(f"Average CV Accuracy: {cv_results['average_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
#     logger.info(f"Average CV F1 Score: {cv_results['average_f1']:.3f} ± {cv_results['std_f1']:.3f}")
    
#     # Clean up
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     elif torch.backends.mps.is_available():
#         torch.mps.empty_cache()

# if __name__ == "__main__":
#     main()























































































# #!/usr/bin/env python3
# """
# Enhanced VideoMAE Training Pipeline for Nepali Sign Language Recognition
# Cleaned version with no unused variables
# """

# import os
# import gc
# import json
# import random
# import warnings
# import traceback
# from pathlib import Path
# from datetime import datetime
# from collections import defaultdict, Counter

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# import cv2
# from PIL import Image
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import f1_score, accuracy_score

# from transformers import (
#     VideoMAEForVideoClassification, 
#     VideoMAEImageProcessor,
#     get_cosine_schedule_with_warmup
# )

# warnings.filterwarnings('ignore')

# import logging
# logging.basicConfig(
#     level=logging.INFO, 
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def set_random_seeds(seed=42):
#     """Set random seeds for reproducibility"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

# def get_device():
#     """Get the best available device"""
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     elif torch.backends.mps.is_available():
#         return torch.device('mps')
#     else:
#         return torch.device('cpu')

# def cleanup_memory(device):
#     """Clean up GPU/MPS memory"""
#     gc.collect()
#     if device.type == 'cuda':
#         torch.cuda.empty_cache()
#     elif device.type == 'mps':
#         torch.mps.empty_cache()

# # ============================================================================
# # CUSTOM LOSS FUNCTIONS
# # ============================================================================

# class FocalLoss(nn.Module):
#     """Focal Loss for addressing class imbalance"""
#     def __init__(self, gamma=2.0, alpha=None):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
    
#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = (1 - pt) ** self.gamma * ce_loss
        
#         if self.alpha is not None:
#             if self.alpha.device != focal_loss.device:
#                 self.alpha = self.alpha.to(focal_loss.device)
#             focal_loss = self.alpha[targets] * focal_loss
        
#         return focal_loss.mean()

# # ============================================================================
# # MODEL COMPONENTS
# # ============================================================================

# class ImprovedClassifierHead(nn.Module):
#     """Enhanced classifier head with better regularization"""
#     def __init__(self, hidden_size, num_classes, dropout=0.5):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.LayerNorm(hidden_size),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.GELU(),
#             nn.LayerNorm(hidden_size // 2),
#             nn.Dropout(dropout * 0.8),
#             nn.Linear(hidden_size // 2, hidden_size // 4),
#             nn.GELU(),
#             nn.Dropout(dropout * 0.6),
#             nn.Linear(hidden_size // 4, num_classes)
#         )
    
#     def forward(self, x):
#         return self.layers(x)

# class EnhancedVideoMAEModel(nn.Module):
#     """Enhanced VideoMAE model with improved classifier head"""
#     def __init__(self, config):
#         super().__init__()
#         self.videomae = VideoMAEForVideoClassification.from_pretrained(
#             config['model_name'],
#             num_labels=config['num_classes'],
#             ignore_mismatched_sizes=True
#         )
        
#         # Replace classifier
#         hidden_size = self.videomae.config.hidden_size
#         self.videomae.classifier = ImprovedClassifierHead(
#             hidden_size=hidden_size,
#             num_classes=config['num_classes'],
#             dropout=config['dropout_rate']
#         )
        
#         # Freeze backbone layers
#         freeze_layers = config['freeze_backbone_layers']
#         for i, layer in enumerate(self.videomae.videomae.encoder.layer):
#             if i < freeze_layers:
#                 for param in layer.parameters():
#                     param.requires_grad = False
#         logger.info(f"Froze first {freeze_layers} backbone layers")
    
#     def forward(self, pixel_values):
#         return self.videomae(pixel_values=pixel_values)

# # ============================================================================
# # DATASET CLASS
# # ============================================================================

# class EnhancedNSLDataset(Dataset):
#     """Dataset with stronger augmentation"""
    
#     def __init__(self, data_paths, labels, processor, augment=True, config=None):
#         self.data_paths = [Path(p) for p in data_paths]
#         self.labels = labels
#         self.processor = processor
#         self.augment = augment
#         self.config = config or {}
        
#         # Class statistics
#         self.class_counts = Counter(labels)
#         self.num_classes = len(set(labels))
        
#         # Verify data and calculate weights
#         self._verify_data()
#         self.sample_weights = self._calculate_sample_weights()
#         self.class_weights = self._calculate_class_weights()
        
#         logger.info(f"Dataset: {len(self.data_paths)} samples, {self.num_classes} classes")
    
#     def _verify_data(self):
#         """Verify all data files exist"""
#         self.detection_rates = []
        
#         for path in self.data_paths:
#             if path.exists():
#                 try:
#                     data = np.load(str(path), allow_pickle=True)
#                     if 'metadata' in data:
#                         metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
#                         if isinstance(metadata, dict):
#                             self.detection_rates.append(metadata.get('detection_rate', 1.0))
#                         else:
#                             self.detection_rates.append(1.0)
#                     else:
#                         self.detection_rates.append(1.0)
#                 except:
#                     self.detection_rates.append(0.5)
#             else:
#                 logger.warning(f"File not found: {path}")
#                 self.detection_rates.append(0.5)
    
#     def _calculate_sample_weights(self):
#         """Calculate sampling weights"""
#         weights = []
#         for idx in range(len(self.data_paths)):
#             label = self.labels[idx]
#             class_weight = 1.0 / max(self.class_counts[label], 1)
#             detection_weight = self.detection_rates[idx] if idx < len(self.detection_rates) else 0.5
#             weights.append(class_weight * detection_weight)
        
#         weights = np.array(weights)
#         return weights / weights.sum()
    
#     def _calculate_class_weights(self):
#         """Calculate class weights for loss function"""
#         weights = []
#         total = len(self.labels)
        
#         for i in range(self.num_classes):
#             count = max(self.class_counts[i], 1)
#             weights.append(np.sqrt(total / (self.num_classes * count)))
        
#         weights = torch.tensor(weights, dtype=torch.float32)
#         return weights / weights.mean()
    
#     def __len__(self):
#         return len(self.data_paths)
    
#     def __getitem__(self, idx):
#         try:
#             # Load data
#             data = np.load(str(self.data_paths[idx]), allow_pickle=True)
#             frames = data['frames'].copy()
#             label = self.labels[idx]
            
#             # Prepare frames
#             frames = self._prepare_frames(frames)
            
#             # Apply augmentations
#             if self.augment:
#                 frames = self._apply_augmentations(frames)
                
#                 # MixUp or CutMix
#                 if random.random() < 0.3:
#                     if random.random() < 0.5:
#                         frames, label = self._apply_mixup(frames, label, idx)
#                     else:
#                         frames, label = self._apply_cutmix(frames, label, idx)
            
#             # Convert to PIL and process
#             pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
#             inputs = self.processor(pil_frames, return_tensors="pt")
#             pixel_values = inputs['pixel_values'].squeeze(0)
            
#             # Prepare label
#             if not isinstance(label, tuple):
#                 label = torch.tensor(label, dtype=torch.long)
            
#             return pixel_values, label
            
#         except Exception as e:
#             logger.error(f"Error loading sample {idx}: {e}")
#             # Return dummy data
#             return torch.zeros((3, 16, 224, 224)), torch.tensor(0, dtype=torch.long)
    
#     def _prepare_frames(self, frames):
#         """Ensure frames have correct shape"""
#         # Temporal dimension
#         if frames.shape[0] < 16:
#             frames = np.concatenate([frames, np.repeat(frames[-1:], 16 - frames.shape[0], axis=0)])
#         elif frames.shape[0] > 16:
#             frames = frames[:16]
        
#         # Spatial dimensions
#         if frames.shape[1:3] != (224, 224):
#             frames = np.array([cv2.resize(f, (224, 224)) for f in frames])
        
#         # Data type
#         return np.clip(frames, 0, 255).astype(np.uint8)
    
#     def _apply_augmentations(self, frames):
#         """Apply augmentations"""
#         # Temporal shift
#         if random.random() < 0.3:
#             shift = random.randint(-2, 2)
#             if shift != 0:
#                 shifted = np.zeros_like(frames)
#                 if shift > 0:
#                     shifted[shift:] = frames[:-shift]
#                     shifted[:shift] = frames[0]
#                 else:
#                     shifted[:shift] = frames[-shift:]
#                     shifted[shift:] = frames[-1]
#                 frames = shifted
        
#         # Color jitter
#         if random.random() < 0.4:
#             factor = random.uniform(0.8, 1.2)
#             frames = np.clip(frames * factor, 0, 255)
        
#         # Random erasing
#         if random.random() < 0.25:
#             h, w = frames.shape[1:3]
#             area = random.uniform(0.02, 0.1) * h * w
#             aspect = random.uniform(0.3, 3.3)
            
#             h_erase = int(np.sqrt(area / aspect))
#             w_erase = int(np.sqrt(area * aspect))
            
#             if h_erase < h and w_erase < w:
#                 top = random.randint(0, h - h_erase)
#                 left = random.randint(0, w - w_erase)
                
#                 for idx in random.sample(range(16), random.randint(4, 12)):
#                     frames[idx, top:top+h_erase, left:left+w_erase] = random.randint(0, 255)
        
#         return frames.astype(np.uint8)
    
#     def _apply_mixup(self, frames, label, idx):
#         """Apply MixUp"""
#         mix_idx = random.randint(0, len(self.data_paths) - 1)
#         if mix_idx == idx:
#             mix_idx = (idx + 1) % len(self.data_paths)
        
#         mix_data = np.load(str(self.data_paths[mix_idx]), allow_pickle=True)
#         mix_frames = self._prepare_frames(mix_data['frames'].copy())
        
#         lam = np.random.beta(0.2, 0.2)
#         lam = max(0.6, min(0.9, lam))
        
#         frames = (lam * frames + (1 - lam) * mix_frames).astype(np.uint8)
        
#         return frames, (torch.tensor(label, dtype=torch.long), 
#                        torch.tensor(self.labels[mix_idx], dtype=torch.long), 
#                        lam)
    
#     def _apply_cutmix(self, frames, label, idx):
#         """Apply CutMix"""
#         mix_idx = random.randint(0, len(self.data_paths) - 1)
#         if mix_idx == idx:
#             mix_idx = (idx + 1) % len(self.data_paths)
        
#         mix_data = np.load(str(self.data_paths[mix_idx]), allow_pickle=True)
#         mix_frames = self._prepare_frames(mix_data['frames'].copy())
        
#         lam = np.random.beta(1.0, 1.0)
#         h, w = frames.shape[1:3]
        
#         cut_rat = np.sqrt(1. - lam)
#         cut_w = int(w * cut_rat)
#         cut_h = int(h * cut_rat)
        
#         cx = np.random.randint(w)
#         cy = np.random.randint(h)
        
#         x1 = np.clip(cx - cut_w // 2, 0, w)
#         y1 = np.clip(cy - cut_h // 2, 0, h)
#         x2 = np.clip(cx + cut_w // 2, 0, w)
#         y2 = np.clip(cy + cut_h // 2, 0, h)
        
#         frames[:, y1:y2, x1:x2] = mix_frames[:, y1:y2, x1:x2]
#         lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        
#         return frames, (torch.tensor(label, dtype=torch.long),
#                        torch.tensor(self.labels[mix_idx], dtype=torch.long),
#                        lam)

# def collate_fn(batch):
#     """Custom collate function"""
#     pixel_values = torch.stack([item[0] for item in batch])
#     labels = [item[1] for item in batch]
    
#     # Check if any labels are tuples (mixup/cutmix)
#     if any(isinstance(l, tuple) for l in labels):
#         return pixel_values, labels
#     else:
#         return pixel_values, torch.stack(labels)

# # ============================================================================
# # TRAINER CLASS
# # ============================================================================

# class Trainer:
#     """Enhanced trainer"""
    
#     def __init__(self, config):
#         self.config = config
#         self.device = get_device()
#         logger.info(f"Using device: {self.device}")
        
#         self.processor = VideoMAEImageProcessor.from_pretrained(config['model_name'])
#         self.fold_results = []
    
#     def compute_loss(self, outputs, labels, class_weights):
#         """Compute loss"""
#         if self.config['use_focal_loss']:
#             criterion = FocalLoss(gamma=2.0, alpha=class_weights)
#         else:
#             criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)
        
#         # Handle mixed labels
#         if isinstance(labels, list):
#             total_loss = 0
#             for i, label in enumerate(labels):
#                 if isinstance(label, tuple):
#                     label_a, label_b, lam = label
#                     loss = lam * criterion(outputs[i:i+1], label_a.unsqueeze(0).to(self.device))
#                     loss += (1 - lam) * criterion(outputs[i:i+1], label_b.unsqueeze(0).to(self.device))
#                 else:
#                     loss = criterion(outputs[i:i+1], label.unsqueeze(0).to(self.device))
#                 total_loss += loss
#             return total_loss / len(labels)
#         else:
#             return criterion(outputs, labels)
    
#     def train_fold(self, fold_idx, train_paths, train_labels, val_paths, val_labels):
#         """Train on a single fold"""
#         logger.info(f"\nTraining Fold {fold_idx + 1}/{self.config['num_folds']}")
        
#         # Create model
#         model = EnhancedVideoMAEModel(self.config).to(self.device)
        
#         # Create datasets
#         train_dataset = EnhancedNSLDataset(
#             train_paths, train_labels, self.processor, 
#             augment=True, config=self.config
#         )
        
#         val_dataset = EnhancedNSLDataset(
#             val_paths, val_labels, self.processor, 
#             augment=False, config=self.config
#         )
        
#         # Create loaders
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=self.config['batch_size'],
#             sampler=WeightedRandomSampler(train_dataset.sample_weights, len(train_dataset)),
#             collate_fn=collate_fn
#         )
        
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=self.config['batch_size'],
#             shuffle=False,
#             collate_fn=collate_fn
#         )
        
#         # Setup training
#         optimizer = torch.optim.AdamW(
#             model.parameters(), 
#             lr=self.config['learning_rate'],
#             weight_decay=self.config['weight_decay']
#         )
        
#         scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
#         class_weights = train_dataset.class_weights.to(self.device)
        
#         # Training loop
#         best_f1 = 0
#         patience_counter = 0
#         history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        
#         for epoch in range(self.config['epochs']):
#             # Train
#             model.train()
#             train_loss = 0
            
#             for pixel_values, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
#                 pixel_values = pixel_values.to(self.device)
                
#                 optimizer.zero_grad()
#                 outputs = model(pixel_values)
#                 loss = self.compute_loss(outputs.logits, labels, class_weights)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
                
#                 train_loss += loss.item()
            
#             scheduler.step()
            
#             # Validate
#             model.eval()
#             val_loss = 0
#             all_preds = []
#             all_labels = []
            
#             with torch.no_grad():
#                 for pixel_values, labels in val_loader:
#                     pixel_values = pixel_values.to(self.device)
#                     labels = labels.to(self.device)
                    
#                     outputs = model(pixel_values)
#                     loss = F.cross_entropy(outputs.logits, labels)
#                     val_loss += loss.item()
                    
#                     preds = outputs.logits.argmax(dim=1)
#                     all_preds.extend(preds.cpu().numpy())
#                     all_labels.extend(labels.cpu().numpy())
            
#             # Calculate metrics
#             train_loss /= len(train_loader)
#             val_loss /= len(val_loader)
#             val_acc = accuracy_score(all_labels, all_preds)
#             val_f1 = f1_score(all_labels, all_preds, average='weighted')
            
#             history['train_loss'].append(train_loss)
#             history['val_loss'].append(val_loss)
#             history['val_acc'].append(val_acc)
#             history['val_f1'].append(val_f1)
            
#             logger.info(f'Epoch {epoch+1}: Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
#                        f'Val Acc={val_acc:.3f}, Val F1={val_f1:.3f}')
            
#             # Save best model
#             if val_f1 > best_f1 + 0.01:
#                 best_f1 = val_f1
#                 patience_counter = 0
                
#                 save_path = Path(self.config['save_dir']) / f'best_fold_{fold_idx}.pth'
#                 save_path.parent.mkdir(exist_ok=True, parents=True)
#                 torch.save(model.state_dict(), save_path)
#                 logger.info(f'Saved best model with F1={val_f1:.3f}')
#             else:
#                 patience_counter += 1
#                 if patience_counter >= self.config['patience']:
#                     logger.info('Early stopping')
#                     break
        
#         self.fold_results.append({
#             'fold': fold_idx,
#             'best_f1': best_f1,
#             'history': history
#         })
        
#         del model
#         cleanup_memory(self.device)
        
#         return best_f1
    
#     def train_cv(self, data_paths, labels):
#         """Train with cross-validation"""
#         skf = StratifiedKFold(n_splits=self.config['num_folds'], shuffle=True, random_state=42)
        
#         for fold_idx, (train_idx, val_idx) in enumerate(skf.split(data_paths, labels)):
#             train_paths = [data_paths[i] for i in train_idx]
#             train_labels = [labels[i] for i in train_idx]
#             val_paths = [data_paths[i] for i in val_idx]
#             val_labels = [labels[i] for i in val_idx]
            
#             self.train_fold(fold_idx, train_paths, train_labels, val_paths, val_labels)
        
#         # Summary
#         avg_f1 = np.mean([r['best_f1'] for r in self.fold_results])
#         logger.info(f"\nAverage CV F1: {avg_f1:.3f}")
        
#         return self.fold_results

# # ============================================================================
# # MAIN
# # ============================================================================

# def main():
#     config = {
#         'data_dir': 'mediapipe_preprocessed_enhanced',
#         'save_dir': 'models/videomae_clean',
#         'model_name': 'MCG-NJU/videomae-base',
#         'num_classes': 10,
#         'batch_size': 2,
#         'epochs': 30,
#         'learning_rate': 5e-5,
#         'weight_decay': 0.05,
#         'dropout_rate': 0.5,
#         'freeze_backbone_layers': 12,
#         'use_focal_loss': True,
#         'patience': 10,
#         'num_folds': 3
#     }
    
#     set_random_seeds(42)
#     os.environ['OMP_NUM_THREADS'] = '1'
    
#     # Load data
#     data_dir = Path(config['data_dir'])
#     all_paths = []
#     all_labels = []
    
#     for class_idx, class_dir in enumerate(sorted(data_dir.iterdir())):
#         if class_dir.is_dir():
#             for npz_file in class_dir.glob('*.npz'):
#                 all_paths.append(npz_file)
#                 all_labels.append(class_idx)
    
#     logger.info(f"Loaded {len(all_paths)} samples")
    
#     # Train
#     trainer = Trainer(config)
#     trainer.train_cv(all_paths, all_labels)

# if __name__ == "__main__":
#     main()




# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, Subset
# import mediapipe as mp
# from transformers import VideoMAEForVideoClassification, VideoMAEConfig
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# import warnings
# import json
# import random
# from pathlib import Path
# import logging
# from typing import List, Tuple, Dict, Any
# from collections import defaultdict

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Suppress warnings
# warnings.filterwarnings('ignore')

# class MediaPipePreprocessor:
#     """Smart MediaPipe-based video preprocessing with padded cropping."""
    
#     def __init__(self, target_size=(224, 224), padding_ratio=0.3):
#         self.target_size = target_size
#         self.padding_ratio = padding_ratio
#         self.mp_hands = None  # Initialize later to avoid pickling issues
        
#     def _init_mediapipe(self):
#         """Initialize MediaPipe hands detector (called when needed)."""
#         if self.mp_hands is None:
#             self.mp_hands = mp.solutions.hands.Hands(
#                 static_image_mode=False,
#                 max_num_hands=2,
#                 min_detection_confidence=0.7,
#                 min_tracking_confidence=0.5
#             )
        
#     def get_hand_bbox(self, landmarks, frame_shape):
#         """Extract bounding box from hand landmarks with padding."""
#         h, w = frame_shape[:2]
#         x_coords = [lm.x * w for lm in landmarks.landmark]
#         y_coords = [lm.y * h for lm in landmarks.landmark]
        
#         x_min, x_max = min(x_coords), max(x_coords)
#         y_min, y_max = min(y_coords), max(y_coords)
        
#         # Add padding
#         width = x_max - x_min
#         height = y_max - y_min
#         padding_x = width * self.padding_ratio
#         padding_y = height * self.padding_ratio
        
#         x_min = max(0, int(x_min - padding_x))
#         y_min = max(0, int(y_min - padding_y))
#         x_max = min(w, int(x_max + padding_x))
#         y_max = min(h, int(y_max + padding_y))
        
#         return x_min, y_min, x_max, y_max
    
#     def smart_crop_frame(self, frame):
#         """Apply smart cropping based on hand detection."""
#         self._init_mediapipe()  # Initialize MediaPipe when needed
        
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.mp_hands.process(rgb_frame)
        
#         if results.multi_hand_landmarks:
#             if len(results.multi_hand_landmarks) == 1:
#                 # Single hand detected - crop around hand with padding
#                 bbox = self.get_hand_bbox(results.multi_hand_landmarks[0], frame.shape)
#                 x_min, y_min, x_max, y_max = bbox
#                 cropped = frame[y_min:y_max, x_min:x_max]
#             else:
#                 # Two hands detected - crop upper body region
#                 bboxes = [self.get_hand_bbox(hand, frame.shape) for hand in results.multi_hand_landmarks]
                
#                 # Get combined bounding box for both hands
#                 x_min = min(bbox[0] for bbox in bboxes)
#                 y_min = min(bbox[1] for bbox in bboxes)
#                 x_max = max(bbox[2] for bbox in bboxes)
#                 y_max = max(bbox[3] for bbox in bboxes)
                
#                 # Expand to upper body region
#                 h, w = frame.shape[:2]
#                 x_center = (x_min + x_max) // 2
#                 y_center = (y_min + y_max) // 2
                
#                 # Calculate upper body crop dimensions
#                 crop_width = int((x_max - x_min) * 1.5)
#                 crop_height = int((y_max - y_min) * 1.2)
                
#                 x_min = max(0, x_center - crop_width // 2)
#                 y_min = max(0, y_center - crop_height // 2)
#                 x_max = min(w, x_center + crop_width // 2)
#                 y_max = min(h, y_center + crop_height // 2)
                
#                 cropped = frame[y_min:y_max, x_min:x_max]
#         else:
#             # No hands detected - return center crop
#             h, w = frame.shape[:2]
#             crop_size = min(h, w)
#             start_h = (h - crop_size) // 2
#             start_w = (w - crop_size) // 2
#             cropped = frame[start_h:start_h + crop_size, start_w:start_w + crop_size]
        
#         # Resize to target size
#         if cropped.size > 0:
#             return cv2.resize(cropped, self.target_size)
#         else:
#             return cv2.resize(frame, self.target_size)

# class NSLDataset(Dataset):
#     """Dataset class for Nepali Sign Language videos."""
    
#     def __init__(self, video_paths, labels, preprocessor, max_frames=16):
#         self.video_paths = video_paths
#         self.labels = labels
#         self.preprocessor = preprocessor
#         self.max_frames = max_frames
        
#     def __len__(self):
#         return len(self.video_paths)
    
#     def load_video(self, path):
#         """Load and preprocess video frames."""
#         cap = cv2.VideoCapture(path)
#         frames = []
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Apply smart cropping
#             processed_frame = self.preprocessor.smart_crop_frame(frame)
#             frames.append(processed_frame)
        
#         cap.release()
        
#         if len(frames) == 0:
#             logger.warning(f"No frames loaded from {path}")
#             # Create dummy frame
#             frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        
#         # Sample or pad frames to max_frames
#         if len(frames) > self.max_frames:
#             # Sample frames uniformly
#             indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
#             frames = [frames[i] for i in indices]
#         else:
#             # Pad with last frame
#             while len(frames) < self.max_frames:
#                 frames.append(frames[-1])
        
#         # Normalize to [0, 1] and convert to tensor format
#         frames = np.array(frames, dtype=np.float32) / 255.0
#         frames = torch.from_numpy(frames).permute(3, 0, 1, 2)  # (C, T, H, W)
        
#         return frames
    
#     def __getitem__(self, idx):
#         video_path = self.video_paths[idx]
#         label = self.labels[idx]
        
#         try:
#             frames = self.load_video(video_path)
#         except Exception as e:
#             logger.error(f"Error loading video {video_path}: {e}")
#             # Return dummy data
#             frames = torch.zeros((3, self.max_frames, 224, 224))
        
#         return frames, label

# class VideoMAEClassifier(nn.Module):
#     """VideoMAE-based classifier with anti-overfitting measures."""
    
#     def __init__(self, num_classes, dropout_rate=0.5):
#         super(VideoMAEClassifier, self).__init__()
        
#         # Load pretrained VideoMAE configuration (reduced for CPU training)
#         config = VideoMAEConfig(
#             num_frames=8,  # Reduced frames
#             image_size=224,
#             patch_size=16,
#             num_channels=3,
#             hidden_size=384,  # Reduced hidden size for CPU
#             num_hidden_layers=6,  # Reduced layers for CPU
#             num_attention_heads=6,  # Reduced attention heads
#             intermediate_size=1536,  # Reduced intermediate size
#             hidden_dropout_prob=dropout_rate,
#             attention_probs_dropout_prob=dropout_rate,
#             num_labels=num_classes
#         )
        
#         self.videomae = VideoMAEForVideoClassification(config)
        
#         # Additional dropout layers for regularization
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.dropout2 = nn.Dropout(dropout_rate * 0.5)
        
#         # Classification head with regularization (adjusted for smaller hidden size)
#         self.classifier = nn.Sequential(
#             nn.Linear(config.hidden_size, 256),  # Reduced from 512
#             nn.ReLU(),
#             self.dropout1,
#             nn.Linear(256, 128),  # Reduced from 256
#             nn.ReLU(),
#             self.dropout2,
#             nn.Linear(128, num_classes)  # Reduced from 256
#         )
        
#     def forward(self, pixel_values, labels=None):
#         # Get features from VideoMAE backbone
#         outputs = self.videomae.videomae(pixel_values)
#         sequence_output = outputs.last_hidden_state
        
#         # Global average pooling
#         pooled_output = sequence_output.mean(dim=1)
        
#         # Classification
#         logits = self.classifier(pooled_output)
        
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
#             loss = loss_fct(logits, labels)
#             return {'loss': loss, 'logits': logits}
        
#         return {'logits': logits}

# class NSLTrainer:
#     """Main training class with anti-overfitting strategies."""
    
#     def __init__(self, model, device, config):
#         self.model = model
#         self.device = device
#         self.config = config
        
#         # Optimizer with L2 regularization
#         self.optimizer = optim.AdamW(
#             model.parameters(),
#             lr=config['learning_rate'],
#             weight_decay=config['weight_decay']
#         )
        
#         # Learning rate scheduler
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer,
#             mode='min',
#             factor=0.5,
#             patience=5,
#             verbose=True
#         )
        
#         # Early stopping
#         self.best_val_loss = float('inf')
#         self.patience_counter = 0
#         self.early_stopping_patience = config['early_stopping_patience']
        
#     def train_epoch(self, dataloader):
#         """Train for one epoch."""
#         self.model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         progress_bar = tqdm(dataloader, desc="Training")
        
#         for batch_idx, (data, targets) in enumerate(progress_bar):
#             data, targets = data.to(self.device), targets.to(self.device)
            
#             self.optimizer.zero_grad()
            
#             # Forward pass
#             outputs = self.model(data, labels=targets)
#             loss = outputs['loss']
            
#             # Backward pass with gradient clipping
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#             self.optimizer.step()
            
#             # Statistics
#             total_loss += loss.item()
#             _, predicted = outputs['logits'].max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
            
#             # Update progress bar
#             progress_bar.set_postfix({
#                 'Loss': f'{loss.item():.4f}',
#                 'Acc': f'{100.*correct/total:.2f}%'
#             })
        
#         avg_loss = total_loss / len(dataloader)
#         accuracy = 100. * correct / total
        
#         return avg_loss, accuracy
    
#     def validate(self, dataloader):
#         """Validate the model."""
#         self.model.eval()
#         total_loss = 0
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for data, targets in tqdm(dataloader, desc="Validation"):
#                 data, targets = data.to(self.device), targets.to(self.device)
                
#                 outputs = self.model(data, labels=targets)
#                 loss = outputs['loss']
                
#                 total_loss += loss.item()
#                 _, predicted = outputs['logits'].max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()
        
#         avg_loss = total_loss / len(dataloader)
#         accuracy = 100. * correct / total
        
#         return avg_loss, accuracy
    
#     def train_fold(self, train_loader, val_loader, fold_num):
#         """Train one fold with anti-overfitting strategies."""
#         logger.info(f"Training Fold {fold_num}")
        
#         train_losses, val_losses = [], []
#         train_accuracies, val_accuracies = [], []
        
#         self.best_val_loss = float('inf')
#         self.patience_counter = 0
        
#         for epoch in range(self.config['epochs']):
#             logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            
#             # Train
#             train_loss, train_acc = self.train_epoch(train_loader)
            
#             # Validate
#             val_loss, val_acc = self.validate(val_loader)
            
#             # Store metrics
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             train_accuracies.append(train_acc)
#             val_accuracies.append(val_acc)
            
#             # Learning rate scheduling
#             self.scheduler.step(val_loss)
            
#             # Early stopping check
#             if val_loss < self.best_val_loss:
#                 self.best_val_loss = val_loss
#                 self.patience_counter = 0
#                 # Save best model
#                 torch.save(self.model.state_dict(), f'best_model_fold_{fold_num}.pth')
#             else:
#                 self.patience_counter += 1
#                 if self.patience_counter >= self.early_stopping_patience:
#                     logger.info(f"Early stopping triggered at epoch {epoch+1}")
#                     break
            
#             logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
#             logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
#         return train_losses, val_losses, train_accuracies, val_accuracies

# def plot_training_history(train_losses, val_losses, train_accs, val_accs, fold_num, save_dir):
#     """Plot and save training history for a fold."""
#     plt.style.use('seaborn-v0_8')
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
#     # Loss plot
#     epochs = range(1, len(train_losses) + 1)
#     ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
#     ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
#     ax1.set_title(f'Training and Validation Loss - Fold {fold_num}', fontsize=14, fontweight='bold')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Accuracy plot
#     ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
#     ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
#     ax2.set_title(f'Training and Validation Accuracy - Fold {fold_num}', fontsize=14, fontweight='bold')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Accuracy (%)')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/fold_{fold_num}_training_history.png', dpi=300, bbox_inches='tight')
#     plt.close()

# def load_dataset(data_dir):
#     """Load NSL dataset with proper organization."""
#     video_paths = []
#     labels = []
#     class_names = []
    
#     data_path = Path(data_dir)
    
#     # Check if data directory exists
#     if not data_path.exists():
#         logger.error(f"Data directory does not exist: {data_dir}")
#         return video_paths, labels, class_names
    
#     logger.info(f"Scanning directory: {data_dir}")
    
#     # List all items in the directory
#     all_items = list(data_path.iterdir())
#     logger.info(f"Found {len(all_items)} items in directory")
    
#     # Check different possible structures
#     video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    
#     # Structure 1: data_dir/class_name/video_files
#     class_dirs = [item for item in all_items if item.is_dir()]
#     if class_dirs:
#         logger.info(f"Found {len(class_dirs)} class directories: {[d.name for d in class_dirs]}")
        
#         for class_dir in sorted(class_dirs):
#             class_name = class_dir.name
            
#             # Get all video files in this class directory
#             video_files = []
#             all_files_in_class = list(class_dir.iterdir())
            
#             logger.info(f"Class '{class_name}': Found {len(all_files_in_class)} files")
            
#             for file_path in all_files_in_class:
#                 if file_path.is_file() and file_path.suffix in video_extensions:
#                     video_files.append(file_path)
            
#             logger.info(f"Class '{class_name}': {len(video_files)} video files")
            
#             if video_files:  # Only add class if it has videos
#                 class_names.append(class_name)
#                 for video_file in video_files:
#                     video_paths.append(str(video_file))
#                     labels.append(len(class_names) - 1)
    
#     # Structure 2: All videos directly in data_dir (fallback)
#     elif not class_dirs:
#         logger.info("No class directories found, looking for videos directly in data directory")
#         direct_videos = [item for item in all_items if item.is_file() and item.suffix in video_extensions]
        
#         if direct_videos:
#             logger.info(f"Found {len(direct_videos)} videos directly in data directory")
#             # Create a single class for all videos
#             class_names.append("default_class")
#             for video_file in direct_videos:
#                 video_paths.append(str(video_file))
#                 labels.append(0)
    
#     logger.info(f"Final result: Loaded {len(video_paths)} videos from {len(class_names)} classes")
#     logger.info(f"Classes: {class_names}")
    
#     # Additional debugging information
#     if len(video_paths) == 0:
#         logger.error("No videos found! Please check:")
#         logger.error(f"1. Directory path: {data_dir}")
#         logger.error(f"2. Expected structure: {data_dir}/class_name/video_files")
#         logger.error(f"3. Supported extensions: {video_extensions}")
#         logger.error("4. Example of expected structure:")
#         logger.error("   data/")
#         logger.error("   ├── CHA/")
#         logger.error("   │   ├── video1.mp4")
#         logger.error("   │   └── video2.mp4")
#         logger.error("   ├── CHHA/")
#         logger.error("   │   └── video3.mp4")
#         logger.error("   └── ...")
    
#     return video_paths, labels, class_names

# def main():
#     """Main training pipeline."""
    
#     # Configuration
#     config = {
#         'data_dir': 'main_dataset',  # Updated path based on your error
#         'batch_size': 2,  # Reduced batch size for CPU training
#         'epochs': 50,  # Reduced epochs for faster testing
#         'learning_rate': 1e-4,
#         'weight_decay': 1e-4,  # L2 regularization
#         'dropout_rate': 0.3,  # Reduced dropout for small dataset
#         'early_stopping_patience': 10,  # Reduced patience
#         'n_folds': 3,  # Reduced folds for faster testing
#         'max_frames': 8,  # Reduced frames for memory efficiency on CPU
#         'target_size': (224, 224),
#         'results_dir': 'training_results'
#     }
    
#     # Create results directory
#     os.makedirs(config['results_dir'], exist_ok=True)
    
#     # Setup device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")
    
#     # Load dataset
#     logger.info("Loading dataset...")
#     video_paths, labels, class_names = load_dataset(config['data_dir'])
    
#     # Initialize preprocessor
#     preprocessor = MediaPipePreprocessor(
#         target_size=config['target_size'],
#         padding_ratio=0.3
#     )
    
#     # Create full dataset
#     dataset = NSLDataset(
#         video_paths=video_paths,
#         labels=labels,
#         preprocessor=preprocessor,
#         max_frames=config['max_frames']
#     )
    
#     # Stratified K-Fold Cross Validation
#     skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
#     fold_results = []
    
#     for fold, (train_idx, val_idx) in enumerate(skf.split(video_paths, labels)):
#         logger.info(f"\n{'='*50}")
#         logger.info(f"FOLD {fold + 1}/{config['n_folds']}")
#         logger.info(f"{'='*50}")
        
#         # Create data loaders with extensive shuffling
#         train_dataset = Subset(dataset, train_idx)
#         val_dataset = Subset(dataset, val_idx)
        
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=config['batch_size'],
#             shuffle=True,  # Extensive shuffling
#             num_workers=0,  # Set to 0 to avoid multiprocessing issues with MediaPipe
#             pin_memory=False,  # Disable for CPU or when num_workers=0
#             drop_last=True
#         )
        
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=config['batch_size'],
#             shuffle=False,
#             num_workers=0,  # Set to 0 to avoid multiprocessing issues with MediaPipe
#             pin_memory=False  # Disable for CPU or when num_workers=0
#         )
        
#         # Initialize model
#         model = VideoMAEClassifier(
#             num_classes=len(class_names),
#             dropout_rate=config['dropout_rate']
#         ).to(device)
        
#         # Initialize trainer
#         trainer = NSLTrainer(model, device, config)
        
#         # Train fold
#         train_losses, val_losses, train_accs, val_accs = trainer.train_fold(
#             train_loader, val_loader, fold + 1
#         )
        
#         # Plot and save training history
#         plot_training_history(
#             train_losses, val_losses, train_accs, val_accs,
#             fold + 1, config['results_dir']
#         )
        
#         # Store fold results
#         fold_results.append({
#             'fold': fold + 1,
#             'best_val_loss': trainer.best_val_loss,
#             'best_val_acc': max(val_accs),
#             'train_losses': train_losses,
#             'val_losses': val_losses,
#             'train_accs': train_accs,
#             'val_accs': val_accs
#         })
        
#         logger.info(f"Fold {fold + 1} completed.")
#         logger.info(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
#         logger.info(f"Best Validation Accuracy: {max(val_accs):.2f}%")
    
#     # Save overall results
#     with open(f"{config['results_dir']}/fold_results.json", 'w') as f:
#         json.dump(fold_results, f, indent=2)
    
#     # Calculate and log average performance
#     avg_val_loss = np.mean([fold['best_val_loss'] for fold in fold_results])
#     avg_val_acc = np.mean([fold['best_val_acc'] for fold in fold_results])
#     std_val_loss = np.std([fold['best_val_loss'] for fold in fold_results])
#     std_val_acc = np.std([fold['best_val_acc'] for fold in fold_results])
    
#     logger.info(f"\n{'='*50}")
#     logger.info("FINAL RESULTS")
#     logger.info(f"{'='*50}")
#     logger.info(f"Average Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
#     logger.info(f"Average Validation Accuracy: {avg_val_acc:.2f}% ± {std_val_acc:.2f}%")
    
#     # Create summary plot
#     plt.figure(figsize=(12, 8))
    
#     plt.subplot(2, 2, 1)
#     fold_nums = [fold['fold'] for fold in fold_results]
#     val_losses = [fold['best_val_loss'] for fold in fold_results]
#     plt.bar(fold_nums, val_losses, alpha=0.7)
#     plt.title('Best Validation Loss per Fold')
#     plt.xlabel('Fold')
#     plt.ylabel('Validation Loss')
    
#     plt.subplot(2, 2, 2)
#     val_accs = [fold['best_val_acc'] for fold in fold_results]
#     plt.bar(fold_nums, val_accs, alpha=0.7, color='orange')
#     plt.title('Best Validation Accuracy per Fold')
#     plt.xlabel('Fold')
#     plt.ylabel('Validation Accuracy (%)')
    
#     plt.subplot(2, 1, 2)
#     for fold in fold_results:
#         epochs = range(1, len(fold['val_losses']) + 1)
#         plt.plot(epochs, fold['val_losses'], alpha=0.7, label=f'Fold {fold["fold"]}')
#     plt.title('Validation Loss Across All Folds')
#     plt.xlabel('Epoch')
#     plt.ylabel('Validation Loss')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig(f"{config['results_dir']}/cross_validation_summary.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
#     logger.info("Training pipeline completed successfully!")
#     logger.info(f"Results saved to: {config['results_dir']}")

# if __name__ == "__main__":
#     main()
















































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