"""
VideoMAE Inference Script for Nepali Sign Language Recognition
Supports both test set evaluation and unlabeled video classification
"""

import torch
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import numpy as np
from pathlib import Path
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from tqdm import tqdm
import logging
from datetime import datetime
import warnings
import mediapipe as mp
from collections import defaultdict
import pandas as pd

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fix for PyTorch 2.6 compatibility
torch.serialization.add_safe_globals([np.core.multiarray.scalar])


class MediaPipeProcessor:
    """Simple MediaPipe processor for inference"""
    
    def __init__(self, confidence=0.3, hand_padding_ratio=0.35):
        self.confidence = confidence
        self.hand_padding_ratio = hand_padding_ratio
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5
        )
    
    def process_frame(self, frame):
        """Process a single frame to detect and crop hand region"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = self.mp_hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get bounding box
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Add padding
            pad_x = (max_x - min_x) * self.hand_padding_ratio
            pad_y = (max_y - min_y) * self.hand_padding_ratio
            
            min_x = max(0, min_x - pad_x)
            max_x = min(1, max_x + pad_x)
            min_y = max(0, min_y - pad_y)
            max_y = min(1, max_y + pad_y)
            
            # Convert to pixel coordinates
            x1, x2 = int(min_x * w), int(max_x * w)
            y1, y2 = int(min_y * h), int(max_y * h)
            
            # Crop
            cropped = frame[y1:y2, x1:x2]
            
            # Resize to 224x224
            if cropped.size > 0:
                return cv2.resize(cropped, (224, 224))
        
        # If no hand detected, return center crop
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
        return cv2.resize(cropped, (224, 224))
    
    def close(self):
        self.mp_hands.close()


class VideoMAEInference:
    """Inference class for VideoMAE gesture classification"""
    
    def __init__(self, model_dir, device=None):
        """
        Initialize inference engine
        
        Args:
            model_dir: Directory containing trained models and config
            device: Device to use (None for auto-detect)
        """
        self.model_dir = Path(model_dir)
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        self.load_config()
        
        # Load class mapping
        self.load_class_mapping()
        
        # Initialize processor
        self.processor = VideoMAEImageProcessor.from_pretrained(
            self.config.get('model_name', 'MCG-NJU/videomae-base')
        )
        
        # Load models for ensemble
        self.models = self.load_models()
        
        # Initialize MediaPipe processor
        self.mp_processor = MediaPipeProcessor()
    
    def load_config(self):
        """Load training configuration"""
        config_path = self.model_dir / 'cv_results.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                results = json.load(f)
                self.config = results.get('config', {})
                self.cv_results = results
        else:
            # Default config
            self.config = {
                'model_name': 'MCG-NJU/videomae-base',
                'num_classes': 10,
                'num_folds': 3
            }
            logger.warning("Config file not found, using defaults")
    
    def load_class_mapping(self):
        """Load class name mapping"""
        mapping_path = self.model_dir / 'class_mapping.json'
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                self.class_mapping = json.load(f)
                # Create reverse mapping
                self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        else:
            # Default mapping for 10 classes
            self.class_mapping = {
                'CHA': 0, 'CHHA': 1, 'GA': 2, 'GHA': 3, 'JA': 4,
                'JHA': 5, 'KA': 6, 'KHA': 7, 'NGA': 8, 'YAN': 9
            }
            self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
            logger.warning("Class mapping not found, using defaults")
    
    def load_models(self):
        """Load all fold models for ensemble prediction"""
        models = []
        
        for fold in range(self.config.get('num_folds', 3)):
            model_path = self.model_dir / f'best_model_fold_{fold}.pth'
            
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                continue
            
            # Load model
            model = VideoMAEForVideoClassification.from_pretrained(
                self.config.get('model_name', 'MCG-NJU/videomae-base'),
                num_labels=self.config.get('num_classes', 10),
                ignore_mismatched_sizes=True
            )
            
            # Load checkpoint with weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            models.append(model)
            logger.info(f"Loaded model from fold {fold+1}")
        
        if not models:
            raise ValueError("No models could be loaded!")
        
        logger.info(f"Successfully loaded {len(models)} models for ensemble")
        return models
    
    def preprocess_video(self, video_path, num_frames=16, use_mediapipe=True):
        """
        Preprocess a video for inference
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            use_mediapipe: Whether to use MediaPipe for hand detection
        
        Returns:
            Preprocessed frames ready for model input
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Calculate frame indices to sample
        if total_frames <= num_frames:
            indices = list(range(total_frames))
            # Pad if needed
            while len(indices) < num_frames:
                indices.append(total_frames - 1)
        else:
            # Sample frames uniformly
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Process with MediaPipe if enabled
                if use_mediapipe:
                    frame = self.mp_processor.process_frame(frame)
                else:
                    # Simple resize
                    frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        cap.release()
        
        if len(frames) < num_frames:
            # Pad with last frame
            while len(frames) < num_frames:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Convert to PIL Images
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        
        # Process with VideoMAE processor
        inputs = self.processor(pil_frames, return_tensors="pt")
        
        return inputs['pixel_values']
    
    def predict_single(self, video_path, use_mediapipe=True):
        """
        Predict gesture for a single video
        
        Args:
            video_path: Path to video file
            use_mediapipe: Whether to use MediaPipe preprocessing
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess video
        pixel_values = self.preprocess_video(video_path, use_mediapipe=use_mediapipe)
        
        if pixel_values is None:
            logger.error(f"Failed to process video: {video_path}")
            return None
        
        pixel_values = pixel_values.to(self.device)
        
        # Ensemble prediction
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(pixel_values=pixel_values)
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        
        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)[0]  # Remove batch dimension
        
        # Get prediction
        pred_idx = np.argmax(avg_probs)
        pred_class = self.idx_to_class[pred_idx]
        confidence = avg_probs[pred_idx]
        
        # Get top-3 predictions
        top3_indices = np.argsort(avg_probs)[-3:][::-1]
        top3_predictions = [
            {
                'class': self.idx_to_class[idx],
                'confidence': float(avg_probs[idx])
            }
            for idx in top3_indices
        ]
        
        return {
            'predicted_class': pred_class,
            'confidence': float(confidence),
            'top3_predictions': top3_predictions,
            'all_probabilities': {self.idx_to_class[i]: float(p) for i, p in enumerate(avg_probs)}
        }
    
    def evaluate_test_folder(self, test_folder, plot_confusion=True, save_results=True):
        """
        CASE 1: Evaluate model on labeled test folder
        
        Args:
            test_folder: Path to test folder with subfolders for each class
            plot_confusion: Whether to plot confusion matrix
            save_results: Whether to save results to file
        
        Returns:
            Dictionary with evaluation metrics
        """
        test_path = Path(test_folder)
        
        if not test_path.exists():
            raise ValueError(f"Test folder not found: {test_path}")
        
        # Collect test videos
        test_videos = []
        true_labels = []
        
        for class_dir in sorted(test_path.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            
            class_name = class_dir.name
            if class_name not in self.class_mapping:
                logger.warning(f"Unknown class: {class_name}")
                continue
            
            class_idx = self.class_mapping[class_name]
            
            # Find all video files
            for ext in ['*.mp4', '*.avi', '*.mov', '*.MOV', '*.mkv']:
                for video_file in class_dir.glob(ext):
                    test_videos.append(video_file)
                    true_labels.append(class_idx)
        
        if not test_videos:
            raise ValueError("No test videos found!")
        
        logger.info(f"Found {len(test_videos)} test videos")
        
        # Predict for all videos
        predictions = []
        confidences = []
        
        for video_path in tqdm(test_videos, desc="Evaluating test videos"):
            result = self.predict_single(video_path)
            
            if result:
                pred_idx = self.class_mapping[result['predicted_class']]
                predictions.append(pred_idx)
                confidences.append(result['confidence'])
            else:
                # Use random prediction if failed
                predictions.append(0)
                confidences.append(0.0)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        report = classification_report(
            true_labels, predictions,
            target_names=class_names,
            output_dict=True
        )
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Accuracy: {accuracy:.3f}")
        logger.info(f"F1 Score: {f1:.3f}")
        logger.info(f"Average Confidence: {np.mean(confidences):.3f}")
        logger.info("\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=class_names))
        
        # Plot confusion matrix
        if plot_confusion:
            self.plot_confusion_matrix(cm, class_names, 
                                      title=f'Test Set Confusion Matrix (Acc: {accuracy:.3f})')
        
        # Save results
        if save_results:
            results = {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'avg_confidence': float(np.mean(confidences)),
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'predictions': predictions,
                'true_labels': true_labels,
                'timestamp': datetime.now().isoformat()
            }
            
            save_path = self.model_dir / 'test_evaluation_results.json'
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {save_path}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'report': report
        }
    
    def classify_unlabeled_folder(self, unlabeled_folder, save_results=True, 
                                 create_summary=True, confidence_threshold=0.5):
        """
        CASE 2: Classify unlabeled videos in a folder
        
        Args:
            unlabeled_folder: Path to folder containing unlabeled videos
            save_results: Whether to save results to file
            create_summary: Whether to create a summary CSV
            confidence_threshold: Minimum confidence for reliable prediction
        
        Returns:
            List of prediction results for each video
        """
        folder_path = Path(unlabeled_folder)
        
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Find all video files
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.MOV', '*.mkv', '*.webm']:
            video_files.extend(folder_path.glob(ext))
            video_files.extend(folder_path.rglob(ext))  # Include subdirectories
        
        if not video_files:
            raise ValueError(f"No video files found in {folder_path}")
        
        logger.info(f"Found {len(video_files)} videos to classify")
        
        # Classify each video
        results = []
        class_distribution = defaultdict(int)
        
        for video_path in tqdm(video_files, desc="Classifying videos"):
            result = self.predict_single(video_path)
            
            if result:
                # Add video path to result
                result['video_path'] = str(video_path)
                result['video_name'] = video_path.name
                result['reliable'] = result['confidence'] >= confidence_threshold
                
                results.append(result)
                class_distribution[result['predicted_class']] += 1
                
                # Log individual result
                logger.info(f"{video_path.name}: {result['predicted_class']} "
                          f"(confidence: {result['confidence']:.3f})")
            else:
                logger.error(f"Failed to classify: {video_path.name}")
                results.append({
                    'video_path': str(video_path),
                    'video_name': video_path.name,
                    'predicted_class': 'UNKNOWN',
                    'confidence': 0.0,
                    'reliable': False,
                    'error': 'Processing failed'
                })
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("CLASSIFICATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total videos processed: {len(results)}")
        logger.info(f"Reliable predictions (>{confidence_threshold:.1f}): "
                   f"{sum(1 for r in results if r.get('reliable', False))}")
        logger.info("\nClass Distribution:")
        for class_name in sorted(class_distribution.keys()):
            count = class_distribution[class_name]
            percentage = (count / len(results)) * 100
            logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Create summary CSV
        if create_summary:
            df_data = []
            for r in results:
                df_data.append({
                    'Video': r['video_name'],
                    'Predicted Class': r['predicted_class'],
                    'Confidence': f"{r['confidence']:.3f}",
                    'Reliable': 'Yes' if r.get('reliable', False) else 'No',
                    'Top 2nd': r.get('top3_predictions', [{}])[1].get('class', 'N/A') if len(r.get('top3_predictions', [])) > 1 else 'N/A',
                    'Top 2nd Conf': f"{r.get('top3_predictions', [{}])[1].get('confidence', 0):.3f}" if len(r.get('top3_predictions', [])) > 1 else '0.000',
                    'Full Path': r['video_path']
                })
            
            df = pd.DataFrame(df_data)
            csv_path = folder_path / 'classification_results.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"\nSummary CSV saved to: {csv_path}")
            
            # Also create a simple text report
            report_path = folder_path / 'classification_report.txt'
            with open(report_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("VIDEO CLASSIFICATION REPORT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Total Videos: {len(results)}\n")
                f.write(f"Confidence Threshold: {confidence_threshold}\n")
                f.write(f"Reliable Predictions: {sum(1 for r in results if r.get('reliable', False))}\n\n")
                
                f.write("CLASS DISTRIBUTION:\n")
                for class_name in sorted(class_distribution.keys()):
                    count = class_distribution[class_name]
                    f.write(f"  {class_name}: {count}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("DETAILED RESULTS:\n")
                f.write("="*60 + "\n\n")
                
                for r in results:
                    f.write(f"Video: {r['video_name']}\n")
                    f.write(f"  Predicted: {r['predicted_class']} (confidence: {r['confidence']:.3f})\n")
                    f.write(f"  Reliable: {'Yes' if r.get('reliable', False) else 'No'}\n")
                    if 'top3_predictions' in r:
                        f.write("  Top 3 predictions:\n")
                        for i, pred in enumerate(r['top3_predictions'], 1):
                            f.write(f"    {i}. {pred['class']}: {pred['confidence']:.3f}\n")
                    f.write("\n")
            
            logger.info(f"Detailed report saved to: {report_path}")
        
        # Save JSON results
        if save_results:
            json_path = folder_path / 'classification_results.json'
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"JSON results saved to: {json_path}")
        
        # Plot class distribution
        self.plot_class_distribution(class_distribution, 
                                    title=f"Predictions for {len(results)} Videos")
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names, title='Confusion Matrix'):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        # Normalize for percentage display
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotation text with both count and percentage
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)'
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        save_path = self.model_dir / f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    def plot_class_distribution(self, class_distribution, title='Class Distribution'):
        """Plot bar chart of class distribution"""
        plt.figure(figsize=(12, 6))
        
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())
        
        # Sort by class name
        sorted_data = sorted(zip(classes, counts))
        classes, counts = zip(*sorted_data) if sorted_data else ([], [])
        
        # Create bar plot
        bars = plt.bar(range(len(classes)), counts, color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Gesture Class', fontsize=12)
        plt.ylabel('Number of Videos', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        save_path = self.model_dir / f'class_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Class distribution plot saved to: {save_path}")
    
    def close(self):
        """Clean up resources"""
        self.mp_processor.close()


def main():
    """Main function with two cases for inference"""
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    # Path to your trained models directory
    MODEL_DIR = "models/videomae_nsl_fixed"
    
    # CASE 1: Test folder with labeled data (for evaluation)
    TEST_FOLDER = "splitted_dataset/test"  # Update this path
    
    # CASE 2: Unlabeled videos folder (for classification)
    UNLABELED_FOLDER = "unseen_videos"  # Update this path
    
    # ============================================================
    # SELECT WHICH CASE TO RUN (Comment/Uncomment as needed)
    # ============================================================
    
    # Choose which case to run (set one to True)
    # RUN_CASE_1 = False
    # RUN_CASE_2 = True

    RUN_CASE_1 = True
    RUN_CASE_2 = False
    
    # ============================================================
    # INFERENCE
    # ============================================================
    
    try:
        # Initialize inference engine
        logger.info("Initializing VideoMAE inference engine...")
        inference = VideoMAEInference(MODEL_DIR)
        
        # --------------------------------------------------------
        # CASE 1: Evaluate on labeled test folder
        # --------------------------------------------------------
        if RUN_CASE_1:
            logger.info("\n" + "="*60)
            logger.info("CASE 1: Evaluating on labeled test set")
            logger.info("="*60)
            
            if not Path(TEST_FOLDER).exists():
                logger.error(f"Test folder not found: {TEST_FOLDER}")
                logger.info("Please update TEST_FOLDER path in the script")
            else:
                # Run evaluation
                results = inference.evaluate_test_folder(
                    test_folder=TEST_FOLDER,
                    plot_confusion=True,
                    save_results=True
                )
                
                # Print summary
                logger.info("\n" + "="*60)
                logger.info("EVALUATION COMPLETE")
                logger.info(f"Final Accuracy: {results['accuracy']:.3f}")
                logger.info(f"Final F1 Score: {results['f1_score']:.3f}")
                logger.info("="*60)
        
        # --------------------------------------------------------
        # CASE 2: Classify unlabeled videos
        # --------------------------------------------------------
        if RUN_CASE_2:
            logger.info("\n" + "="*60)
            logger.info("CASE 2: Classifying unlabeled videos")
            logger.info("="*60)
            
            if not Path(UNLABELED_FOLDER).exists():
                logger.error(f"Unlabeled folder not found: {UNLABELED_FOLDER}")
                logger.info("Please update UNLABELED_FOLDER path in the script")
            else:
                # Run classification
                results = inference.classify_unlabeled_folder(
                    unlabeled_folder=UNLABELED_FOLDER,
                    save_results=True,
                    create_summary=True,
                    confidence_threshold=0.6  # Adjust as needed
                )
                
                # Print summary
                logger.info("\n" + "="*60)
                logger.info("CLASSIFICATION COMPLETE")
                logger.info(f"Processed {len(results)} videos")
                logger.info(f"Results saved in: {UNLABELED_FOLDER}")
                logger.info("="*60)
        
        # --------------------------------------------------------
        # BONUS: Single video prediction example
        # --------------------------------------------------------
        # Uncomment to test a single video
        """
        single_video_path = "path/to/your/video.mp4"
        if Path(single_video_path).exists():
            logger.info("\nTesting single video prediction...")
            result = inference.predict_single(single_video_path)
            if result:
                logger.info(f"Predicted: {result['predicted_class']} "
                          f"(confidence: {result['confidence']:.3f})")
                logger.info("Top 3 predictions:")
                for i, pred in enumerate(result['top3_predictions'], 1):
                    logger.info(f"  {i}. {pred['class']}: {pred['confidence']:.3f}")
        """
        
        # Clean up
        inference.close()
        logger.info("\nInference complete!")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()















































#!/usr/bin/env python3
"""
VideoMAE Testing Script for Nepali Sign Language Recognition
Fixed version with improved preprocessing alignment
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import numpy as np
from pathlib import Path
import json
import cv2
from PIL import Image
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import warnings
import gc
import os
import mediapipe as mp

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================== CONFIGURATION ==================
RUN_CASE_1 = True   # Test on labeled test dataset
RUN_CASE_2 = False    # Test on unlabeled videos

CONFIG = {
    'model_dir': 'models/videomae_nsl_improved',
    'test_dir': 'main_dataset',
    'unseen_dir': 'unseen_videos',
    'batch_size': 1,  # Reduced for better processing
    'num_frames': 16,
    'frame_size': (224, 224),
    'num_folds': 3,
    'device': 'auto',
    'save_predictions': True,
    'visualize_results': True,
    'output_dir': 'test_results',
    'debug_mode': True,  # Enable debug logging
    
    # MediaPipe settings - MUST match training exactly
    'mediapipe_confidence': 0.3,
    'hand_padding_ratio': 0.35,
    'use_pose_fallback': True,
    'stabilize_crops': True,
    'prioritize_signing_space': True,
    'signing_space_bounds': (0.1, 0.7),
    'min_hand_size_ratio': 0.05,
}

# ================== ENHANCED MEDIAPIPE PREPROCESSOR ==================
class EnhancedMediaPipePreprocessor:
    """Enhanced preprocessor matching training exactly"""
    
    def __init__(self, config):
        self.config = config
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=config['mediapipe_confidence'],
            min_tracking_confidence=0.5
        )
        
        if config['use_pose_fallback']:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5
            )
        
        self.crop_history = deque(maxlen=5)
        self.debug_mode = config.get('debug_mode', False)
    
    def process_video(self, video_path):
        """Process video with enhanced detection"""
        if self.debug_mode:
            logger.info(f"Processing video: {video_path}")
        
        frames = self._load_video_frames(video_path)
        if frames is None:
            logger.warning(f"Failed to load frames from {video_path}")
            return None, None
        
        # Process frames with full detection pipeline
        processed_frames, detection_info = self._process_frames_enhanced(frames, video_path)
        
        if self.debug_mode:
            detection_rate = sum(1 for d in detection_info if d['detected']) / len(detection_info)
            logger.info(f"Detection rate for {Path(video_path).name}: {detection_rate:.1%}")
        
        return processed_frames, detection_info
    
    def _load_video_frames(self, video_path):
        """Load and sample frames matching training"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Load all frames
        all_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        
        if len(all_frames) == 0:
            return None
        
        # Use adaptive sampling matching training
        if len(all_frames) <= self.config['num_frames']:
            selected_frames = all_frames
        else:
            indices = self._get_adaptive_sampling_indices(
                len(all_frames), 
                self.config['num_frames'],
                duration=total_frames/fps if fps > 0 else 0
            )
            selected_frames = [all_frames[i] for i in indices]
        
        # Pad if necessary
        while len(selected_frames) < self.config['num_frames']:
            selected_frames.append(selected_frames[-1] if selected_frames else 
                                 np.zeros((480, 640, 3), dtype=np.uint8))
        
        return selected_frames[:self.config['num_frames']]
    
    def _get_adaptive_sampling_indices(self, total_frames, target_frames, duration=None):
        """Adaptive sampling matching training preprocessing"""
        if duration and duration > 3:
            start_ratio = 0.15
            middle_ratio = 0.7
            end_ratio = 0.15
        else:
            start_ratio = 0.2
            middle_ratio = 0.6
            end_ratio = 0.2
        
        start_count = max(2, int(target_frames * start_ratio))
        middle_count = max(8, int(target_frames * middle_ratio))
        end_count = target_frames - start_count - middle_count
        
        indices = []
        
        # Sample from different parts
        start_end = int(total_frames * 0.25)
        indices.extend(np.linspace(0, start_end, start_count, dtype=int))
        
        middle_start = int(total_frames * 0.25)
        middle_end = int(total_frames * 0.75)
        indices.extend(np.linspace(middle_start, middle_end, middle_count, dtype=int))
        
        end_start = int(total_frames * 0.75)
        indices.extend(np.linspace(end_start, total_frames - 1, end_count, dtype=int))
        
        return sorted(set(indices))[:target_frames]
    
    def _process_frames_enhanced(self, frames, video_path=None):
        """Enhanced frame processing matching training"""
        detections = []
        
        # First pass: detect hands in all frames
        for i, frame in enumerate(frames):
            detection = self._detect_hand_enhanced(frame, frame_idx=i)
            detections.append(detection)
        
        # Calculate detection statistics
        successful_detections = sum(1 for d in detections if d['success'])
        detection_rate = successful_detections / len(detections)
        
        if self.debug_mode:
            logger.info(f"Raw detection rate: {detection_rate:.1%}")
        
        # Apply pose fallback if needed
        if detection_rate < 0.3 and self.config['use_pose_fallback']:
            if self.debug_mode:
                logger.info("Low detection rate, enhancing with pose")
            detections = self._enhance_with_pose(frames, detections)
        
        # Apply temporal smoothing
        if self.config['stabilize_crops']:
            smoothed_detections = self._apply_smoothing(detections)
        else:
            smoothed_detections = detections
        
        # Second pass: crop frames
        processed_frames = []
        detection_info = []
        
        for frame, detection in zip(frames, smoothed_detections):
            if detection['success']:
                cropped = self._intelligent_crop(frame, detection)
                detection_info.append({
                    'detected': True,
                    'method': detection.get('method', 'hand'),
                    'confidence': detection.get('confidence', 0)
                })
            else:
                cropped = self._fallback_crop(frame, detection_info)
                detection_info.append({'detected': False, 'method': 'fallback'})
            
            # Resize to target size
            resized = cv2.resize(cropped, self.config['frame_size'], 
                               interpolation=cv2.INTER_CUBIC)
            processed_frames.append(resized)
        
        return np.array(processed_frames, dtype=np.uint8), detection_info
    
    def _detect_hand_enhanced(self, frame, frame_idx=0):
        """Enhanced hand detection matching training"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        signing_space_top = self.config['signing_space_bounds'][0]
        signing_space_bottom = self.config['signing_space_bounds'][1]
        
        # Try hand detection
        hand_results = self.mp_hands.process(rgb_frame)
        
        if hand_results.multi_hand_landmarks:
            all_hands_info = []
            
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                
                hand_info = {
                    'min_x': min(x_coords),
                    'max_x': max(x_coords),
                    'min_y': min(y_coords),
                    'max_y': max(y_coords),
                    'center_x': center_x,
                    'center_y': center_y,
                    'in_signing_space': signing_space_top <= center_y <= signing_space_bottom,
                    'y_position': center_y,
                    'hand_idx': hand_idx
                }
                
                hand_width = hand_info['max_x'] - hand_info['min_x']
                hand_height = hand_info['max_y'] - hand_info['min_y']
                hand_info['area'] = hand_width * hand_height
                hand_info['width'] = hand_width
                hand_info['height'] = hand_height
                
                all_hands_info.append(hand_info)
            
            # Select best hand (matching training logic)
            if len(all_hands_info) == 1:
                selected_hand = all_hands_info[0]
            else:
                # Multiple hands - prioritize signing space
                hands_in_signing_space = [h for h in all_hands_info if h['in_signing_space']]
                
                if hands_in_signing_space:
                    selected_hand = min(hands_in_signing_space, key=lambda h: h['y_position'])
                else:
                    selected_hand = min(all_hands_info, key=lambda h: h['y_position'])
            
            # Check minimum size
            if selected_hand['area'] < self.config['min_hand_size_ratio']:
                larger_hands = [h for h in all_hands_info if h['area'] >= self.config['min_hand_size_ratio']]
                if larger_hands:
                    selected_hand = min(larger_hands, key=lambda h: h['y_position'])
                else:
                    return {'success': False, 'reason': 'hands_too_small'}
            
            # Calculate confidence
            position_score = 1.0 if selected_hand['in_signing_space'] else 0.7
            size_score = min(1.0, selected_hand['area'] / 0.15)
            confidence = position_score * size_score
            
            return {
                'success': True,
                'method': 'hand',
                'center': (selected_hand['center_x'], selected_hand['center_y']),
                'size': (selected_hand['width'], selected_hand['height']),
                'bbox': (selected_hand['min_x'], selected_hand['min_y'],
                        selected_hand['max_x'], selected_hand['max_y']),
                'confidence': confidence,
                'frame_idx': frame_idx
            }
        
        return {'success': False, 'frame_idx': frame_idx}
    
    def _enhance_with_pose(self, frames, detections):
        """Enhance with pose detection"""
        enhanced = []
        
        for frame, detection in zip(frames, detections):
            if not detection['success']:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = self.mp_pose.process(rgb_frame)
                
                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    
                    # Get wrist positions
                    right_wrist = landmarks[16]
                    left_wrist = landmarks[15]
                    right_elbow = landmarks[14]
                    left_elbow = landmarks[13]
                    
                    # Choose better wrist
                    if right_wrist.visibility > left_wrist.visibility:
                        wrist = right_wrist
                        elbow = right_elbow
                    else:
                        wrist = left_wrist
                        elbow = left_elbow
                    
                    if wrist.visibility > 0.5:
                        # Estimate hand size from arm length
                        arm_length = np.sqrt((wrist.x - elbow.x)**2 + (wrist.y - elbow.y)**2)
                        hand_size = arm_length * 0.5
                        
                        enhanced.append({
                            'success': True,
                            'method': 'pose',
                            'center': (wrist.x, wrist.y),
                            'size': (hand_size, hand_size),
                            'confidence': wrist.visibility,
                            'frame_idx': detection.get('frame_idx', 0)
                        })
                    else:
                        enhanced.append(detection)
                else:
                    enhanced.append(detection)
            else:
                enhanced.append(detection)
        
        return enhanced
    
    def _apply_smoothing(self, detections):
        """Apply temporal smoothing"""
        smoothed = []
        
        for i, detection in enumerate(detections):
            if detection['success']:
                if i > 0 and smoothed and smoothed[-1]['success']:
                    # Smooth with previous detection
                    alpha = 0.7
                    prev = smoothed[-1]
                    
                    smoothed_detection = {
                        'success': True,
                        'center': (
                            alpha * detection['center'][0] + (1-alpha) * prev['center'][0],
                            alpha * detection['center'][1] + (1-alpha) * prev['center'][1]
                        ),
                        'size': (
                            alpha * detection['size'][0] + (1-alpha) * prev['size'][0],
                            alpha * detection['size'][1] + (1-alpha) * prev['size'][1]
                        ),
                        'confidence': detection['confidence'],
                        'method': detection.get('method', 'hand')
                    }
                    smoothed.append(smoothed_detection)
                else:
                    smoothed.append(detection)
            else:
                # Try interpolation for missing frames
                if i > 0 and i < len(detections) - 1:
                    prev_det = detections[i-1] if i > 0 else None
                    next_det = detections[i+1] if i < len(detections) - 1 else None
                    
                    if prev_det and next_det and prev_det['success'] and next_det['success']:
                        # Interpolate
                        interpolated = {
                            'success': True,
                            'center': (
                                (prev_det['center'][0] + next_det['center'][0]) / 2,
                                (prev_det['center'][1] + next_det['center'][1]) / 2
                            ),
                            'size': (
                                (prev_det['size'][0] + next_det['size'][0]) / 2,
                                (prev_det['size'][1] + next_det['size'][1]) / 2
                            ),
                            'confidence': min(prev_det['confidence'], next_det['confidence']) * 0.8,
                            'method': 'interpolated'
                        }
                        smoothed.append(interpolated)
                    else:
                        smoothed.append(detection)
                else:
                    smoothed.append(detection)
        
        return smoothed
    
    def _intelligent_crop(self, frame, detection):
        """Intelligent cropping matching training"""
        h, w = frame.shape[:2]
        cx, cy = detection['center']
        
        # Convert to pixels
        cx_px = int(cx * w)
        cy_px = int(cy * h)
        
        # Calculate crop size with adaptive padding
        hand_w, hand_h = detection['size']
        hand_size_px = max(hand_w * w, hand_h * h)
        
        # Adaptive padding based on detection method
        if detection.get('method') == 'pose':
            padding_ratio = self.config['hand_padding_ratio'] * 1.5
        elif detection.get('confidence', 1.0) < 0.5:
            padding_ratio = self.config['hand_padding_ratio'] * 1.3
        else:
            padding_ratio = self.config['hand_padding_ratio']
        
        crop_size = int(hand_size_px * (1 + 2 * padding_ratio))
        
        # Ensure minimum crop size
        min_crop = int(min(w, h) * 0.4)
        crop_size = max(crop_size, min_crop)
        
        # Calculate boundaries
        x1 = cx_px - crop_size // 2
        y1 = cy_px - crop_size // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        # Apply stabilization if enabled
        if self.config['stabilize_crops'] and self.crop_history:
            recent_crops = list(self.crop_history)
            avg_x1 = np.mean([c[0] for c in recent_crops])
            avg_y1 = np.mean([c[1] for c in recent_crops])
            
            smooth_factor = 0.3
            x1 = int(x1 * smooth_factor + avg_x1 * (1 - smooth_factor))
            y1 = int(y1 * smooth_factor + avg_y1 * (1 - smooth_factor))
            x2 = x1 + crop_size
            y2 = y1 + crop_size
        
        # Ensure within bounds
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        # Store in history
        self.crop_history.append((x1, y1, x2, y2))
        
        return frame[y1:y2, x1:x2]
    
    def _fallback_crop(self, frame, previous_detections):
        """Fallback cropping when no hand detected"""
        h, w = frame.shape[:2]
        
        # Use recent successful detection if available
        if previous_detections:
            recent_detected = [d for d in previous_detections[-5:] if d.get('detected', False)]
            if recent_detected and self.crop_history:
                # Use last known crop with slight expansion
                last_crop = self.crop_history[-1]
                x1, y1, x2, y2 = last_crop
                
                # Expand by 10%
                width = x2 - x1
                height = y2 - y1
                x1 = max(0, int(x1 - width * 0.1))
                y1 = max(0, int(y1 - height * 0.1))
                x2 = min(w, int(x2 + width * 0.1))
                y2 = min(h, int(y2 + height * 0.1))
                
                return frame[y1:y2, x1:x2]
        
        # Default center crop
        crop_ratio = 0.7
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        y_start = (h - new_h) // 2
        x_start = (w - new_w) // 2
        
        return frame[y_start:y_start+new_h, x_start:x_start+new_w]
    
    def close(self):
        """Clean up resources"""
        self.mp_hands.close()
        if hasattr(self, 'mp_pose'):
            self.mp_pose.close()

# ================== DATASET CLASSES ==================
class UnseenVideoDataset(Dataset):
    """Dataset for processing unseen videos"""
    
    def __init__(self, video_paths, preprocessor, processor):
        self.video_paths = video_paths
        self.preprocessor = preprocessor
        self.processor = processor
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        # Preprocess video
        frames, detection_info = self.preprocessor.process_video(video_path)
        
        if frames is None:
            # Return dummy data if processing fails
            frames = np.zeros((16, 224, 224, 3), dtype=np.uint8)
            detection_info = [{'detected': False}] * 16
        
        # Ensure correct dtype
        frames = frames.astype(np.uint8)
        
        # Convert to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Process with VideoMAE processor - match training settings
        inputs = self.processor(
            pil_frames, 
            return_tensors="pt",
            do_resize=False,  # Already resized
            do_center_crop=False  # Already cropped
        )
        
        detection_rate = sum(1 for d in detection_info if d['detected']) / len(detection_info) if detection_info else 0
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'path': str(video_path),
            'detection_rate': detection_rate
        }

# ================== Rest of the code remains the same but with fix for MPS ==================

# ================== DATASET CLASSES ==================
class TestDataset(Dataset):
    """Dataset for preprocessed test data"""
    
    def __init__(self, data_paths, labels, processor):
        self.data_paths = data_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # Load preprocessed data
        data = np.load(self.data_paths[idx], allow_pickle=True)
        frames = data['frames']
        
        # Ensure correct shape
        if frames.shape[0] != 16:
            if frames.shape[0] < 16:
                padding = np.repeat(frames[-1:], 16 - frames.shape[0], axis=0)
                frames = np.concatenate([frames, padding], axis=0)
            else:
                frames = frames[:16]
        
        # Convert to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Process with VideoMAE processor
        inputs = self.processor(pil_frames, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'path': str(self.data_paths[idx])
        }


# ================== MODEL ENSEMBLE ==================
class ModelEnsemble:
    """Ensemble of fold models for prediction"""
    
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.models = []
        self.processor = None
        self._load_models()
    
    def _get_device(self):
        """Get computing device"""
        if self.config['device'] == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(self.config['device'])
    
    def _load_models(self):
        """Load all fold models"""
        model_dir = Path(self.config['model_dir'])
        
        # Load class mapping
        class_mapping_path = model_dir / 'class_mapping.json'
        if class_mapping_path.exists():
            with open(class_mapping_path, 'r') as f:
                self.class_mapping = json.load(f)
                self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        else:
            raise FileNotFoundError(f"Class mapping not found in {model_dir}")
        
        # Load processor
        self.processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
        
        # Load fold models
        for fold in range(self.config['num_folds']):
            model_path = model_dir / f'best_model_fold_{fold}.pth'
            
            if not model_path.exists():
                logger.warning(f"Model for fold {fold} not found: {model_path}")
                continue
            
            # Load model
            model = VideoMAEForVideoClassification.from_pretrained(
                'MCG-NJU/videomae-base',
                num_labels=len(self.class_mapping),
                ignore_mismatched_sizes=True
            )
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            logger.info(f"Loaded model for fold {fold}")
        
        if not self.models:
            raise ValueError("No models loaded!")
        
        logger.info(f"Loaded {len(self.models)} fold models for ensemble")
    
    def predict(self, dataloader, return_probs=False):
        """Make predictions using ensemble"""
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_paths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                pixel_values = batch['pixel_values'].to(self.device)
                
                # Get predictions from each model
                ensemble_probs = []
                for model in self.models:
                    outputs = model(pixel_values=pixel_values)
                    probs = F.softmax(outputs.logits, dim=-1)
                    ensemble_probs.append(probs.cpu().numpy())
                
                # Average probabilities
                avg_probs = np.mean(ensemble_probs, axis=0)
                predictions = np.argmax(avg_probs, axis=1)
                
                all_predictions.extend(predictions)
                all_probabilities.extend(avg_probs)
                
                # Get labels if available
                if 'labels' in batch:
                    all_labels.extend(batch['labels'].cpu().numpy())
                
                # Get paths
                if 'path' in batch:
                    all_paths.extend(batch['path'])
        
        results = {
            'predictions': np.array(all_predictions),
            'paths': all_paths
        }
        
        if return_probs:
            results['probabilities'] = np.array(all_probabilities)
        
        if all_labels:
            results['labels'] = np.array(all_labels)
        
        return results

# ================== EVALUATION FUNCTIONS ==================
def evaluate_test_set(config):
    """Evaluate on labeled test set (CASE 1)"""
    logger.info("\n" + "="*60)
    logger.info("CASE 1: Evaluating on Labeled Test Set")
    logger.info("="*60)
    
    # Create output directory
    output_dir = Path(config['output_dir']) / 'case1_test_set'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    test_dir = Path(config['test_dir'])
    test_data = []
    test_labels = []
    
    # Load ensemble model
    ensemble = ModelEnsemble(config)
    
    # Collect test files
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        
        class_name = class_dir.name
        if class_name not in ensemble.class_mapping:
            logger.warning(f"Class {class_name} not in training classes, skipping")
            continue
        
        class_idx = ensemble.class_mapping[class_name]
        
        # # Check for preprocessed files first
        # preprocessed_dir = Path('mediapipe_preprocessed') / class_name
        # if preprocessed_dir.exists():
        #     npz_files = list(preprocessed_dir.glob('*_processed.npz'))
        #     for npz_file in npz_files:
        #         test_data.append(npz_file)
        #         test_labels.append(class_idx)
        # else:
        #     # Process videos on the fly
        video_files = list(class_dir.glob('*.mp4')) + list(class_dir.glob('*.avi')) + list(class_dir.glob('*.MOV'))
        for video_file in video_files:
            test_data.append(video_file)
            test_labels.append(class_idx)
    
    if not test_data:
        logger.error("No test data found!")
        return
    
    logger.info(f"Found {len(test_data)} test samples")
    
    # Create dataset and dataloader
    if test_data[0].suffix == '.npz':
        # Preprocessed data
        dataset = TestDataset(test_data, test_labels, ensemble.processor)
    else:
        # Raw videos - need preprocessing
        logger.info("Preprocessing test videos...")
        preprocessor = EnhancedMediaPipePreprocessor(config)
        
        # Preprocess and save
        preprocessed_data = []
        for video_path, label in tqdm(zip(test_data, test_labels), total=len(test_data), desc="Preprocessing"):
            frames, detection_info = preprocessor.process_video(video_path)
            if frames is not None:
                # Save preprocessed data
                temp_path = output_dir / f"temp_{video_path.stem}.npz"
                np.savez_compressed(temp_path, frames=frames, detection_info=detection_info)
                preprocessed_data.append(temp_path)
        
        preprocessor.close()
        dataset = TestDataset(preprocessed_data, test_labels, ensemble.processor)
    
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Make predictions
    logger.info("Making predictions...")
    results = ensemble.predict(dataloader, return_probs=True)
    
    # Calculate metrics
    predictions = results['predictions']
    labels = results['labels']
    probabilities = results['probabilities']
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    # Generate classification report
    class_names = [ensemble.idx_to_class[i] for i in range(len(ensemble.class_mapping))]
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("TEST SET RESULTS")
    logger.info("="*60)
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info(f"F1 Score: {f1:.3f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(labels, predictions, target_names=class_names))
    
    # Save results
    if config['save_predictions']:
        results_dict = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': predictions.tolist(),
            'labels': labels.tolist(),
            'probabilities': probabilities.tolist(),
            'paths': results['paths'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_dir / 'test_results.json'}")
    
    # Visualize confusion matrix
    if config['visualize_results']:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Test Set (Accuracy: {accuracy:.3f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=100)
        plt.close()
        logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    
    return results_dict

def predict_unseen_videos(config):
    """Predict on unlabeled videos (CASE 2)"""
    logger.info("\n" + "="*60)
    logger.info("CASE 2: Predicting on Unseen Videos")
    logger.info("="*60)
    
    # Create output directory
    output_dir = Path(config['output_dir']) / 'case2_unseen_videos'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ensemble model
    ensemble = ModelEnsemble(config)
    
    # Find all videos in unseen directory
    unseen_dir = Path(config['unseen_dir'])
    if not unseen_dir.exists():
        logger.error(f"Unseen videos directory not found: {unseen_dir}")
        return
    
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MOV', '*.mkv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(unseen_dir.glob(ext)))
        video_files.extend(list(unseen_dir.glob(f'**/{ext}')))  # Recursive search

    video_files = list(set(video_files))
    
    if not video_files:
        logger.error(f"No video files found in {unseen_dir}")
        return
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Create preprocessor and dataset
    preprocessor = EnhancedMediaPipePreprocessor(config)
    dataset = UnseenVideoDataset(video_files, preprocessor, ensemble.processor)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Make predictions
    logger.info("Processing and predicting...")
    results = ensemble.predict(dataloader, return_probs=True)
    
    # Process results
    predictions = results['predictions']
    probabilities = results['probabilities']
    paths = results['paths']
    
    # Create detailed predictions
    detailed_predictions = []
    for i, (path, pred, probs) in enumerate(zip(paths, predictions, probabilities)):
        video_name = Path(path).name
        predicted_class = ensemble.idx_to_class[pred]
        confidence = float(probs[pred])
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probs)[-3:][::-1]
        top_3_predictions = [
            {
                'class': ensemble.idx_to_class[idx],
                'confidence': float(probs[idx])
            }
            for idx in top_3_idx
        ]
        
        detailed_predictions.append({
            'video': video_name,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_3_predictions': top_3_predictions,
            'all_probabilities': {
                ensemble.idx_to_class[j]: float(probs[j]) 
                for j in range(len(probs))
            }
        })
        
        # Print prediction
        logger.info(f"\n{video_name}:")
        logger.info(f"  Predicted: {predicted_class} (confidence: {confidence:.2%})")
        logger.info("  Top 3: " + ', '.join([f"{p['class']} ({p['confidence']:.1%})" for p in top_3_predictions]))
    
    # Save results
    if config['save_predictions']:
        results_dict = {
            'num_videos': len(video_files),
            'predictions': detailed_predictions,
            'timestamp': datetime.now().isoformat(),
            'model_dir': str(config['model_dir']),
            'class_mapping': ensemble.class_mapping
        }
        
        output_file = output_dir / 'unseen_predictions.json'
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"\nPredictions saved to {output_file}")
        
        # Also save a simplified CSV for easy viewing
        import csv
        csv_file = output_dir / 'unseen_predictions.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Video', 'Predicted Class', 'Confidence', 'Top 3 Classes'])
            for pred in detailed_predictions:
                top_3_str = ', '.join([f"{p['class']} ({p['confidence']:.1%})" 
                                      for p in pred['top_3_predictions']])
                writer.writerow([
                    pred['video'],
                    pred['predicted_class'],
                    f"{pred['confidence']:.2%}",
                    top_3_str
                ])
        logger.info(f"CSV summary saved to {csv_file}")
    
    # Visualize prediction distribution
    if config['visualize_results']:
        class_counts = defaultdict(int)
        for pred in detailed_predictions:
            class_counts[pred['predicted_class']] += 1
        
        plt.figure(figsize=(12, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        plt.bar(classes, counts)
        plt.xlabel('Predicted Class')
        plt.ylabel('Count')
        plt.title('Distribution of Predictions for Unseen Videos')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_distribution.png', dpi=100)
        plt.close()
        logger.info(f"Prediction distribution saved to {output_dir / 'prediction_distribution.png'}")
    
    # Clean up
    preprocessor.close()
    
    return results_dict

def main():
    """Main testing function"""
    
    # Set environment variables for macOS
    os.environ['OMP_NUM_THREADS'] = '1'
    
    logger.info("="*60)
    logger.info("VideoMAE Testing Script for NSL Recognition")
    logger.info("="*60)
    
    # Check which case to run
    if RUN_CASE_1 and RUN_CASE_2:
        logger.error("Both RUN_CASE_1 and RUN_CASE_2 are True. Please set only one to True.")
        return
    
    if not RUN_CASE_1 and not RUN_CASE_2:
        logger.error("Both RUN_CASE_1 and RUN_CASE_2 are False. Please set one to True.")
        return
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Run selected case
    if RUN_CASE_1:
        # Test on labeled test set
        try:
            results = evaluate_test_set(CONFIG)
            logger.info("\n" + "="*60)
            logger.info("CASE 1 COMPLETED SUCCESSFULLY")
            logger.info("="*60)
        except Exception as e:
            logger.error(f"Error in Case 1: {str(e)}")
            import traceback
            traceback.print_exc()
    
    elif RUN_CASE_2:
        # Predict on unseen videos
        try:
            results = predict_unseen_videos(CONFIG)
            logger.info("\n" + "="*60)
            logger.info("CASE 2 COMPLETED SUCCESSFULLY")
            logger.info("="*60)
        except Exception as e:
            logger.error(f"Error in Case 2: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Clean up - FIX for MPS
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        # MPS doesn't have empty_cache method, just pass
        pass
    
    logger.info("\nTesting complete!")

if __name__ == "__main__":
   main()