# import cv2
# import numpy as np
# import mediapipe as mp
# from pathlib import Path
# import random
# from tqdm import tqdm
# import albumentations as A
# import os
# import json
# from collections import defaultdict

# class SignLanguageSafeAugmenter:
#     """Safe augmentation for sign language videos with enhanced multi-hand handling"""
    
#     def __init__(self, target_size=(224, 224), padding_ratio=0.3, debug_mode=False):
#         self.target_size = target_size
#         self.padding_ratio = padding_ratio
#         self.debug_mode = debug_mode
        
#         # Initialize MediaPipe
#         self.mp_hands = mp.solutions.hands.Hands(
#             static_image_mode=False,  # Process video frames
#             max_num_hands=2,
#             model_complexity=1,
#             min_detection_confidence=0.5
#         )
        
#         self.mp_pose = mp.solutions.pose.Pose(
#             static_image_mode=False,
#             model_complexity=1,
#             min_detection_confidence=0.5
#         )
        
#         # Debug visualization tools
#         if self.debug_mode:
#             self.mp_drawing = mp.solutions.drawing_utils
#             self.mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS
        
#         # Define safe augmentations (no rotation or flipping)
#         self.spatial_augmentation = A.Compose([
#             A.Affine(
#                 translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
#                 scale=(0.95, 1.05),
#                 rotate=0,
#                 shear=0,
#                 mode=cv2.BORDER_CONSTANT,
#                 p=0.5
#             ),
#         ])
        
#         self.color_augmentation = A.Compose([
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
#             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.6),
#             A.RandomGamma(gamma_limit=(90, 110), p=0.4),
#         ])
    
#     def apply_specific_augmentation(self, frames, technique):
#         """Apply a specific augmentation technique with controlled variations"""
#         aug_frames = []
        
#         if technique == 'bright_high':
#             # Higher brightness increase
#             factor = random.uniform(1.3, 1.5)
#             for frame in frames:
#                 bright = cv2.convertScaleAbs(frame, alpha=factor, beta=10)
#                 aug_frames.append(bright)
                
#         elif technique == 'bright_low':
#             # Brightness decrease
#             factor = random.uniform(0.6, 0.8)
#             for frame in frames:
#                 dark = cv2.convertScaleAbs(frame, alpha=factor, beta=-10)
#                 aug_frames.append(dark)
                
#         elif technique == 'contrast_high':
#             # High contrast
#             factor = random.uniform(1.4, 1.8)
#             for frame in frames:
#                 contrast = cv2.convertScaleAbs(frame, alpha=factor, beta=128 * (1 - factor))
#                 aug_frames.append(contrast)
                
#         elif technique == 'contrast_low':
#             # Low contrast
#             factor = random.uniform(0.5, 0.7)
#             for frame in frames:
#                 contrast = cv2.convertScaleAbs(frame, alpha=factor, beta=128 * (1 - factor))
#                 aug_frames.append(contrast)
                
#         elif technique == 'saturation_high':
#             # High saturation
#             for frame in frames:
#                 hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
#                 hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(1.3, 1.6)
#                 hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
#                 saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
#                 aug_frames.append(saturated)
                
#         elif technique == 'saturation_low':
#             # Low saturation (more grayscale)
#             for frame in frames:
#                 hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
#                 hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.3, 0.6)
#                 desaturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
#                 aug_frames.append(desaturated)
                
#         elif technique == 'blur_motion':
#             # Motion blur effect
#             kernel_size = random.choice([5, 7, 9])
#             for frame in frames:
#                 kernel = np.zeros((kernel_size, kernel_size))
#                 kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
#                 kernel = kernel / kernel_size
#                 blurred = cv2.filter2D(frame, -1, kernel)
#                 aug_frames.append(blurred)
                
#         elif technique == 'blur_gaussian':
#             # Very light blur
#             kernel_size = random.choice([5, 7])
#             for frame in frames:
#                 blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 1)
#                 aug_frames.append(blurred)
                
#         elif technique == 'noise_moderate':
#             # Moderate noise (still clearly visible hands)
#             for frame in frames:
#                 noise = np.random.normal(0, 15, frame.shape).astype(np.float32)
#                 noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
#                 aug_frames.append(noisy)
                
#         elif technique == 'noise_salt_pepper':
#             # Salt and pepper noise
#             for frame in frames:
#                 noisy = frame.copy()
#                 h, w, c = noisy.shape
#                 num_salt = np.ceil(0.015 * h * w)
#                 coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in (h, w)]
#                 noisy[coords_salt[0], coords_salt[1], :] = 255
#                 num_pepper = np.ceil(0.015 * h * w)
#                 coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in (h, w)]
#                 noisy[coords_pepper[0], coords_pepper[1], :] = 0
#                 aug_frames.append(noisy)
                
#         elif technique == 'gamma':
#             # Gamma correction - smaller range
#             gamma = random.uniform(0.9, 1.1)
#             for frame in frames:
#                 inv_gamma = 1.0 / gamma
#                 table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#                 gamma_corrected = cv2.LUT(frame, table)
#                 aug_frames.append(gamma_corrected)
                
#         else:
#             aug_frames = frames.copy()
        
#         return aug_frames
    
#     def augment_video(self, video_path, video_idx, num_augmentations=5):
#         """Augment a video with safe transformations"""
#         cap = cv2.VideoCapture(str(video_path))
#         fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
#         frames = [frame for ret, frame in iter(lambda: cap.read(), (False, None)) if ret]
#         cap.release()
        
#         if not frames:
#             print(f"    ⚠️  No frames read from {Path(video_path).name}")
#             return [], fps, []
        
#         print(f"  Processing: {Path(video_path).name}")
        
#         augmented_videos = [(frames, 'original', video_idx)]
        
#         techniques = [
#             'bright_high', 'bright_low', 'contrast_high', 'contrast_low',
#             'saturation_high', 'saturation_low', 'blur_motion', 'blur_gaussian',
#             'noise_moderate', 'noise_salt_pepper', 'gamma'
#         ]
#         selected = random.sample(techniques, min(num_augmentations - 1, len(techniques)))
        
#         for technique in selected:
#             aug_frames = self.apply_specific_augmentation(frames, technique)
#             if aug_frames and len(aug_frames) > 0:
#                 augmented_videos.append((aug_frames, technique, video_idx))
        
#         technique_names = [tech for _, tech, _ in augmented_videos]
#         print(f"    ✓ Created {len(augmented_videos)} versions: {', '.join(technique_names)}")
#         return augmented_videos, fps, technique_names
    
#     def save_video(self, frames, output_path, fps=30):
#         """Save frames as a video"""
#         if not frames:
#             return False
#         h, w = frames[0].shape[:2]
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
#         for frame in frames:
#             out.write(frame)
#         out.release()
#         return True
    
#     def process_dataset(self, input_dir, output_dir, num_augmentations=5):
#         """Process an entire dataset of sign language videos"""
#         input_path = Path(input_dir)
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
        
#         video_extensions = ['*.mp4', '*.mov', '*.mkv', '*.avi', "*.MOV"]
#         total_processed = 0
#         total_augmented = 0
#         augmentation_counts = defaultdict(int)
        
#         gesture_folders = sorted([f for f in input_path.iterdir() if f.is_dir()])
#         for gesture_folder in gesture_folders:
#             gesture_name = gesture_folder.name
#             print(f"\nProcessing {gesture_name}...")
            
#             output_gesture_path = output_path / gesture_name
#             output_gesture_path.mkdir(exist_ok=True)
            
#             video_files = sorted(sum([list(gesture_folder.glob(ext)) for ext in video_extensions], []))
#             if not video_files:
#                 print(f"  No video files found in {gesture_folder}")
#                 continue
            
#             for video_idx, video_file in enumerate(tqdm(video_files, desc=f"  {gesture_name}"), 1):
#                 try:
#                     augmented_videos, fps = self.augment_video(video_file, video_idx, num_augmentations)
#                     for aug_frames, technique, idx in augmented_videos:
#                         output_name = f"{gesture_name}_{technique}_{idx}.mp4"
#                         if self.save_video(aug_frames, output_gesture_path / output_name, fps):
#                             total_augmented += 1
#                             if technique != 'original':
#                                 augmentation_counts[technique] += 1
#                     total_processed += 1
#                 except Exception as e:
#                     print(f"  Error processing {video_file.name}: {e}")
        
#         summary = {
#             'input_directory': str(input_dir),
#             'output_directory': str(output_dir),
#             'total_original_videos': total_processed,
#             'total_augmented_videos': total_augmented,
#             'augmentation_counts': dict(augmentation_counts)
#         }
#         with open(output_path / 'summary.json', 'w') as f:
#             json.dump(summary, f, indent=2)
        
#         print(f"\nProcessed {total_processed} videos, created {total_augmented} augmented videos.")
    
#     def __del__(self):
#         """Clean up MediaPipe resources"""
#         if hasattr(self, 'mp_hands'):
#             self.mp_hands.close()
#         if hasattr(self, 'mp_pose'):
#             self.mp_pose.close()

# def main():
#     """Main function"""
#     INPUT_DIR = "main_dataset_all"
#     OUTPUT_DIR = "augmented_videos_3"
#     NUM_AUGMENTATIONS = 5
#     DEBUG_MODE = False
    
#     print("="*60)
#     print("SIGN LANGUAGE SAFE AUGMENTATION")
#     print("="*60)
#     print(f"Input: {INPUT_DIR}")
#     print(f"Output: {OUTPUT_DIR}")
#     print(f"Augmentations per video: {NUM_AUGMENTATIONS}")
#     print("\nAugmentations used:")
#     print("  ✅ Brightness (high/low)")
#     print("  ✅ Contrast (high/low)")
#     print("  ✅ Saturation (high/low)")
#     print("  ✅ Motion and Gaussian blur")
#     print("  ✅ Moderate and salt-and-pepper noise")
#     print("  ✅ Gamma correction")
#     print("\nAugmentations avoided:")
#     print("  ❌ Rotation/flipping")
#     print("  ❌ Heavy noise/blur")
#     print("="*60)
    
#     augmenter = SignLanguageSafeAugmenter(debug_mode=DEBUG_MODE)
#     augmenter.process_dataset(INPUT_DIR, OUTPUT_DIR, num_augmentations=NUM_AUGMENTATIONS)

# if __name__ == "__main__":
#     main()
































































# import cv2
# import numpy as np
# import os
# import random
# from pathlib import Path
# import logging
# from typing import List, Tuple, Dict, Callable
# from tqdm import tqdm

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class NSLVideoAugmenter:
#     """
#     Nepali Sign Language Video Augmentation Pipeline
#     Preserves spatial integrity while applying various augmentations
#     """
    
#     def __init__(self, input_dataset_path: str, output_dataset_path: str, augmentation_probability: float = 0.7):
#         """
#         Initialize the augmenter
        
#         Args:
#             input_dataset_path: Path to the main_dataset containing class folders
#             output_dataset_path: Path to the augmented_videos output folder
#             augmentation_probability: Probability of applying each augmentation (0.0 to 1.0)
#         """
#         self.input_path = Path(input_dataset_path)
#         self.output_path = Path(output_dataset_path)
#         self.aug_prob = augmentation_probability
        
#         # Create output directory if it doesn't exist
#         self.output_path.mkdir(parents=True, exist_ok=True)
        
#         # Define augmentation functions with conservative parameters
#         self.augmentations = {
#             'bright_high': self._brightness_high,
#             'bright_low': self._brightness_low,
#             'contrast_high': self._contrast_high,
#             'contrast_low': self._contrast_low,
#             'saturation_high': self._saturation_high,
#             'saturation_low': self._saturation_low,
#             'blur_motion': self._motion_blur,
#             'blur_gaussian': self._gaussian_blur,
#             'noise_moderate': self._moderate_noise,
#             'noise_salt_pepper': self._salt_pepper_noise,
#             'gamma_high': self._gamma_high,
#             'gamma_low': self._gamma_low
#         }
        
#         # Supported video formats (input and output)
#         self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
#         # Force output to MP4 format for consistency
#         self.output_format = '.mp4'
        
#     def _brightness_high(self, frame: np.ndarray) -> np.ndarray:
#         """Increase brightness while preserving hand visibility"""
#         beta = random.uniform(20, 40)  # Conservative range
#         return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
    
#     def _brightness_low(self, frame: np.ndarray) -> np.ndarray:
#         """Decrease brightness while maintaining hand contrast"""
#         beta = random.uniform(-30, -10)  # Conservative range
#         return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
    
#     def _contrast_high(self, frame: np.ndarray) -> np.ndarray:
#         """Increase contrast moderately"""
#         alpha = random.uniform(1.1, 1.3)  # Conservative range
#         return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    
#     def _contrast_low(self, frame: np.ndarray) -> np.ndarray:
#         """Decrease contrast slightly"""
#         alpha = random.uniform(0.7, 0.9)  # Conservative range
#         return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    
#     def _saturation_high(self, frame: np.ndarray) -> np.ndarray:
#         """Increase saturation"""
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         saturation_scale = random.uniform(1.1, 1.3)
#         hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255).astype(np.uint8)
#         return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
#     def _saturation_low(self, frame: np.ndarray) -> np.ndarray:
#         """Decrease saturation"""
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         saturation_scale = random.uniform(0.7, 0.9)
#         hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255).astype(np.uint8)
#         return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
#     def _motion_blur(self, frame: np.ndarray) -> np.ndarray:
#         """Apply slight motion blur"""
#         kernel_size = random.randint(3, 7)  # Small kernel to preserve detail
#         kernel = np.zeros((kernel_size, kernel_size))
        
#         # Random direction for motion blur
#         direction = random.choice(['horizontal', 'vertical', 'diagonal'])
#         if direction == 'horizontal':
#             kernel[kernel_size // 2, :] = 1
#         elif direction == 'vertical':
#             kernel[:, kernel_size // 2] = 1
#         else:  # diagonal
#             np.fill_diagonal(kernel, 1)
        
#         kernel = kernel / kernel_size
#         return cv2.filter2D(frame, -1, kernel)
    
#     def _gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
#         """Apply light gaussian blur"""
#         kernel_size = random.choice([3, 5])  # Small kernel sizes
#         sigma = random.uniform(0.5, 1.0)  # Low sigma values
#         return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    
#     def _moderate_noise(self, frame: np.ndarray) -> np.ndarray:
#         """Add moderate gaussian noise"""
#         noise = np.random.normal(0, random.uniform(5, 15), frame.shape).astype(np.int16)
#         noisy_frame = frame.astype(np.int16) + noise
#         return np.clip(noisy_frame, 0, 255).astype(np.uint8)
    
#     def _salt_pepper_noise(self, frame: np.ndarray) -> np.ndarray:
#         """Add salt and pepper noise with low density"""
#         noise_density = random.uniform(0.001, 0.005)  # Very low density
#         noisy_frame = frame.copy()
        
#         # Salt noise (white pixels)
#         salt_coords = np.random.random(frame.shape[:2]) < (noise_density / 2)
#         noisy_frame[salt_coords] = 255
        
#         # Pepper noise (black pixels)
#         pepper_coords = np.random.random(frame.shape[:2]) < (noise_density / 2)
#         noisy_frame[pepper_coords] = 0
        
#         return noisy_frame
    
#     def _gamma_high(self, frame: np.ndarray) -> np.ndarray:
#         """Apply gamma correction (lighter)"""
#         gamma = random.uniform(0.7, 0.9)  # Values < 1 make image brighter
#         inv_gamma = 1.0 / gamma
#         table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
#         return cv2.LUT(frame, table)
    
#     def _gamma_low(self, frame: np.ndarray) -> np.ndarray:
#         """Apply gamma correction (darker)"""
#         gamma = random.uniform(1.1, 1.3)  # Values > 1 make image darker
#         inv_gamma = 1.0 / gamma
#         table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
#         return cv2.LUT(frame, table)
    
#     def _apply_random_augmentations(self, frame: np.ndarray, min_augmentations: int = 2, max_augmentations: int = 4) -> Tuple[np.ndarray, List[str]]:
#         """
#         Apply random combination of augmentations to a frame
        
#         Args:
#             frame: Input frame
#             min_augmentations: Minimum number of augmentations to apply
#             max_augmentations: Maximum number of augmentations to apply
            
#         Returns:
#             Tuple of (augmented_frame, list_of_applied_augmentations)
#         """
#         num_augmentations = random.randint(min_augmentations, max_augmentations)
        
#         available_augs = list(self.augmentations.keys())
#         selected_augs = random.sample(available_augs, min(num_augmentations, len(available_augs)))
        
#         augmented_frame = frame.copy()
#         applied_augs = []
        
#         for aug_name in selected_augs:
#             if random.random() < self.aug_prob:
#                 augmented_frame = self.augmentations[aug_name](augmented_frame)
#                 applied_augs.append(aug_name)
        
#         return augmented_frame, applied_augs
    
#     def augment_video(self, video_path: Path, output_path: Path, augmentation_suffix: str) -> bool:
#         """
#         Augment a single video file and save as MP4
        
#         Args:
#             video_path: Path to input video
#             output_path: Path to save augmented video (will be forced to .mp4)
#             augmentation_suffix: Suffix to add to filename
            
#         Returns:
#             True if successful, False otherwise
#         """
#         try:
#             # Ensure output path has .mp4 extension
#             if output_path.suffix.lower() != '.mp4':
#                 output_path = output_path.with_suffix('.mp4')
            
#             # Open input video
#             cap = cv2.VideoCapture(str(video_path))
#             if not cap.isOpened():
#                 logger.error(f"Could not open video: {video_path}")
#                 return False
            
#             # Get video properties
#             fps = int(cap.get(cv2.CAP_PROP_FPS))
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
#             # Ensure valid FPS (default to 30 if invalid)
#             if fps <= 0:
#                 fps = 30
#                 logger.warning(f"Invalid FPS detected, defaulting to {fps}")
            
#             # Use H.264 codec for better compatibility and quality
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             # Alternative codecs to try if mp4v fails
#             alternative_codecs = ['XVID', 'MJPG', 'X264']
            
#             out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
#             # Try alternative codecs if the first one fails
#             if not out.isOpened():
#                 for codec in alternative_codecs:
#                     fourcc_alt = cv2.VideoWriter_fourcc(*codec)
#                     out = cv2.VideoWriter(str(output_path), fourcc_alt, fps, (width, height))
#                     if out.isOpened():
#                         logger.info(f"Using {codec} codec for {output_path.name}")
#                         break
            
#             if not out.isOpened():
#                 logger.error(f"Could not create output video with any codec: {output_path}")
#                 cap.release()
#                 return False
            
#             frame_count = 0
#             applied_augmentations = []
            
#             # Process each frame
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 # Apply augmentations (2-4 augmentations per frame)
#                 augmented_frame, frame_augs = self._apply_random_augmentations(frame, 2, 4)
#                 applied_augmentations.extend(frame_augs)
                
#                 # Write frame
#                 out.write(augmented_frame)
#                 frame_count += 1
            
#             # Clean up
#             cap.release()
#             out.release()
            
#             # Verify output file was created and has valid size
#             if output_path.exists() and output_path.stat().st_size > 0:
#                 # Log augmentation summary
#                 unique_augs = list(set(applied_augmentations))
#                 logger.info(f"Augmented {video_path.name} -> {output_path.name}")
#                 logger.info(f"Applied augmentations: {', '.join(unique_augs)}")
#                 logger.info(f"Processed {frame_count}/{total_frames} frames")
#                 return True
#             else:
#                 logger.error(f"Output video file is invalid or empty: {output_path}")
#                 if output_path.exists():
#                     output_path.unlink()  # Remove invalid file
#                 return False
            
#         except Exception as e:
#             logger.error(f"Error augmenting video {video_path}: {str(e)}")
#             return False
    
#     def process_class_folder(self, class_folder_name: str, augmentations_per_video: int = 5) -> Dict[str, int]:
#         """
#         Process all videos in a class folder
        
#         Args:
#             class_folder_name: Name of the class folder
#             augmentations_per_video: Number of augmented versions to create per original video (minimum 5)
            
#         Returns:
#             Dictionary with processing statistics
#         """
#         # Ensure minimum 5 augmentations
#         if augmentations_per_video < 5:
#             augmentations_per_video = 5
#             logger.info(f"Minimum 5 augmentations required. Set to {augmentations_per_video}")
        
#         input_class_folder = self.input_path / class_folder_name
#         output_class_folder = self.output_path / class_folder_name
        
#         # Create output class folder
#         output_class_folder.mkdir(parents=True, exist_ok=True)
        
#         stats = {
#             'original_videos': 0,
#             'augmented_videos': 0,
#             'failed_videos': 0
#         }
        
#         if not input_class_folder.exists():
#             logger.error(f"Input class folder does not exist: {input_class_folder}")
#             return stats
        
#         # Find all video files in the input folder
#         video_files = []
#         for ext in self.video_extensions:
#             video_files.extend(input_class_folder.glob(f'*{ext}'))
#             video_files.extend(input_class_folder.glob(f'*{ext.upper()}'))
        
#         # Filter original videos (no augmentation suffixes)
#         original_videos = []
#         for video_file in video_files:
#             stem = video_file.stem
#             # Check if it's an original video (no augmentation suffix)
#             is_augmented = any(aug_name in stem for aug_name in self.augmentations.keys())
#             is_augmented = is_augmented or '_aug_' in stem or '_augmented' in stem
            
#             if not is_augmented:
#                 original_videos.append(video_file)
        
#         stats['original_videos'] = len(original_videos)
        
#         logger.info(f"Processing class folder: {class_folder_name}")
#         logger.info(f"Input folder: {input_class_folder}")
#         logger.info(f"Output folder: {output_class_folder}")
#         logger.info(f"Found {len(original_videos)} original videos")
#         logger.info(f"Creating {augmentations_per_video} augmentations per video")
        
#         # Process each original video
#         for video_path in tqdm(original_videos, desc=f"Augmenting {class_folder_name}"):
#             for aug_idx in range(augmentations_per_video):
#                 # Create augmented filename (always .mp4)
#                 stem = video_path.stem
#                 augmented_name = f"{stem}_aug_{aug_idx + 1:02d}{self.output_format}"
#                 output_path = output_class_folder / augmented_name
                
#                 # Skip if augmented version already exists
#                 if output_path.exists():
#                     logger.info(f"Skipping existing augmented video: {augmented_name}")
#                     stats['augmented_videos'] += 1
#                     continue
                
#                 # Augment the video
#                 success = self.augment_video(video_path, output_path, f"aug_{aug_idx + 1:02d}")
                
#                 if success:
#                     stats['augmented_videos'] += 1
#                 else:
#                     stats['failed_videos'] += 1
#                     # Try to remove failed output file if it exists
#                     if output_path.exists():
#                         try:
#                             output_path.unlink()
#                         except:
#                             pass
        
#         return stats
    
#     def process_entire_dataset(self, augmentations_per_video: int = 5) -> Dict[str, Dict[str, int]]:
#         """
#         Process the entire dataset
        
#         Args:
#             augmentations_per_video: Number of augmented versions per original video (minimum 5)
            
#         Returns:
#             Dictionary with statistics for each class
#         """
#         # Ensure minimum 5 augmentations
#         if augmentations_per_video < 5:
#             augmentations_per_video = 5
#             logger.info(f"Minimum 5 augmentations required. Set to {augmentations_per_video}")
        
#         if not self.input_path.exists():
#             raise ValueError(f"Input dataset path does not exist: {self.input_path}")
        
#         # Find all class folders in input directory
#         class_folders = [f for f in self.input_path.iterdir() if f.is_dir()]
        
#         if not class_folders:
#             raise ValueError(f"No class folders found in: {self.input_path}")
        
#         logger.info(f"Found {len(class_folders)} class folders in input dataset")
#         logger.info(f"Input path: {self.input_path}")
#         logger.info(f"Output path: {self.output_path}")
#         logger.info(f"Processing with {augmentations_per_video} augmentations per video")
#         logger.info(f"All output videos will be in MP4 format")
        
#         all_stats = {}
        
#         for class_folder in sorted(class_folders):
#             class_name = class_folder.name
#             try:
#                 stats = self.process_class_folder(class_name, augmentations_per_video)
#                 all_stats[class_name] = stats
                
#                 logger.info(f"Class '{class_name}' completed:")
#                 logger.info(f"  Original videos: {stats['original_videos']}")
#                 logger.info(f"  Augmented videos: {stats['augmented_videos']}")
#                 logger.info(f"  Failed videos: {stats['failed_videos']}")
                
#             except Exception as e:
#                 logger.error(f"Error processing class folder {class_name}: {str(e)}")
#                 all_stats[class_name] = {
#                     'original_videos': 0,
#                     'augmented_videos': 0,
#                     'failed_videos': 0,
#                     'error': str(e)
#                 }
        
#         return all_stats
    
#     def print_dataset_summary(self, stats: Dict[str, Dict[str, int]]):
#         """Print a summary of the augmentation process"""
#         print("\n" + "="*60)
#         print("NEPALI SIGN LANGUAGE DATASET AUGMENTATION SUMMARY")
#         print("="*60)
        
#         total_original = 0
#         total_augmented = 0
#         total_failed = 0
        
#         for class_name, class_stats in stats.items():
#             if 'error' in class_stats:
#                 print(f"{class_name:20s} ERROR: {class_stats['error']}")
#             else:
#                 orig = class_stats['original_videos']
#                 aug = class_stats['augmented_videos']
#                 fail = class_stats['failed_videos']
                
#                 total_original += orig
#                 total_augmented += aug
#                 total_failed += fail
                
#                 print(f"{class_name:20s} Original: {orig:3d} | Augmented: {aug:3d} | Failed: {fail:3d}")
        
#         print("-" * 60)
#         print(f"{'TOTAL':20s} Original: {total_original:3d} | Augmented: {total_augmented:3d} | Failed: {total_failed:3d}")
#         print(f"Dataset expansion: {total_original} -> {total_original + total_augmented} videos")
#         print(f"Expansion factor: {(total_original + total_augmented) / total_original:.2f}x")
#         print("="*60)


# # Example usage
# if __name__ == "__main__":
#     # Configuration
#     INPUT_DATASET_PATH = "main_dataset"  # Input folder name
#     OUTPUT_DATASET_PATH = "augmented_videos"  # Output folder name
#     AUGMENTATIONS_PER_VIDEO = 5  # Minimum 5 augmented versions per original video
#     AUGMENTATION_PROBABILITY = 0.8  # 80% chance of applying each augmentation
    
#     # Initialize augmenter
#     augmenter = NSLVideoAugmenter(
#         input_dataset_path=INPUT_DATASET_PATH,
#         output_dataset_path=OUTPUT_DATASET_PATH,
#         augmentation_probability=AUGMENTATION_PROBABILITY
#     )
    
#     try:
#         # Process the entire dataset
#         print("Starting NSL dataset augmentation...")
#         print(f"Input: {INPUT_DATASET_PATH}")
#         print(f"Output: {OUTPUT_DATASET_PATH}")
#         print(f"Augmentations per video: {AUGMENTATIONS_PER_VIDEO}")
#         print(f"All output videos will be in MP4 format")
#         print("-" * 60)
        
#         stats = augmenter.process_entire_dataset(augmentations_per_video=AUGMENTATIONS_PER_VIDEO)
        
#         # Print summary
#         augmenter.print_dataset_summary(stats)
        
#         print(f"\nAll augmented videos saved to: {OUTPUT_DATASET_PATH}/")
#         print("Each class folder contains augmented videos in MP4 format")
        
#     except Exception as e:
#         logger.error(f"Failed to process dataset: {str(e)}")
#         print(f"Error: {str(e)}")










































































import cv2
import numpy as np
import os
import random
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Callable
from tqdm import tqdm
import subprocess
import json
import mediapipe as mp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NSLVideoAugmenter:
    """
    Enhanced Nepali Sign Language Video Augmentation Pipeline
    Generates 15-20 variations per video with robust hand detection
    """
    
    def __init__(self, input_dataset_path: str, output_dataset_path: str, 
                 augmentation_probability: float = 0.7, enable_hand_detection: bool = True):
        """
        Initialize the augmenter with enhanced augmentation techniques
        
        Args:
            input_dataset_path: Path to the main_dataset containing class folders
            output_dataset_path: Path to the augmented_videos output folder
            augmentation_probability: Probability of applying each augmentation
            enable_hand_detection: Whether to use MediaPipe for robust hand detection
        """
        self.input_path = Path(input_dataset_path)
        self.output_path = Path(output_dataset_path)
        self.aug_prob = augmentation_probability
        self.enable_hand_detection = enable_hand_detection
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe for hand detection if enabled
        if self.enable_hand_detection:
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Define enhanced augmentation functions - 25+ techniques for variety
        self.augmentations = {
            # Brightness variations
            'bright_high': self._brightness_high,
            'bright_low': self._brightness_low,
            'bright_medium_high': self._brightness_medium_high,
            'bright_medium_low': self._brightness_medium_low,
            
            # Contrast variations
            'contrast_high': self._contrast_high,
            'contrast_low': self._contrast_low,
            'contrast_adaptive': self._adaptive_contrast,
            
            # Saturation variations
            'saturation_high': self._saturation_high,
            'saturation_low': self._saturation_low,
            'saturation_selective': self._selective_saturation,
            
            # Blur variations
            'blur_motion': self._motion_blur,
            'blur_gaussian': self._gaussian_blur,
            'blur_median': self._median_blur,
            'blur_bilateral': self._bilateral_blur,
            
            # Noise variations
            'noise_gaussian': self._gaussian_noise,
            'noise_salt_pepper': self._salt_pepper_noise,
            'noise_poisson': self._poisson_noise,
            'noise_speckle': self._speckle_noise,
            
            # Gamma corrections
            'gamma_high': self._gamma_high,
            'gamma_low': self._gamma_low,
            
            # Color adjustments
            'color_warm': self._color_temperature_warm,
            'color_cool': self._color_temperature_cool,
            'hue_shift': self._hue_shift,
            
            # Exposure adjustments
            'exposure_high': self._exposure_high,
            'exposure_low': self._exposure_low,
            
            # Special effects
            'vignette': self._vignette,
            'chromatic_aberration': self._chromatic_aberration,
            'grain': self._film_grain,
            'edge_enhance': self._edge_enhancement,
            
            # Channel manipulations
            'channel_shift_r': self._channel_shift_red,
            'channel_shift_g': self._channel_shift_green,
            'channel_shift_b': self._channel_shift_blue,
            
            # Shadow/Highlight adjustments
            'shadow_adjust': self._shadow_adjustment,
            'highlight_adjust': self._highlight_adjustment,
        }
        
        # Supported video formats
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        self.output_format = '.mp4'
        
        # Check if ffmpeg is available
        self.ffmpeg_available = self._check_ffmpeg_availability()
    
    def _check_two_hands_present(self, frame: np.ndarray) -> Tuple[bool, List]:
        """
        Detect if two hands are present in the frame using MediaPipe
        Returns: (two_hands_detected, hand_landmarks_list)
        """
        if not self.enable_hand_detection:
            return False, []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            return len(results.multi_hand_landmarks) >= 2, results.multi_hand_landmarks
        return False, []
    
    def _get_expanded_crop_region(self, frame: np.ndarray, hand_landmarks) -> Tuple[int, int, int, int]:
        """
        Get expanded crop region for two-hand scenarios (includes upper body)
        """
        h, w = frame.shape[:2]
        
        # Get all hand points
        all_x = []
        all_y = []
        
        for landmarks in hand_landmarks:
            for landmark in landmarks.landmark:
                all_x.append(landmark.x * w)
                all_y.append(landmark.y * h)
        
        if not all_x:
            return 0, 0, w, h
        
        # Calculate bounding box
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Expand region for two hands (include upper body)
        hand_width = max_x - min_x
        hand_height = max_y - min_y
        
        # Expand more vertically to include upper body/torso
        expand_x = hand_width * 0.3
        expand_y = hand_height * 0.6  # More expansion vertically
        
        # Add upper body space (extend upward more than downward)
        crop_x1 = max(0, int(min_x - expand_x))
        crop_y1 = max(0, int(min_y - expand_y * 1.5))  # More space above
        crop_x2 = min(w, int(max_x + expand_x))
        crop_y2 = min(h, int(max_y + expand_y * 0.5))  # Less space below
        
        return crop_x1, crop_y1, crop_x2, crop_y2
    
    def _check_ffmpeg_availability(self) -> bool:
        """Check if ffmpeg is available for metadata operations"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            logger.info("FFmpeg is available for metadata preservation")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("FFmpeg not available - metadata preservation will be limited")
            return False
    
    # ============= Enhanced Augmentation Functions =============
    
    def _brightness_high(self, frame: np.ndarray) -> np.ndarray:
        """Increase brightness significantly"""
        beta = random.uniform(30, 50)
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
    
    def _brightness_low(self, frame: np.ndarray) -> np.ndarray:
        """Decrease brightness significantly"""
        beta = random.uniform(-40, -20)
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
    
    def _brightness_medium_high(self, frame: np.ndarray) -> np.ndarray:
        """Moderate brightness increase"""
        beta = random.uniform(15, 25)
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
    
    def _brightness_medium_low(self, frame: np.ndarray) -> np.ndarray:
        """Moderate brightness decrease"""
        beta = random.uniform(-20, -10)
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
    
    def _contrast_high(self, frame: np.ndarray) -> np.ndarray:
        """Increase contrast"""
        alpha = random.uniform(1.2, 1.5)
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    
    def _contrast_low(self, frame: np.ndarray) -> np.ndarray:
        """Decrease contrast"""
        alpha = random.uniform(0.6, 0.85)
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    
    def _adaptive_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE for adaptive contrast"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=random.uniform(2.0, 4.0), tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    def _saturation_high(self, frame: np.ndarray) -> np.ndarray:
        """Increase saturation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(1.3, 1.6)
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _saturation_low(self, frame: np.ndarray) -> np.ndarray:
        """Decrease saturation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.4, 0.7)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _selective_saturation(self, frame: np.ndarray) -> np.ndarray:
        """Selectively adjust saturation based on color ranges"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Boost warm colors, reduce cool colors
        mask = (hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 150)
        hsv[:, :, 1][mask] *= 1.3
        hsv[:, :, 1][~mask] *= 0.8
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _motion_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply motion blur"""
        kernel_size = random.choice([5, 7, 9, 11])
        kernel = np.zeros((kernel_size, kernel_size))
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        if direction == 'horizontal':
            kernel[kernel_size // 2, :] = 1
        elif direction == 'vertical':
            kernel[:, kernel_size // 2] = 1
        else:
            np.fill_diagonal(kernel, 1)
        kernel = kernel / np.sum(kernel)
        return cv2.filter2D(frame, -1, kernel)
    
    def _gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur"""
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 1.5)
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    
    def _median_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply median blur"""
        kernel_size = random.choice([3, 5, 7])
        return cv2.medianBlur(frame, kernel_size)
    
    def _bilateral_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply bilateral filter (edge-preserving blur)"""
        d = random.choice([5, 7, 9])
        sigma_color = random.uniform(50, 100)
        sigma_space = random.uniform(50, 100)
        return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
    
    def _gaussian_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, random.uniform(8, 20), frame.shape).astype(np.float32)
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)
    
    def _salt_pepper_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add salt and pepper noise"""
        noise_density = random.uniform(0.002, 0.01)
        noisy = frame.copy()
        h, w = noisy.shape[:2]
        
        # Salt
        num_salt = int(noise_density * h * w)
        coords = [np.random.randint(0, i - 1, num_salt) for i in (h, w)]
        noisy[coords[0], coords[1], :] = 255
        
        # Pepper
        num_pepper = int(noise_density * h * w)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in (h, w)]
        noisy[coords[0], coords[1], :] = 0
        
        return noisy
    
    def _poisson_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add Poisson noise"""
        vals = len(np.unique(frame))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(frame * vals) / float(vals)
        return np.clip(noisy * 255, 0, 255).astype(np.uint8)
    
    def _speckle_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add speckle noise"""
        noise = np.random.randn(*frame.shape) * random.uniform(0.02, 0.08)
        noisy = frame + frame * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def _gamma_high(self, frame: np.ndarray) -> np.ndarray:
        """Apply gamma correction (brighter)"""
        gamma = random.uniform(0.6, 0.85)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(frame, table)
    
    def _gamma_low(self, frame: np.ndarray) -> np.ndarray:
        """Apply gamma correction (darker)"""
        gamma = random.uniform(1.2, 1.5)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(frame, table)
    
    def _color_temperature_warm(self, frame: np.ndarray) -> np.ndarray:
        """Apply warm color temperature"""
        b, g, r = cv2.split(frame)
        r = cv2.add(r, random.randint(10, 30))
        b = cv2.subtract(b, random.randint(10, 30))
        return cv2.merge([b, g, r])
    
    def _color_temperature_cool(self, frame: np.ndarray) -> np.ndarray:
        """Apply cool color temperature"""
        b, g, r = cv2.split(frame)
        b = cv2.add(b, random.randint(10, 30))
        r = cv2.subtract(r, random.randint(10, 30))
        return cv2.merge([b, g, r])
    
    def _hue_shift(self, frame: np.ndarray) -> np.ndarray:
        """Shift hue slightly"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _exposure_high(self, frame: np.ndarray) -> np.ndarray:
        """Simulate overexposure"""
        return cv2.addWeighted(frame, random.uniform(1.1, 1.3), 
                              np.ones(frame.shape, frame.dtype) * 255, 
                              random.uniform(0.02, 0.05), 0)
    
    def _exposure_low(self, frame: np.ndarray) -> np.ndarray:
        """Simulate underexposure"""
        return cv2.addWeighted(frame, random.uniform(0.7, 0.9), 
                              np.zeros(frame.shape, frame.dtype), 0, 0)
    
    def _vignette(self, frame: np.ndarray) -> np.ndarray:
        """Add vignette effect"""
        h, w = frame.shape[:2]
        kernel_x = cv2.getGaussianKernel(w, w * 0.5)
        kernel_y = cv2.getGaussianKernel(h, h * 0.5)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        mask = np.stack([mask] * 3, axis=2)
        
        strength = random.uniform(0.3, 0.7)
        vignette = frame * (1 - strength + strength * mask)
        return np.clip(vignette, 0, 255).astype(np.uint8)
    
    def _chromatic_aberration(self, frame: np.ndarray) -> np.ndarray:
        """Simulate chromatic aberration"""
        b, g, r = cv2.split(frame)
        shift = random.randint(1, 3)
        
        # Shift channels slightly
        M_r = np.float32([[1, 0, shift], [0, 1, 0]])
        M_b = np.float32([[1, 0, -shift], [0, 1, 0]])
        
        r = cv2.warpAffine(r, M_r, (frame.shape[1], frame.shape[0]))
        b = cv2.warpAffine(b, M_b, (frame.shape[1], frame.shape[0]))
        
        return cv2.merge([b, g, r])
    
    def _film_grain(self, frame: np.ndarray) -> np.ndarray:
        """Add film grain effect"""
        grain = np.random.normal(0, random.uniform(3, 8), frame.shape)
        grainy = np.clip(frame.astype(np.float32) + grain, 0, 255)
        return grainy.astype(np.uint8)
    
    def _edge_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """Enhance edges"""
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        return cv2.addWeighted(frame, 0.7, sharpened, 0.3, 0)
    
    def _channel_shift_red(self, frame: np.ndarray) -> np.ndarray:
        """Shift red channel"""
        b, g, r = cv2.split(frame)
        r = np.clip(r.astype(np.int16) + random.randint(-20, 20), 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])
    
    def _channel_shift_green(self, frame: np.ndarray) -> np.ndarray:
        """Shift green channel"""
        b, g, r = cv2.split(frame)
        g = np.clip(g.astype(np.int16) + random.randint(-20, 20), 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])
    
    def _channel_shift_blue(self, frame: np.ndarray) -> np.ndarray:
        """Shift blue channel"""
        b, g, r = cv2.split(frame)
        b = np.clip(b.astype(np.int16) + random.randint(-20, 20), 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])
    
    def _shadow_adjustment(self, frame: np.ndarray) -> np.ndarray:
        """Adjust shadows"""
        # Convert to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Adjust shadows (lower values)
        shadow_thresh = 50
        mask = l < shadow_thresh
        l[mask] = np.clip(l[mask] * random.uniform(0.7, 0.9), 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    def _highlight_adjustment(self, frame: np.ndarray) -> np.ndarray:
        """Adjust highlights"""
        # Convert to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Adjust highlights (higher values)
        highlight_thresh = 200
        mask = l > highlight_thresh
        l[mask] = np.clip(l[mask] * random.uniform(1.1, 1.3), 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # ============= Metadata Functions (unchanged) =============
    
    def _extract_video_metadata(self, video_path: str) -> Dict:
        """Extract comprehensive metadata from video using ffprobe"""
        metadata = {}
        
        if not self.ffmpeg_available:
            return metadata
        
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            
            if 'format' in probe_data:
                format_info = probe_data['format']
                metadata['duration'] = format_info.get('duration')
                metadata['size'] = format_info.get('size')
                metadata['bit_rate'] = format_info.get('bit_rate')
                
                if 'tags' in format_info:
                    metadata['creation_time'] = format_info['tags'].get('creation_time')
                    metadata['encoder'] = format_info['tags'].get('encoder')
            
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    metadata['width'] = stream.get('width')
                    metadata['height'] = stream.get('height')
                    metadata['fps'] = stream.get('r_frame_rate')
                    metadata['codec_name'] = stream.get('codec_name')
                    metadata['pix_fmt'] = stream.get('pix_fmt')
                    
                    if 'tags' in stream:
                        metadata['rotate'] = stream['tags'].get('rotate', '0')
                    
                    if 'side_data_list' in stream:
                        for side_data in stream['side_data_list']:
                            if side_data.get('side_data_type') == 'Display Matrix':
                                metadata['display_matrix_rotation'] = side_data.get('rotation', '0')
                    break
            
            return metadata
            
        except Exception as e:
            logger.debug(f"Could not extract metadata from {video_path}: {e}")
            return metadata
    
    def _copy_metadata_to_video(self, source_video: str, target_video: str, metadata: Dict) -> bool:
        """Copy metadata from source to target video using ffmpeg"""
        if not self.ffmpeg_available or not metadata:
            return False
        
        try:
            temp_video = str(target_video) + "_temp.mp4"
            
            cmd = ['ffmpeg', '-i', str(target_video), '-c', 'copy']
            
            if metadata.get('creation_time'):
                cmd.extend(['-metadata', f"creation_time={metadata['creation_time']}"])
            
            if metadata.get('encoder'):
                cmd.extend(['-metadata', f"encoder=NSL_Augmented_{metadata['encoder']}"])
            else:
                cmd.extend(['-metadata', 'encoder=NSL_Video_Augmenter'])
            
            if metadata.get('rotate') and metadata['rotate'] != '0':
                cmd.extend(['-metadata:s:v:0', f"rotate={metadata['rotate']}"])
            
            cmd.extend(['-metadata', 'comment=Augmented NSL video for gesture recognition'])
            cmd.extend(['-metadata', 'title=NSL Augmented Dataset'])
            cmd.extend(['-y', temp_video])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                os.replace(temp_video, target_video)
                logger.debug(f"Successfully copied metadata to {target_video}")
                return True
            else:
                logger.debug(f"FFmpeg metadata copy failed: {result.stderr}")
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                return False
                
        except Exception as e:
            logger.debug(f"Error copying metadata: {e}")
            temp_video = str(target_video) + "_temp.mp4"
            if os.path.exists(temp_video):
                os.remove(temp_video)
            return False
    
    def _preserve_video_metadata(self, original_path: str, augmented_path: str) -> bool:
        """Preserve metadata from original video to augmented video"""
        try:
            metadata = self._extract_video_metadata(original_path)
            
            if not metadata:
                logger.debug(f"No metadata to preserve for {original_path}")
                return False
            
            success = self._copy_metadata_to_video(original_path, augmented_path, metadata)
            
            if success:
                logger.debug(f"Metadata preserved from {original_path} to {augmented_path}")
            
            return success
            
        except Exception as e:
            logger.debug(f"Error preserving metadata: {e}")
            return False
    
    def _get_video_rotation(self, video_path: str) -> int:
        """Get video rotation angle from metadata"""
        try:
            if self.ffmpeg_available:
                metadata = self._extract_video_metadata(video_path)
                
                if 'rotate' in metadata and metadata['rotate']:
                    rotation = int(float(metadata['rotate']))
                    return rotation % 360
                
                if 'display_matrix_rotation' in metadata and metadata['display_matrix_rotation']:
                    rotation = int(float(metadata['display_matrix_rotation']))
                    return abs(rotation) % 360
            
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
                cap.release()
                
                if rotation == 90:
                    return 90
                elif rotation == 180:
                    return 180
                elif rotation == 270:
                    return 270
            
            return 0
            
        except Exception as e:
            logger.debug(f"Could not get rotation metadata from {video_path}: {e}")
            return 0
    
    def _apply_selected_augmentations(self, frame: np.ndarray, 
                                     selected_techniques: List[str]) -> np.ndarray:
        """Apply specific augmentation techniques to a frame"""
        augmented_frame = frame.copy()
        
        for technique in selected_techniques:
            if technique in self.augmentations and random.random() < self.aug_prob:
                augmented_frame = self.augmentations[technique](augmented_frame)
        
        return augmented_frame
    
    def augment_video(self, video_path: Path, output_path: Path, 
                     augmentation_techniques: List[str]) -> bool:
        """
        Augment a single video file with specific techniques
        
        Args:
            video_path: Path to input video
            output_path: Path to save augmented video
            augmentation_techniques: List of augmentation techniques to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if output_path.suffix.lower() != '.mp4':
                output_path = output_path.with_suffix('.mp4')
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0:
                fps = 30
                logger.warning(f"Invalid FPS detected, defaulting to {fps}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Could not create output video: {output_path}")
                cap.release()
                return False
            
            frame_count = 0
            two_hands_detected = False
            
            # First pass: check if video has two-hand gestures
            sample_frames = []
            sample_interval = max(1, total_frames // 10)  # Sample 10 frames
            
            for i in range(0, total_frames, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    two_hands, _ = self._check_two_hands_present(frame)
                    if two_hands:
                        two_hands_detected = True
                        break
            
            # Reset to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Process each frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply robust cropping for two-hand scenarios
                if self.enable_hand_detection and two_hands_detected:
                    _, hand_landmarks = self._check_two_hands_present(frame)
                    if hand_landmarks:
                        x1, y1, x2, y2 = self._get_expanded_crop_region(frame, hand_landmarks)
                        # Apply crop with safety margins
                        frame_cropped = frame[y1:y2, x1:x2]
                        if frame_cropped.size > 0:
                            frame_cropped = cv2.resize(frame_cropped, (width, height))
                            frame = frame_cropped
                
                # Apply augmentations
                augmented_frame = self._apply_selected_augmentations(frame, augmentation_techniques)
                
                out.write(augmented_frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            if output_path.exists() and output_path.stat().st_size > 0:
                self._preserve_video_metadata(str(video_path), str(output_path))
                
                logger.info(f"Augmented {video_path.name} -> {output_path.name}")
                logger.info(f"Applied: {', '.join(augmentation_techniques[:5])}...")
                logger.info(f"Processed {frame_count}/{total_frames} frames")
                if two_hands_detected:
                    logger.info(f"Two-hand gesture detected - expanded crop applied")
                
                return True
            else:
                logger.error(f"Output video file is invalid or empty: {output_path}")
                if output_path.exists():
                    output_path.unlink()
                return False
            
        except Exception as e:
            logger.error(f"Error augmenting video {video_path}: {str(e)}")
            return False
    
    def generate_augmentation_combinations(self, num_augmentations: int = 18) -> List[List[str]]:
        """
        Generate diverse combinations of augmentation techniques
        Returns list of augmentation combinations (15-20 unique combinations)
        """
        all_techniques = list(self.augmentations.keys())
        combinations = []
        
        # Create diverse combinations with 2-5 techniques each
        for i in range(num_augmentations):
            num_techniques = random.randint(2, 5)
            
            # Ensure diversity by categorizing techniques
            brightness_techs = [t for t in all_techniques if 'bright' in t]
            contrast_techs = [t for t in all_techniques if 'contrast' in t]
            saturation_techs = [t for t in all_techniques if 'saturation' in t]
            blur_techs = [t for t in all_techniques if 'blur' in t]
            noise_techs = [t for t in all_techniques if 'noise' in t]
            color_techs = [t for t in all_techniques if 'color' in t or 'hue' in t]
            special_techs = [t for t in all_techniques if t not in brightness_techs + contrast_techs + 
                           saturation_techs + blur_techs + noise_techs + color_techs]
            
            # Build combination with diversity
            combination = []
            categories = [brightness_techs, contrast_techs, saturation_techs, 
                         blur_techs, noise_techs, color_techs, special_techs]
            
            for _ in range(num_techniques):
                available_cats = [c for c in categories if c]
                if available_cats:
                    chosen_cat = random.choice(available_cats)
                    if chosen_cat:
                        technique = random.choice(chosen_cat)
                        if technique not in combination:
                            combination.append(technique)
            
            # Ensure minimum techniques
            while len(combination) < 2:
                technique = random.choice(all_techniques)
                if technique not in combination:
                    combination.append(technique)
            
            if combination not in combinations:
                combinations.append(combination)
        
        # Ensure we have 15-20 unique combinations
        while len(combinations) < 15:
            num_techniques = random.randint(2, 5)
            combination = random.sample(all_techniques, num_techniques)
            if combination not in combinations:
                combinations.append(combination)
        
        return combinations[:min(20, len(combinations))]
    
    def process_class_folder(self, class_folder_name: str, 
                            augmentations_per_video: int = 18) -> Dict[str, int]:
        """
        Process all videos in a class folder with 15-20 augmentations per video
        """
        if augmentations_per_video < 15:
            augmentations_per_video = 18
            logger.info(f"Set to {augmentations_per_video} augmentations per video for diversity")
        
        input_class_folder = self.input_path / class_folder_name
        output_class_folder = self.output_path / class_folder_name
        output_class_folder.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'original_videos': 0,
            'augmented_videos': 0,
            'failed_videos': 0
        }
        
        if not input_class_folder.exists():
            logger.error(f"Input class folder does not exist: {input_class_folder}")
            return stats
        
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(input_class_folder.glob(f'*{ext}'))
            video_files.extend(input_class_folder.glob(f'*{ext.upper()}'))
        
        original_videos = []
        for video_file in video_files:
            stem = video_file.stem
            is_augmented = any(aug_name in stem for aug_name in self.augmentations.keys())
            is_augmented = is_augmented or '_aug_' in stem or '_augmented' in stem
            
            if not is_augmented:
                original_videos.append(video_file)
        
        stats['original_videos'] = len(original_videos)
        
        logger.info(f"Processing class: {class_folder_name}")
        logger.info(f"Found {len(original_videos)} original videos")
        logger.info(f"Creating {augmentations_per_video} augmentations per video")
        
        for video_path in tqdm(original_videos, desc=f"Augmenting {class_folder_name}"):
            # Generate unique augmentation combinations for this video
            augmentation_combinations = self.generate_augmentation_combinations(augmentations_per_video)
            
            for aug_idx, techniques in enumerate(augmentation_combinations, 1):
                stem = video_path.stem
                technique_suffix = '_'.join([t.split('_')[0][:3] for t in techniques[:3]])
                augmented_name = f"{stem}_aug_{aug_idx:02d}_{technique_suffix}{self.output_format}"
                output_path = output_class_folder / augmented_name
                
                if output_path.exists():
                    logger.debug(f"Skipping existing: {augmented_name}")
                    stats['augmented_videos'] += 1
                    continue
                
                success = self.augment_video(video_path, output_path, techniques)
                
                if success:
                    stats['augmented_videos'] += 1
                else:
                    stats['failed_videos'] += 1
                    if output_path.exists():
                        try:
                            output_path.unlink()
                        except:
                            pass
        
        return stats
    
    def process_entire_dataset(self, augmentations_per_video: int = 18) -> Dict[str, Dict[str, int]]:
        """Process the entire dataset with 15-20 augmentations per video"""
        if augmentations_per_video < 15:
            augmentations_per_video = 18
        
        if not self.input_path.exists():
            raise ValueError(f"Input dataset path does not exist: {self.input_path}")
        
        class_folders = [f for f in self.input_path.iterdir() if f.is_dir()]
        
        if not class_folders:
            raise ValueError(f"No class folders found in: {self.input_path}")
        
        logger.info(f"Found {len(class_folders)} class folders")
        logger.info(f"Creating {augmentations_per_video} augmentations per video")
        logger.info(f"Total augmentation techniques available: {len(self.augmentations)}")
        
        all_stats = {}
        
        for class_folder in sorted(class_folders):
            class_name = class_folder.name
            try:
                stats = self.process_class_folder(class_name, augmentations_per_video)
                all_stats[class_name] = stats
                
                logger.info(f"Class '{class_name}' completed:")
                logger.info(f"  Original: {stats['original_videos']}")
                logger.info(f"  Augmented: {stats['augmented_videos']}")
                logger.info(f"  Failed: {stats['failed_videos']}")
                
            except Exception as e:
                logger.error(f"Error processing class {class_name}: {str(e)}")
                all_stats[class_name] = {
                    'original_videos': 0,
                    'augmented_videos': 0,
                    'failed_videos': 0,
                    'error': str(e)
                }
        
        return all_stats
    
    def print_dataset_summary(self, stats: Dict[str, Dict[str, int]]):
        """Print summary of augmentation process"""
        print("\n" + "="*70)
        print("NSL DATASET AUGMENTATION SUMMARY (15-20 Variations)")
        print("="*70)
        
        total_original = 0
        total_augmented = 0
        total_failed = 0
        
        for class_name, class_stats in stats.items():
            if 'error' in class_stats:
                print(f"{class_name:20s} ERROR: {class_stats['error']}")
            else:
                orig = class_stats['original_videos']
                aug = class_stats['augmented_videos']
                fail = class_stats['failed_videos']
                
                total_original += orig
                total_augmented += aug
                total_failed += fail
                
                avg_per_video = aug / orig if orig > 0 else 0
                print(f"{class_name:20s} Orig: {orig:3d} | Aug: {aug:4d} | Failed: {fail:3d} | Avg/video: {avg_per_video:.1f}")
        
        print("-" * 70)
        print(f"{'TOTAL':20s} Orig: {total_original:3d} | Aug: {total_augmented:4d} | Failed: {total_failed:3d}")
        
        if total_original > 0:
            print(f"\nDataset expansion: {total_original} → {total_original + total_augmented} videos")
            print(f"Expansion factor: {(total_original + total_augmented) / total_original:.1f}x")
            print(f"Average augmentations per video: {total_augmented / total_original:.1f}")
        
        print("="*70)
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'mp_hands'):
            self.mp_hands.close()
        if hasattr(self, 'mp_pose'):
            self.mp_pose.close()


# Example usage
if __name__ == "__main__":
    # Configuration
    INPUT_DATASET_PATH = "main_dataset"
    OUTPUT_DATASET_PATH = "augmented_videos_enhanced"
    AUGMENTATIONS_PER_VIDEO = 18  # Will generate 15-20 variations
    AUGMENTATION_PROBABILITY = 0.8
    ENABLE_HAND_DETECTION = True  # Enable robust two-hand detection
    
    # Initialize augmenter
    augmenter = NSLVideoAugmenter(
        input_dataset_path=INPUT_DATASET_PATH,
        output_dataset_path=OUTPUT_DATASET_PATH,
        augmentation_probability=AUGMENTATION_PROBABILITY,
        enable_hand_detection=ENABLE_HAND_DETECTION
    )
    
    try:
        print("="*70)
        print("NSL VIDEO AUGMENTATION - ENHANCED VERSION")
        print("="*70)
        print(f"Input: {INPUT_DATASET_PATH}")
        print(f"Output: {OUTPUT_DATASET_PATH}")
        print(f"Target augmentations per video: 15-20")
        print(f"Total techniques available: {len(augmenter.augmentations)}")
        print(f"Hand detection: {'Enabled (with two-hand support)' if ENABLE_HAND_DETECTION else 'Disabled'}")
        
        if augmenter.ffmpeg_available:
            print("✓ FFmpeg detected - Full metadata preservation")
        else:
            print("⚠ FFmpeg not detected - Limited metadata")
        
        print("-" * 70)
        
        stats = augmenter.process_entire_dataset(augmentations_per_video=AUGMENTATIONS_PER_VIDEO)
        
        augmenter.print_dataset_summary(stats)
        
        print(f"\n✅ Augmentation complete!")
        print(f"Output saved to: {OUTPUT_DATASET_PATH}/")
        
    except Exception as e:
        logger.error(f"Failed to process dataset: {str(e)}")
        print(f"Error: {str(e)}")