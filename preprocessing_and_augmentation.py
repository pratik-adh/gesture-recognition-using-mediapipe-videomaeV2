# import cv2
# import numpy as np
# import os
# import random
# from pathlib import Path
# import logging
# from typing import List, Tuple, Dict, Callable
# from tqdm import tqdm
# import subprocess
# import json
# import mediapipe as mp
# from collections import deque

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # MediaPipe setup
# mp_hands = mp.solutions.hands

# class OptimizedHandProcessor:
#     """Optimized hand region extraction with extra context around hands"""
    
#     def __init__(self, target_size=(224, 224), base_margin=0.5, 
#                  confidence_threshold=0.6, smooth_window=5):
#         """
#         Args:
#             target_size: Final resize size (width, height).
#             base_margin: Fraction of box size added as context margin (0.5 = 50% extra).
#             confidence_threshold: MediaPipe detection threshold.
#             smooth_window: Number of frames for temporal smoothing.
#         """
#         self.target_size = target_size
#         self.base_margin = base_margin
#         self.confidence_threshold = confidence_threshold
        
#         # Temporal smoothing setup
#         self.smooth_window = smooth_window
#         self.history = deque(maxlen=smooth_window)
#         self.ema_box = None
#         self.ema_alpha = 0.7
        
#         # Hand tracking state
#         self.last_valid_box = None
#         self.frames_without_hand = 0
#         self.max_frames_without_hand = 10
        
#         # Initialize MediaPipe hands
#         self.hands = mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=confidence_threshold,
#             min_tracking_confidence=0.5
#         )
    
#     def reset_state(self):
#         """Reset processor state for new video"""
#         self.history.clear()
#         self.ema_box = None
#         self.last_valid_box = None
#         self.frames_without_hand = 0
    
#     def temporal_smooth(self, box):
#         """Apply temporal smoothing to reduce jitter"""
#         if box is None:
#             return self.ema_box if self.ema_box is not None else None
        
#         self.history.append(box)
        
#         # Exponential moving average
#         if self.ema_box is None:
#             self.ema_box = box
#         else:
#             self.ema_box = tuple(
#                 self.ema_alpha * b + (1 - self.ema_alpha) * e 
#                 for b, e in zip(box, self.ema_box)
#             )
        
#         # Blend with median for stability
#         if len(self.history) >= 3:
#             boxes = np.array(list(self.history))
#             median_box = np.median(boxes, axis=0)
#             return tuple(0.6 * self.ema_box[i] + 0.4 * median_box[i] 
#                         for i in range(4))
        
#         return self.ema_box
    
#     def detect_hand(self, frame):
#         """Detect hand and return bounding box"""
#         h, w, _ = frame.shape
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(rgb_frame)
        
#         if not results.multi_hand_landmarks:
#             return None
        
#         # Process all detected hands
#         hand_boxes = []
#         for hand_landmarks in results.multi_hand_landmarks:
#             x_coords = [lm.x * w for lm in hand_landmarks.landmark]
#             y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
#             min_x, max_x = min(x_coords), max(x_coords)
#             min_y, max_y = min(y_coords), max(y_coords)
            
#             # Store box with y-position for selection
#             cy = (min_y + max_y) / 2
#             hand_boxes.append({
#                 'box': (min_x, min_y, max_x, max_y),
#                 'y_pos': cy
#             })
        
#         # Select uppermost hand (typically the signing hand)
#         if hand_boxes:
#             selected = min(hand_boxes, key=lambda x: x['y_pos'])
#             return selected['box']
        
#         return None
    
#     def crop_and_resize(self, frame, box):
#         """Crop frame around hand with margin (context) and resize"""
#         h, w, _ = frame.shape
        
#         if box is None:
#             # Use last valid box or center crop
#             if self.last_valid_box and self.frames_without_hand < self.max_frames_without_hand:
#                 self.frames_without_hand += 1
#                 box = self.last_valid_box
#             else:
#                 size = min(h, w)
#                 x1 = (w - size) // 2
#                 y1 = (h - size) // 2
#                 cropped = frame[y1:y1+size, x1:x1+size]
#                 return cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LANCZOS4)
#         else:
#             self.frames_without_hand = 0
#             self.last_valid_box = box
        
#         min_x, min_y, max_x, max_y = box
        
#         # Calculate box dimensions
#         box_w = max_x - min_x
#         box_h = max_y - min_y
#         max_dim = max(box_w, box_h)
        
#         # Apply extra context margin
#         margin_size = max_dim * (1 + self.base_margin)
        
#         # Center the crop around hand
#         cx = (min_x + max_x) / 2
#         cy = (min_y + max_y) / 2
#         half_size = margin_size / 2
        
#         crop_x1 = int(max(0, cx - half_size))
#         crop_x2 = int(min(w, cx + half_size))
#         crop_y1 = int(max(0, cy - half_size))
#         crop_y2 = int(min(h, cy + half_size))
        
#         # Ensure square crop
#         crop_w = crop_x2 - crop_x1
#         crop_h = crop_y2 - crop_y1
#         if crop_w != crop_h:
#             target_size = max(crop_w, crop_h)
#             if crop_w < target_size:
#                 diff = target_size - crop_w
#                 crop_x1 = max(0, crop_x1 - diff // 2)
#                 crop_x2 = min(w, crop_x2 + diff // 2)
#             if crop_h < target_size:
#                 diff = target_size - crop_h
#                 crop_y1 = max(0, crop_y1 - diff // 2)
#                 crop_y2 = min(h, crop_y2 + diff // 2)
        
#         cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
#         if cropped.size == 0:
#             cropped = frame
        
#         return cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LANCZOS4)
    
#     def process_frame(self, frame):
#         """Process single frame"""
#         box = self.detect_hand(frame)
        
#         # Apply temporal smoothing
#         if box is not None:
#             box = self.temporal_smooth(box)
        
#         return self.crop_and_resize(frame, box)




# class NSLVideoAugmenterWithCrop:
#     """
#     Combined Nepali Sign Language Video Augmentation and Hand Cropping Pipeline
#     """

    
    
#     def __init__(self, input_dataset_path: str, output_dataset_path: str, 
#                  augmentation_probability: float = 0.7, apply_hand_crop: bool = True,
#                  crop_target_size: Tuple[int, int] = (224, 224)):
#         """
#         Initialize the augmenter with hand cropping
        
#         Args:
#             input_dataset_path: Path to the main_dataset containing class folders
#             output_dataset_path: Path to the augmented_videos output folder
#             augmentation_probability: Probability of applying each augmentation (0.0 to 1.0)
#             apply_hand_crop: Whether to apply hand detection and cropping
#             crop_target_size: Target size for cropped videos
#         """
#         self.input_path = Path(input_dataset_path)
#         self.output_path = Path(output_dataset_path)
#         self.aug_prob = augmentation_probability
#         self.apply_hand_crop = apply_hand_crop
#         self.crop_target_size = crop_target_size
        
#         # Create output directory if it doesn't exist
#         self.output_path.mkdir(parents=True, exist_ok=True)
        
#         # Initialize hand processor if needed
#         if self.apply_hand_crop:
#             self.hand_processor = OptimizedHandProcessor(target_size=crop_target_size)
#             logger.info(f"Hand cropping enabled - output size: {crop_target_size}")
        
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
        
#         # Check if ffmpeg is available for metadata handling
#         self.ffmpeg_available = self._check_ffmpeg_availability()
        
#     def _check_ffmpeg_availability(self) -> bool:
#         """Check if ffmpeg is available for metadata operations"""
#         try:
#             subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
#             logger.info("FFmpeg is available for metadata preservation")
#             return True
#         except (subprocess.CalledProcessError, FileNotFoundError):
#             logger.warning("FFmpeg not available - metadata preservation will be limited")
#             return False
        
#     def _extract_video_metadata(self, video_path: str) -> Dict:
#         """Extract comprehensive metadata from video using ffprobe"""
#         metadata = {}
        
#         if not self.ffmpeg_available:
#             return metadata
        
#         try:
#             # Use ffprobe to extract metadata
#             cmd = [
#                 'ffprobe', '-v', 'quiet', '-print_format', 'json',
#                 '-show_format', '-show_streams', str(video_path)
#             ]
            
#             result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#             probe_data = json.loads(result.stdout)
            
#             # Extract relevant metadata
#             if 'format' in probe_data:
#                 format_info = probe_data['format']
#                 metadata['duration'] = format_info.get('duration')
#                 metadata['size'] = format_info.get('size')
#                 metadata['bit_rate'] = format_info.get('bit_rate')
                
#                 # Extract format tags
#                 if 'tags' in format_info:
#                     metadata['creation_time'] = format_info['tags'].get('creation_time')
#                     metadata['encoder'] = format_info['tags'].get('encoder')
            
#             # Extract video stream metadata
#             for stream in probe_data.get('streams', []):
#                 if stream.get('codec_type') == 'video':
#                     metadata['width'] = stream.get('width')
#                     metadata['height'] = stream.get('height')
#                     metadata['fps'] = stream.get('r_frame_rate')
#                     metadata['codec_name'] = stream.get('codec_name')
#                     metadata['pix_fmt'] = stream.get('pix_fmt')
                    
#                     # Check for rotation in stream tags
#                     if 'tags' in stream:
#                         metadata['rotate'] = stream['tags'].get('rotate', '0')
                    
#                     # Check for side_data rotation
#                     if 'side_data_list' in stream:
#                         for side_data in stream['side_data_list']:
#                             if side_data.get('side_data_type') == 'Display Matrix':
#                                 metadata['display_matrix_rotation'] = side_data.get('rotation', '0')
                    
#                     break
            
#             logger.debug(f"Extracted metadata from {video_path}: {metadata}")
#             return metadata
            
#         except Exception as e:
#             logger.debug(f"Could not extract metadata from {video_path}: {e}")
#             return metadata
    
#     def _copy_metadata_to_video(self, source_video: str, target_video: str, metadata: Dict) -> bool:
#         """Copy metadata from source to target video using ffmpeg"""
#         if not self.ffmpeg_available or not metadata:
#             return False
        
#         try:
#             # Create a temporary file with metadata
#             temp_video = str(target_video) + "_temp.mp4"
            
#             # Build ffmpeg command to copy metadata
#             cmd = ['ffmpeg', '-i', str(target_video), '-c', 'copy']
            
#             # Add metadata parameters
#             if metadata.get('creation_time'):
#                 cmd.extend(['-metadata', f"creation_time={metadata['creation_time']}"])
            
#             if metadata.get('encoder'):
#                 cmd.extend(['-metadata', f"encoder=NSL_Augmented_{metadata['encoder']}"])
#             else:
#                 cmd.extend(['-metadata', 'encoder=NSL_Video_Augmenter'])
            
#             # Add rotation metadata if present
#             if metadata.get('rotate') and metadata['rotate'] != '0':
#                 cmd.extend(['-metadata:s:v:0', f"rotate={metadata['rotate']}"])
            
#             # Add custom metadata for augmented videos
#             cmd.extend(['-metadata', 'comment=Augmented NSL video with hand cropping'])
#             cmd.extend(['-metadata', 'title=NSL Augmented Dataset'])
            
#             if self.apply_hand_crop:
#                 cmd.extend(['-metadata', f'processed=augmented_cropped_{self.crop_target_size[0]}x{self.crop_target_size[1]}'])
#             else:
#                 cmd.extend(['-metadata', 'processed=augmented_only'])
            
#             cmd.extend(['-y', temp_video])  # -y to overwrite without asking
            
#             # Execute ffmpeg command
#             result = subprocess.run(cmd, capture_output=True, text=True)
            
#             if result.returncode == 0:
#                 # Replace original with metadata-enhanced version
#                 os.replace(temp_video, target_video)
#                 logger.debug(f"Successfully copied metadata to {target_video}")
#                 return True
#             else:
#                 logger.debug(f"FFmpeg metadata copy failed: {result.stderr}")
#                 # Clean up temp file if it exists
#                 if os.path.exists(temp_video):
#                     os.remove(temp_video)
#                 return False
                
#         except Exception as e:
#             logger.debug(f"Error copying metadata: {e}")
#             # Clean up temp file if it exists
#             temp_video = str(target_video) + "_temp.mp4"
#             if os.path.exists(temp_video):
#                 os.remove(temp_video)
#             return False
    
#     def _preserve_video_metadata(self, original_path: str, augmented_path: str) -> bool:
#         """Preserve metadata from original video to augmented video"""
#         try:
#             # Extract metadata from original video
#             metadata = self._extract_video_metadata(original_path)
            
#             if not metadata:
#                 logger.debug(f"No metadata to preserve for {original_path}")
#                 return False
            
#             # Copy metadata to augmented video
#             success = self._copy_metadata_to_video(original_path, augmented_path, metadata)
            
#             if success:
#                 logger.debug(f"Metadata preserved from {original_path} to {augmented_path}")
            
#             return success
            
#         except Exception as e:
#             logger.debug(f"Error preserving metadata: {e}")
#             return False
    
#     def _get_video_rotation(self, video_path: str) -> int:
#         """Get video rotation angle from metadata using ffprobe or OpenCV"""
#         try:
#             # First try ffprobe for more reliable metadata reading
#             if self.ffmpeg_available:
#                 metadata = self._extract_video_metadata(video_path)
                
#                 # Check for rotation in metadata
#                 if 'rotate' in metadata and metadata['rotate']:
#                     rotation = int(float(metadata['rotate']))
#                     return rotation % 360
                
#                 if 'display_matrix_rotation' in metadata and metadata['display_matrix_rotation']:
#                     rotation = int(float(metadata['display_matrix_rotation']))
#                     return abs(rotation) % 360
            
#             # Fallback to OpenCV
#             cap = cv2.VideoCapture(str(video_path))
#             if cap.isOpened():
#                 rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
#                 cap.release()
                
#                 # Convert to standard rotation angles
#                 if rotation == 90:
#                     return 90
#                 elif rotation == 180:
#                     return 180
#                 elif rotation == 270:
#                     return 270
            
#             return 0
            
#         except Exception as e:
#             logger.debug(f"Could not get rotation metadata from {video_path}: {e}")
#             return 0
    
#     def _rotate_frame(self, frame: np.ndarray, angle: int) -> np.ndarray:
#         """Rotate frame by specified angle (90, 180, 270 degrees)"""
#         if angle == 0:
#             return frame
#         elif angle == 90:
#             return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#         elif angle == 180:
#             return cv2.rotate(frame, cv2.ROTATE_180)
#         elif angle == 270:
#             return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
#         else:
#             return frame
    
#     def _brightness_high(self, frame: np.ndarray) -> np.ndarray:
#         """Increase brightness while preserving hand visibility"""
#         beta = random.uniform(20, 40)
#         return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
    
#     def _brightness_low(self, frame: np.ndarray) -> np.ndarray:
#         """Decrease brightness while maintaining hand contrast"""
#         beta = random.uniform(-30, -10)
#         return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
    
#     def _contrast_high(self, frame: np.ndarray) -> np.ndarray:
#         """Increase contrast moderately"""
#         alpha = random.uniform(1.1, 1.3)
#         return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    
#     def _contrast_low(self, frame: np.ndarray) -> np.ndarray:
#         """Decrease contrast slightly"""
#         alpha = random.uniform(0.7, 0.9)
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
#         kernel_size = random.randint(3, 7)
#         kernel = np.zeros((kernel_size, kernel_size))
        
#         direction = random.choice(['horizontal', 'vertical', 'diagonal'])
#         if direction == 'horizontal':
#             kernel[kernel_size // 2, :] = 1
#         elif direction == 'vertical':
#             kernel[:, kernel_size // 2] = 1
#         else:
#             np.fill_diagonal(kernel, 1)
        
#         kernel = kernel / kernel_size
#         return cv2.filter2D(frame, -1, kernel)
    
#     def _gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
#         """Apply light gaussian blur"""
#         kernel_size = random.choice([3, 5])
#         sigma = random.uniform(0.5, 1.0)
#         return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    
#     def _moderate_noise(self, frame: np.ndarray) -> np.ndarray:
#         """Add moderate gaussian noise"""
#         noise = np.random.normal(0, random.uniform(5, 15), frame.shape).astype(np.int16)
#         noisy_frame = frame.astype(np.int16) + noise
#         return np.clip(noisy_frame, 0, 255).astype(np.uint8)
    
#     def _salt_pepper_noise(self, frame: np.ndarray) -> np.ndarray:
#         """Add salt and pepper noise with low density"""
#         noise_density = random.uniform(0.001, 0.005)
#         noisy_frame = frame.copy()
        
#         salt_coords = np.random.random(frame.shape[:2]) < (noise_density / 2)
#         noisy_frame[salt_coords] = 255
        
#         pepper_coords = np.random.random(frame.shape[:2]) < (noise_density / 2)
#         noisy_frame[pepper_coords] = 0
        
#         return noisy_frame
    
#     def _gamma_high(self, frame: np.ndarray) -> np.ndarray:
#         """Apply gamma correction (lighter)"""
#         gamma = random.uniform(0.7, 0.9)
#         inv_gamma = 1.0 / gamma
#         table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
#         return cv2.LUT(frame, table)
    
#     def _gamma_low(self, frame: np.ndarray) -> np.ndarray:
#         """Apply gamma correction (darker)"""
#         gamma = random.uniform(1.1, 1.3)
#         inv_gamma = 1.0 / gamma
#         table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
#         return cv2.LUT(frame, table)
    
#     def _apply_random_augmentations(self, frame: np.ndarray, min_augmentations: int = 2, 
#                                    max_augmentations: int = 4) -> Tuple[np.ndarray, List[str]]:
#         """Apply random combination of augmentations to a frame"""
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
    
#     def augment_and_crop_video(self, video_path: Path, output_path: Path, 
#                                augmentation_suffix: str) -> bool:
#         """
#         Augment a single video file, apply hand cropping, preserve metadata, and save as MP4
        
#         Args:
#             video_path: Path to input video
#             output_path: Path to save augmented and cropped video
#             augmentation_suffix: Suffix to add to filename
            
#         Returns:
#             True if successful, False otherwise
#         """
#         try:
#             # Ensure output path has .mp4 extension
#             if output_path.suffix.lower() != '.mp4':
#                 output_path = output_path.with_suffix('.mp4')
            
#             # Extract original video metadata for preservation
#             original_metadata = self._extract_video_metadata(str(video_path))
            
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
            
#             # Detect rotation needed
#             rotation_angle = self._get_video_rotation(str(video_path))
            
#             # Adjust dimensions if rotation will change them
#             if rotation_angle in [90, 270]:
#                 width, height = height, width
#                 logger.debug(f"Video {video_path.name} rotation detected: {rotation_angle}°")
            
#             # Determine output dimensions
#             if self.apply_hand_crop:
#                 output_width, output_height = self.crop_target_size
#             else:
#                 output_width, output_height = width, height
            
#             # Ensure valid FPS (default to 30 if invalid)
#             if fps <= 0:
#                 fps = 30
#                 logger.warning(f"Invalid FPS detected, defaulting to {fps}")
            
#             # Setup video writer
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             alternative_codecs = ['XVID', 'MJPG', 'X264']
            
#             out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
            
#             # Try alternative codecs if the first one fails
#             if not out.isOpened():
#                 for codec in alternative_codecs:
#                     fourcc_alt = cv2.VideoWriter_fourcc(*codec)
#                     out = cv2.VideoWriter(str(output_path), fourcc_alt, fps, 
#                                         (output_width, output_height))
#                     if out.isOpened():
#                         logger.info(f"Using {codec} codec for {output_path.name}")
#                         break
            
#             if not out.isOpened():
#                 logger.error(f"Could not create output video with any codec: {output_path}")
#                 cap.release()
#                 return False
            
#             # Reset hand processor state for new video
#             if self.apply_hand_crop:
#                 self.hand_processor.reset_state()
            
#             frame_count = 0
#             applied_augmentations = []
            
#             # Process each frame
#             pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", leave=False)
            
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 # Step 1: Apply rotation if needed to correct orientation
#                 if rotation_angle != 0:
#                     frame = self._rotate_frame(frame, rotation_angle)
                
#                 # Step 2: Apply augmentations
#                 augmented_frame, frame_augs = self._apply_random_augmentations(frame, 2, 4)
#                 applied_augmentations.extend(frame_augs)
                
#                 # Step 3: Apply hand cropping if enabled
#                 if self.apply_hand_crop:
#                     processed_frame = self.hand_processor.process_frame(augmented_frame)
#                 else:
#                     processed_frame = augmented_frame
                
#                 # Write frame
#                 out.write(processed_frame)
#                 frame_count += 1
#                 pbar.update(1)
            
#             pbar.close()
            
#             # Clean up
#             cap.release()
#             out.release()
            
#             # Verify output file was created and has valid size
#             if output_path.exists() and output_path.stat().st_size > 0:
#                 # Preserve metadata from original video
#                 metadata_preserved = self._preserve_video_metadata(str(video_path), str(output_path))
                
#                 # Log processing summary
#                 unique_augs = list(set(applied_augmentations))
#                 logger.info(f"Processed {video_path.name} -> {output_path.name}")
#                 logger.info(f"  Applied augmentations: {', '.join(unique_augs[:5])}...")
#                 if self.apply_hand_crop:
#                     logger.info(f"  Hand cropping applied: {self.crop_target_size}")
#                 if rotation_angle != 0:
#                     logger.info(f"  Rotation corrected: {rotation_angle}°")
#                 if metadata_preserved:
#                     logger.info(f"  Metadata preserved: Yes")
#                 logger.info(f"  Processed {frame_count}/{total_frames} frames")
                
#                 return True
#             else:
#                 logger.error(f"Output video file is invalid or empty: {output_path}")
#                 if output_path.exists():
#                     output_path.unlink()
#                 return False
            
#         except Exception as e:
#             logger.error(f"Error processing video {video_path}: {str(e)}")
#             return False
    
#     def process_class_folder(self, class_folder_name: str, augmentations_per_video: int = 5) -> Dict[str, int]:
#         """
#         Process all videos in a class folder
        
#         Args:
#             class_folder_name: Name of the class folder
#             augmentations_per_video: Number of augmented versions to create per original video
            
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
        
#         logger.info(f"\nProcessing class: {class_folder_name}")
#         logger.info(f"  Found {len(original_videos)} original videos")
#         logger.info(f"  Creating {augmentations_per_video} augmentations per video")
#         if self.apply_hand_crop:
#             logger.info(f"  Hand cropping enabled: {self.crop_target_size}")
        
#         # Process each original video
#         for video_path in tqdm(original_videos, desc=f"Class {class_folder_name}"):
#             for aug_idx in range(augmentations_per_video):
#                 # Create augmented filename (always .mp4)
#                 stem = video_path.stem
#                 if self.apply_hand_crop:
#                     augmented_name = f"{stem}_aug_{aug_idx + 1:02d}_crop{self.output_format}"
#                 else:
#                     augmented_name = f"{stem}_aug_{aug_idx + 1:02d}{self.output_format}"
#                 output_path = output_class_folder / augmented_name
                
#                 # Skip if augmented version already exists
#                 if output_path.exists():
#                     logger.debug(f"Skipping existing: {augmented_name}")
#                     stats['augmented_videos'] += 1
#                     continue
                
#                 # Augment and crop the video
#                 success = self.augment_and_crop_video(
#                     video_path, output_path, f"aug_{aug_idx + 1:02d}"
#                 )
                
#                 if success:
#                     stats['augmented_videos'] += 1
#                 else:
#                     stats['failed_videos'] += 1
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
#             augmentations_per_video: Number of augmented versions per original video
            
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
        
#         logger.info("="*70)
#         logger.info("NSL VIDEO AUGMENTATION AND HAND CROPPING PIPELINE")
#         logger.info("="*70)
#         logger.info(f"Input path: {self.input_path}")
#         logger.info(f"Output path: {self.output_path}")
#         logger.info(f"Classes found: {len(class_folders)}")
#         logger.info(f"Augmentations per video: {augmentations_per_video}")
#         logger.info(f"Hand cropping: {'Enabled' if self.apply_hand_crop else 'Disabled'}")
#         if self.apply_hand_crop:
#             logger.info(f"Crop target size: {self.crop_target_size}")
#         logger.info(f"Output format: MP4")
#         logger.info("="*70)
        
#         all_stats = {}
        
#         for class_folder in sorted(class_folders):
#             class_name = class_folder.name
#             try:
#                 stats = self.process_class_folder(class_name, augmentations_per_video)
#                 all_stats[class_name] = stats
                
#                 logger.info(f"\n✓ Class '{class_name}' completed:")
#                 logger.info(f"    Original: {stats['original_videos']}")
#                 logger.info(f"    Augmented: {stats['augmented_videos']}")
#                 logger.info(f"    Failed: {stats['failed_videos']}")
                
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
#         """Print a summary of the augmentation and cropping process"""
#         print("\n" + "="*70)
#         print("PROCESSING SUMMARY")
#         print("="*70)
        
#         total_original = 0
#         total_augmented = 0
#         total_failed = 0
        
#         for class_name, class_stats in stats.items():
#             if 'error' in class_stats:
#                 print(f"{class_name:25s} ERROR: {class_stats['error']}")
#             else:
#                 orig = class_stats['original_videos']
#                 aug = class_stats['augmented_videos']
#                 fail = class_stats['failed_videos']
                
#                 total_original += orig
#                 total_augmented += aug
#                 total_failed += fail
                
#                 print(f"{class_name:25s} Orig: {orig:3d} | Aug: {aug:4d} | Fail: {fail:3d}")
        
#         print("-" * 70)
#         print(f"{'TOTAL':25s} Orig: {total_original:3d} | Aug: {total_augmented:4d} | Fail: {total_failed:3d}")
        
#         if total_original > 0:
#             print(f"\nDataset expansion: {total_original} -> {total_original + total_augmented} videos")
#             print(f"Expansion factor: {(total_original + total_augmented) / total_original:.2f}x")
        
#         if self.apply_hand_crop:
#             print(f"All videos cropped to: {self.crop_target_size[0]}x{self.crop_target_size[1]} pixels")
        
#         print(f"Success rate: {(total_augmented / max(1, total_augmented + total_failed)) * 100:.1f}%")
#         print("="*70)


# def main():
#     """Main execution function"""
#     # ======================== CONFIGURATION ========================
    
#     # Paths
#     INPUT_DATASET_PATH = "main_dataset"        # Input folder containing class folders
#     OUTPUT_DATASET_PATH = "preprocessed_and_augmented_videos_new"  # Output folder for processed videos
    
#     # Augmentation settings
#     AUGMENTATIONS_PER_VIDEO = 5      # Number of augmented versions per original video
#     AUGMENTATION_PROBABILITY = 0.8   # Probability of applying each augmentation
    
#     # Hand cropping settings
#     APPLY_HAND_CROP = True           # Enable/disable hand detection and cropping
#     CROP_TARGET_SIZE = (224, 224)    # Target size for cropped videos
    
#     # ===============================================================
    
#     print("\n" + "="*70)
#     print("NSL VIDEO PROCESSING PIPELINE")
#     print("="*70)
#     print("\nInitializing pipeline...")
    
#     # Initialize the combined augmenter and cropper
#     processor = NSLVideoAugmenterWithCrop(
#         input_dataset_path=INPUT_DATASET_PATH,
#         output_dataset_path=OUTPUT_DATASET_PATH,
#         augmentation_probability=AUGMENTATION_PROBABILITY,
#         apply_hand_crop=APPLY_HAND_CROP,
#         crop_target_size=CROP_TARGET_SIZE
#     )
    
#     # Check dependencies
#     print("\nChecking dependencies:")
#     if processor.ffmpeg_available:
#         print("  ✓ FFmpeg detected - Metadata preservation enabled")
#     else:
#         print("  ⚠ FFmpeg not detected - Limited metadata preservation")
#         print("    Install FFmpeg for complete metadata support:")
#         print("    - Windows: Download from https://ffmpeg.org/download.html")
#         print("    - macOS: brew install ffmpeg")
#         print("    - Ubuntu/Debian: sudo apt install ffmpeg")
    
#     if APPLY_HAND_CROP:
#         print("  ✓ MediaPipe hand detection enabled")
#         print(f"    Output size: {CROP_TARGET_SIZE[0]}x{CROP_TARGET_SIZE[1]} pixels")
#     else:
#         print("  ⚠ Hand cropping disabled - videos will be augmented only")
    
#     try:
#         # Process the entire dataset
#         print("\n" + "-"*70)
#         print("Starting processing...")
#         print("-"*70)
        
#         stats = processor.process_entire_dataset(
#             augmentations_per_video=AUGMENTATIONS_PER_VIDEO
#         )
        
#         # Print summary
#         processor.print_dataset_summary(stats)
        
#         print(f"\n✅ Processing complete!")
#         print(f"   Output location: {OUTPUT_DATASET_PATH}/")
#         print(f"   Each class folder contains augmented {'and cropped ' if APPLY_HAND_CROP else ''}videos")
#         print(f"   All videos are in MP4 format")
        
#     except KeyboardInterrupt:
#         print("\n\n⚠ Processing interrupted by user")
#         print("Partial results may have been saved")
        
#     except Exception as e:
#         print(f"\n❌ Failed to process dataset: {str(e)}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()



























    





    


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
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands

class OptimizedHandProcessor:
    """Optimized hand region extraction with robust hand selection"""
    
    def __init__(self, target_size=(224, 224), base_margin=0.5, 
                 confidence_threshold=0.3, smooth_window=5):
        """
        Args:
            target_size: Final resize size (width, height).
            base_margin: Fraction of box size added as context margin (0.5 = 50% extra).
            confidence_threshold: MediaPipe detection threshold.
            smooth_window: Number of frames for temporal smoothing.
        """
        self.target_size = target_size
        self.base_margin = base_margin
        self.confidence_threshold = confidence_threshold
        
        # Temporal smoothing setup
        self.smooth_window = smooth_window
        self.history = deque(maxlen=smooth_window)
        self.ema_box = None
        self.ema_alpha = 0.7
        
        # Hand tracking state
        self.last_valid_box = None
        self.frames_without_hand = 0
        self.max_frames_without_hand = 10
        
        # Initialize MediaPipe hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=0.5
        )
    
    def reset_state(self):
        """Reset processor state for new video"""
        self.history.clear()
        self.ema_box = None
        self.last_valid_box = None
        self.frames_without_hand = 0
    
    def temporal_smooth(self, box):
        """Apply temporal smoothing to reduce jitter"""
        if box is None:
            return self.ema_box if self.ema_box is not None else None
        
        self.history.append(box)
        
        # Exponential moving average
        if self.ema_box is None:
            self.ema_box = box
        else:
            self.ema_box = tuple(
                self.ema_alpha * b + (1 - self.ema_alpha) * e 
                for b, e in zip(box, self.ema_box)
            )
        
        # Blend with median for stability
        if len(self.history) >= 3:
            boxes = np.array(list(self.history))
            median_box = np.median(boxes, axis=0)
            return tuple(0.6 * self.ema_box[i] + 0.4 * median_box[i] 
                        for i in range(4))
        
        return self.ema_box
    
    def stable_select(self, new_box):
        """Stickiness: prevent sudden jumps between hands"""
        if self.last_valid_box is None:
            return new_box
        
        # Compute distance between centers
        def center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)
        cx_prev, cy_prev = center(self.last_valid_box)
        cx_new, cy_new = center(new_box)
        dist = np.hypot(cx_prev - cx_new, cy_prev - cy_new)
        
        # If distance too large (sudden switch), keep previous
        if dist > 0.25 * self.target_size[0]:
            return self.last_valid_box
        return new_box
    
    def detect_hand(self, frame):
        """Detect hand and return bounding box"""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
        
        hand_boxes = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            hand_boxes.append({
                'box': (min_x, min_y, max_x, max_y),
                'top': min_y   # use top of the hand
            })
        
        # If both hands are visible, always pick the uppermost one (smallest top)
        if hand_boxes:
            selected = min(hand_boxes, key=lambda x: x['top'])
            return selected['box']
        
        return None
    
    def crop_and_resize(self, frame, box):
        """Crop frame around hand with margin (context) and resize"""
        h, w, _ = frame.shape
        
        if box is None:
            # Use last valid box or center crop
            if self.last_valid_box and self.frames_without_hand < self.max_frames_without_hand:
                self.frames_without_hand += 1
                box = self.last_valid_box
            else:
                size = min(h, w)
                x1 = (w - size) // 2
                y1 = (h - size) // 2
                cropped = frame[y1:y1+size, x1:x1+size]
                return cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        else:
            self.frames_without_hand = 0
            self.last_valid_box = box
        
        min_x, min_y, max_x, max_y = box
        
        # Calculate box dimensions
        box_w = max_x - min_x
        box_h = max_y - min_y
        max_dim = max(box_w, box_h)
        
        # Apply extra context margin
        margin_size = max_dim * (1 + self.base_margin)
        
        # Center the crop around hand
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        half_size = margin_size / 2
        
        crop_x1 = int(max(0, cx - half_size))
        crop_x2 = int(min(w, cx + half_size))
        crop_y1 = int(max(0, cy - half_size))
        crop_y2 = int(min(h, cy + half_size))
        
        # Ensure square crop
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        if crop_w != crop_h:
            target_size = max(crop_w, crop_h)
            if crop_w < target_size:
                diff = target_size - crop_w
                crop_x1 = max(0, crop_x1 - diff // 2)
                crop_x2 = min(w, crop_x2 + diff // 2)
            if crop_h < target_size:
                diff = target_size - crop_h
                crop_y1 = max(0, crop_y1 - diff // 2)
                crop_y2 = min(h, crop_y2 + diff // 2)
        
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped.size == 0:
            cropped = frame
        
        return cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LANCZOS4)
    
    def process_frame(self, frame):
        """Process single frame"""
        box = self.detect_hand(frame)
        
        # Apply temporal smoothing
        if box is not None:
            box = self.temporal_smooth(box)
            box = self.stable_select(box)
        
        return self.crop_and_resize(frame, box)



class NSLVideoAugmenterWithCrop:
    """
    Combined Nepali Sign Language Video Augmentation and Hand Cropping Pipeline
    """

    def __init__(self, input_dataset_path: str, output_dataset_path: str, 
                 augmentation_probability: float = 0.7, apply_hand_crop: bool = True,
                 crop_target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the augmenter with hand cropping
        
        Args:
            input_dataset_path: Path to the main_dataset containing class folders
            output_dataset_path: Path to the augmented_videos output folder
            augmentation_probability: Probability of applying each augmentation (0.0 to 1.0)
            apply_hand_crop: Whether to apply hand detection and cropping
            crop_target_size: Target size for cropped videos
        """
        self.input_path = Path(input_dataset_path)
        self.output_path = Path(output_dataset_path)
        self.aug_prob = augmentation_probability
        self.apply_hand_crop = apply_hand_crop
        self.crop_target_size = crop_target_size
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize hand processor if needed
        if self.apply_hand_crop:
            self.hand_processor = OptimizedHandProcessor(target_size=crop_target_size)
            logger.info(f"Hand cropping enabled - output size: {crop_target_size}")
        
        # Define augmentation functions with conservative parameters
        self.augmentations = {
            'bright_high': self._brightness_high,
            'bright_low': self._brightness_low,
            'contrast_high': self._contrast_high,
            'contrast_low': self._contrast_low,
            'saturation_high': self._saturation_high,
            'saturation_low': self._saturation_low,
            'blur_motion': self._motion_blur,
            'blur_gaussian': self._gaussian_blur,
            'noise_moderate': self._moderate_noise,
            'noise_salt_pepper': self._salt_pepper_noise,
            'gamma_high': self._gamma_high,
            'gamma_low': self._gamma_low,
            # New augmentations
            'hue_shift': self._hue_shift,
            'sharpen': self._sharpen,
            'edge_enhance': self._edge_enhance,
            'bilateral_filter': self._bilateral_filter,
            'clahe': self._clahe,
            'color_temp_warm': self._color_temperature_warm,
            'color_temp_cool': self._color_temperature_cool,
            'vignette': self._vignette,
            'chromatic_aberration': self._chromatic_aberration,
            'lens_distortion': self._lens_distortion,
            'film_grain': self._film_grain,
            'compression_artifacts': self._compression_artifacts,
            'shadow_highlight': self._shadow_highlight,
            'channel_shift': self._channel_shift,
            'exposure_comp': self._exposure_compensation,
            'color_jitter': self._color_jitter,
            'defocus_blur': self._defocus_blur,
            'emboss': self._emboss_subtle,
            'posterize': self._posterize,
            'solarize': self._solarize
        }
        
        # Supported video formats (input and output)
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        # Force output to MP4 format for consistency
        self.output_format = '.mp4'
        
        # Check if ffmpeg is available for metadata handling
        self.ffmpeg_available = self._check_ffmpeg_availability()
        
    def _check_ffmpeg_availability(self) -> bool:
        """Check if ffmpeg is available for metadata operations"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            logger.info("FFmpeg is available for metadata preservation")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("FFmpeg not available - metadata preservation will be limited")
            return False
        
    def _extract_video_metadata(self, video_path: str) -> Dict:
        """Extract comprehensive metadata from video using ffprobe"""
        metadata = {}
        
        if not self.ffmpeg_available:
            return metadata
        
        try:
            # Use ffprobe to extract metadata
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
            
            # Extract relevant metadata
            if 'format' in probe_data:
                format_info = probe_data['format']
                metadata['duration'] = format_info.get('duration')
                metadata['size'] = format_info.get('size')
                metadata['bit_rate'] = format_info.get('bit_rate')
                
                # Extract format tags
                if 'tags' in format_info:
                    metadata['creation_time'] = format_info['tags'].get('creation_time')
                    metadata['encoder'] = format_info['tags'].get('encoder')
            
            # Extract video stream metadata
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    metadata['width'] = stream.get('width')
                    metadata['height'] = stream.get('height')
                    metadata['fps'] = stream.get('r_frame_rate')
                    metadata['codec_name'] = stream.get('codec_name')
                    metadata['pix_fmt'] = stream.get('pix_fmt')
                    
                    # Check for rotation in stream tags
                    if 'tags' in stream:
                        metadata['rotate'] = stream['tags'].get('rotate', '0')
                    
                    # Check for side_data rotation
                    if 'side_data_list' in stream:
                        for side_data in stream['side_data_list']:
                            if side_data.get('side_data_type') == 'Display Matrix':
                                metadata['display_matrix_rotation'] = side_data.get('rotation', '0')
                    
                    break
            
            logger.debug(f"Extracted metadata from {video_path}: {metadata}")
            return metadata
            
        except Exception as e:
            logger.debug(f"Could not extract metadata from {video_path}: {e}")
            return metadata
    
    def _copy_metadata_to_video(self, source_video: str, target_video: str, metadata: Dict) -> bool:
        """Copy metadata from source to target video using ffmpeg"""
        if not self.ffmpeg_available or not metadata:
            return False
        
        try:
            # Create a temporary file with metadata
            temp_video = str(target_video) + "_temp.mp4"
            
            # Build ffmpeg command to copy metadata
            cmd = ['ffmpeg', '-i', str(target_video), '-c', 'copy']
            
            # Add metadata parameters
            if metadata.get('creation_time'):
                cmd.extend(['-metadata', f"creation_time={metadata['creation_time']}"])
            
            if metadata.get('encoder'):
                cmd.extend(['-metadata', f"encoder=NSL_Augmented_{metadata['encoder']}"])
            else:
                cmd.extend(['-metadata', 'encoder=NSL_Video_Augmenter'])
            
            # Add rotation metadata if present
            if metadata.get('rotate') and metadata['rotate'] != '0':
                cmd.extend(['-metadata:s:v:0', f"rotate={metadata['rotate']}"])
            
            # Add custom metadata for augmented videos
            cmd.extend(['-metadata', 'comment=Augmented NSL video with hand cropping'])
            cmd.extend(['-metadata', 'title=NSL Augmented Dataset'])
            
            if self.apply_hand_crop:
                cmd.extend(['-metadata', f'processed=augmented_cropped_{self.crop_target_size[0]}x{self.crop_target_size[1]}'])
            else:
                cmd.extend(['-metadata', 'processed=augmented_only'])
            
            cmd.extend(['-y', temp_video])  # -y to overwrite without asking
            
            # Execute ffmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original with metadata-enhanced version
                os.replace(temp_video, target_video)
                logger.debug(f"Successfully copied metadata to {target_video}")
                return True
            else:
                logger.debug(f"FFmpeg metadata copy failed: {result.stderr}")
                # Clean up temp file if it exists
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                return False
                
        except Exception as e:
            logger.debug(f"Error copying metadata: {e}")
            # Clean up temp file if it exists
            temp_video = str(target_video) + "_temp.mp4"
            if os.path.exists(temp_video):
                os.remove(temp_video)
            return False
    
    def _preserve_video_metadata(self, original_path: str, augmented_path: str) -> bool:
        """Preserve metadata from original video to augmented video"""
        try:
            # Extract metadata from original video
            metadata = self._extract_video_metadata(original_path)
            
            if not metadata:
                logger.debug(f"No metadata to preserve for {original_path}")
                return False
            
            # Copy metadata to augmented video
            success = self._copy_metadata_to_video(original_path, augmented_path, metadata)
            
            if success:
                logger.debug(f"Metadata preserved from {original_path} to {augmented_path}")
            
            return success
            
        except Exception as e:
            logger.debug(f"Error preserving metadata: {e}")
            return False
    
    def _get_video_rotation(self, video_path: str) -> int:
        """Get video rotation angle from metadata using ffprobe or OpenCV"""
        try:
            # First try ffprobe for more reliable metadata reading
            if self.ffmpeg_available:
                metadata = self._extract_video_metadata(video_path)
                
                # Check for rotation in metadata
                if 'rotate' in metadata and metadata['rotate']:
                    rotation = int(float(metadata['rotate']))
                    return rotation % 360
                
                if 'display_matrix_rotation' in metadata and metadata['display_matrix_rotation']:
                    rotation = int(float(metadata['display_matrix_rotation']))
                    return abs(rotation) % 360
            
            # Fallback to OpenCV
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
                cap.release()
                
                # Convert to standard rotation angles
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
    
    def _rotate_frame(self, frame: np.ndarray, angle: int) -> np.ndarray:
        """Rotate frame by specified angle (90, 180, 270 degrees)"""
        if angle == 0:
            return frame
        elif angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return frame
    
    def _brightness_high(self, frame: np.ndarray) -> np.ndarray:
        """Increase brightness while preserving hand visibility"""
        beta = random.uniform(20, 40)
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
    
    def _brightness_low(self, frame: np.ndarray) -> np.ndarray:
        """Decrease brightness while maintaining hand contrast"""
        beta = random.uniform(-30, -10)
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
    
    def _contrast_high(self, frame: np.ndarray) -> np.ndarray:
        """Increase contrast moderately"""
        alpha = random.uniform(1.1, 1.3)
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    
    def _contrast_low(self, frame: np.ndarray) -> np.ndarray:
        """Decrease contrast slightly"""
        alpha = random.uniform(0.7, 0.9)
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    
    def _saturation_high(self, frame: np.ndarray) -> np.ndarray:
        """Increase saturation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_scale = random.uniform(1.1, 1.3)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _saturation_low(self, frame: np.ndarray) -> np.ndarray:
        """Decrease saturation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_scale = random.uniform(0.7, 0.9)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _motion_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply slight motion blur"""
        kernel_size = random.randint(3, 7)
        kernel = np.zeros((kernel_size, kernel_size))
        
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        if direction == 'horizontal':
            kernel[kernel_size // 2, :] = 1
        elif direction == 'vertical':
            kernel[:, kernel_size // 2] = 1
        else:
            np.fill_diagonal(kernel, 1)
        
        kernel = kernel / kernel_size
        return cv2.filter2D(frame, -1, kernel)
    
    def _gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply light gaussian blur"""
        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.0)
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    
    def _moderate_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add moderate gaussian noise"""
        noise = np.random.normal(0, random.uniform(5, 15), frame.shape).astype(np.int16)
        noisy_frame = frame.astype(np.int16) + noise
        return np.clip(noisy_frame, 0, 255).astype(np.uint8)
    
    def _salt_pepper_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add salt and pepper noise with low density"""
        noise_density = random.uniform(0.001, 0.005)
        noisy_frame = frame.copy()
        
        salt_coords = np.random.random(frame.shape[:2]) < (noise_density / 2)
        noisy_frame[salt_coords] = 255
        
        pepper_coords = np.random.random(frame.shape[:2]) < (noise_density / 2)
        noisy_frame[pepper_coords] = 0
        
        return noisy_frame
    
    def _gamma_high(self, frame: np.ndarray) -> np.ndarray:
        """Apply gamma correction (lighter)"""
        gamma = random.uniform(0.7, 0.9)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(frame, table)
    
    def _gamma_low(self, frame: np.ndarray) -> np.ndarray:
        """Apply gamma correction (darker)"""
        gamma = random.uniform(1.1, 1.3)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(frame, table)
    
    # Extended augmentation methods
    def _hue_shift(self, frame: np.ndarray) -> np.ndarray:
        """Shift hue values"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_shift = random.uniform(-20, 20)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _sharpen(self, frame: np.ndarray) -> np.ndarray:
        """Apply sharpening filter"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        alpha = random.uniform(0.3, 0.7)
        return cv2.addWeighted(frame, 1 - alpha, sharpened, alpha, 0)
    
    def _edge_enhance(self, frame: np.ndarray) -> np.ndarray:
        """Enhance edges"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        alpha = random.uniform(0.1, 0.3)
        return cv2.addWeighted(frame, 1 - alpha, edges_colored, alpha, 0)
    
    def _bilateral_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for noise reduction while preserving edges"""
        d = random.choice([5, 9])
        sigma_color = random.uniform(40, 80)
        sigma_space = random.uniform(40, 80)
        return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
    
    def _clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clip_limit = random.uniform(2.0, 4.0)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _color_temperature_warm(self, frame: np.ndarray) -> np.ndarray:
        """Apply warm color temperature"""
        temp_matrix = np.array([[1.0, 0.0, 0.0],
                               [0.0, 0.95, 0.0],
                               [0.0, 0.0, 0.8]])
        warm_frame = frame.astype(np.float32) / 255.0
        warm_frame = np.dot(warm_frame, temp_matrix.T)
        return np.clip(warm_frame * 255, 0, 255).astype(np.uint8)
    
    def _color_temperature_cool(self, frame: np.ndarray) -> np.ndarray:
        """Apply cool color temperature"""
        temp_matrix = np.array([[0.8, 0.0, 0.0],
                               [0.0, 0.95, 0.0],
                               [0.0, 0.0, 1.2]])
        cool_frame = frame.astype(np.float32) / 255.0
        cool_frame = np.dot(cool_frame, temp_matrix.T)
        return np.clip(cool_frame * 255, 0, 255).astype(np.uint8)
    
    def _vignette(self, frame: np.ndarray) -> np.ndarray:
        """Add vignette effect"""
        h, w = frame.shape[:2]
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        X, Y = np.meshgrid(x, y)
        radius = np.sqrt(X**2 + Y**2)
        max_radius = np.sqrt((w//2)**2 + (h//2)**2)
        
        vignette_strength = random.uniform(0.3, 0.7)
        vignette = 1 - vignette_strength * (radius / max_radius)**2
        vignette = np.clip(vignette, 0, 1)
        
        return (frame * vignette[..., np.newaxis]).astype(np.uint8)
    
    def _chromatic_aberration(self, frame: np.ndarray) -> np.ndarray:
        """Add chromatic aberration effect"""
        h, w = frame.shape[:2]
        shift = random.randint(1, 3)
        
        b, g, r = cv2.split(frame)
        
        # Shift red channel
        r_shifted = np.roll(r, shift, axis=1)
        r_shifted = np.roll(r_shifted, shift//2, axis=0)
        
        # Shift blue channel opposite direction
        b_shifted = np.roll(b, -shift, axis=1)
        b_shifted = np.roll(b_shifted, -shift//2, axis=0)
        
        return cv2.merge([b_shifted, g, r_shifted])
    
    def _lens_distortion(self, frame: np.ndarray) -> np.ndarray:
        """Add barrel distortion"""
        h, w = frame.shape[:2]
        k = random.uniform(-0.0002, 0.0002)  # Distortion coefficient
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        # Normalize coordinates to [-1, 1]
        x_norm = (x - w/2) / (w/2)
        y_norm = (y - h/2) / (h/2)
        
        # Apply distortion
        r2 = x_norm**2 + y_norm**2
        x_dist = x_norm * (1 + k * r2)
        y_dist = y_norm * (1 + k * r2)
        
        # Convert back to image coordinates
        x_new = (x_dist * w/2 + w/2).astype(np.float32)
        y_new = (y_dist * h/2 + h/2).astype(np.float32)
        
        return cv2.remap(frame, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    def _film_grain(self, frame: np.ndarray) -> np.ndarray:
        """Add film grain effect"""
        h, w = frame.shape[:2]
        grain_strength = random.uniform(10, 25)
        
        # Generate grain pattern
        grain = np.random.normal(0, grain_strength, (h, w)).astype(np.int16)
        grain_colored = np.stack([grain, grain, grain], axis=2)
        
        # Apply grain
        grainy_frame = frame.astype(np.int16) + grain_colored
        return np.clip(grainy_frame, 0, 255).astype(np.uint8)
    
    def _compression_artifacts(self, frame: np.ndarray) -> np.ndarray:
        """Simulate compression artifacts"""
        # Encode and decode with JPEG compression
        quality = random.randint(60, 85)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        
        # Create temporary encoded data
        _, encimg = cv2.imencode('.jpg', frame, encode_param)
        return cv2.imdecode(encimg, 1)
    
    def _shadow_highlight(self, frame: np.ndarray) -> np.ndarray:
        """Adjust shadows and highlights"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Shadow adjustment
        shadow_adj = random.uniform(0.8, 1.2)
        shadow_mask = (l_channel < 85).astype(np.float32)
        l_channel = l_channel * (1 + (shadow_adj - 1) * shadow_mask)
        
        # Highlight adjustment
        highlight_adj = random.uniform(0.8, 1.2)
        highlight_mask = (l_channel > 170).astype(np.float32)
        l_channel = l_channel * (1 + (highlight_adj - 1) * highlight_mask)
        
        lab[:, :, 0] = np.clip(l_channel, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _channel_shift(self, frame: np.ndarray) -> np.ndarray:
        """Shift color channels"""
        b, g, r = cv2.split(frame)
        
        # Random channel shifts
        shifts = [random.randint(-2, 2) for _ in range(3)]
        
        if shifts[0] != 0:
            b = np.roll(b, shifts[0], axis=random.choice([0, 1]))
        if shifts[1] != 0:
            g = np.roll(g, shifts[1], axis=random.choice([0, 1]))
        if shifts[2] != 0:
            r = np.roll(r, shifts[2], axis=random.choice([0, 1]))
        
        return cv2.merge([b, g, r])
    
    def _exposure_compensation(self, frame: np.ndarray) -> np.ndarray:
        """Adjust exposure"""
        exposure_value = random.uniform(-1.0, 1.0)  # EV adjustment
        multiplier = 2 ** exposure_value
        
        adjusted = frame.astype(np.float32) * multiplier
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def _color_jitter(self, frame: np.ndarray) -> np.ndarray:
        """Apply color jittering"""
        # Convert to float for calculations
        img_float = frame.astype(np.float32) / 255.0
        
        # Random adjustments
        brightness_adj = random.uniform(0.8, 1.2)
        contrast_adj = random.uniform(0.8, 1.2)
        saturation_adj = random.uniform(0.8, 1.2)
        
        # Apply brightness
        img_float = img_float * brightness_adj
        
        # Apply contrast
        mean_val = np.mean(img_float)
        img_float = (img_float - mean_val) * contrast_adj + mean_val
        
        # Apply saturation (in HSV space)
        hsv = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_adj
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        
        return np.clip(img_float * 255, 0, 255).astype(np.uint8)
    
    def _defocus_blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply defocus blur"""
        kernel_size = random.choice([5, 7, 9])
        sigma = random.uniform(1.0, 3.0)
        
        # Create circular kernel for defocus effect
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        y, x = np.ogrid[:kernel_size, :kernel_size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= (kernel_size // 2) ** 2
        kernel[mask] = 1
        kernel = kernel / np.sum(kernel)
        
        return cv2.filter2D(frame, -1, kernel)
    
    def _emboss_subtle(self, frame: np.ndarray) -> np.ndarray:
        """Apply subtle emboss effect"""
        kernel = np.array([[-1, -1, 0],
                          [-1, 0, 1],
                          [0, 1, 1]])
        embossed = cv2.filter2D(frame, -1, kernel)
        embossed = cv2.add(embossed, 128)  # Add gray offset
        
        # Blend with original
        alpha = random.uniform(0.1, 0.3)
        return cv2.addWeighted(frame, 1 - alpha, embossed, alpha, 0)
    
    def _posterize(self, frame: np.ndarray) -> np.ndarray:
        """Apply posterization effect"""
        levels = random.randint(4, 8)
        factor = 256 // levels
        
        posterized = (frame // factor) * factor
        return np.clip(posterized, 0, 255).astype(np.uint8)
    
    def _solarize(self, frame: np.ndarray) -> np.ndarray:
        """Apply solarization effect"""
        threshold = random.randint(128, 200)
        solarized = frame.copy()
        solarized[frame >= threshold] = 255 - solarized[frame >= threshold]
        return solarized
    
    def _apply_random_augmentations(self, frame: np.ndarray, min_augmentations: int = 2, 
                                   max_augmentations: int = 4) -> Tuple[np.ndarray, List[str]]:
        """Apply random combination of augmentations to a frame"""
        num_augmentations = random.randint(min_augmentations, max_augmentations)
        
        available_augs = list(self.augmentations.keys())
        selected_augs = random.sample(available_augs, min(num_augmentations, len(available_augs)))
        
        augmented_frame = frame.copy()
        applied_augs = []
        
        for aug_name in selected_augs:
            if random.random() < self.aug_prob:
                augmented_frame = self.augmentations[aug_name](augmented_frame)
                applied_augs.append(aug_name)
        
        return augmented_frame, applied_augs
    
    def augment_and_crop_video(self, video_path: Path, output_path: Path, 
                               augmentation_suffix: str) -> bool:
        """
        Augment a single video file, apply hand cropping, preserve metadata, and save as MP4
        
        Args:
            video_path: Path to input video
            output_path: Path to save augmented and cropped video
            augmentation_suffix: Suffix to add to filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output path has .mp4 extension
            if output_path.suffix.lower() != '.mp4':
                output_path = output_path.with_suffix('.mp4')
            
            # Extract original video metadata for preservation
            original_metadata = self._extract_video_metadata(str(video_path))
            
            # Open input video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Detect rotation needed
            rotation_angle = self._get_video_rotation(str(video_path))
            
            # Adjust dimensions if rotation will change them
            if rotation_angle in [90, 270]:
                width, height = height, width
                logger.debug(f"Video {video_path.name} rotation detected: {rotation_angle}°")
            
            # Determine output dimensions
            if self.apply_hand_crop:
                output_width, output_height = self.crop_target_size
            else:
                output_width, output_height = width, height
            
            # Ensure valid FPS (default to 30 if invalid)
            if fps <= 0:
                fps = 30
                logger.warning(f"Invalid FPS detected, defaulting to {fps}")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            alternative_codecs = ['XVID', 'MJPG', 'X264']
            
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
            
            # Try alternative codecs if the first one fails
            if not out.isOpened():
                for codec in alternative_codecs:
                    fourcc_alt = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(str(output_path), fourcc_alt, fps, 
                                        (output_width, output_height))
                    if out.isOpened():
                        logger.info(f"Using {codec} codec for {output_path.name}")
                        break
            
            if not out.isOpened():
                logger.error(f"Could not create output video with any codec: {output_path}")
                cap.release()
                return False
            
            # Reset hand processor state for new video
            if self.apply_hand_crop:
                self.hand_processor.reset_state()
            
            frame_count = 0
            applied_augmentations = []
            
            # Process each frame
            pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", leave=False)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Step 1: Apply rotation if needed to correct orientation
                if rotation_angle != 0:
                    frame = self._rotate_frame(frame, rotation_angle)
                
                # Step 2: Apply augmentations
                augmented_frame, frame_augs = self._apply_random_augmentations(frame, 2, 4)
                applied_augmentations.extend(frame_augs)
                
                # Step 3: Apply hand cropping if enabled
                if self.apply_hand_crop:
                    processed_frame = self.hand_processor.process_frame(augmented_frame)
                else:
                    processed_frame = augmented_frame
                
                # Write frame
                out.write(processed_frame)
                frame_count += 1
                pbar.update(1)
            
            pbar.close()
            
            # Clean up
            cap.release()
            out.release()
            
            # Verify output file was created and has valid size
            if output_path.exists() and output_path.stat().st_size > 0:
                # Preserve metadata from original video
                metadata_preserved = self._preserve_video_metadata(str(video_path), str(output_path))
                
                # Log processing summary
                unique_augs = list(set(applied_augmentations))
                logger.info(f"Processed {video_path.name} -> {output_path.name}")
                logger.info(f"  Applied augmentations: {', '.join(unique_augs[:5])}...")
                if self.apply_hand_crop:
                    logger.info(f"  Hand cropping applied: {self.crop_target_size}")
                if rotation_angle != 0:
                    logger.info(f"  Rotation corrected: {rotation_angle}°")
                if metadata_preserved:
                    logger.info(f"  Metadata preserved: Yes")
                logger.info(f"  Processed {frame_count}/{total_frames} frames")
                
                return True
            else:
                logger.error(f"Output video file is invalid or empty: {output_path}")
                if output_path.exists():
                    output_path.unlink()
                return False
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return False
    
    def process_class_folder(self, class_folder_name: str, augmentations_per_video: int = 18) -> Dict[str, int]:
        """
        Process all videos in a class folder
        
        Args:
            class_folder_name: Name of the class folder
            augmentations_per_video: Number of augmented versions to create per original video
            
        Returns:
            Dictionary with processing statistics
        """
        # Ensure minimum 5 augmentations
        if augmentations_per_video < 5:
            augmentations_per_video = 5
            logger.info(f"Minimum 5 augmentations required. Set to {augmentations_per_video}")
        
        input_class_folder = self.input_path / class_folder_name
        output_class_folder = self.output_path / class_folder_name
        
        # Create output class folder
        output_class_folder.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'original_videos': 0,
            'augmented_videos': 0,
            'failed_videos': 0
        }
        
        if not input_class_folder.exists():
            logger.error(f"Input class folder does not exist: {input_class_folder}")
            return stats
        
        # Find all video files in the input folder
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(input_class_folder.glob(f'*{ext}'))
            video_files.extend(input_class_folder.glob(f'*{ext.upper()}'))
        
        # Filter original videos (no augmentation suffixes)
        original_videos = []
        for video_file in video_files:
            stem = video_file.stem
            # Check if it's an original video (no augmentation suffix)
            is_augmented = any(aug_name in stem for aug_name in self.augmentations.keys())
            is_augmented = is_augmented or '_aug_' in stem or '_augmented' in stem
            
            if not is_augmented:
                original_videos.append(video_file)
        
        stats['original_videos'] = len(original_videos)
        
        logger.info(f"\nProcessing class: {class_folder_name}")
        logger.info(f"  Found {len(original_videos)} original videos")
        logger.info(f"  Creating {augmentations_per_video} augmentations per video")
        if self.apply_hand_crop:
            logger.info(f"  Hand cropping enabled: {self.crop_target_size}")
        
        # Process each original video
        for video_path in tqdm(original_videos, desc=f"Class {class_folder_name}"):
            for aug_idx in range(augmentations_per_video):
                # Create augmented filename (always .mp4)
                stem = video_path.stem
                if self.apply_hand_crop:
                    augmented_name = f"{stem}_aug_{aug_idx + 1:02d}_crop{self.output_format}"
                else:
                    augmented_name = f"{stem}_aug_{aug_idx + 1:02d}{self.output_format}"
                output_path = output_class_folder / augmented_name
                
                # Skip if augmented version already exists
                if output_path.exists():
                    logger.debug(f"Skipping existing: {augmented_name}")
                    stats['augmented_videos'] += 1
                    continue
                
                # Augment and crop the video
                success = self.augment_and_crop_video(
                    video_path, output_path, f"aug_{aug_idx + 1:02d}"
                )
                
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
    
    def process_entire_dataset(self, augmentations_per_video: int = 5) -> Dict[str, Dict[str, int]]:
        """
        Process the entire dataset
        
        Args:
            augmentations_per_video: Number of augmented versions per original video
            
        Returns:
            Dictionary with statistics for each class
        """
        # Ensure minimum 5 augmentations
        if augmentations_per_video < 5:
            augmentations_per_video = 5
            logger.info(f"Minimum 5 augmentations required. Set to {augmentations_per_video}")
        
        if not self.input_path.exists():
            raise ValueError(f"Input dataset path does not exist: {self.input_path}")
        
        # Find all class folders in input directory
        class_folders = [f for f in self.input_path.iterdir() if f.is_dir()]
        
        if not class_folders:
            raise ValueError(f"No class folders found in: {self.input_path}")
        
        logger.info("="*70)
        logger.info("NSL VIDEO AUGMENTATION AND HAND CROPPING PIPELINE")
        logger.info("="*70)
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Classes found: {len(class_folders)}")
        logger.info(f"Augmentations per video: {augmentations_per_video}")
        logger.info(f"Hand cropping: {'Enabled' if self.apply_hand_crop else 'Disabled'}")
        if self.apply_hand_crop:
            logger.info(f"Crop target size: {self.crop_target_size}")
        logger.info(f"Output format: MP4")
        logger.info("="*70)
        
        all_stats = {}
        
        for class_folder in sorted(class_folders):
            class_name = class_folder.name
            try:
                stats = self.process_class_folder(class_name, augmentations_per_video)
                all_stats[class_name] = stats
                
                logger.info(f"\n✓ Class '{class_name}' completed:")
                logger.info(f"    Original: {stats['original_videos']}")
                logger.info(f"    Augmented: {stats['augmented_videos']}")
                logger.info(f"    Failed: {stats['failed_videos']}")
                
            except Exception as e:
                logger.error(f"Error processing class folder {class_name}: {str(e)}")
                all_stats[class_name] = {
                    'original_videos': 0,
                    'augmented_videos': 0,
                    'failed_videos': 0,
                    'error': str(e)
                }
        
        return all_stats
    
    def print_dataset_summary(self, stats: Dict[str, Dict[str, int]]):
        """Print a summary of the augmentation and cropping process"""
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        
        total_original = 0
        total_augmented = 0
        total_failed = 0
        
        for class_name, class_stats in stats.items():
            if 'error' in class_stats:
                print(f"{class_name:25s} ERROR: {class_stats['error']}")
            else:
                orig = class_stats['original_videos']
                aug = class_stats['augmented_videos']
                fail = class_stats['failed_videos']
                
                total_original += orig
                total_augmented += aug
                total_failed += fail
                
                print(f"{class_name:25s} Orig: {orig:3d} | Aug: {aug:4d} | Fail: {fail:3d}")
        
        print("-" * 70)
        print(f"{'TOTAL':25s} Orig: {total_original:3d} | Aug: {total_augmented:4d} | Fail: {total_failed:3d}")
        
        if total_original > 0:
            print(f"\nDataset expansion: {total_original} -> {total_original + total_augmented} videos")
            print(f"Expansion factor: {(total_original + total_augmented) / total_original:.2f}x")
        
        if self.apply_hand_crop:
            print(f"All videos cropped to: {self.crop_target_size[0]}x{self.crop_target_size[1]} pixels")
        
        print(f"Success rate: {(total_augmented / max(1, total_augmented + total_failed)) * 100:.1f}%")
        print("="*70)

def main():
    """Main execution function"""
    # ======================== CONFIGURATION ========================
    
    # Paths
    INPUT_DATASET_PATH = "main_dataset"        # Input folder containing class folders
    OUTPUT_DATASET_PATH = "preprocessed_and_augmented_videos"  # Output folder for processed videos
    
    # Augmentation settings
    AUGMENTATIONS_PER_VIDEO = 18      # Number of augmented versions per original video
    AUGMENTATION_PROBABILITY = 0.8   # Probability of applying each augmentation
    
    # Hand cropping settings
    APPLY_HAND_CROP = True           # Enable/disable hand detection and cropping
    CROP_TARGET_SIZE = (224, 224)    # Target size for cropped videos
    
    # ===============================================================
    
    print("\n" + "="*70)
    print("NSL VIDEO PROCESSING PIPELINE")
    print("="*70)
    print("\nInitializing pipeline...")
    
    # Initialize the combined augmenter and cropper
    processor = NSLVideoAugmenterWithCrop(
        input_dataset_path=INPUT_DATASET_PATH,
        output_dataset_path=OUTPUT_DATASET_PATH,
        augmentation_probability=AUGMENTATION_PROBABILITY,
        apply_hand_crop=APPLY_HAND_CROP,
        crop_target_size=CROP_TARGET_SIZE
    )
    
    # Check dependencies
    print("\nChecking dependencies:")
    if processor.ffmpeg_available:
        print("  ✓ FFmpeg detected - Metadata preservation enabled")
    else:
        print("  ⚠ FFmpeg not detected - Limited metadata preservation")
        print("    Install FFmpeg for complete metadata support:")
        print("    - Windows: Download from https://ffmpeg.org/download.html")
        print("    - macOS: brew install ffmpeg")
        print("    - Ubuntu/Debian: sudo apt install ffmpeg")
    
    if APPLY_HAND_CROP:
        print("  ✓ MediaPipe hand detection enabled")
        print(f"    Output size: {CROP_TARGET_SIZE[0]}x{CROP_TARGET_SIZE[1]} pixels")
    else:
        print("  ⚠ Hand cropping disabled - videos will be augmented only")
    
    try:
        # Process the entire dataset
        print("\n" + "-"*70)
        print("Starting processing...")
        print("-"*70)
        
        stats = processor.process_entire_dataset(
            augmentations_per_video=AUGMENTATIONS_PER_VIDEO
        )
        
        # Print summary
        processor.print_dataset_summary(stats)
        
        print(f"\n✅ Processing complete!")
        print(f"   Output location: {OUTPUT_DATASET_PATH}/")
        print(f"   Each class folder contains augmented {'and cropped ' if APPLY_HAND_CROP else ''}videos")
        print(f"   All videos are in MP4 format")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user")
        print("Partial results may have been saved")
        
    except Exception as e:
        print(f"\n❌ Failed to process dataset: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()