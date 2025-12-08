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
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pick_device():
    """Return the fastest *available* device without starving other tasks."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = pick_device()
cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logger.info(f"Using device: {DEVICE}")


class NSLVideoAugmenter:
    """
    Nepali Sign Language Video Augmentation Pipeline
    """
    
    def __init__(self, input_dataset_path: str, output_dataset_path: str, 
                 augmentation_probability: float = 0.85, 
                 target_size: tuple = None):
        self.input_path = Path(input_dataset_path)
        self.output_path = Path(output_dataset_path)
        self.aug_prob = augmentation_probability
        self.target_size = target_size  # Optional: (width, height) for resizing
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Define augmentation functions
        self.augmentations = {
            'bright_high':        self._brightness_high,
            'bright_low':         self._brightness_low,
            'contrast_high':      self._contrast_high,
            'contrast_low':       self._contrast_low,
            'saturation_high':    self._saturation_high,
            'saturation_low':     self._saturation_low,
            'blur_motion':        self._motion_blur,
            'blur_gaussian':      self._gaussian_blur,
            'noise_moderate':     self._moderate_noise,
            'noise_salt_pepper':  self._salt_pepper_noise,
            'gamma_high':         self._gamma_high,
            'gamma_low':          self._gamma_low,
            'hue_shift':          self._hue_shift,
            'bilateral_filter':   self._bilateral_filter,
            'clahe':              self._clahe,
            'color_temp_warm':    self._color_temperature_warm,
            'color_temp_cool':    self._color_temperature_cool,
            'film_grain':         self._film_grain,
            'compression_artifacts': self._compression_artifacts,
            'shadow_highlight':   self._shadow_highlight,
            'exposure_comp':      self._exposure_compensation,
            'color_jitter':       self._color_jitter,
            'defocus_blur':       self._defocus_blur
        }
        
        # Supported video formats
        self.video_extensions = {'.mp4', '.avi', '.mov', '.MOV', '.wmv'}
        self.output_format = '.mp4'
        
        # Check if ffmpeg is available
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
            
            cmd.extend(['-metadata', 'comment=Augmented NSL video'])
            cmd.extend(['-metadata', 'title=NSL Augmented Dataset'])
            cmd.extend(['-metadata', 'processed=augmented_only'])
            
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
            
            cap = cv2.VideoCapture(str(video_path))
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
    
    def _rotate_frame(self, frame: np.ndarray, angle: int) -> np.ndarray:
        """Rotate frame by specified angle"""
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
    
    # Augmentation methods
    def _brightness_high(self, frame): 
        beta = random.uniform(15, 35)
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)

    def _brightness_low(self, frame):  
        beta = random.uniform(-25, -10)
        return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
    
    def _contrast_high(self, frame):   
        alpha = random.uniform(1.1, 1.25)
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    
    def _contrast_low(self, frame):    
        alpha = random.uniform(0.75, 0.9)
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    
    def _saturation_high(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(1.1, 1.25), 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _saturation_low(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.75, 0.9), 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _motion_blur(self, frame):
        k = random.randint(3, 5)
        kernel = np.zeros((k, k))
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        if direction == 'horizontal': kernel[k // 2, :] = 1
        elif direction == 'vertical': kernel[:, k // 2] = 1
        else: np.fill_diagonal(kernel, 1)
        kernel = kernel / k
        return cv2.filter2D(frame, -1, kernel)
    
    def _gaussian_blur(self, frame):
        k = random.choice([3, 5])
        return cv2.GaussianBlur(frame, (k, k), random.uniform(0.5, 1.2))
    
    def _moderate_noise(self, frame):
        noise = np.random.normal(0, random.uniform(5, 12), frame.shape).astype(np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    def _salt_pepper_noise(self, frame: np.ndarray) -> np.ndarray:
        density = random.uniform(0.001, 0.003)
        out = frame.copy()
        mask = np.random.random(out.shape[:2]) < density
        salt = np.random.random(mask.sum()) < 0.5
        out[mask] = np.where(salt[:, None], 255, 0)
        return out
    
    def _gamma_high(self, frame):
        gamma = random.uniform(0.75, 0.9)
        inv = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(frame, table)
    
    def _gamma_low(self, frame):
        gamma = random.uniform(1.1, 1.25)
        inv = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(frame, table)
    
    def _hue_shift(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-15, 15)) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _bilateral_filter(self, frame):
        return cv2.bilateralFilter(frame, random.choice([5, 7]), random.uniform(30, 60), random.uniform(30, 60))
    
    def _clahe(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=random.uniform(2.0, 3.5), tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _color_temperature_warm(self, frame):
        m = np.array([[1.0, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 0.8]])
        f = frame.astype(np.float32) / 255.0
        return np.clip(np.dot(f, m.T) * 255, 0, 255).astype(np.uint8)
    
    def _color_temperature_cool(self, frame):
        m = np.array([[0.8, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 1.2]])
        f = frame.astype(np.float32) / 255.0
        return np.clip(np.dot(f, m.T) * 255, 0, 255).astype(np.uint8)
    
    def _film_grain(self, frame):
        h, w = frame.shape[:2]
        grain = np.random.normal(0, random.uniform(8, 20), (h, w)).astype(np.int16)
        return np.clip(frame.astype(np.int16) + np.stack([grain, grain, grain], axis=2), 0, 255).astype(np.uint8)
    
    def _compression_artifacts(self, frame):
        _, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(65, 85)])
        return cv2.imdecode(enc, 1)
    
    def _shadow_highlight(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)
        l = l * (1 + (random.uniform(0.85, 1.15) - 1) * (l < 85))
        l = l * (1 + (random.uniform(0.85, 1.15) - 1) * (l > 170))
        lab[:, :, 0] = np.clip(l, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _exposure_compensation(self, frame):
        return np.clip(frame.astype(np.float32) * (2 ** random.uniform(-0.8, 0.8)), 0, 255).astype(np.uint8)
    
    def _color_jitter(self, frame):
        f = frame.astype(np.float32) / 255.0
        f = f * random.uniform(0.85, 1.15)
        mean = np.mean(f)
        f = (f - mean) * random.uniform(0.85, 1.15) + mean
        hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.85, 1.15)
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _defocus_blur(self, frame):
        k = random.choice([5, 7])
        kernel = np.zeros((k, k))
        c = k // 2
        y, x = np.ogrid[:k, :k]
        kernel[(x - c) ** 2 + (y - c) ** 2 <= c ** 2] = 1
        kernel /= kernel.sum()
        return cv2.filter2D(frame, -1, kernel)
    
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
    
    def augment_video(self, video_path: Path, output_path: Path, 
                     augmentation_suffix: str) -> bool:
        """
        Augment a single video file, preserve metadata, and save as MP4
        
        Args:
            video_path: Path to input video
            output_path: Path to save augmented video
            augmentation_suffix: Suffix to add to filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if output_path.suffix.lower() != '.mp4':
                output_path = output_path.with_suffix('.mp4')
            
            original_metadata = self._extract_video_metadata(str(video_path))
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            rotation_angle = self._get_video_rotation(str(video_path))
            
            if rotation_angle in [90, 270]:
                width, height = height, width
                logger.debug(f"Video {video_path.name} rotation detected: {rotation_angle}°")
            
            # Apply target size if specified
            if self.target_size:
                output_width, output_height = self.target_size
            else:
                output_width, output_height = width, height
            
            if fps <= 0:
                fps = 30
                logger.warning(f"Invalid FPS detected, defaulting to {fps}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            alternative_codecs = ['XVID', 'MJPG', 'X264']
            
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
            
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
            
            frame_count = 0
            applied_augmentations = []
            
            pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", leave=False)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply rotation if needed
                if rotation_angle != 0:
                    frame = self._rotate_frame(frame, rotation_angle)
                
                # Apply augmentations
                augmented_frame, frame_augs = self._apply_random_augmentations(frame, 2, 4)
                applied_augmentations.extend(frame_augs)
                
                # Resize if target size is specified
                if self.target_size and (augmented_frame.shape[1] != output_width or augmented_frame.shape[0] != output_height):
                    augmented_frame = cv2.resize(augmented_frame, (output_width, output_height), 
                                                interpolation=cv2.INTER_LANCZOS4)
                
                out.write(augmented_frame)
                frame_count += 1
                pbar.update(1)
            
            pbar.close()
            
            cap.release()
            out.release()
            
            if output_path.exists() and output_path.stat().st_size > 0:
                metadata_preserved = self._preserve_video_metadata(str(video_path), str(output_path))
                
                unique_augs = list(set(applied_augmentations))
                logger.info(f"Processed {video_path.name} -> {output_path.name}")
                logger.info(f"  Applied augmentations: {', '.join(unique_augs[:5])}...")
                if self.target_size:
                    logger.info(f"  Resized to: {self.target_size}")
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
    
    def _get_optimal_worker_count(self) -> int:
        """Calculate optimal number of workers based on system resources"""
        try:
            cpu_count = multiprocessing.cpu_count()
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            optimal_workers = max(1, min(cpu_count // 2, 8))
            memory_limited_workers = max(1, int(available_gb // 2))
            optimal_workers = min(optimal_workers, memory_limited_workers)
            
            if torch.backends.mps.is_available():
                optimal_workers = 1
            
            logger.info(f"Optimal workers calculated: {optimal_workers} (CPU: {cpu_count}, RAM: {available_gb:.1f}GB)")
            return optimal_workers
            
        except Exception:
            return 2
    
    def _process_video_batch(self, video_batch: List[Tuple[Path, Path, str]], worker_id: int) -> List[bool]:
        """Process a batch of videos in a worker process"""
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        
        results = []
        for video_path, output_path, suffix in video_batch:
            try:
                success = self.augment_video(video_path, output_path, suffix)
                results.append(success)
            except Exception as e:
                logger.error(f"Worker {worker_id} failed to process {video_path}: {e}")
                results.append(False)
        return results
    
    def process_class_folder_parallel(self, class_folder_name: str, augmentations_per_video: int = 18) -> Dict[str, int]:
        """Process all videos in a class folder using parallel processing"""
        if augmentations_per_video < 5:
            augmentations_per_video = 5
            logger.info(f"Minimum 5 augmentations required. Set to {augmentations_per_video}")
        
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
        
        if not original_videos:
            logger.warning(f"No original videos found in {class_folder_name}")
            return stats
        
        logger.info(f"\nProcessing class: {class_folder_name}")
        logger.info(f"  Found {len(original_videos)} original videos")
        logger.info(f"  Creating {augmentations_per_video} augmentations per video")
        if self.target_size:
            logger.info(f"  Target size: {self.target_size}")
        
        tasks = []
        for video_path in original_videos:
            for aug_idx in range(augmentations_per_video):
                stem = video_path.stem
                augmented_name = f"{stem}_aug_{aug_idx + 1:02d}{self.output_format}"
                output_path = output_class_folder / augmented_name
                
                if output_path.exists():
                    logger.debug(f"Skipping existing: {augmented_name}")
                    stats['augmented_videos'] += 1
                    continue
                
                tasks.append((video_path, output_path, f"aug_{aug_idx + 1:02d}"))
        
        if not tasks:
            logger.info(f"All videos already processed for {class_folder_name}")
            return stats
        
        num_workers = self._get_optimal_worker_count()
        
        logger.info(f"  Processing {len(tasks)} tasks with {num_workers} workers")
        
        if num_workers == 1:
            for video_path, output_path, suffix in tqdm(tasks, desc=f"Class {class_folder_name}"):
                success = self.augment_video(video_path, output_path, suffix)
                if success:
                    stats['augmented_videos'] += 1
                else:
                    stats['failed_videos'] += 1
        else:
            try:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    batch_size = max(1, len(tasks) // num_workers)
                    batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]
                    
                    future_to_batch = {
                        executor.submit(self._process_video_batch, batch, idx): batch
                        for idx, batch in enumerate(batches)
                    }
                    
                    for future in tqdm(as_completed(future_to_batch), 
                                     total=len(batches), 
                                     desc=f"Class {class_folder_name}"):
                        try:
                            batch_results = future.result()
                            stats['augmented_videos'] += sum(batch_results)
                            stats['failed_videos'] += len(batch_results) - sum(batch_results)
                        except Exception as e:
                            logger.error(f"Batch processing failed: {e}")
                            stats['failed_videos'] += len(future_to_batch[future])
                            
            except Exception as e:
                logger.error(f"Parallel processing failed, falling back to sequential: {e}")
                for video_path, output_path, suffix in tqdm(tasks, desc=f"Class {class_folder_name}"):
                    success = self.augment_video(video_path, output_path, suffix)
                    if success:
                        stats['augmented_videos'] += 1
                    else:
                        stats['failed_videos'] += 1
        
        return stats
    
    def process_class_folder(self, class_folder_name: str, augmentations_per_video: int = 18) -> Dict[str, int]:
        """Process all videos in a class folder"""
        if augmentations_per_video < 5:
            augmentations_per_video = 5
            logger.info(f"Minimum 5 augmentations required. Set to {augmentations_per_video}")
        
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
        
        logger.info(f"\nProcessing class: {class_folder_name}")
        logger.info(f"  Found {len(original_videos)} original videos")
        logger.info(f"  Creating {augmentations_per_video} augmentations per video")
        if self.target_size:
            logger.info(f"  Target size: {self.target_size}")
        
        for video_path in tqdm(original_videos, desc=f"Class {class_folder_name}"):
            for aug_idx in range(augmentations_per_video):
                stem = video_path.stem
                augmented_name = f"{stem}_aug_{aug_idx + 1:02d}{self.output_format}"
                output_path = output_class_folder / augmented_name
                
                if output_path.exists():
                    logger.debug(f"Skipping existing: {augmented_name}")
                    stats['augmented_videos'] += 1
                    continue
                
                success = self.augment_video(
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
    
    def process_entire_dataset_parallel(self, augmentations_per_video: int = 5) -> Dict[str, Dict[str, int]]:
        """Process the entire dataset using parallel processing"""
        if augmentations_per_video < 5:
            augmentations_per_video = 5
            logger.info(f"Minimum 5 augmentations required. Set to {augmentations_per_video}")
        
        if not self.input_path.exists():
            raise ValueError(f"Input dataset path does not exist: {self.input_path}")
        
        class_folders = [f for f in self.input_path.iterdir() if f.is_dir()]
        
        if not class_folders:
            raise ValueError(f"No class folders found in: {self.input_path}")
        
        logger.info("="*70)
        logger.info("NSL VIDEO AUGMENTATION PIPELINE (PARALLEL)")
        logger.info("="*70)
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Classes found: {len(class_folders)}")
        logger.info(f"Augmentations per video: {augmentations_per_video}")
        if self.target_size:
            logger.info(f"Target size: {self.target_size}")
        logger.info(f"Output format: MP4")
        logger.info(f"Parallel processing: {'Enabled' if self._get_optimal_worker_count() > 1 else 'Disabled'}")
        logger.info("="*70)
        
        all_stats = {}
        
        for class_folder in sorted(class_folders):
            class_name = class_folder.name
            try:
                stats = self.process_class_folder_parallel(class_name, augmentations_per_video)
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

    def process_entire_dataset(self, augmentations_per_video: int = 5) -> Dict[str, Dict[str, int]]:
        """Process the entire dataset"""
        if augmentations_per_video < 5:
            augmentations_per_video = 5
            logger.info(f"Minimum 5 augmentations required. Set to {augmentations_per_video}")
        
        if not self.input_path.exists():
            raise ValueError(f"Input dataset path does not exist: {self.input_path}")
        
        class_folders = [f for f in self.input_path.iterdir() if f.is_dir()]
        
        if not class_folders:
            raise ValueError(f"No class folders found in: {self.input_path}")
        
        logger.info("="*70)
        logger.info("NSL VIDEO AUGMENTATION PIPELINE")
        logger.info("="*70)
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Classes found: {len(class_folders)}")
        logger.info(f"Augmentations per video: {augmentations_per_video}")
        if self.target_size:
            logger.info(f"Target size: {self.target_size}")
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
        """Print a summary of the augmentation process"""
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
        
        if self.target_size:
            print(f"All videos resized to: {self.target_size[0]}x{self.target_size[1]} pixels")
        
        print(f"Success rate: {(total_augmented / max(1, total_augmented + total_failed)) * 100:.1f}%")
        print("="*70)


def main():
    """Main execution function"""
    # ======================== CONFIGURATION ========================
    
    # Paths
    INPUT_DATASET_PATH = "main_dataset"
    OUTPUT_DATASET_PATH = "preprocessed_and_augmented_videos_without_cropping"
    
    # Augmentation settings
    AUGMENTATIONS_PER_VIDEO = 20
    AUGMENTATION_PROBABILITY = 0.85
    
    # Resize settings (set to None to keep original size)
    TARGET_SIZE = (224, 224)  # (width, height) or None
    
    # Enable/disable parallel processing
    ENABLE_PARALLEL = True
    
    # ===============================================================
    
    print("\n" + "="*70)
    print("NSL VIDEO AUGMENTATION PIPELINE")
    print("="*70)
    print("\nInitializing pipeline...")
    
    # Initialize the augmenter
    processor = NSLVideoAugmenter(
        input_dataset_path=INPUT_DATASET_PATH,
        output_dataset_path=OUTPUT_DATASET_PATH,
        augmentation_probability=AUGMENTATION_PROBABILITY,
        target_size=TARGET_SIZE
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
    
    if TARGET_SIZE:
        print(f"  ✓ Video resizing enabled: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} pixels")
    else:
        print("  ⚠ Video resizing disabled - original dimensions preserved")
    
    optimal_workers = processor._get_optimal_worker_count() if ENABLE_PARALLEL else 1
    print(f"  Parallel processing: {'Enabled' if optimal_workers > 1 else 'Disabled'} ({optimal_workers} workers)")
    
    try:
        print("\n" + "-"*70)
        print("Starting processing...")
        print("-"*70)
        
        if ENABLE_PARALLEL and optimal_workers > 1:
            stats = processor.process_entire_dataset_parallel(
                augmentations_per_video=AUGMENTATIONS_PER_VIDEO
            )
        else:
            stats = processor.process_entire_dataset(
                augmentations_per_video=AUGMENTATIONS_PER_VIDEO
            )
        
        # Print summary
        processor.print_dataset_summary(stats)
        
        print(f"\n✅ Processing complete!")
        print(f"   Output location: {OUTPUT_DATASET_PATH}/")
        print(f"   Each class folder contains augmented videos")
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