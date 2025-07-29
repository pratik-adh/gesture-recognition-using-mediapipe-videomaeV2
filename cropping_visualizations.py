import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
import shutil
from datetime import datetime
import multiprocessing as mp_proc
from functools import partial

class VideoPreprocessor:
    """Preprocess videos with MediaPipe hand detection and smart cropping"""
    
    def __init__(self, config):
        self.config = config
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=config['model_complexity'],
            min_detection_confidence=config['confidence_threshold'],
            min_tracking_confidence=0.5
        )
        self.stats = defaultdict(int)
    
    def process_video(self, video_path, output_video_path, output_frames_dir=None):
        """Process a single video with MediaPipe cropping"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"Warning: Empty video {video_path}")
            return None
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        # Process frames with MediaPipe
        cropped_frames = self._process_frames(frames)
        
        # Save cropped video
        if len(cropped_frames) > 0:
            h, w = cropped_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
            
            for frame in cropped_frames:
                out.write(frame)
            out.release()
            
            # Save individual frames if requested
            if output_frames_dir:
                output_frames_dir.mkdir(parents=True, exist_ok=True)
                for i, frame in enumerate(cropped_frames):
                    frame_path = output_frames_dir / f"frame_{i:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
            
            # Calculate statistics
            detection_rate = self.stats['hands_detected'] / max(1, self.stats['frames_processed']) * 100
            
            return {
                'success': True,
                'original_frames': total_frames,
                'processed_frames': len(cropped_frames),
                'detection_rate': detection_rate,
                'output_video': str(output_video_path),
                'output_frames': str(output_frames_dir) if output_frames_dir else None
            }
        
        return {'success': False, 'error': 'No frames processed'}
    
    def _process_frames(self, frames):
        """Process frames with MediaPipe detection and cropping"""
        cropped_frames = []
        all_detections = []
        
        # First pass: detect hands
        for frame in frames:
            detection = self._detect_hand(frame)
            all_detections.append(detection)
            self.stats['frames_processed'] += 1
            if detection['success']:
                self.stats['hands_detected'] += 1
        
        # Apply temporal smoothing
        if self.config['temporal_smoothing']:
            all_detections = self._smooth_detections(all_detections)
        
        # Second pass: crop frames
        for frame, detection in zip(frames, all_detections):
            if detection['success']:
                cropped = self._crop_with_padding(frame, detection)
            else:
                cropped = self._center_crop(frame)
                self.stats['center_crops'] += 1
            
            # Resize to target size
            resized = cv2.resize(cropped, tuple(self.config['output_size']), 
                               interpolation=cv2.INTER_CUBIC)
            cropped_frames.append(resized)
        
        return cropped_frames
    
    def _detect_hand(self, frame):
        """Detect hand in frame using MediaPipe"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.mp_hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            
            # Get bounding box
            x_coords = [lm.x for lm in landmarks.landmark]
            y_coords = [lm.y for lm in landmarks.landmark]
            
            return {
                'success': True,
                'bbox': (min(x_coords), min(y_coords), max(x_coords), max(y_coords)),
                'center': ((min(x_coords) + max(x_coords)) / 2, 
                          (min(y_coords) + max(y_coords)) / 2),
                'size': (max(x_coords) - min(x_coords), 
                        max(y_coords) - min(y_coords))
            }
        
        return {'success': False}
    
    def _smooth_detections(self, detections):
        """Apply temporal smoothing to detections"""
        smoothed = []
        valid_indices = [i for i, d in enumerate(detections) if d['success']]
        
        if len(valid_indices) < 2:
            return detections
        
        for i, detection in enumerate(detections):
            if detection['success']:
                smoothed.append(detection)
            else:
                # Find nearest valid detections for interpolation
                prev_valid = next((j for j in reversed(valid_indices) if j < i), None)
                next_valid = next((j for j in valid_indices if j > i), None)
                
                if prev_valid is not None and next_valid is not None:
                    # Interpolate
                    alpha = (i - prev_valid) / (next_valid - prev_valid)
                    prev_det = detections[prev_valid]
                    next_det = detections[next_valid]
                    
                    interpolated = {
                        'success': True,
                        'center': (
                            prev_det['center'][0] * (1-alpha) + next_det['center'][0] * alpha,
                            prev_det['center'][1] * (1-alpha) + next_det['center'][1] * alpha
                        ),
                        'size': (
                            prev_det['size'][0] * (1-alpha) + next_det['size'][0] * alpha,
                            prev_det['size'][1] * (1-alpha) + next_det['size'][1] * alpha
                        )
                    }
                    smoothed.append(interpolated)
                else:
                    smoothed.append(detection)
        
        return smoothed
    
    def _crop_with_padding(self, frame, detection):
        """Crop frame with padding around detected hand"""
        h, w = frame.shape[:2]
        cx, cy = detection['center']
        hand_w, hand_h = detection['size']
        
        # Apply padding
        padding = self.config['padding_ratio']
        crop_size = max(hand_w, hand_h) * (1 + 2 * padding)
        crop_size = int(crop_size * max(w, h))
        
        # Ensure minimum size
        min_crop = int(min(w, h) * 0.4)
        crop_size = max(crop_size, min_crop)
        
        # Calculate boundaries
        x1 = int(max(0, cx * w - crop_size // 2))
        y1 = int(max(0, cy * h - crop_size // 2))
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        
        # Adjust if needed
        if x2 - x1 < crop_size:
            x1 = max(0, x2 - crop_size)
        if y2 - y1 < crop_size:
            y1 = max(0, y2 - crop_size)
        
        return frame[y1:y2, x1:x2]
    
    def _center_crop(self, frame):
        """Fallback center crop"""
        h, w = frame.shape[:2]
        crop_ratio = 0.8
        
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        
        y_start = (h - new_h) // 2
        x_start = (w - new_w) // 2
        
        return frame[y_start:y_start+new_h, x_start:x_start+new_w]
    
    def __del__(self):
        if hasattr(self, 'mp_hands'):
            self.mp_hands.close()


def process_video_wrapper(args):
    """Wrapper for multiprocessing"""
    video_path, output_video_path, output_frames_dir, config = args
    
    preprocessor = VideoPreprocessor(config)
    result = preprocessor.process_video(video_path, output_video_path, output_frames_dir)
    
    return {
        'video_path': str(video_path),
        'result': result,
        'stats': dict(preprocessor.stats)
    }


def preprocess_dataset(
    input_dir,
    output_dir,
    config=None,
    save_frames=False,
    num_workers=4,
    max_videos_per_class=None
):
    """
    Preprocess entire dataset with MediaPipe cropping
    
    Parameters:
    -----------
    input_dir : str
        Path to input dataset directory
    output_dir : str
        Path to output directory for processed videos
    config : dict
        Preprocessing configuration
    save_frames : bool
        Whether to save individual frames
    num_workers : int
        Number of parallel workers
    max_videos_per_class : int
        Maximum videos to process per class (for testing)
    """
    
    # Default configuration
    if config is None:
        config = {
            'model_complexity': 1,
            'confidence_threshold': 0.3,
            'padding_ratio': 0.35,
            'output_size': [224, 224],
            'temporal_smoothing': True,
            'min_detection_rate': 0.5  # Minimum detection rate to keep video
        }
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all videos
    print("Scanning for videos...")
    all_tasks = []
    class_stats = defaultdict(int)
    
    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(exist_ok=True)
        
        # Find videos in class
        videos = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
        
        if max_videos_per_class:
            videos = videos[:max_videos_per_class]
        
        for video_path in videos:
            output_video_path = output_class_dir / video_path.name
            
            output_frames_dir = None
            if save_frames:
                output_frames_dir = output_class_dir / f"{video_path.stem}_frames"
            
            all_tasks.append((video_path, output_video_path, output_frames_dir, config))
            class_stats[class_name] += 1
    
    print(f"\nFound {len(all_tasks)} videos across {len(class_stats)} classes:")
    for class_name, count in sorted(class_stats.items()):
        print(f"  {class_name}: {count} videos")
    
    # Process videos
    print(f"\nProcessing videos with {num_workers} workers...")
    
    successful = 0
    failed = 0
    low_detection = 0
    all_results = []
    
    # Single-threaded for macOS compatibility
    if num_workers == 1:
        preprocessor = VideoPreprocessor(config)
        
        for task in tqdm(all_tasks, desc="Processing videos"):
            video_path, output_video_path, output_frames_dir, _ = task
            
            result = preprocessor.process_video(video_path, output_video_path, output_frames_dir)
            
            if result and result.get('success'):
                if result.get('detection_rate', 0) < config['min_detection_rate'] * 100:
                    low_detection += 1
                    print(f"\n⚠️  Low detection rate ({result['detection_rate']:.1f}%) for {video_path.name}")
                successful += 1
            else:
                failed += 1
                print(f"\n❌ Failed to process {video_path.name}")
            
            all_results.append({
                'video_path': str(video_path),
                'result': result,
                'stats': dict(preprocessor.stats)
            })
    else:
        # Multiprocessing
        with mp_proc.Pool(num_workers) as pool:
            for result_data in tqdm(
                pool.imap_unordered(process_video_wrapper, all_tasks),
                total=len(all_tasks),
                desc="Processing videos"
            ):
                all_results.append(result_data)
                
                if result_data['result'] and result_data['result'].get('success'):
                    if result_data['result'].get('detection_rate', 0) < config['min_detection_rate'] * 100:
                        low_detection += 1
                    successful += 1
                else:
                    failed += 1
    
    # Generate summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'statistics': {
            'total_videos': len(all_tasks),
            'successful': successful,
            'failed': failed,
            'low_detection_rate': low_detection,
            'classes': dict(class_stats)
        },
        'results': all_results
    }
    
    # Save summary
    summary_path = output_path / 'preprocessing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos: {len(all_tasks)}")
    print(f"Successful: {successful} ({successful/len(all_tasks)*100:.1f}%)")
    print(f"Failed: {failed}")
    print(f"Low detection rate: {low_detection}")
    print(f"\nOutput saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    
    # Generate list of problematic videos
    if failed > 0 or low_detection > 0:
        problematic_videos = []
        for result_data in all_results:
            if not result_data['result'].get('success'):
                problematic_videos.append(result_data['video_path'])
            elif result_data['result'].get('detection_rate', 0) < config['min_detection_rate'] * 100:
                problematic_videos.append(f"{result_data['video_path']} (low detection: {result_data['result']['detection_rate']:.1f}%)")
        
        prob_path = output_path / 'problematic_videos.txt'
        with open(prob_path, 'w') as f:
            f.write("Problematic Videos:\n")
            f.write("="*50 + "\n")
            for video in problematic_videos:
                f.write(f"{video}\n")
        
        print(f"\nProblematic videos listed in: {prob_path}")
    
    return summary


if __name__ == "__main__":
    # Configuration
    config = {
        'model_complexity': 1,
        'confidence_threshold': 0.3,
        'padding_ratio': 0.35,
        'output_size': [224, 224],  # VideoMAE input size
        'temporal_smoothing': True,
        'min_detection_rate': 0.5  # Skip videos with <50% hand detection
    }
    
    # Example 1: Preprocess entire dataset
    preprocess_dataset(
        input_dir='splitted_dataset/train',
        output_dir='preprocessed_dataset/train',
        config=config,
        save_frames=False,  # Set to True to save individual frames
        num_workers=1,  # Use 1 for macOS
        max_videos_per_class=None  # Process all videos
    )
    
    # Example 2: Preprocess test set
    preprocess_dataset(
        input_dir='splitted_dataset/test',
        output_dir='preprocessed_dataset/test',
        config=config,
        save_frames=False,
        num_workers=1
    )
    
    # Example 3: Test on few videos
    # preprocess_dataset(
    #     input_dir='splitted_dataset/train',
    #     output_dir='preprocessed_test',
    #     config=config,
    #     save_frames=True,
    #     num_workers=1,
    #     max_videos_per_class=2  # Only 2 videos per class for testing
    # )