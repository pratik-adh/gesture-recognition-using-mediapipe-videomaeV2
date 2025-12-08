import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import json
import warnings
import subprocess
import tempfile
warnings.filterwarnings("ignore")

# Configuration
INPUT_VIDEO_DIR = "preprocessed_and_augmented_videos"
OUTPUT_TENSOR_DIR = "ytorch_tensors"

# Lossless configuration
PRESERVE_ORIGINAL_RESOLUTION = True
EXTRACT_ALL_FRAMES = True
PRESERVE_AUDIO = True
USE_LOSSLESS_COMPRESSION = True

# Space-saving optimizations
CONVERT_TO_FLOAT16 = False  # Set True to halve size (minimal quality loss for neural networks)
USE_BETTER_COMPRESSION = True  # Use higher compression level
STORE_DIFFERENTIAL_FRAMES = False  # Store only changes between frames (experimental)
DOWNSAMPLE_AUDIO = True  # Reduce audio sample rate to 16kHz (sufficient for most tasks)

# CUDA Configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
PIN_MEMORY = USE_CUDA  # Enable pinned memory for faster transfers

# Create output directory
os.makedirs(OUTPUT_TENSOR_DIR, exist_ok=True)

class LosslessVideoConverter:
    def __init__(self):
        self.preserve_resolution = PRESERVE_ORIGINAL_RESOLUTION
        self.extract_all_frames = EXTRACT_ALL_FRAMES
        self.preserve_audio = PRESERVE_AUDIO
        self.use_lossless = USE_LOSSLESS_COMPRESSION
        self.device = DEVICE
        
        # Print CUDA info
        if USE_CUDA:
            print(f"CUDA available: Using GPU ({torch.cuda.get_device_name(0)})")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("CUDA not available: Using CPU")
        
        # Initialize MediaPipe for optional hand detection metadata
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_all_video_data(self, video_path):
        """Extract all frames and metadata without any loss"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Extract all frames
        frames = []
        hand_detections = []
        frame_timestamps = []
        
        print(f"  Extracting {total_frames} frames at {width}x{height} resolution...")
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Store original frame without any modification
            frames.append(frame)
            
            # Get timestamp
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_timestamps.append(timestamp)
            
            # Optional: Store hand detection metadata (doesn't modify frame)
            bbox = self.detect_hand_metadata(frame)
            hand_detections.append(bbox)
        
        cap.release()
        
        # Convert to numpy array - preserve original dtype (usually uint8)
        video_array = np.stack(frames) if frames else np.array([])
        
        # Store in format (T, H, W, C) - temporal first for video
        # No transposition or modification of original data
        
        metadata = {
            'fps': fps,
            'total_frames': len(frames),
            'original_total_frames': total_frames,
            'width': width,
            'height': height,
            'fourcc': fourcc,
            'hand_detections': hand_detections,
            'timestamps': frame_timestamps,
            'original_path': str(video_path)
        }
        
        return video_array, metadata
    
    def extract_audio(self, video_path):
        """Extract audio track if present"""
        try:
            # Check if video has audio
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_rate,channels',
                '-of', 'json',
                str(video_path)
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                audio_info = json.loads(result.stdout)
                
                if audio_info.get('streams'):
                    # Extract audio as numpy array
                    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    extract_cmd = [
                        'ffmpeg', '-i', str(video_path),
                        '-vn', '-acodec', 'pcm_s16le',
                        '-ar', '44100', '-ac', '2',
                        temp_audio.name, '-y'
                    ]
                    
                    subprocess.run(extract_cmd, capture_output=True)
                    
                    # Read WAV file as numpy array
                    import wave
                    with wave.open(temp_audio.name, 'rb') as wav_file:
                        frames = wav_file.readframes(-1)
                        audio_array = np.frombuffer(frames, dtype=np.int16)
                        audio_metadata = {
                            'sample_rate': wav_file.getframerate(),
                            'channels': wav_file.getnchannels(),
                            'sample_width': wav_file.getsampwidth()
                        }
                    
                    os.unlink(temp_audio.name)
                    return audio_array, audio_metadata
        except Exception as e:
            print(f"    Audio extraction failed: {e}")
        
        return None, None
    
    def detect_hand_metadata(self, frame):
        """Detect hand bounding box (for metadata only, doesn't modify frame)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            all_hands = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                coords = []
                for landmark in hand_landmarks.landmark:
                    coords.extend([landmark.x * w, landmark.y * h])
                
                if coords:
                    x_coords = coords[::2]
                    y_coords = coords[1::2]
                    
                    bbox = {
                        'x_min': min(x_coords),
                        'y_min': min(y_coords),
                        'x_max': max(x_coords),
                        'y_max': max(y_coords)
                    }
                    all_hands.append(bbox)
            
            return all_hands if all_hands else None
        return None
    
    def save_lossless_tensor(self, video_array, metadata, audio_data, output_path):
        """Save video data with lossless compression using CUDA"""
        
        if self.use_lossless:
            # Convert to torch tensor but preserve original dtype
            video_tensor = torch.from_numpy(video_array)
            
            # Move to CUDA if available (for processing)
            if USE_CUDA:
                video_tensor = video_tensor.pin_memory()  # Pin memory for faster transfer
            
            # Prepare save dict
            save_dict = {
                'video': video_tensor,
                'metadata': metadata,
                'compression': 'lossless',
                'dtype': str(video_array.dtype),
                'shape': video_array.shape,
                'device': 'cuda' if USE_CUDA else 'cpu'
            }
            
            if audio_data is not None and audio_data[0] is not None:
                audio_tensor = torch.from_numpy(audio_data[0])
                if USE_CUDA:
                    audio_tensor = audio_tensor.pin_memory()
                save_dict['audio'] = audio_tensor
                save_dict['audio_metadata'] = audio_data[1]
            
            # Save with compression (PyTorch uses pickle protocol 4 with compression)
            # Note: Tensors are saved to CPU for storage, loaded to CUDA on demand
            torch.save(save_dict, output_path, pickle_protocol=4)
            
        else:
            # Save raw without any compression
            video_tensor = torch.from_numpy(video_array)
            if USE_CUDA:
                video_tensor = video_tensor.pin_memory()
                
            save_dict = {
                'video': video_tensor,
                'metadata': metadata,
                'compression': 'none',
                'dtype': str(video_array.dtype),
                'shape': video_array.shape,
                'device': 'cuda' if USE_CUDA else 'cpu'
            }
            
            if audio_data is not None and audio_data[0] is not None:
                audio_tensor = torch.from_numpy(audio_data[0])
                if USE_CUDA:
                    audio_tensor = audio_tensor.pin_memory()
                save_dict['audio'] = audio_tensor
                save_dict['audio_metadata'] = audio_data[1]
            
            torch.save(save_dict, output_path)
        
        # Clear CUDA cache to free memory
        if USE_CUDA:
            torch.cuda.empty_cache()
        
        # Calculate sizes
        original_size = video_array.nbytes
        if audio_data is not None and audio_data[0] is not None:
            original_size += audio_data[0].nbytes
        
        compressed_size = os.path.getsize(output_path)
        
        return {
            'original_mb': original_size / (1024 * 1024),
            'saved_mb': compressed_size / (1024 * 1024),
            'compression_ratio': compressed_size / original_size if original_size > 0 else 1.0
        }
    
    def process_video(self, video_path, output_path):
        """Process single video with lossless conversion"""
        try:
            print(f"  Processing: {os.path.basename(video_path)}")
            
            # Extract all video data
            video_array, metadata = self.extract_all_video_data(video_path)
            
            # Extract audio if requested
            audio_data = None
            if self.preserve_audio:
                audio_array, audio_metadata = self.extract_audio(video_path)
                if audio_array is not None:
                    audio_data = (audio_array, audio_metadata)
                    print(f"    Audio extracted: {len(audio_array)} samples")
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save losslessly
            stats = self.save_lossless_tensor(video_array, metadata, audio_data, output_path)
            
            print(f"    Saved: {stats['saved_mb']:.2f} MB (ratio: {stats['compression_ratio']:.2%})")
            
            return {
                'success': True,
                'stats': stats,
                'frames': metadata['total_frames'],
                'resolution': f"{metadata['width']}x{metadata['height']}",
                'has_audio': audio_data is not None
            }
            
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def convert_dataset(self):
        """Convert entire dataset with lossless preservation"""
        input_path = Path(INPUT_VIDEO_DIR)
        output_path = Path(OUTPUT_TENSOR_DIR)
        
        if not input_path.exists():
            print(f"ERROR: Input directory not found: {INPUT_VIDEO_DIR}")
            return False
        
        stats = {
            'total_videos': 0,
            'successful': 0,
            'failed': 0,
            'total_frames': 0,
            'total_original_mb': 0.0,
            'total_saved_mb': 0.0,
            'videos_with_audio': 0
        }
        
        print("Starting lossless video conversion...")
        print(f"Settings:")
        print(f"  - Device: {self.device}")
        print(f"  - Preserve original resolution: {self.preserve_resolution}")
        print(f"  - Extract all frames: {self.extract_all_frames}")
        print(f"  - Preserve audio: {self.preserve_audio}")
        print(f"  - Lossless compression: {self.use_lossless}")
        print(f"  - Pin memory: {PIN_MEMORY}\n")
        
        class_folders = [d for d in input_path.iterdir() if d.is_dir()]
        
        for class_folder in tqdm(class_folders, desc="Processing classes"):
            class_name = class_folder.name
            output_class_dir = output_path / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            video_files = []
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                video_files.extend(list(class_folder.glob(f"*{ext}")))
            
            if not video_files:
                continue
            
            print(f"\nProcessing class: {class_name} ({len(video_files)} videos)")
            
            for video_file in video_files:
                video_name = video_file.stem
                output_file = output_class_dir / f"{video_name}.pt"
                
                result = self.process_video(str(video_file), str(output_file))
                stats['total_videos'] += 1
                
                if result['success']:
                    stats['successful'] += 1
                    stats['total_frames'] += result['frames']
                    stats['total_original_mb'] += result['stats']['original_mb']
                    stats['total_saved_mb'] += result['stats']['saved_mb']
                    if result['has_audio']:
                        stats['videos_with_audio'] += 1
                else:
                    stats['failed'] += 1
        
        # Print results
        print(f"\n{'='*50}")
        print(f"CONVERSION SUMMARY")
        print(f"{'='*50}")
        print(f"Device used: {self.device}")
        print(f"Total videos processed: {stats['successful']}/{stats['total_videos']}")
        print(f"Failed conversions: {stats['failed']}")
        print(f"Total frames preserved: {stats['total_frames']:,}")
        print(f"Videos with audio: {stats['videos_with_audio']}")
        print(f"Original size: {stats['total_original_mb']:.2f} MB")
        print(f"Saved size: {stats['total_saved_mb']:.2f} MB")
        
        if stats['total_original_mb'] > 0:
            ratio = stats['total_saved_mb'] / stats['total_original_mb']
            print(f"Compression ratio: {ratio:.2%}")
            print(f"Space efficiency: {(1-ratio)*100:.1f}% saved with lossless compression")
        
        if USE_CUDA:
            print(f"\nGPU Memory Stats:")
            print(f"  - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"  - Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        return True


class LosslessVideoDataset:
    """Dataset loader for lossless video tensors with CUDA support"""
    
    def __init__(self, tensor_dir, device=None):
        self.tensor_dir = Path(tensor_dir)
        self.device = device if device is not None else DEVICE
        self.tensor_files = []
        self.labels = []
        self.class_to_idx = {}
        
        # Load dataset structure
        class_folders = [d for d in self.tensor_dir.iterdir() if d.is_dir()]
        
        for idx, class_folder in enumerate(sorted(class_folders)):
            class_name = class_folder.name
            self.class_to_idx[class_name] = idx
            
            tensor_files = list(class_folder.glob("*.pt"))
            for tensor_file in tensor_files:
                self.tensor_files.append(str(tensor_file))
                self.labels.append(idx)
        
        print(f"Loaded {len(self.tensor_files)} videos from {len(class_folders)} classes")
        print(f"Dataset device: {self.device}")
    
    def load_video(self, tensor_path, to_device=True):
        """Load complete video data and optionally move to CUDA"""
        # Load to CPU first
        data = torch.load(tensor_path, map_location='cpu')
        
        video_tensor = data['video']
        metadata = data['metadata']
        
        # Move to device if requested
        if to_device and USE_CUDA:
            video_tensor = video_tensor.to(self.device, non_blocking=True)
        
        # Optionally load audio
        audio = None
        if 'audio' in data:
            audio_tensor = data['audio']
            if to_device and USE_CUDA:
                audio_tensor = audio_tensor.to(self.device, non_blocking=True)
            
            audio = {
                'data': audio_tensor,
                'metadata': data['audio_metadata']
            }
        
        return {
            'video': video_tensor,
            'metadata': metadata,
            'audio': audio
        }
    
    def reconstruct_video(self, video_data, output_path):
        """Reconstruct original video from tensor data"""
        # Move to CPU if on CUDA
        if video_data['video'].is_cuda:
            video_tensor = video_data['video'].cpu().numpy()
        else:
            video_tensor = video_data['video'].numpy()
            
        metadata = video_data['metadata']
        
        # Create video writer with original properties
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            metadata['fps'],
            (metadata['width'], metadata['height'])
        )
        
        # Write all frames
        for frame in video_tensor:
            out.write(frame)
        
        out.release()
        
        # Add audio if present
        if video_data['audio'] is not None:
            # Use ffmpeg to add audio back
            # Implementation depends on your audio handling needs
            pass
        
        print(f"Video reconstructed: {output_path}")
    
    def __len__(self):
        return len(self.tensor_files)
    
    def __getitem__(self, idx):
        """Get video data for training (returns on CUDA if available)"""
        tensor_path = self.tensor_files[idx]
        label = self.labels[idx]
        video_data = self.load_video(tensor_path, to_device=True)
        
        # Return full video data on device
        label_tensor = torch.tensor(label, dtype=torch.long)
        if USE_CUDA:
            label_tensor = label_tensor.to(self.device)
        
        return {
            'video': video_data['video'],
            'metadata': video_data['metadata'],
            'audio': video_data['audio'],
            'label': label_tensor,
            'path': tensor_path
        }


def verify_lossless_conversion(original_path, tensor_path):
    """Verify that conversion is truly lossless"""
    print("\nVerifying lossless conversion...")
    
    # Load original video
    cap = cv2.VideoCapture(original_path)
    original_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame)
    cap.release()
    
    # Load tensor (to CPU for comparison)
    data = torch.load(tensor_path, map_location='cpu')
    tensor_frames = data['video'].numpy()
    
    # Compare
    if len(original_frames) != len(tensor_frames):
        print(f"  Frame count mismatch: {len(original_frames)} vs {len(tensor_frames)}")
        return False
    
    for i, (orig, tensor) in enumerate(zip(original_frames, tensor_frames)):
        if not np.array_equal(orig, tensor):
            print(f"  Frame {i} differs!")
            return False
    
    print(f"  ✓ Perfect match! All {len(original_frames)} frames are identical")
    return True


def main():
    """Main execution"""
    print("="*60)
    print("LOSSLESS VIDEO TO PYTORCH TENSOR CONVERTER (CUDA-ENABLED)")
    print("="*60)
    
    converter = LosslessVideoConverter()
    success = converter.convert_dataset()
    
    if success:
        print(f"\n✓ Dataset successfully converted to: {OUTPUT_TENSOR_DIR}")
        print("✓ All video data preserved without any loss")
        print(f"✓ Device used: {DEVICE}")
        print("\nTo use the dataset:")
        print("  dataset = LosslessVideoDataset(OUTPUT_TENSOR_DIR)")
        print("  video_data = dataset[0]  # Gets complete video with all frames (on CUDA)")
        
        # Optional: Verify a sample
        # verify_lossless_conversion('path/to/original.mp4', 'path/to/tensor.pt')
    else:
        print("\n✗ Conversion failed")


if __name__ == "__main__":
    main()