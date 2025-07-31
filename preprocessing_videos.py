import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import deque

# MediaPipe setup
mp_hands = mp.solutions.hands

class OptimizedHandProcessor:
    """Optimized hand region extraction and processing for VideoMAE"""
    
    def __init__(self, target_size=(224, 224), base_margin=0.3, 
                 confidence_threshold=0.6, smooth_window=5):
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
    
    def detect_hand(self, frame):
        """Detect hand and return bounding box"""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Process all detected hands
        hand_boxes = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Store box with y-position for selection
            cy = (min_y + max_y) / 2
            hand_boxes.append({
                'box': (min_x, min_y, max_x, max_y),
                'y_pos': cy
            })
        
        # Select uppermost hand (typically the signing hand)
        if hand_boxes:
            selected = min(hand_boxes, key=lambda x: x['y_pos'])
            return selected['box']
        
        return None
    
    def crop_and_resize(self, frame, box):
        """Crop frame around hand with margin and resize"""
        if box is None:
            # Use last valid box or center crop
            if self.last_valid_box and self.frames_without_hand < self.max_frames_without_hand:
                self.frames_without_hand += 1
                box = self.last_valid_box
            else:
                # Center crop fallback
                h, w, _ = frame.shape
                size = min(h, w)
                x1 = (w - size) // 2
                y1 = (h - size) // 2
                cropped = frame[y1:y1+size, x1:x1+size]
                return cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        else:
            self.frames_without_hand = 0
            self.last_valid_box = box
        
        h, w, _ = frame.shape
        min_x, min_y, max_x, max_y = box
        
        # Calculate box dimensions
        box_w = max_x - min_x
        box_h = max_y - min_y
        max_dim = max(box_w, box_h)
        
        # Apply margin for context
        margin_size = max_dim * (1 + self.base_margin)
        
        # Center the crop
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        half_size = margin_size / 2
        
        # Calculate crop boundaries
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
        
        # Crop and resize
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if cropped.size == 0:
            cropped = frame
        
        return cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LANCZOS4)
    
    def process_frame(self, frame):
        """Process single frame"""
        # Detect hand
        box = self.detect_hand(frame)
        
        # Apply temporal smoothing
        if box is not None:
            box = self.temporal_smooth(box)
        
        # Crop and resize
        processed = self.crop_and_resize(frame, box)
        
        return processed
    
    def process_video(self, input_path, output_path):
        """Process entire video"""
        cap = cv2.VideoCapture(str(input_path))
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            self.target_size,
            isColor=True
        )
        
        # Process frames
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed = self.process_frame(frame)
            
            # Write frame
            out.write(processed)
            frame_count += 1
        
        cap.release()
        out.release()
        
        # Reset state for next video
        self.history.clear()
        self.ema_box = None
        self.last_valid_box = None
        self.frames_without_hand = 0
        
        return frame_count


def preprocess_dataset(input_root, output_root, target_size=(224, 224)):
    """Main preprocessing function - outputs only videos in class folders"""
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = OptimizedHandProcessor(target_size=target_size)
    
    # Collect all videos
    video_files = list(input_root.glob("**/*.mp4"))
    
    # Organize by class
    classes = {}
    for video_file in video_files:
        class_name = video_file.parent.name
        if class_name not in classes:
            classes[class_name] = []
        classes[class_name].append(video_file)
    
    print(f"Found {len(classes)} gesture classes")
    for class_name, videos in classes.items():
        print(f"  {class_name}: {len(videos)} videos")
    
    # Process videos by class
    total_processed = 0
    
    for class_name, class_videos in classes.items():
        print(f"\nProcessing class: {class_name}")
        
        # Create output class folder
        class_output = output_root / class_name
        class_output.mkdir(parents=True, exist_ok=True)
        
        # Process each video
        for video_file in tqdm(class_videos, desc=f"Class {class_name}"):
            # Maintain filename
            output_path = class_output / video_file.name
            
            # Process video
            frames_processed = processor.process_video(video_file, output_path)
            total_processed += 1
    
    print(f"\n✓ Processing complete!")
    print(f"  Total videos processed: {total_processed}")
    print(f"  Output directory: {output_root}")
    print(f"  All videos are 224x224 RGB format")


if __name__ == "__main__":
    # Configure paths
    INPUT_ROOT = "augmented_videos"  # Your augmented dataset
    OUTPUT_ROOT = "preprocessed_videos"  # Output directory
    
    # Run preprocessing
    preprocess_dataset(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        target_size=(224, 224)
    )