# #!/usr/bin/env python3
# """
# Improved MediaPipe Video Preprocessing Script
# Optimized for Nepali Sign Language gesture recognition on M3 Pro
# """

# import cv2
# import numpy as np
# import mediapipe as mp
# from pathlib import Path
# import json
# from tqdm import tqdm
# import multiprocessing as mp_proc
# from functools import partial
# from datetime import datetime
# import argparse
# import logging
# from collections import deque, defaultdict
# import os
# from typing import List, Dict, Tuple, Optional

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class ImprovedMediaPipeProcessor:
#     """Enhanced MediaPipe processor with better hand tracking and stabilization"""
    
#     def __init__(self, 
#                  confidence=0.3,
#                  hand_padding_ratio=0.35,
#                  use_pose_fallback=True,
#                  stabilize_crops=True,
#                  min_hand_size_ratio=0.05,
#                  prioritize_signing_space=True,
#                  signing_space_bounds=(0.1, 0.7)):
#         """
#         Initialize improved processor
        
#         Args:
#             confidence: Min detection confidence
#             hand_padding_ratio: Padding around hand (0.35 = 35%)
#             use_pose_fallback: Use pose detection when hand detection fails
#             stabilize_crops: Apply temporal stabilization
#             min_hand_size_ratio: Minimum hand size relative to frame
#             prioritize_signing_space: Prioritize hands in upper body signing space
#             signing_space_bounds: (top, bottom) bounds for signing space (0-1 normalized)
#         """
#         self.confidence = confidence
#         self.hand_padding_ratio = hand_padding_ratio
#         self.use_pose_fallback = use_pose_fallback
#         self.stabilize_crops = stabilize_crops
#         self.min_hand_size_ratio = min_hand_size_ratio
#         self.prioritize_signing_space = prioritize_signing_space
#         self.signing_space_bounds = signing_space_bounds
        
#         # Initialize MediaPipe components
#         self.mp_hands = mp.solutions.hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,  # Detect both hands
#             model_complexity=1,
#             min_detection_confidence=confidence,
#             min_tracking_confidence=0.5
#         )
        
#         # Pose detection as fallback
#         if use_pose_fallback:
#             self.mp_pose = mp.solutions.pose.Pose(
#                 static_image_mode=False,
#                 model_complexity=1,
#                 min_detection_confidence=0.5
#             )
        
#         # Stabilization buffer
#         self.crop_history = deque(maxlen=5)
    
#     def process_frames(self, frames, video_path=None):
#         """Process frames with improved detection and stabilization"""
#         detections = []
        
#         # First pass: detect hands/pose in all frames
#         logger.debug(f"Processing {len(frames)} frames from {video_path}")
        
#         for i, frame in enumerate(frames):
#             detection = self._detect_hand_or_pose(frame, frame_idx=i)
#             detections.append(detection)
        
#         # Calculate detection statistics
#         successful_detections = sum(1 for d in detections if d['success'])
#         detection_rate = successful_detections / len(detections)
        
#         # Apply different strategies based on detection rate
#         if detection_rate < 0.3 and self.use_pose_fallback:
#             logger.warning(f"Low hand detection rate ({detection_rate:.1%}), enhancing with pose")
#             detections = self._enhance_with_pose(frames, detections)
        
#         # Apply temporal smoothing
#         if self.stabilize_crops:
#             smoothed_detections = self._advanced_smoothing(detections)
#         else:
#             smoothed_detections = detections
        
#         # Second pass: crop frames
#         cropped_frames = []
#         detection_info = []
        
#         for frame, detection in zip(frames, smoothed_detections):
#             if detection['success']:
#                 cropped = self._intelligent_crop(frame, detection)
#                 detection_info.append({
#                     'detected': True,
#                     'method': detection.get('method', 'hand'),
#                     'confidence': detection.get('confidence', 0),
#                     'bbox': detection.get('crop_bbox', None)
#                 })
#             else:
#                 cropped = self._smart_fallback_crop(frame, detection_info)
#                 detection_info.append({
#                     'detected': False,
#                     'method': 'fallback'
#                 })
            
#             cropped_frames.append(cropped)
        
#         # Post-processing: ensure consistent sizing
#         cropped_frames = self._ensure_consistent_size(cropped_frames)
        
#         return cropped_frames, detection_info, {
#             'detection_rate': detection_rate,
#             'enhanced_rate': sum(1 for d in detection_info if d['detected']) / len(detection_info)
#         }
    
#     def _detect_hand_or_pose(self, frame, frame_idx=0):
#         """Detect hand with pose fallback - prioritize hands in signing space"""
#         h, w = frame.shape[:2]
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Define signing space (upper body region)
#         # Typically hands are used in the area from chest to head level
#         signing_space_top = self.signing_space_bounds[0]  # Top bound
#         signing_space_bottom = self.signing_space_bounds[1]  # Bottom bound
        
#         # Try hand detection first
#         hand_results = self.mp_hands.process(rgb_frame)
        
#         if hand_results.multi_hand_landmarks:
#             # Process all detected hands
#             all_hands_info = []
            
#             for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
#                 x_coords = [lm.x for lm in hand_landmarks.landmark]
#                 y_coords = [lm.y for lm in hand_landmarks.landmark]
                
#                 center_x = np.mean(x_coords)
#                 center_y = np.mean(y_coords)
                
#                 # Calculate hand info
#                 hand_info = {
#                     'min_x': min(x_coords),
#                     'max_x': max(x_coords),
#                     'min_y': min(y_coords),
#                     'max_y': max(y_coords),
#                     'center_x': center_x,
#                     'center_y': center_y,
#                     'in_signing_space': signing_space_top <= center_y <= signing_space_bottom,
#                     'y_position': center_y,
#                     'hand_idx': hand_idx
#                 }
                
#                 # Calculate hand area
#                 hand_width = hand_info['max_x'] - hand_info['min_x']
#                 hand_height = hand_info['max_y'] - hand_info['min_y']
#                 hand_info['area'] = hand_width * hand_height
#                 hand_info['width'] = hand_width
#                 hand_info['height'] = hand_height
                
#                 all_hands_info.append(hand_info)
            
#             # Filter and prioritize hands based on signing space
#             # HAND SELECTION LOGIC:
#             # 1. If BOTH hands visible -> Select hand in upper body signing space
#             # 2. If SINGLE hand visible -> Use it regardless of position
#             # 3. Priority within signing space -> Higher hand (closer to face)
            
#             # Check if we have exactly one hand
#             if len(all_hands_info) == 1:
#                 # Single hand detected - use it regardless of position
#                 selected_hand = all_hands_info[0]
#                 logger.debug(f"Single hand detected at position {selected_hand['y_position']:.2f}")
#             else:
#                 # Multiple hands detected - apply signing space priority
#                 hands_in_signing_space = [h for h in all_hands_info if h['in_signing_space']]
                
#                 if hands_in_signing_space:
#                     # If multiple hands in signing space, choose the higher one (closer to face)
#                     selected_hand = min(hands_in_signing_space, key=lambda h: h['y_position'])
#                     logger.debug(f"Selected hand in signing space from {len(all_hands_info)} hands")
#                 else:
#                     # If no hands in signing space, choose the highest hand
#                     selected_hand = min(all_hands_info, key=lambda h: h['y_position'])
#                     logger.debug(f"No hands in signing space, selected highest hand")
            
#             # Check if hand is too small
#             if selected_hand['area'] < self.min_hand_size_ratio:
#                 # Try to find a larger hand
#                 larger_hands = [h for h in all_hands_info if h['area'] >= self.min_hand_size_ratio]
#                 if larger_hands:
#                     # Among larger hands, still prefer those in signing space
#                     larger_in_space = [h for h in larger_hands if h['in_signing_space']]
#                     if larger_in_space:
#                         selected_hand = min(larger_in_space, key=lambda h: h['y_position'])
#                     else:
#                         selected_hand = min(larger_hands, key=lambda h: h['y_position'])
#                 else:
#                     return {'success': False, 'reason': 'all_hands_too_small'}
            
#             # Calculate confidence based on hand position and size
#             position_score = 1.0 if selected_hand['in_signing_space'] else 0.7
#             size_score = min(1.0, selected_hand['area'] / 0.15)
#             confidence = position_score * size_score
            
#             return {
#                 'success': True,
#                 'method': 'hand',
#                 'center': (selected_hand['center_x'], selected_hand['center_y']),
#                 'size': (selected_hand['width'], selected_hand['height']),
#                 'bbox': (selected_hand['min_x'], selected_hand['min_y'], 
#                          selected_hand['max_x'], selected_hand['max_y']),
#                 'confidence': confidence,
#                 'frame_idx': frame_idx,
#                 'num_hands': len(all_hands_info),
#                 'in_signing_space': selected_hand['in_signing_space'],
#                 'hand_position': 'upper' if selected_hand['center_y'] < 0.4 else 'middle'
#             }
        
#         # Fallback to pose detection
#         if self.use_pose_fallback:
#             pose_results = self.mp_pose.process(rgb_frame)
            
#             if pose_results.pose_landmarks:
#                 # Get pose landmarks
#                 landmarks = pose_results.pose_landmarks.landmark
                
#                 # Get shoulder positions to determine upper body region
#                 right_shoulder = landmarks[12]
#                 left_shoulder = landmarks[11]
#                 avg_shoulder_y = (right_shoulder.y + left_shoulder.y) / 2
                
#                 # Get wrist and elbow positions
#                 right_wrist = landmarks[16]
#                 right_elbow = landmarks[14]
#                 left_wrist = landmarks[15]
#                 left_elbow = landmarks[13]
                
#                 # Evaluate both hands
#                 hands_info = []
                
#                 # Right hand
#                 if right_wrist.visibility > 0.5:
#                     hands_info.append({
#                         'wrist': right_wrist,
#                         'elbow': right_elbow,
#                         'side': 'right',
#                         'in_upper_body': right_wrist.y <= avg_shoulder_y + 0.3  # Allow some margin below shoulders
#                     })
                
#                 # Left hand
#                 if left_wrist.visibility > 0.5:
#                     hands_info.append({
#                         'wrist': left_wrist,
#                         'elbow': left_elbow,
#                         'side': 'left',
#                         'in_upper_body': left_wrist.y <= avg_shoulder_y + 0.3
#                     })
                
#                 if not hands_info:
#                     return {'success': False, 'reason': 'no_visible_hands_in_pose'}
                
#                 # Prioritize hands in upper body region
#                 upper_body_hands = [h for h in hands_info if h['in_upper_body']]
                
#                 if upper_body_hands:
#                     # Choose the hand with better visibility
#                     selected_hand = max(upper_body_hands, key=lambda h: h['wrist'].visibility)
#                 else:
#                     # If no hands in upper body, choose the highest hand
#                     selected_hand = min(hands_info, key=lambda h: h['wrist'].y)
                
#                 wrist = selected_hand['wrist']
#                 elbow = selected_hand['elbow']
                
#                 # Estimate hand region from wrist and elbow
#                 wrist_x, wrist_y = wrist.x, wrist.y
#                 elbow_x, elbow_y = elbow.x, elbow.y
                
#                 # Estimate hand size based on arm length
#                 arm_length = np.sqrt((wrist_x - elbow_x)**2 + (wrist_y - elbow_y)**2)
#                 estimated_hand_size = arm_length * 0.5
                
#                 # Adjust confidence based on position
#                 position_confidence = 1.0 if selected_hand['in_upper_body'] else 0.7
                
#                 return {
#                     'success': True,
#                     'method': 'pose',
#                     'center': (wrist_x, wrist_y),
#                     'size': (estimated_hand_size, estimated_hand_size),
#                     'confidence': wrist.visibility * position_confidence,
#                     'frame_idx': frame_idx,
#                     'side': selected_hand['side'],
#                     'in_upper_body': selected_hand['in_upper_body']
#                 }
        
#         return {'success': False, 'frame_idx': frame_idx}
    
#     def _advanced_smoothing(self, detections):
#         """Apply advanced temporal smoothing"""
#         smoothed = []
        
#         # Find valid detection sequences
#         valid_sequences = []
#         current_sequence = []
        
#         for i, det in enumerate(detections):
#             if det['success']:
#                 current_sequence.append(i)
#             else:
#                 if len(current_sequence) > 0:
#                     valid_sequences.append(current_sequence)
#                     current_sequence = []
        
#         if current_sequence:
#             valid_sequences.append(current_sequence)
        
#         # Process each frame
#         for i, detection in enumerate(detections):
#             if detection['success']:
#                 # Apply exponential moving average for smooth transitions
#                 if i > 0 and smoothed and smoothed[-1]['success']:
#                     alpha = 0.7  # Smoothing factor
#                     prev = smoothed[-1]
                    
#                     smoothed_detection = {
#                         'success': True,
#                         'center': (
#                             alpha * detection['center'][0] + (1-alpha) * prev['center'][0],
#                             alpha * detection['center'][1] + (1-alpha) * prev['center'][1]
#                         ),
#                         'size': (
#                             alpha * detection['size'][0] + (1-alpha) * prev['size'][0],
#                             alpha * detection['size'][1] + (1-alpha) * prev['size'][1]
#                         ),
#                         'confidence': detection['confidence'],
#                         'method': detection.get('method', 'hand')
#                     }
#                     smoothed.append(smoothed_detection)
#                 else:
#                     smoothed.append(detection)
#             else:
#                 # Try to interpolate
#                 interpolated = self._try_interpolate(i, detections, valid_sequences)
#                 smoothed.append(interpolated if interpolated else detection)
        
#         return smoothed
    
#     def _try_interpolate(self, idx, detections, valid_sequences):
#         """Try to interpolate missing detection"""
#         # Find surrounding valid detections
#         for seq in valid_sequences:
#             if seq[0] <= idx <= seq[-1]:
#                 # We're within a valid sequence, interpolate
#                 prev_idx = max([i for i in seq if i < idx], default=None)
#                 next_idx = min([i for i in seq if i > idx], default=None)
                
#                 if prev_idx is not None and next_idx is not None:
#                     prev_det = detections[prev_idx]
#                     next_det = detections[next_idx]
                    
#                     # Linear interpolation
#                     alpha = (idx - prev_idx) / (next_idx - prev_idx)
                    
#                     return {
#                         'success': True,
#                         'center': (
#                             prev_det['center'][0] * (1-alpha) + next_det['center'][0] * alpha,
#                             prev_det['center'][1] * (1-alpha) + next_det['center'][1] * alpha
#                         ),
#                         'size': (
#                             prev_det['size'][0] * (1-alpha) + next_det['size'][0] * alpha,
#                             prev_det['size'][1] * (1-alpha) + next_det['size'][1] * alpha
#                         ),
#                         'confidence': min(prev_det['confidence'], next_det['confidence']) * 0.8,
#                         'method': 'interpolated'
#                     }
        
#         return None
    
#     def _intelligent_crop(self, frame, detection):
#         """Crop with intelligent padding and stabilization"""
#         h, w = frame.shape[:2]
#         cx, cy = detection['center']
        
#         # Convert normalized coordinates to pixels
#         cx_px = int(cx * w)
#         cy_px = int(cy * h)
        
#         # Calculate crop size with adaptive padding
#         hand_w, hand_h = detection['size']
#         hand_size_px = max(hand_w * w, hand_h * h)
        
#         # Adaptive padding based on detection confidence and method
#         if detection.get('method') == 'pose':
#             padding_ratio = self.hand_padding_ratio * 1.5  # More padding for pose
#         elif detection.get('confidence', 1.0) < 0.5:
#             padding_ratio = self.hand_padding_ratio * 1.3  # More padding for low confidence
#         else:
#             padding_ratio = self.hand_padding_ratio
        
#         crop_size = int(hand_size_px * (1 + 2 * padding_ratio))
        
#         # Ensure minimum crop size
#         min_crop = int(min(w, h) * 0.4)
#         crop_size = max(crop_size, min_crop)
        
#         # Calculate crop boundaries
#         x1 = cx_px - crop_size // 2
#         y1 = cy_px - crop_size // 2
#         x2 = x1 + crop_size
#         y2 = y1 + crop_size
        
#         # Apply stabilization
#         if self.stabilize_crops and self.crop_history:
#             # Average recent crop positions
#             recent_crops = list(self.crop_history)
#             avg_x1 = np.mean([c[0] for c in recent_crops])
#             avg_y1 = np.mean([c[1] for c in recent_crops])
            
#             # Smooth transition
#             smooth_factor = 0.3
#             x1 = int(x1 * smooth_factor + avg_x1 * (1 - smooth_factor))
#             y1 = int(y1 * smooth_factor + avg_y1 * (1 - smooth_factor))
#             x2 = x1 + crop_size
#             y2 = y1 + crop_size
        
#         # Ensure crop stays within frame boundaries
#         if x1 < 0:
#             x2 = min(w, x2 - x1)
#             x1 = 0
#         if y1 < 0:
#             y2 = min(h, y2 - y1)
#             y1 = 0
#         if x2 > w:
#             x1 = max(0, x1 - (x2 - w))
#             x2 = w
#         if y2 > h:
#             y1 = max(0, y1 - (y2 - h))
#             y2 = h
        
#         # Store in history
#         self.crop_history.append((x1, y1, x2, y2))
        
#         # Store crop bbox for reference
#         detection['crop_bbox'] = [x1, y1, x2, y2]
        
#         # Crop and resize
#         cropped = frame[y1:y2, x1:x2]
        
#         return cropped
    
#     def _smart_fallback_crop(self, frame, previous_detections):
#         """Smart fallback when no hand detected"""
#         h, w = frame.shape[:2]
        
#         # If we have recent successful detections, use their position
#         if previous_detections:
#             recent_detected = [d for d in previous_detections[-5:] if d.get('detected', False)]
#             if recent_detected:
#                 # Use the last known position with slight expansion
#                 last_bbox = recent_detected[-1].get('bbox')
#                 if last_bbox:
#                     x1, y1, x2, y2 = last_bbox
                    
#                     # Expand slightly in case hand moved
#                     expansion = 0.1
#                     width = x2 - x1
#                     height = y2 - y1
                    
#                     x1 = max(0, int(x1 - width * expansion))
#                     y1 = max(0, int(y1 - height * expansion))
#                     x2 = min(w, int(x2 + width * expansion))
#                     y2 = min(h, int(y2 + height * expansion))
                    
#                     return frame[y1:y2, x1:x2]
        
#         # Default center crop
#         crop_ratio = 0.7
#         new_h = int(h * crop_ratio)
#         new_w = int(w * crop_ratio)
        
#         y_start = (h - new_h) // 2
#         x_start = (w - new_w) // 2
        
#         return frame[y_start:y_start+new_h, x_start:x_start+new_w]
    
#     def _ensure_consistent_size(self, frames, target_size=(224, 224)):
#         """Ensure all frames have consistent size"""
#         resized_frames = []
        
#         for frame in frames:
#             if frame.shape[:2] != target_size:
#                 # Use high-quality resizing
#                 resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
#                 resized_frames.append(resized)
#             else:
#                 resized_frames.append(frame)
        
#         return resized_frames
    
#     def _enhance_with_pose(self, frames, detections):
#         """Enhance detections using pose information"""
#         enhanced_detections = []
        
#         for frame, detection in zip(frames, detections):
#             if not detection['success']:
#                 # Try pose detection
#                 pose_detection = self._detect_hand_or_pose(frame, frame_idx=detection.get('frame_idx', 0))
#                 if pose_detection['success'] and pose_detection.get('method') == 'pose':
#                     enhanced_detections.append(pose_detection)
#                 else:
#                     enhanced_detections.append(detection)
#             else:
#                 enhanced_detections.append(detection)
        
#         return enhanced_detections
    
#     def close(self):
#         """Clean up resources"""
#         self.mp_hands.close()
#         if hasattr(self, 'mp_pose'):
#             self.mp_pose.close()

# def load_video_frames_adaptive(video_path, target_frames=16, gesture_type=None):
#     """
#     Load frames with adaptive sampling based on gesture characteristics
    
#     Args:
#         video_path: Path to video file
#         target_frames: Number of frames to extract
#         gesture_type: Optional gesture type for optimized sampling
#     """
#     cap = cv2.VideoCapture(str(video_path))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     if total_frames == 0:
#         cap.release()
#         return None, None
    
#     # Calculate video duration
#     duration = total_frames / fps if fps > 0 else 0
    
#     # Load all frames first
#     all_frames = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         all_frames.append(frame)
    
#     cap.release()
    
#     if len(all_frames) == 0:
#         return None, None
    
#     # Adaptive sampling based on video characteristics
#     if duration < 1.0:  # Very short video
#         # Take all frames for short videos
#         indices = list(range(len(all_frames)))
#     elif len(all_frames) <= target_frames:
#         # Take all frames if we have fewer than target
#         indices = list(range(len(all_frames)))
#     else:
#         # Smart sampling for longer videos
#         indices = get_adaptive_sampling_indices(
#             len(all_frames), 
#             target_frames,
#             video_duration=duration,
#             gesture_type=gesture_type
#         )
    
#     # Select frames
#     selected_frames = [all_frames[i] for i in indices]
    
#     # Pad if necessary
#     while len(selected_frames) < target_frames:
#         if selected_frames:
#             selected_frames.append(selected_frames[-1])
#         else:
#             # Create black frame
#             selected_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
    
#     metadata = {
#         'original_frames': total_frames,
#         'fps': fps,
#         'duration': duration,
#         'selected_indices': indices[:target_frames]
#     }
    
#     return selected_frames[:target_frames], metadata

# def get_adaptive_sampling_indices(total_frames, target_frames, video_duration=None, gesture_type=None):
#     """
#     Get frame indices with adaptive sampling
    
#     Different sampling strategies for different types of gestures:
#     - Static gestures: More uniform sampling
#     - Dynamic gestures: Focus on middle portion
#     - Repetitive gestures: Sample full cycles
#     """
    
#     # Default importance sampling for sign language
#     # Sign language gestures typically have:
#     # - Setup phase (hand positioning)
#     # - Main gesture phase
#     # - Hold/completion phase
    
#     if video_duration and video_duration > 3:
#         # For longer videos, skip more of the beginning and end
#         start_ratio = 0.15
#         middle_ratio = 0.7
#         end_ratio = 0.15
#     else:
#         # Standard distribution
#         start_ratio = 0.2
#         middle_ratio = 0.6
#         end_ratio = 0.2
    
#     start_count = max(2, int(target_frames * start_ratio))
#     middle_count = max(8, int(target_frames * middle_ratio))
#     end_count = target_frames - start_count - middle_count
    
#     indices = []
    
#     # Start frames (setup phase)
#     start_end = int(total_frames * 0.25)
#     indices.extend(np.linspace(0, start_end, start_count, dtype=int))
    
#     # Middle frames (main gesture - denser sampling)
#     middle_start = int(total_frames * 0.25)
#     middle_end = int(total_frames * 0.75)
#     indices.extend(np.linspace(middle_start, middle_end, middle_count, dtype=int))
    
#     # End frames (completion phase)
#     end_start = int(total_frames * 0.75)
#     indices.extend(np.linspace(end_start, total_frames - 1, end_count, dtype=int))
    
#     # Remove duplicates and sort
#     indices = sorted(set(indices))
    
#     # Ensure we have exactly target_frames
#     if len(indices) > target_frames:
#         # Subsample to target
#         step = len(indices) / target_frames
#         indices = [indices[int(i * step)] for i in range(target_frames)]
#     elif len(indices) < target_frames:
#         # Add more frames from middle section
#         middle_indices = list(range(middle_start, middle_end))
#         additional = np.random.choice(
#             middle_indices, 
#             size=target_frames - len(indices), 
#             replace=False
#         )
#         indices.extend(additional)
#         indices = sorted(indices)
    
#     return indices[:target_frames]

# def process_single_video_improved(args):
#     """Process a single video with improved error handling"""
#     video_info, config = args
#     input_path = video_info['input']
#     output_path = video_info['output']
#     gesture_type = video_info.get('gesture_type', None)
    
#     # Skip if already processed (unless forced)
#     if output_path.exists() and not config.get('force_reprocess', False):
#         # Verify the file is valid
#         try:
#             data = np.load(output_path)
#             if 'frames' in data and len(data['frames']) == config['num_frames']:
#                 return {
#                     'input': str(input_path),
#                     'output': str(output_path),
#                     'success': True,
#                     'cached': True
#                 }
#         except:
#             # Corrupted file, reprocess
#             logger.warning(f"Corrupted preprocessed file: {output_path}, reprocessing...")
    
#     try:
#         # Load video frames with adaptive sampling
#         frames, video_metadata = load_video_frames_adaptive(
#             input_path, 
#             config['num_frames'],
#             gesture_type=gesture_type
#         )
        
#         if frames is None:
#             return {
#                 'input': str(input_path),
#                 'output': str(output_path),
#                 'success': False,
#                 'error': 'Failed to load video'
#             }
        
#         # Process with MediaPipe
#         processor = ImprovedMediaPipeProcessor(
#             confidence=config['mediapipe_confidence'],
#             hand_padding_ratio=config['hand_padding_ratio'],
#             use_pose_fallback=config.get('use_pose_fallback', True),
#             stabilize_crops=config.get('stabilize_crops', True),
#             prioritize_signing_space=config.get('prioritize_signing_space', True),
#             signing_space_bounds=config.get('signing_space_bounds', (0.1, 0.7))
#         )
        
#         cropped_frames, detection_info, stats = processor.process_frames(frames, video_path=input_path)
#         processor.close()
        
#         # Resize frames to target size
#         target_size = config['frame_size']
#         resized_frames = []
#         for frame in cropped_frames:
#             resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
#             resized_frames.append(resized)
        
#         # Convert to numpy array
#         frames_array = np.array(resized_frames, dtype=np.uint8)
        
#         # Save as compressed npz file with metadata
#         np.savez_compressed(
#             output_path,
#             frames=frames_array,
#             detection_info=detection_info,
#             metadata={
#                 'original_video': str(input_path),
#                 'num_frames': len(frames_array),
#                 'frame_size': config['frame_size'],
#                 'detection_rate': stats['detection_rate'],
#                 'enhanced_rate': stats['enhanced_rate'],
#                 'video_metadata': video_metadata,
#                 'processed_date': datetime.now().isoformat(),
#                 'processor_version': '2.0'
#             }
#         )
        
#         return {
#             'input': str(input_path),
#             'output': str(output_path),
#             'success': True,
#             'shape': frames_array.shape,
#             'detection_rate': stats['detection_rate'],
#             'enhanced_rate': stats['enhanced_rate']
#         }
        
#     except Exception as e:
#         logger.error(f"Error processing {input_path}: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return {
#             'input': str(input_path),
#             'output': str(output_path),
#             'success': False,
#             'error': str(e)
#         }

# def preprocess_dataset(input_dir, output_dir, config):
#     """Preprocess entire dataset with improved processing"""
#     input_path = Path(input_dir)
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)
    
#     # Validate input directory
#     if not input_path.exists():
#         raise ValueError(f"Input directory {input_path} does not exist!")
    
#     # Collect all videos
#     all_videos = []
#     class_stats = {}
    
#     for class_dir in sorted(input_path.iterdir()):
#         if not class_dir.is_dir() or class_dir.name.startswith('.'):
#             continue
        
#         class_name = class_dir.name
#         class_output = output_path / class_name
#         class_output.mkdir(exist_ok=True)
        
#         # Find all video files
#         video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MOV', '*.mkv', '*.webm']
#         video_files = []
#         for ext in video_extensions:
#             video_files.extend(list(class_dir.glob(ext)))
        
#         class_stats[class_name] = len(video_files)
        
#         for video_file in video_files:
#             output_file = class_output / f"{video_file.stem}_processed.npz"
#             all_videos.append({
#                 'input': video_file,
#                 'output': output_file,
#                 'class': class_name,
#                 'gesture_type': class_name  # Can be used for gesture-specific processing
#             })
    
#     logger.info(f"\nDataset summary:")
#     logger.info(f"Total classes: {len(class_stats)}")
#     logger.info(f"Total videos: {len(all_videos)}")
#     for class_name, count in class_stats.items():
#         logger.info(f"  {class_name}: {count} videos")
    
#     # Process videos
#     if config['num_workers'] > 1:
#         # Multiprocessing for faster processing
#         with mp_proc.Pool(config['num_workers']) as pool:
#             args = [(video_info, config) for video_info in all_videos]
            
#             results = list(tqdm(
#                 pool.imap(process_single_video_improved, args),
#                 total=len(all_videos),
#                 desc="Processing videos"
#             ))
#     else:
#         # Single process (easier to debug)
#         results = []
#         for video_info in tqdm(all_videos, desc="Processing videos"):
#             result = process_single_video_improved((video_info, config))
#             results.append(result)
    
#     # Calculate statistics
#     successful = sum(1 for r in results if r['success'])
#     failed = len(results) - successful
#     cached = sum(1 for r in results if r.get('cached', False))
    
#     detection_rates = [r['detection_rate'] for r in results 
#                       if r['success'] and 'detection_rate' in r]
#     enhanced_rates = [r.get('enhanced_rate', r['detection_rate']) for r in results 
#                      if r['success'] and 'detection_rate' in r]
    
#     avg_detection_rate = np.mean(detection_rates) if detection_rates else 0
#     avg_enhanced_rate = np.mean(enhanced_rates) if enhanced_rates else 0
    
#     # Per-class statistics
#     class_results = defaultdict(lambda: {'total': 0, 'successful': 0, 'failed': 0, 'detection_rates': []})
    
#     for video_info, result in zip(all_videos, results):
#         class_name = video_info['class']
#         class_results[class_name]['total'] += 1
        
#         if result['success']:
#             class_results[class_name]['successful'] += 1
#             if 'detection_rate' in result:
#                 class_results[class_name]['detection_rates'].append(result['detection_rate'])
#         else:
#             class_results[class_name]['failed'] += 1
    
#     # Calculate per-class average detection rates
#     for class_name, stats in class_results.items():
#         if stats['detection_rates']:
#             stats['avg_detection_rate'] = np.mean(stats['detection_rates'])
#         else:
#             stats['avg_detection_rate'] = 0.0
    
#     # Save processing summary
#     summary = {
#         'input_dir': str(input_dir),
#         'output_dir': str(output_dir),
#         'total_videos': len(all_videos),
#         'successful': successful,
#         'failed': failed,
#         'cached': cached,
#         'average_detection_rate': float(avg_detection_rate),
#         'average_enhanced_rate': float(avg_enhanced_rate),
#         'improvement': float(avg_enhanced_rate - avg_detection_rate) if detection_rates else 0,
#         'config': config,
#         'class_statistics': dict(class_results),
#         'processed_date': datetime.now().isoformat(),
#         'failed_videos': [r['input'] for r in results if not r['success']]
#     }
    
#     summary_path = output_path / 'preprocessing_summary.json'
#     with open(summary_path, 'w') as f:
#         json.dump(summary, f, indent=2)
    
#     # Print detailed summary
#     logger.info(f"\n{'='*60}")
#     logger.info(f"PREPROCESSING COMPLETE!")
#     logger.info(f"{'='*60}")
#     logger.info(f"Total videos: {len(all_videos)}")
#     logger.info(f"Successful: {successful} ({successful/len(all_videos)*100:.1f}%)")
#     logger.info(f"Failed: {failed}")
#     logger.info(f"Cached: {cached}")
#     logger.info(f"Average detection rate: {avg_detection_rate:.1%}")
#     logger.info(f"Average enhanced rate: {avg_enhanced_rate:.1%}")
#     logger.info(f"Improvement: +{(avg_enhanced_rate - avg_detection_rate)*100:.1f}%")
    
#     logger.info(f"\nPer-class results:")
#     for class_name, stats in sorted(class_results.items()):
#         logger.info(f"  {class_name}: {stats['successful']}/{stats['total']} "
#                    f"(detection: {stats['avg_detection_rate']:.1%})")
    
#     if summary['failed_videos']:
#         logger.warning(f"\nFailed videos:")
#         for video in summary['failed_videos'][:10]:  # Show first 10
#             logger.warning(f"  - {video}")
#         if len(summary['failed_videos']) > 10:
#             logger.warning(f"  ... and {len(summary['failed_videos']) - 10} more")
    
#     logger.info(f"\nOutput directory: {output_path}")
    
#     return output_path

# def validate_preprocessing(output_dir, sample_size=5):
#     """Validate preprocessed data quality"""
#     output_path = Path(output_dir)
    
#     logger.info("\nValidating preprocessed data...")
    
#     issues = []
    
#     # Check each class
#     for class_dir in output_path.iterdir():
#         if not class_dir.is_dir() or class_dir.name.startswith('.'):
#             continue
        
#         npz_files = list(class_dir.glob('*.npz'))
        
#         if not npz_files:
#             issues.append(f"No preprocessed files in {class_dir.name}")
#             continue
        
#         # Sample validation
#         sample_files = np.random.choice(
#             npz_files, 
#             size=min(sample_size, len(npz_files)), 
#             replace=False
#         )
        
#         for npz_file in sample_files:
#             try:
#                 # Load with allow_pickle=True since we store metadata as objects
#                 data = np.load(npz_file, allow_pickle=True)
#                 frames = data['frames']
                
#                 # Check shape
#                 if frames.shape[0] != 16:  # Expected number of frames
#                     issues.append(f"{npz_file.name}: Wrong number of frames {frames.shape[0]}")
                
#                 if frames.shape[1:3] != (224, 224):  # Expected frame size
#                     issues.append(f"{npz_file.name}: Wrong frame size {frames.shape[1:3]}")
                
#                 # Check data range
#                 if frames.min() < 0 or frames.max() > 255:
#                     issues.append(f"{npz_file.name}: Invalid pixel values")
                
#                 # Check metadata
#                 if 'metadata' in data:
#                     metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
#                     if isinstance(metadata, dict):
#                         if metadata.get('detection_rate', 1.0) < 0.1:
#                             issues.append(f"{npz_file.name}: Very low detection rate ({metadata['detection_rate']:.1%})")
                        
#                         # Check if hand was in signing space
#                         if 'enhanced_rate' in metadata and metadata['enhanced_rate'] < 0.3:
#                             issues.append(f"{npz_file.name}: Low enhanced detection rate ({metadata['enhanced_rate']:.1%})")
                
#             except Exception as e:
#                 issues.append(f"{npz_file.name}: Error loading - {str(e)}")
    
#     if issues:
#         logger.warning(f"\nValidation issues found:")
#         for issue in issues[:20]:  # Show first 20 issues
#             logger.warning(f"  - {issue}")
#         if len(issues) > 20:
#             logger.warning(f"  ... and {len(issues) - 20} more issues")
#     else:
#         logger.info("✓ All validation checks passed!")
    
#     return len(issues) == 0

# def main():
#     """Main preprocessing function"""
    
#     # Configuration
#     config = {
#         'input_dir':'splitted_dataset/train',
#         'output_dir': 'mediapipe_preprocessed',
#         'num_frames': 16,
#         'frame_size': tuple([224, 224]),
#         'mediapipe_confidence': 0.3,
#         'hand_padding_ratio': 0.35,
#         'num_workers': 4,
#         'force_reprocess': True,
#         'use_pose_fallback': True,
#         'stabilize_crops': True,
#         'validate': True,
#         'prioritize_signing_space': True,
#         'signing_space_bounds': (0.1, 0.7),
#     }
    
#     logger.info("Improved MediaPipe Preprocessing")
#     logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
#     # Run preprocessing
#     output_path = preprocess_dataset(config["input_dir"], config["output_dir"], config)
    
#     # Validate if requested
#     if config["validate"]:
#         is_valid = validate_preprocessing(output_path)
#         if not is_valid:
#             logger.warning("\n⚠️  Some validation issues were found. Please review the warnings above.")
#         else:
#             logger.info("\n✅ All preprocessing completed successfully!")

# if __name__ == "__main__":
#     main()








































































#!/usr/bin/env python3
"""
Sign Language Video Augmentation Pipeline
Applies non-spatial augmentations to preserve gesture meanings
Optimized for GPU execution with thread-safe operations
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import multiprocessing as mp_proc
from datetime import datetime
import argparse
import logging
import os
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class VideoAugmenter:
    """
    Video augmentation pipeline with anti-overfitting properties
    Applies various non-spatial augmentations to preserve sign language meanings
    """
    
    def __init__(self, 
                 augmentation_probability=0.8,
                 max_augmentations_per_video=3,
                 randomize_params=True,
                 preserve_original=True):
        """
        Initialize augmenter with anti-overfitting strategies
        
        Args:
            augmentation_probability: Overall probability of applying augmentations
            max_augmentations_per_video: Maximum number of augmentations to combine
            randomize_params: Whether to randomize augmentation parameters
            preserve_original: Whether to keep original video along with augmented versions
        """
        self.augmentation_probability = augmentation_probability
        self.max_augmentations_per_video = max_augmentations_per_video
        self.randomize_params = randomize_params
        self.preserve_original = preserve_original
        
        # Define augmentation functions with their parameter ranges
        self.augmentations = {
            'bright_high': self.apply_brightness,
            'bright_low': self.apply_brightness,
            'contrast_high': self.apply_contrast,
            'contrast_low': self.apply_contrast,
            'saturation_high': self.apply_saturation,
            'saturation_low': self.apply_saturation,
            'blur_motion': self.apply_motion_blur,
            'blur_gaussian': self.apply_gaussian_blur,
            'noise_moderate': self.apply_gaussian_noise,
            'noise_salt_pepper': self.apply_salt_pepper_noise,
            'gamma': self.apply_gamma_correction
        }
        
        # Parameter ranges for each augmentation
        self.param_ranges = {
            'bright_high': {'factor': (1.2, 1.5)},
            'bright_low': {'factor': (0.5, 0.8)},
            'contrast_high': {'factor': (1.2, 1.8)},
            'contrast_low': {'factor': (0.5, 0.8)},
            'saturation_high': {'factor': (1.3, 2.0)},
            'saturation_low': {'factor': (0.3, 0.7)},
            'blur_motion': {'kernel_size': (5, 15), 'angle': (0, 360)},
            'blur_gaussian': {'kernel_size': (3, 9), 'sigma': (0.5, 2.0)},
            'noise_moderate': {'var': (0.001, 0.01)},
            'noise_salt_pepper': {'prob': (0.01, 0.05)},
            'gamma': {'gamma': (0.5, 2.0)}
        }
    
    def apply_brightness(self, frame, factor=None):
        """Apply brightness adjustment"""
        if factor is None:
            factor = np.random.uniform(0.5, 1.5)
        
        # Convert to float for processing
        frame_float = frame.astype(np.float32)
        adjusted = cv2.convertScaleAbs(frame_float, alpha=factor, beta=0)
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def apply_contrast(self, frame, factor=None):
        """Apply contrast adjustment"""
        if factor is None:
            factor = np.random.uniform(0.5, 1.8)
        
        # Calculate mean for contrast adjustment
        mean = np.mean(frame, axis=(0, 1), keepdims=True)
        adjusted = mean + factor * (frame.astype(np.float32) - mean)
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def apply_saturation(self, frame, factor=None):
        """Apply saturation adjustment"""
        if factor is None:
            factor = np.random.uniform(0.3, 2.0)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def apply_motion_blur(self, frame, kernel_size=None, angle=None):
        """Apply motion blur"""
        if kernel_size is None:
            kernel_size = np.random.choice([5, 7, 9, 11])
        if angle is None:
            angle = np.random.uniform(0, 360)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        
        return cv2.filter2D(frame, -1, kernel)
    
    def apply_gaussian_blur(self, frame, kernel_size=None, sigma=None):
        """Apply Gaussian blur"""
        if kernel_size is None:
            kernel_size = np.random.choice([3, 5, 7])
        if sigma is None:
            sigma = np.random.uniform(0.5, 2.0)
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    
    def apply_gaussian_noise(self, frame, var=None):
        """Apply Gaussian noise"""
        if var is None:
            var = np.random.uniform(0.001, 0.01)
        
        noise = np.random.normal(0, var ** 0.5 * 255, frame.shape)
        noisy = frame.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def apply_salt_pepper_noise(self, frame, prob=None):
        """Apply salt and pepper noise"""
        if prob is None:
            prob = np.random.uniform(0.01, 0.05)
        
        output = frame.copy()
        noise = np.random.random(frame.shape[:2])
        
        # Salt noise (white pixels)
        output[noise < prob / 2] = 255
        
        # Pepper noise (black pixels)
        output[noise > 1 - prob / 2] = 0
        
        return output
    
    def apply_gamma_correction(self, frame, gamma=None):
        """Apply gamma correction"""
        if gamma is None:
            gamma = np.random.uniform(0.5, 2.0)
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        
        return cv2.LUT(frame, table)
    
    def get_random_augmentation_params(self, aug_name):
        """Get randomized parameters for an augmentation"""
        if not self.randomize_params:
            # Use default middle values
            params = {}
            for param, range_val in self.param_ranges[aug_name].items():
                if isinstance(range_val, tuple):
                    params[param] = np.mean(range_val)
                else:
                    params[param] = range_val
            return params
        
        # Randomize parameters within ranges
        params = {}
        for param, range_val in self.param_ranges[aug_name].items():
            if isinstance(range_val, tuple):
                if param in ['kernel_size']:
                    # For kernel sizes, use odd integers
                    choices = list(range(int(range_val[0]), int(range_val[1]) + 1, 2))
                    if not choices:
                        choices = [int(range_val[0])]
                    params[param] = np.random.choice(choices)
                else:
                    params[param] = np.random.uniform(range_val[0], range_val[1])
            else:
                params[param] = range_val
        
        return params
    
    def augment_video_frames(self, frames, video_path=None):
        """
        Apply augmentations to video frames with anti-overfitting strategies
        
        Returns multiple augmented versions of the video
        """
        augmented_versions = []
        
        # Always include original if specified
        if self.preserve_original:
            augmented_versions.append({
                'frames': frames.copy(),
                'augmentations': [],
                'params': {}
            })
        
        # Decide whether to augment
        if np.random.random() > self.augmentation_probability:
            return augmented_versions
        
        # Select random augmentations
        available_augs = list(self.augmentations.keys())
        num_augmentations = np.random.randint(1, min(self.max_augmentations_per_video + 1, len(available_augs) + 1))
        
        # Create different augmentation combinations
        num_versions = np.random.randint(1, 4)  # Create 1-3 augmented versions
        
        for version in range(num_versions):
            # Randomly select augmentations for this version
            selected_augs = np.random.choice(available_augs, size=num_augmentations, replace=False)
            
            # Apply augmentations
            augmented_frames = frames.copy()
            applied_augs = []
            applied_params = {}
            
            for aug_name in selected_augs:
                # Get augmentation function
                aug_func = self.augmentations[aug_name]
                
                # Get parameters
                params = self.get_random_augmentation_params(aug_name)
                
                # Apply to all frames
                for i in range(len(augmented_frames)):
                    # Apply with slight frame-to-frame variation for realism
                    frame_params = params.copy()
                    if self.randomize_params and np.random.random() < 0.1:
                        # 10% chance to slightly vary parameters between frames
                        for key in frame_params:
                            if isinstance(frame_params[key], (int, float)):
                                frame_params[key] *= np.random.uniform(0.95, 1.05)
                    
                    augmented_frames[i] = aug_func(augmented_frames[i], **frame_params)
                
                applied_augs.append(aug_name)
                applied_params[aug_name] = params
            
            augmented_versions.append({
                'frames': augmented_frames,
                'augmentations': applied_augs,
                'params': applied_params
            })
        
        return augmented_versions

def load_video_frames(video_path, target_frames=16):
    """
    Load video frames with uniform sampling
    No cropping or spatial manipulation
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        cap.release()
        return None, None
    
    # Calculate video duration
    duration = total_frames / fps if fps > 0 else 0
    
    # Load all frames first
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    
    cap.release()
    
    if len(all_frames) == 0:
        return None, None
    
    # Uniform sampling
    if len(all_frames) <= target_frames:
        # Take all frames if we have fewer than target
        indices = list(range(len(all_frames)))
    else:
        # Uniform sampling for longer videos
        indices = np.linspace(0, len(all_frames) - 1, target_frames, dtype=int)
    
    # Select frames
    selected_frames = [all_frames[i] for i in indices]
    
    # Pad if necessary
    while len(selected_frames) < target_frames:
        if selected_frames:
            selected_frames.append(selected_frames[-1])
        else:
            # Create black frame with standard size
            selected_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
    
    metadata = {
        'original_frames': total_frames,
        'fps': fps,
        'duration': duration,
        'selected_indices': indices[:target_frames].tolist(),
        'original_shape': all_frames[0].shape if all_frames else (480, 640, 3)
    }
    
    return selected_frames[:target_frames], metadata

def process_single_video_with_augmentation(args):
    """Process a single video with augmentations"""
    video_info, config = args
    input_path = video_info['input']
    output_base = video_info['output_base']
    gesture_class = video_info.get('class', 'unknown')
    
    try:
        # Load video frames
        frames, video_metadata = load_video_frames(
            input_path, 
            config['num_frames']
        )
        
        if frames is None:
            return [{
                'input': str(input_path),
                'output': None,
                'success': False,
                'error': 'Failed to load video'
            }]
        
        # Convert frames to numpy array
        frames_array = np.array(frames, dtype=np.uint8)
        
        # Create augmenter
        augmenter = VideoAugmenter(
            augmentation_probability=config.get('augmentation_probability', 0.8),
            max_augmentations_per_video=config.get('max_augmentations_per_video', 3),
            randomize_params=config.get('randomize_params', True),
            preserve_original=config.get('preserve_original', True)
        )
        
        # Generate augmented versions
        augmented_versions = augmenter.augment_video_frames(frames_array, video_path=input_path)
        
        results = []
        
        # Save each augmented version
        for idx, aug_data in enumerate(augmented_versions):
            if idx == 0 and config.get('preserve_original', True):
                # Original version
                output_path = Path(str(output_base) + '_original.npz')
                aug_suffix = 'original'
            else:
                # Augmented version
                aug_names = '_'.join(aug_data['augmentations'][:2]) if aug_data['augmentations'] else 'aug'
                output_path = Path(str(output_base) + f'_aug{idx}_{aug_names}.npz')
                aug_suffix = f'aug{idx}'
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as compressed npz file with metadata
            np.savez_compressed(
                output_path,
                frames=aug_data['frames'],
                augmentations=aug_data['augmentations'],
                augmentation_params=aug_data['params'],
                metadata={
                    'original_video': str(input_path),
                    'num_frames': len(aug_data['frames']),
                    'frame_shape': aug_data['frames'][0].shape,
                    'video_metadata': video_metadata,
                    'processed_date': datetime.now().isoformat(),
                    'augmentation_version': aug_suffix,
                    'class': gesture_class
                }
            )
            
            results.append({
                'input': str(input_path),
                'output': str(output_path),
                'success': True,
                'shape': aug_data['frames'].shape,
                'augmentations': aug_data['augmentations'],
                'version': aug_suffix
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return [{
            'input': str(input_path),
            'output': None,
            'success': False,
            'error': str(e)
        }]

def preprocess_dataset_with_augmentation(input_dir, output_dir, config):
    """Preprocess dataset with augmentations"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Validate input directory
    if not input_path.exists():
        raise ValueError(f"Input directory {input_path} does not exist!")
    
    # Collect all videos
    all_videos = []
    class_stats = {}
    
    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        
        class_name = class_dir.name
        class_output = output_path / class_name
        class_output.mkdir(exist_ok=True)
        
        # Find all video files
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MOV', '*.mkv', '*.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(class_dir.glob(ext)))
        
        class_stats[class_name] = len(video_files)
        
        for video_file in video_files:
            output_base = class_output / video_file.stem
            all_videos.append({
                'input': video_file,
                'output_base': output_base,
                'class': class_name
            })
    
    logger.info(f"\nDataset summary:")
    logger.info(f"Total classes: {len(class_stats)}")
    logger.info(f"Total videos: {len(all_videos)}")
    for class_name, count in class_stats.items():
        logger.info(f"  {class_name}: {count} videos")
    
    # Process videos with augmentations
    # Use single process to avoid GPU/thread conflicts
    all_results = []
    
    for video_info in tqdm(all_videos, desc="Processing videos with augmentations"):
        results = process_single_video_with_augmentation((video_info, config))
        all_results.extend(results)
    
    # Calculate statistics
    successful = sum(1 for r in all_results if r['success'])
    failed = sum(1 for r in all_results if not r['success'])
    original_count = sum(1 for r in all_results if r['success'] and r.get('version') == 'original')
    augmented_count = successful - original_count
    
    # Per-class statistics
    class_results = defaultdict(lambda: {
        'total_videos': 0, 
        'total_generated': 0,
        'original': 0,
        'augmented': 0,
        'failed': 0,
        'augmentation_types': defaultdict(int)
    })
    
    for video_info in all_videos:
        class_name = video_info['class']
        class_results[class_name]['total_videos'] += 1
    
    for result in all_results:
        # Find class from input path
        input_path = Path(result['input'])
        class_name = input_path.parent.name
        
        if result['success']:
            class_results[class_name]['total_generated'] += 1
            if result.get('version') == 'original':
                class_results[class_name]['original'] += 1
            else:
                class_results[class_name]['augmented'] += 1
                for aug in result.get('augmentations', []):
                    class_results[class_name]['augmentation_types'][aug] += 1
        else:
            class_results[class_name]['failed'] += 1
    
    # Save processing summary
    summary = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'total_original_videos': len(all_videos),
        'total_generated_files': len(all_results),
        'successful': successful,
        'failed': failed,
        'original_preserved': original_count,
        'augmented_generated': augmented_count,
        'augmentation_factor': successful / len(all_videos) if all_videos else 0,
        'config': config,
        'class_statistics': {k: dict(v) for k, v in class_results.items()},
        'processed_date': datetime.now().isoformat()
    }
    
    # Convert defaultdict to regular dict for JSON serialization
    for class_name in summary['class_statistics']:
        if 'augmentation_types' in summary['class_statistics'][class_name]:
            summary['class_statistics'][class_name]['augmentation_types'] = dict(
                summary['class_statistics'][class_name]['augmentation_types']
            )
    
    summary_path = output_path / 'augmentation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print detailed summary
    logger.info(f"\n{'='*60}")
    logger.info(f"AUGMENTATION COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Original videos: {len(all_videos)}")
    logger.info(f"Total generated: {successful}")
    logger.info(f"Augmentation factor: {successful/len(all_videos):.1f}x")
    logger.info(f"Failed: {failed}")
    
    logger.info(f"\nPer-class results:")
    for class_name, stats in sorted(class_results.items()):
        logger.info(f"  {class_name}: {stats['total_videos']} videos → {stats['total_generated']} files "
                   f"({stats['original']} original + {stats['augmented']} augmented)")
    
    logger.info(f"\nAugmentation distribution:")
    aug_totals = defaultdict(int)
    for stats in class_results.values():
        for aug_type, count in stats['augmentation_types'].items():
            aug_totals[aug_type] += count
    
    for aug_type, count in sorted(aug_totals.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {aug_type}: {count} applications")
    
    logger.info(f"\nOutput directory: {output_path}")
    
    return output_path

def validate_augmented_data(output_dir, sample_size=5):
    """Validate augmented data quality"""
    output_path = Path(output_dir)
    
    logger.info("\nValidating augmented data...")
    
    issues = []
    augmentation_stats = defaultdict(int)
    
    # Check each class
    for class_dir in output_path.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        
        npz_files = list(class_dir.glob('*.npz'))
        
        if not npz_files:
            issues.append(f"No preprocessed files in {class_dir.name}")
            continue
        
        # Sample validation
        sample_files = np.random.choice(
            npz_files, 
            size=min(sample_size, len(npz_files)), 
            replace=False
        )
        
        for npz_file in sample_files:
            try:
                # Load with allow_pickle=True for metadata
                data = np.load(npz_file, allow_pickle=True)
                frames = data['frames']
                
                # Check shape
                if frames.shape[0] != 16:  # Expected number of frames
                    issues.append(f"{npz_file.name}: Wrong number of frames {frames.shape[0]}")
                
                # Check data range
                if frames.min() < 0 or frames.max() > 255:
                    issues.append(f"{npz_file.name}: Invalid pixel values")
                
                # Check augmentations
                if 'augmentations' in data:
                    augs = data['augmentations']
                    if hasattr(augs, 'tolist'):
                        augs = augs.tolist()
                    for aug in augs:
                        augmentation_stats[aug] += 1
                
                # Verify frames are different from each other (not all identical)
                if len(frames) > 1:
                    frame_diff = np.std([np.mean(frames[i] - frames[0]) for i in range(1, min(5, len(frames)))])
                    if frame_diff < 0.01:
                        issues.append(f"{npz_file.name}: Frames appear to be identical")
                
            except Exception as e:
                issues.append(f"{npz_file.name}: Error loading - {str(e)}")
    
    if issues:
        logger.warning(f"\nValidation issues found:")
        for issue in issues[:20]:  # Show first 20 issues
            logger.warning(f"  - {issue}")
        if len(issues) > 20:
            logger.warning(f"  ... and {len(issues) - 20} more issues")
    else:
        logger.info("✓ All validation checks passed!")
    
    if augmentation_stats:
        logger.info("\nAugmentation types found in samples:")
        for aug_type, count in sorted(augmentation_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {aug_type}: {count}")
    
    return len(issues) == 0

def main():
    """Main preprocessing function with augmentation"""
    
    # Configuration
    config = {
        'input_dir': 'splitted_dataset/train',
        'output_dir': 'augmented_dataset',
        'num_frames': 16,
        'augmentation_probability': 0.8,  # 80% chance to augment each video
        'max_augmentations_per_video': 3,  # Maximum augmentations to combine
        'randomize_params': True,  # Randomize augmentation parameters
        'preserve_original': True,  # Keep original video along with augmented
        'validate': True
    }
    
    logger.info("Sign Language Video Augmentation Pipeline")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Run preprocessing with augmentation
    output_path = preprocess_dataset_with_augmentation(
        config["input_dir"], 
        config["output_dir"], 
        config
    )
    
    # Validate if requested
    if config["validate"]:
        is_valid = validate_augmented_data(output_path)
        if not is_valid:
            logger.warning("\n⚠️  Some validation issues were found. Please review the warnings above.")
        else:
            logger.info("\n✅ All augmentation completed successfully!")

if __name__ == "__main__":
    main()