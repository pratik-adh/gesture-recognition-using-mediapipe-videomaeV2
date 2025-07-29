import os
import cv2
import numpy as np

def extract_hand_sequence(video_path, hands, num_frames=16, size=224, sample_multiplier=4):
    """
    Extract a sequence of hand-cropped frames from a video in RGB format, robust to various frame qualities.

    Args:
        video_path (str): Path to the video file.
        hands (mp_hands.Hands): MediaPipe Hands object for hand detection.
        num_frames (int): Number of frames to extract (default: 16).
        size (int): Size to resize the cropped frames to (default: 224).
        sample_multiplier (int): Multiplier to sample more frames initially (default: 4).

    Returns:
        list: List of RGB frames, padded with zeros if necessary to meet num_frames.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total-1, num_frames * sample_multiplier, dtype=int)
    frames = []

    print(f"🔍 Starting landmark extraction for {os.path.basename(video_path)}...")
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, img = cap.read()
        if not ret:
            continue

        # Preprocess frame: resize and normalize
        img = cv2.resize(img, (640, 480))  # Standardize resolution
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)  # Normalize lighting

        # Convert to RGB based on frame format
        if len(img.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:  # BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:  # BGRA
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            print(f"Unexpected frame shape: {img.shape}")
            continue

        # Process with MediaPipe for hand detection
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            print(f"⚠️ No hand detected in frame {i} of {video_path}")
            continue

        # Extract landmarks and calculate cropping coordinates
        lm = res.multi_hand_landmarks[0].landmark
        h, w, _ = img_rgb.shape
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        x1, y1 = max(int(min(xs)-20), 0), max(int(min(ys)-20), 0)
        x2, y2 = min(int(max(xs)+20), w), min(int(max(ys)+20), h)

        # Crop and resize the hand region
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (size, size))
        frames.append(crop)
        print(f"✅ Processed frame {i+1}/{num_frames} for {os.path.basename(video_path)}")
        if len(frames) >= num_frames:
            break

    cap.release()
    if len(frames) < num_frames:
        print(f"❌ Insufficient frames for {os.path.basename(video_path)}, padding with zeros...")
        frames += [np.zeros((size, size, 3), dtype=np.uint8)] * (num_frames - len(frames))
    print(f"🎉 Completed landmark extraction for {os.path.basename(video_path)}")
    return frames