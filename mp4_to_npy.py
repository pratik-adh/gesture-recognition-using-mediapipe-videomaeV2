import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========== CONFIG ==========
DATASET_DIR = "preprocessed_and_augmented_videos"
OUTPUT_DIR = "npy_preprocessed_videos"
CSV_PATH = "metadata.csv"


def video_to_numpy(video_path):
    """Read video and return np.array of shape (num_frames, H, W, 3)."""
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (cv2 default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return None
    return np.array(frames, dtype=np.uint8)  # shape: (num_frames, 224, 224, 3)

# ========== MAIN LOOP ==========
metadata = []
video_id = 0

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    # Create class folder inside OUTPUT_DIR
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for video_file in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
        if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        video_path = os.path.join(class_dir, video_file)
        arr = video_to_numpy(video_path)
        if arr is None:
            print(f"⚠️ Skipping {video_path}, no frames found.")
            continue

        # Keep same base filename but with .npy extension
        base_name = os.path.splitext(video_file)[0]
        save_name = f"{base_name}.npy"
        save_path = os.path.join(output_class_dir, save_name)

        np.save(save_path, arr)
        metadata.append([video_id, class_name, save_path, arr.shape[0]])
        video_id += 1

# Save CSV mapping
df = pd.DataFrame(metadata, columns=["video_id", "class", "npy_path", "num_frames"])
df.to_csv(CSV_PATH, index=False)
print(f"✅ Preprocessing complete. Saved {video_id} videos under {OUTPUT_DIR}, metadata in {CSV_PATH}")
