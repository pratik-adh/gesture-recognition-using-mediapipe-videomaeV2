"""
Convert class-organised *.mp4 → VideoMAE-ready *.pt tensors
Shape: (C, T, H, W) = (3, 16, 224, 224)
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

mp_hands = mp.solutions.hands

TARGET_SHAPE = (224, 224)
NUM_FRAMES   = 16


class HandCropper:
    """Detect a single hand, crop & resize to square."""
    def __init__(self,
                 target_size=(224, 224),
                 margin=0.3,
                 conf=0.6):
        self.target_size = target_size
        self.margin = margin
        self.hands = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=1,
                                    min_detection_confidence=conf)

    def __call__(self, frame):
        """Return cropped hand frame or fallback center-crop."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            size = max(max_x - min_x, max_y - min_y) * (1 + self.margin)
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            x1 = max(0, int(cx - size / 2))
            y1 = max(0, int(cy - size / 2))
            x2 = min(w, int(cx + size / 2))
            y2 = min(h, int(cy + size / 2))
            cropped = frame[y1:y2, x1:x2]
        else:
            # fallback: square center crop
            size = min(h, w)
            y1 = (h - size) // 2
            x1 = (w - size) // 2
            cropped = frame[y1:y1 + size, x1:x1 + size]

        return cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LANCZOS4)


def load_video_frames(video_path, num_frames=NUM_FRAMES):
    """Load and uniformly sample exactly `num_frames`."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        return None

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    cropper = HandCropper()

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            return None
        frames.append(cropper(frame))
    cap.release()

    # (T, H, W, C) → (T, C, H, W) → (C, T, H, W)
    tensor = torch.from_numpy(np.array(frames, dtype=np.float32) / 255.0)
    tensor = tensor.permute(0, 3, 1, 2).permute(1, 0, 2, 3)
    return tensor


def preprocess_videos_to_tensors(input_root, output_root):
    """Main entry point."""
    input_root, output_root = Path(input_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    videos = [p for p in input_root.rglob("*.mp4")]
    if not videos:
        raise ValueError("No .mp4 files found.")

    metadata = {"classes": {}, "videos": []}

    for class_dir in sorted(input_root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_idx = len(metadata["classes"])
        class_name = class_dir.name
        metadata["classes"][class_idx] = class_name

        out_class_dir = output_root / class_name
        out_class_dir.mkdir(exist_ok=True)

        for video_path in tqdm(class_dir.glob("*.mp4"), desc=class_name):
            tensor = load_video_frames(video_path)
            if tensor is None:
                continue

            out_file = out_class_dir / f"{video_path.stem}.pt"
            torch.save(tensor, out_file)

            metadata["videos"].append({
                "path": str(out_file),
                "class_idx": class_idx,
                "class_name": class_name
            })

    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"\n✅ Done: {len(metadata['videos'])} tensors → {output_root}")


if __name__ == "__main__":
    preprocess_videos_to_tensors(
        input_root="all_classes_in_mp4",
        output_root="tensor_preprocessed_all"
    )