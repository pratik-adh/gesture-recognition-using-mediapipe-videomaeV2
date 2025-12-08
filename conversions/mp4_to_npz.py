import os
import cv2
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LightweightNPZConverter")


class LightweightVideoToNPZConverter:
    """
    Converts video datasets to lightweight NPZ format suitable for VideoMAE, TCN, etc.
    - Resizes to a fixed resolution (e.g., 224x224)
    - Stores uint8 (saves disk space)
    - Defers normalization to the training pipeline
    """

    def __init__(self, input_dir: str, output_dir: str, num_frames: int = 16, resize_size: int = 224):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_frames = num_frames
        self.resize_size = resize_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _uniform_temporal_sampling(self, total_frames: int) -> np.ndarray:
        """Uniformly sample frame indices from a video."""
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(total_frames - 1)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        return indices

    def _process_video(self, video_path: Path) -> np.ndarray:
        """Read and sample frames from a single video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            logger.warning(f"No frames found in video: {video_path}")
            cap.release()
            return None

        frame_indices = self._uniform_temporal_sampling(total_frames)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1])  # Repeat last valid frame
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.resize_size, self.resize_size), interpolation=cv2.INTER_AREA)
            frames.append(frame.astype(np.uint8))

        cap.release()

        if len(frames) != self.num_frames:
            logger.warning(f"Expected {self.num_frames} frames, got {len(frames)} for {video_path}")
            return None

        return np.stack(frames, axis=0)  # Shape: (T, H, W, C)

    def convert_dataset(self):
        """Convert all videos in the dataset into NPZ files."""
        logger.info("🚀 Starting lightweight NPZ conversion...")

        successful, failed = 0, 0
        stats = {}

        for class_dir in tqdm(sorted(self.input_dir.iterdir()), desc="Processing classes"):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            output_class_dir = self.output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            video_files = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mov"))
            class_success, class_fail = 0, 0

            for vid in tqdm(video_files, desc=f"{class_name}", leave=False):
                video_tensor = self._process_video(vid)
                if video_tensor is None:
                    failed += 1
                    class_fail += 1
                    continue

                output_path = output_class_dir / f"{vid.stem}.npz"
                np.savez_compressed(
                    output_path,
                    video=video_tensor,
                    num_frames=self.num_frames,
                    class_name=class_name,
                    video_name=vid.stem
                )

                successful += 1
                class_success += 1

            stats[class_name] = {"successful": class_success, "failed": class_fail}

        summary = {
            "total_successful": successful,
            "total_failed": failed,
            "total_videos": successful + failed,
            "num_frames": self.num_frames,
            "resize_size": self.resize_size,
            "classes": stats
        }

        with open(self.output_dir / "conversion_stats.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✅ Conversion complete! Successful: {successful}, Failed: {failed}")
        return summary


if __name__ == "__main__":
    INPUT_DIR = "preprocessed_and_augmented_videos_without_cropping"   # Input folder (class subfolders)
    OUTPUT_DIR = "npz_lightweight_videos_without_cropping"             # Output NPZ folder

    converter = LightweightVideoToNPZConverter(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        num_frames=16,        # Typical VideoMAE setting
        resize_size=224       # Can use 160 for smaller models
    )

    converter.convert_dataset()
