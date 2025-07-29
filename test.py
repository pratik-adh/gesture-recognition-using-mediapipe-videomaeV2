import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, VideoMAEImageProcessor, AutoModelForVideoClassification
from PIL import Image
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, GaussianBlur, ColorJitter, RandomAffine
import time
import mediapipe as mp


# Initialize MediaPipe Hands globally
mp_hands = mp.solutions.hands

def extract_hand_sequence(video_path, hands, num_frames=16, size=224):
    print(f"🔍 Processing video: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        print(f"❌ Video {os.path.basename(video_path)} is empty or invalid")
        cap.release()
        return [np.zeros((size, size, 3), dtype=np.uint8)] * num_frames
    
    idxs = np.linspace(0, max(total-1, 0), num_frames, dtype=int)
    frames = []
    for i in range(total):
        ret, img = cap.read()
        if not ret or i not in idxs:
            continue
        # Convert to RGB
        if len(img.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:  # BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:  # BGRA
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            print(f"⚠️ Unexpected frame shape in {os.path.basename(video_path)}: {img.shape}")
            continue
        # Hand detection
        res = hands.process(img_rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            h, w, _ = img_rgb.shape
            xs = [p.x * w for p in lm]
            ys = [p.y * h for p in lm]
            x1, y1 = max(int(min(xs)-30), 0), max(int(min(ys)-30), 0)
            x2, y2 = min(int(max(xs)+30), w), min(int(max(ys)+30), h)
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                print(f"⚠️ Invalid crop in frame {i} of {os.path.basename(video_path)}")
                continue
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(crop)
            print(f"✅ Frame {len(frames)}/{num_frames} extracted")
        else:
            print(f"⚠️ No hand detected in frame {i}")
    
    cap.release()
    if len(frames) < num_frames:
        print(f"⚠️ Only {len(frames)}/{num_frames} frames extracted, padding...")
        last_frame = frames[-1] if frames else np.zeros((size, size, 3), dtype=np.uint8)
        while len(frames) < num_frames:
            frames.append(last_frame.copy())
    elif len(frames) > num_frames:
        frames = frames[:num_frames]
    print(f"🎉 Completed processing {os.path.basename(video_path)} with {len(frames)} frames")
    return frames


class GestureDataset(Dataset):
    """Custom Dataset for loading and processing gesture videos."""
    def __init__(self, split_dir, hands, transform=None, token=None):
        self.vids = []
        self.labels = []
        try:
            self.processor = VideoMAEImageProcessor.from_pretrained(
                "MCG-NJU/videomae-base", token=token, trust_remote_code=True
            )
            print("✅ Successfully loaded VideoMAEImageProcessor")
        except Exception as e:
            print(f"❌ Failed to load processor: {e}")
            raise
        self.hands = hands
        self.transform = transform

        print(f"📂 Loading dataset from {split_dir}...")
        if not os.path.exists(split_dir):
            print(f"❌ Directory {split_dir} does not exist!")
            raise FileNotFoundError(f"Directory {split_dir} not found")
        for label, folder in enumerate(sorted(os.listdir(split_dir))):
            folder_path = os.path.join(split_dir, folder)
            if os.path.isdir(folder_path):
                files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
                print(f"📁 Folder {folder}: {len(files)} videos")
                for file in files:
                    self.vids.append(os.path.join(folder_path, file))
                    self.labels.append(label)
        print(f"✅ Loaded {len(self.vids)} videos from {split_dir}")
        if len(self.vids) == 0:
            raise ValueError(f"No videos found in {split_dir}")

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        max_attempts = 5
        attempts = 0
        while attempts < max_attempts:
            try:
                video_path = self.vids[idx]
                print(f"📥 Loading video: {os.path.basename(video_path)}")
                seq = extract_hand_sequence(video_path, self.hands)
                if self.transform:
                    seq = self.transform(seq)
                for i, frame in enumerate(seq):
                    if frame.shape != (224, 224, 3):
                        raise ValueError(f"Frame {i} in {os.path.basename(video_path)} has shape {frame.shape}")
                inputs = self.processor(seq, return_tensors="pt")
                pixel_values = inputs['pixel_values'].squeeze(0).to(torch.float16)
                label = torch.tensor(self.labels[idx])
                print(f"✅ Processed {os.path.basename(video_path)}")
                return pixel_values, label
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(self.vids[idx])}: {e}")
                attempts += 1
                idx = np.random.randint(len(self.vids))
                print(f"🔄 Retrying with {os.path.basename(self.vids[idx])} (Attempt {attempts}/{max_attempts})")
        raise RuntimeError(f"Failed to process video after {max_attempts} attempts")


def eval_epoch(model, dl, device, kind="Validation"):
    """Evaluate the model on validation or test set."""
    model.eval()
    total, correct = 0, 0
    print(f"🔍 Starting {kind} evaluation...")
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(dl, desc=f"{kind} Batch")):
            X, y = X.to(device), y.to(device)
            outputs = model(pixel_values=X)
            preds = outputs.logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            print(f"📊 Batch {i+1}/{len(dl)} - Acc: {(preds == y).sum().item()/y.size(0):.4f}")
    accuracy = correct / total if total > 0 else 0
    print(f"✅ {kind} Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    # Setup
    base_dir = "./splitted_dataset"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Hugging Face token

    huggingface_token = os.getenv('HUGGINGFACE_TOKEN')  # Default token if not set

    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    test_dataset = GestureDataset(os.path.join(base_dir, "test"), hands, token=huggingface_token)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')

    # Create class map
    class_map = {i: folder for i, folder in enumerate(sorted(os.listdir(os.path.join(base_dir, "test")))) 
                 if os.path.isdir(os.path.join(base_dir, "test", folder))}
    print(f"📋 Class Map: {class_map}")

    config = AutoConfig.from_pretrained("MCG-NJU/videomae-base", token=huggingface_token, trust_remote_code=True)
    config.num_labels = len(class_map)
    config.dropout = 0.3
    config.drop_path_rate = 0.1
    config.label_smoothing = 0.1
    model = AutoModelForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base", config=config, token=huggingface_token, trust_remote_code=True
    ).to(device, dtype=torch.float16)

    # Testing
    print("🔍 Testing best model...")
    if os.path.exists("best_model.pt"):
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
        test_acc = eval_epoch(model, test_loader, device, "Test")
        print(f"🎯 Final Test Accuracy: {test_acc:.4f}")
    else:
        print("❌ No trained model found for testing")