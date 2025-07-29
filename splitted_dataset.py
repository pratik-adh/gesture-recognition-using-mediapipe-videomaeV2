# train, test

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

# Configuration - Change these if needed
SOURCE_DIR = 'main_dataset_all'
OUTPUT_DIR = 'splitted_dataset_all'
# TRAIN_RATIO = 0.70
TRAIN_RATIO = 0.80
# VAL_RATIO = 0.15
TEST_RATIO = 0.20
# TEST_RATIO = 0.15
RANDOM_SEED = 42

def splitted_dataset():
    """Split the dataset into train/test folders"""
    
    # Set random seed
    random.seed(RANDOM_SEED)
    
    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)
    
    print("="*60)
    print("SPLITTING DATASET")
    print("="*60)
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    # print(f"Split: {TRAIN_RATIO*100:.0f}% train, {VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% test")
    print(f"Split: {TRAIN_RATIO*100:.0f}% train, {TEST_RATIO*100:.0f}% test")
    print("="*60)
    
    # Create output directories
    # for split in ['train', 'val', 'test']:
    for split in ['train', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Process each gesture
    gesture_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    # total_stats = {'train': 0, 'val': 0, 'test': 0}
    total_stats = {'train': 0, 'test': 0}
    
    for gesture_dir in tqdm(gesture_dirs, desc="Processing gestures"):
        gesture_name = gesture_dir.name
        
        # Get all videos for this gesture
        videos = list(gesture_dir.glob('*.mp4')) + list(gesture_dir.glob('*.avi')) + list(gesture_dir.glob('*.mov')) + list(gesture_dir.glob('*.MOV'))
        
        if not videos:
            continue
        
        # Shuffle videos
        random.shuffle(videos)
        
        # Calculate split sizes
        n_total = len(videos)
        n_train = int(n_total * TRAIN_RATIO)
        # n_val = int(n_total * VAL_RATIO)
        n_test = int(n_total * TEST_RATIO)
        
        # Split videos
        train_videos = videos[:n_train]
        # val_videos = videos[n_train:n_train + n_val]
        test_videos = videos[n_train:n_train + n_test]
        
        # Create gesture directories in each split
        for split in ['train', 'test']:
        # for split in ['train', 'val', 'test']:
            (output_path / split / gesture_name).mkdir(exist_ok=True)
        
        # Copy videos to respective splits
        for video in train_videos:
            shutil.copy2(video, output_path / 'train' / gesture_name / video.name)
            total_stats['train'] += 1
            
        # for video in val_videos:
        #     shutil.copy2(video, output_path / 'val' / gesture_name / video.name)
        #     total_stats['val'] += 1
            
        for video in test_videos:
            shutil.copy2(video, output_path / 'test' / gesture_name / video.name)
            total_stats['test'] += 1
        
        # print(f"\n{gesture_name}: {n_total} videos → train:{len(train_videos)}, val:{len(val_videos)}, test:{len(test_videos)}")
        print(f"\n{gesture_name}: {n_total} videos → train:{len(train_videos)}, test:{len(test_videos)}")
    
    # Print summary
    print("\n" + "="*60)
    print("SPLIT COMPLETE!")
    print("="*60)
    print(f"Total videos split:")
    print(f"  Train: {total_stats['train']} videos")
    # print(f"  Val: {total_stats['val']} videos")
    print(f"  Test: {total_stats['test']} videos")
    print(f"  Total: {sum(total_stats.values())} videos")
    print(f"\nOutput saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    splitted_dataset()


# train, test, val

# import os
# import shutil
# from pathlib import Path
# import random
# from tqdm import tqdm

# # Configuration - Change these if needed
# SOURCE_DIR = 'augmented_videos_3'
# OUTPUT_DIR = 'splitted_dataset'
# TRAIN_RATIO = 0.70
# VAL_RATIO = 0.15
# TEST_RATIO = 0.15
# RANDOM_SEED = 42

# def splitted_dataset():
#     """Split the dataset into train/test folders"""
    
#     # Set random seed
#     random.seed(RANDOM_SEED)
    
#     source_path = Path(SOURCE_DIR)
#     output_path = Path(OUTPUT_DIR)
    
#     print("="*60)
#     print("SPLITTING DATASET")
#     print("="*60)
#     print(f"Source: {source_path}")
#     print(f"Output: {output_path}")
#     print(f"Split: {TRAIN_RATIO*100:.0f}% train, {VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% test")
#     print(f"Split: {TRAIN_RATIO*100:.0f}% train, {TEST_RATIO*100:.0f}% test")
#     print("="*60)
    
#     # Create output directories
#     for split in ['train', 'val', 'test']:
#     # for split in ['train', 'test']:
#         (output_path / split).mkdir(parents=True, exist_ok=True)
    
#     # Process each gesture
#     gesture_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
#     total_stats = {'train': 0, 'val': 0, 'test': 0}
#     # total_stats = {'train': 0, 'test': 0}
    
#     for gesture_dir in tqdm(gesture_dirs, desc="Processing gestures"):
#         gesture_name = gesture_dir.name
        
#         # Get all videos for this gesture
#         videos = list(gesture_dir.glob('*.mp4')) + list(gesture_dir.glob('*.avi'))
        
#         if not videos:
#             continue
        
#         # Shuffle videos
#         random.shuffle(videos)
        
#         # Calculate split sizes
#         n_total = len(videos)
#         n_train = int(n_total * TRAIN_RATIO)
#         n_val = int(n_total * VAL_RATIO)
#         n_test = int(n_total * TEST_RATIO)
        
#         # Split videos
#         train_videos = videos[:n_train]
#         val_videos = videos[n_train:n_train + n_val]
#         test_videos = videos[n_train: n_train + n_test]
        
#         # Create gesture directories in each split
#         # for split in ['train', 'test']:
#         for split in ['train', 'val', 'test']:
#             (output_path / split / gesture_name).mkdir(exist_ok=True)
        
#         # Copy videos to respective splits
#         for video in train_videos:
#             shutil.copy2(video, output_path / 'train' / gesture_name / video.name)
#             total_stats['train'] += 1
            
#         for video in val_videos:
#             shutil.copy2(video, output_path / 'val' / gesture_name / video.name)
#             total_stats['val'] += 1
            
#         for video in test_videos:
#             shutil.copy2(video, output_path / 'test' / gesture_name / video.name)
#             total_stats['test'] += 1
        
#         print(f"\n{gesture_name}: {n_total} videos → train:{len(train_videos)}, val:{len(val_videos)}, test:{len(test_videos)}")
#         # print(f"\n{gesture_name}: {n_total} videos → train:{len(train_videos)}, test:{len(test_videos)}")
    
#     # Print summary
#     print("\n" + "="*60)
#     print("SPLIT COMPLETE!")
#     print("="*60)
#     print(f"Total videos split:")
#     print(f"  Train: {total_stats['train']} videos")
#     print(f"  Val: {total_stats['val']} videos")
#     print(f"  Test: {total_stats['test']} videos")
#     print(f"  Total: {sum(total_stats.values())} videos")
#     print(f"\nOutput saved to: {output_path}")
#     print("="*60)

# if __name__ == "__main__":
#     splitted_dataset()