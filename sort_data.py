import os
import shutil
import re

# Define the gesture mapping (list of gesture names for reference)
GESTURE_MAP = {
    1: "KA", 2: "KHA", 3: "GA", 4: "GHA", 5: "NGA", 6: "CHA", 7: "CHHA", 8: "JA", 9: "JHA", 10: "YAN"
}

# # Define the gesture mapping (list of gesture names for reference)
# GESTURE_MAP = {
#     1: "KA", 2: "KHA", 3: "GA", 4: "GHA", 5: "NGA", 6: "CHA", 7: "CHHA", 8: "JA",
#     9: "JHA", 10: "YAN", 11: "TA", 12: "THA", 13: "DA", 14: "DHA", 15: "NA",
#     16: "TAA", 17: "THAA", 18: "DAA", 19: "DHAA", 20: "NAA", 21: "PA", 22: "PHA",
#     23: "BA", 24: "BHA", 25: "MA", 26: "YA", 27: "RA", 28: "LA", 29: "WA",
#     30: "T_SHA", 31: "M_SHA", 32: "D_SHA", 33: "HA", 34: "KSHA", 35: "TRA", 36: "GYA"
# }

BASE_DIR = os.path.expanduser("~/Downloads/10478554/NSL_Consonant_Part_1")

# New directory where gesture-based folders will be created
OUTPUT_DIR = "./main_dataset"
GESTURE_NAMES = list(GESTURE_MAP.values())

# Function to extract subject and type from folder name
def parse_folder_name(folder_name):
    parts = folder_name.split("_", 1)  # Split on first underscore
    if len(parts) == 2:
        subject = parts[0]  # e.g., s1, s2
        folder_type = parts[1]  # e.g., NSL_Consonant_Dark
        return subject, folder_type
    return None, None

# Get all folders in the base directory dynamically
source_folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]

# Create output directories for each gesture
for gesture_name in GESTURE_NAMES:
    gesture_dir = os.path.join(OUTPUT_DIR, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

# Process each gesture
for gesture_name in GESTURE_NAMES:
    counter = 1  # To number the files (e.g., KA_1, KA_2, etc.)

    # Iterate through all detected source folders
    for folder_name in source_folders:
        subject, folder_type = parse_folder_name(folder_name)
        if subject and folder_type:  # Valid folder name
            source_folder = os.path.join(BASE_DIR, folder_name)
            gesture_file_prefix = f"{subject}_{gesture_name}"  # e.g., s1_KA

            # Look for files matching the gesture prefix in the source folder
            for file in os.listdir(source_folder):
                if gesture_file_prefix == os.path.splitext(file)[0]:
                    print(gesture_file_prefix, os.path.splitext(file)[0])
                    source_path = os.path.join(source_folder, file)
                    # Preserve the original extension
                    file_extension = os.path.splitext(file)[1]
                    # New filename (e.g., KA_1.mov)
                    new_filename = f"{gesture_name}_{counter}{file_extension}"
                    dest_path = os.path.join(OUTPUT_DIR, gesture_name, new_filename)

                    # Move the file with error handling
                    try:
                        shutil.copy(source_path, dest_path)  # Use shutil.copy() if you want to copy instead
                        print(f"Moved: {source_path} -> {dest_path}")
                        counter += 1
                    except PermissionError as e:
                        print(f"Permission denied: Could not move {source_path} -> {dest_path}. Error: {e}")
                    except Exception as e:
                        print(f"Error moving {source_path} -> {dest_path}: {e}")

print("Reorganization complete!")