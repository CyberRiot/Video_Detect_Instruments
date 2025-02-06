import os
import cv2
import ffmpeg
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------- Configuration ----------------------
VIDEO_PATH = "abc_polyphia.mp4"  # Change this to your video path
OUTPUT_DIR = "dataset2"  # Directory where images & data will be stored
DATA_FILE = "output_binary_file2.data"  # Binary output file
LABEL_FILE = "labels2.csv"  # CSV label file
FRAME_INTERVAL = 24  # Extract every 24th frame (adjust as needed)
RESIZE_DIM = (480, 270)  # Output dimensions
LABEL = 1  # Change based on your dataset (0 = no_guitar, 1 = guitar_detected, etc.)

# ---------------------- Step 1: Extract I-Frames ----------------------
def extract_iframes(video_path, output_dir, frame_interval=FRAME_INTERVAL):
    """Extracts I-frames at the specified interval using FFmpeg."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    (
        ffmpeg
        .input(video_path)
        .output(output_pattern, vf=f"select='not(mod(n\\,{frame_interval}))'", vsync="vfr")
        .run(quiet=True, overwrite_output=True)
    )

    images = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    print(f"Extracted {len(images)} I-frames.")
    return images

# ---------------------- Step 2: Process Images ----------------------
def process_images(image_files, output_dir, resize_dim=RESIZE_DIM):
    """Reads images, converts to grayscale, resizes them, and returns them as binary arrays."""
    processed_images = []
    
    for img_file in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(output_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = cv2.resize(img, resize_dim)  # Resize to target resolution
        
        img_binary = img.tobytes()  # Convert image to binary format
        processed_images.append(img_binary)
    
    return processed_images

# ---------------------- Step 3: Write to Binary File ----------------------
def write_to_binary_file(output_file, images, label):
    """Writes images in binary format with appended labels."""
    with open(output_file, "wb") as f:
        for img in tqdm(images, desc="Writing Binary File"):
            f.write(img)  # Write image binary data
            f.write(np.uint8(label).tobytes())  # Append label as a single byte

# ---------------------- Step 4: Write Labels to CSV ----------------------
def write_labels_csv(label_file, image_files, label):
    """Writes the labels to a CSV file."""
    df = pd.DataFrame({"class_name": ["guitar_detected" if label == 1 else "no_guitar"] * len(image_files), "label": [label] * len(image_files)})
    df.to_csv(label_file, index=False)
    print(f"Labels written to {label_file}")

# ---------------------- Run the Script ----------------------
if __name__ == "__main__":
    print("Extracting I-frames...")
    image_files = extract_iframes(VIDEO_PATH, OUTPUT_DIR)

    print("Processing images...")
    processed_images = process_images(image_files, OUTPUT_DIR)

    print(f"Writing binary file: {DATA_FILE}")
    write_to_binary_file(DATA_FILE, processed_images, LABEL)

    print(f"Writing labels to CSV: {LABEL_FILE}")
    write_labels_csv(LABEL_FILE, image_files, LABEL)

    print("Dataset creation complete!")
