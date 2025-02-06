import os
import cv2
import struct

# Input paths
IMAGE_DIR = "E:\\Code\\0) Projects\\python extractor\\dataset2"  # Change to your image folder path
LABEL_FILE = "E:\\Code\\0) Projects\\python extractor\\labels2.txt"  # Change to your labels file
OUTPUT_FILE = "E:\\Code\\0) Projects\\python extractor\\polyphia.data"  # Output binary file

# Image properties
IMG_WIDTH = 480
IMG_HEIGHT = 270

def load_labels(label_file):
    """Load labels from a text file, assuming each line is a label."""
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def process_images(image_dir, labels, output_file):
    """Process images and store them in a binary .data file along with labels."""
    image_files = sorted(os.listdir(image_dir))  # Ensure images are in order
    assert len(image_files) == len(labels), "Error: Number of images and labels do not match!"

    with open(output_file, "wb") as f:
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)

            # Load image as grayscale and resize
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Flatten the image to a 1D array
            img_data = img.flatten()

            # Convert label to bytes
            label = labels[idx].encode('utf-8')  # Encode label as bytes

            # Write to file: (image binary data, label)
            f.write(img_data.tobytes())  # Write image binary data
            f.write(b",")  # Separate with a comma
            f.write(label)  # Write label
            f.write(b"\n")  # New line for next entry

            print(f"Processed {idx+1}/{len(image_files)}: {image_file} -> {labels[idx]}")

    print(f"Dataset successfully created: {output_file}")

# Load labels and process images
labels = load_labels(LABEL_FILE)
process_images(IMAGE_DIR, labels, OUTPUT_FILE)
