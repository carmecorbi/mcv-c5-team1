import os
import time
import cv2
from detr import DeTR


# Define dataset path
VAL_IMAGES_PATH = "/ghome/c3mcv02/mcv-c5-team1/data/training/val/"
OUTPUT_BASE_DIR = "/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/results/results_inference/"

# Ensure the output base directory exists
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Initialize the DeTR model
model = DeTR()

# Start timing inference
start_time = time.time()

# Process all sequence folders in the VAL_IMAGES_PATH
for sequence_number in os.listdir(VAL_IMAGES_PATH):
    sequence_path = os.path.join(VAL_IMAGES_PATH, sequence_number)

    # Skip if it's not a directory (in case of files or non-folder entries)
    if not os.path.isdir(sequence_path):
        continue

    # Define output directory for this sequence
    output_dir = os.path.join(OUTPUT_BASE_DIR, sequence_number)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing sequence: {sequence_number}")

    # Process all images in the sequence folder
    for filename in os.listdir(sequence_path):
        image_path = os.path.join(sequence_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        # Run inference using the DeTR model
        predictions = model.run_inference(image)
        visualized_image = model.visualize_predictions(image, predictions)

        # Save the visualized image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, visualized_image)
        print(f"Saved: {output_path}")

# Compute inference time
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds.")
