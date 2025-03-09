import os
import time
import cv2
import numpy as np
from detr import DeTR

# Define dataset path
VAL_IMAGES_PATH = "/ghome/c5mcv01/mcv-c5-team1/data/training/val/0013"
OUTPUT_BASE_DIR = "/ghome/c5mcv01/mcv-c5-team1/week1/src/huggingface/results_fine_tuning_test/results_inference/"
TXT_OUTPUT_DIR = "/ghome/c5mcv01/mcv-c5-team1/week1/src/huggingface/results_fine_tuning_test/results_txt/"

# Ensure the output directories exist
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(TXT_OUTPUT_DIR, exist_ok=True)

# Initialize the DeTR model
model_path = "/ghome/c5mcv01/mcv-c5-team1/week1/src/huggingface/detr_finetuned_kitti/checkpoint-13560"
model = DeTR(model_path=model_path)

# Mapping of class names to numerical IDs
CLASS_MAP = {"car": 0, "pedestrian": 1}

# Start timing inference
start_time = time.time()

# Process all sequence folders in the VAL_IMAGES_PATH

sequence_path = os.path.join(VAL_IMAGES_PATH)

# Define output directories for images and txt files
output_dir = os.path.join(OUTPUT_BASE_DIR, "0013")
txt_output_file = os.path.join(TXT_OUTPUT_DIR, "0013")
os.makedirs(output_dir, exist_ok=True)

print("Processing sequence: 0013")

with open(txt_output_file, 'w') as txt_file:
    # Process all images in the sequence folder
    for frame_id, filename in enumerate(sorted(os.listdir(sequence_path))):
        image_path = os.path.join(sequence_path, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        
        # Run inference using the DeTR model
        predictions = model.run_inference(image)
        print(predictions)
        visualized_image, bboxes = model.visualize_predictions(image, predictions, finetuning = True)  # bboxes tiene coords en píxeles
        
        # Save the visualized image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, visualized_image)
        print(f"Saved: {output_path}")
        
        # Save predictions to txt file
        for x1, y1, x2, y2, class_name, score in bboxes:
            if class_name in CLASS_MAP:
                class_id = CLASS_MAP[class_name] 
                txt_file.write(f"{frame_id}, -1, {class_id}, {x1}, {y1}, {x2}, {y2}, {score:.4f}\n")

# Compute inference time
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds.")

