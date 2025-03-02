import os
import time
import cv2
import json
from detr import DeTR

# Define dataset path and output paths
VAL_IMAGES_PATH = "/ghome/c3mcv02/mcv-c5-team1/data/training/val/"
OUTPUT_BASE_DIR = "/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/results_inference/"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Mapping KITTI-MOTS classes to COCO classes
KITTI_TO_COCO = {1: 2, 2: 0}  # 1 (car) -> 2 (COCO car), 2 (pedestrian) -> 0 (COCO person)

# Initialize the DeTR model
model = DeTR()

# Create output directory for COCO predictions
output_dir = os.path.join(OUTPUT_BASE_DIR, "coco_predictions")
os.makedirs(output_dir, exist_ok=True)

# Function to prepare predictions in COCO format
def prepare_predictions_for_coco(predictions, image_id, height, width):
    boxes = predictions.pred_boxes[0].cpu().numpy()
    scores = predictions.logits.softmax(-1)[0, :, :-1].max(-1)[0].cpu().numpy()
    labels = predictions.logits.argmax(-1)[0].cpu().numpy()

    # Map KITTI classes to COCO classes
    mapped_labels = [KITTI_TO_COCO.get(label, label) for label in labels]

    coco_predictions = []
    for box, score, label in zip(boxes, scores, mapped_labels):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        coco_predictions.append({
            "image_id": image_id,
            "category_id": label,
            "bbox": [x_min, y_min, width, height],
            "score": score
        })

    return coco_predictions

# Process all sequence folders in the dataset
start_time = time.time()

for sequence_number in os.listdir(VAL_IMAGES_PATH):
    sequence_path = os.path.join(VAL_IMAGES_PATH, sequence_number)

    if not os.path.isdir(sequence_path):
        continue

    sequence_output_dir = os.path.join(output_dir, sequence_number)
    os.makedirs(sequence_output_dir, exist_ok=True)

    print(f"Processing sequence: {sequence_number}")

    for filename in os.listdir(sequence_path):
        image_path = os.path.join(sequence_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        # Run inference using the DeTR model
        predictions = model.run_inference(image)
        height, width, _ = image.shape
        image_id = int(filename.split(".")[0])  # Get image ID based on filename

        # Prepare predictions in COCO format
        coco_predictions = prepare_predictions_for_coco(predictions, image_id, height, width)

        # Save the COCO predictions as JSON
        coco_output_path = os.path.join(sequence_output_dir, f"{image_id}.json")
        with open(coco_output_path, "w") as f:
            json.dump(coco_predictions, f)
        print(f"Saved: {coco_output_path}")

# Compute inference time
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds.")

