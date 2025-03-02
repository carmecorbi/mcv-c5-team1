from ultralytics import YOLO
import time
import os
import sys

# Check if a sequence number is provided
if len(sys.argv) != 2:
    print("Usage: python inference_c.py <sequence_number>")
    sys.exit(1)

sequence_number = sys.argv[1]

# Load the YOLO model
model = YOLO("/ghome/c3mcv02/mcv-c5-team1/week1/checkpoints/yolo/yolo11n.pt")

# Define dataset paths dynamically based on sequence number
VAL_IMAGES_PATH = f"/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/data/images/val/{sequence_number}"
output_dir = f"/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/results_inference/{sequence_number}"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Start timing inference
start_time = time.time()

# Perform inference and save results
results_inference = model.predict(source=VAL_IMAGES_PATH, save=True, classes=[0, 2], project=output_dir)

# Compute inference time
inference_time = time.time() - start_time
print(f"Inference time: {inference_time} seconds.")



