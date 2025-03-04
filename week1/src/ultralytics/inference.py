import argparse
import time
import os
import sys
from ultralytics import YOLO

# Argument parser setup
parser = argparse.ArgumentParser(description="Run YOLO inference on a specific sequence.")
parser.add_argument("--seq", type=str, required=True, help="Sequence number for inference.")
parser.add_argument("--out", type=str, required=True, help="Output directory for results.")
args = parser.parse_args()

sequence_number = args.seq
output_dir = args.out

# Load the YOLO model
model = YOLO("/ghome/c3mcv02/mcv-c5-team1/week1/checkpoints/yolo/yolo11n.pt")

# Define dataset path dynamically based on sequence number
VAL_IMAGES_PATH = f"/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/data/images/val/{sequence_number}"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Start timing inference
start_time = time.time()

# Perform inference and save results
results_inference = model.predict(source=VAL_IMAGES_PATH, save=True, classes=[0, 2], project=output_dir)

# Compute inference time
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds.")



