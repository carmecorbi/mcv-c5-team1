import argparse
import time
from ultralytics import YOLO

# Argument parser setup
parser = argparse.ArgumentParser(description="Validate a YOLO model.")
parser.add_argument("--m", type=str, required=True, help="Path to the YOLO model file.")
args = parser.parse_args()

# Load the model
model = YOLO(args.m) 
DATASET_PATH = "/ghome/c5mcv01/mcv-c5-team1/week2/src/ultralytics/data/data.yaml"


# Run validation
results = model.val(data=DATASET_PATH)

# Print specific segmentation metrics
print("Average precision:", results.seg.ap)
print("Precision", results.seg.p)
print("Recall",results.seg.r)
print("Mean average precision (mAP@50-95):", results.seg.map)
print("Mean average precision at IoU=0.50 (mAP@50):", results.seg.map50)
print("Mean average precision at IoU=0.75 (mAP@75):", results.seg.map75)



