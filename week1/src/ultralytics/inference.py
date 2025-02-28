from ultralytics import YOLO
import time
import os
import shutil
from pathlib import Path

# Load the YOLO model
model = YOLO("/ghome/c3mcv02/mcv-c5-team1/week1/checkpoints/yolo/yolo11n.pt") 

# Define dataset paths
DATASET_PATH = "/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/data/data.yaml"
#VAL_IMAGES_PATH = "/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/data/images/val/**/*.*"
VAL_IMAGES_PATH = "/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/data/images/val/0002"

# Create the main output directory for results
output_dir = "/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/data/results_inference"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Start timing inference
start_time = time.time()

# Perform inference and save results
results_inference = model.predict(source=VAL_IMAGES_PATH, save=True, classes=[0, 2],project='/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/results_inference/0002')


# Compute inference time
inference_time = time.time() - start_time
print(f"Inference time: {inference_time} seconds.")



'''

results = model.val(data=DATASET_PATH)

# Print specific metrics
print("Class indices with average precision:", results.ap_class_index)
print("Average precision for all classes:", results.box.all_ap)
print("Average precision:", results.box.ap)
print("Average precision at IoU=0.50:", results.box.ap50)


print("F1 score:", results.box.f1)

print("Mean average precision:", results.box.map)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean average precision at IoU=0.75:", results.box.map75)

print("Mean precision:", results.box.mp)
print("Mean recall:", results.box.mr)
print("Precision:", results.box.p)
print("Recall:", results.box.r)

'''