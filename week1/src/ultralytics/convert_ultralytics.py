import os
from pycocotools import mask as maskUtils

# Paths to KITTI-MOTS dataset
KITTI_MOTS_PATH = "/ghome/c3mcv02/mcv-c5-team1/data"
INSTANCES_TXT_PATH = os.path.join(KITTI_MOTS_PATH, "instances_txt")
IMAGES_PATH = os.path.join(KITTI_MOTS_PATH, "training/val")
OUTPUT_DIR = "/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/data/val/labels"  # Path for YOLO annotations

# Create output directory for YOLO annotations if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping KITTI-MOTS classes to COCO classes
KITTI_TO_COCO = {1: 2, 2: 0}  # 1 (car) -> 2 (COCO car), 2 (pedestrian) -> 0 (COCO person)

# Iterate through each .txt annotation file (for each sequence)
for txt_file in sorted(os.listdir(INSTANCES_TXT_PATH)):
    file_path = os.path.join(INSTANCES_TXT_PATH, txt_file)
    seq_id = int(txt_file.split(".")[0])  # Extract sequence ID (e.g., 0000, 0001, etc.)
    
    # Directory for the current sequence of images
    sequence_image_dir = os.path.join(IMAGES_PATH, f"{seq_id:04d}")  # Folder for this sequence of images

    # Check if sequence folder exists
    if not os.path.exists(sequence_image_dir):
        print(f"Warning: Directory for sequence {seq_id:04d} not found.")
        continue

    # Create a subfolder in the 'labels' directory for the current sequence
    sequence_output_dir = os.path.join(OUTPUT_DIR, f"{seq_id:04d}")
    os.makedirs(sequence_output_dir, exist_ok=True)

    # Iterate over all frames in this sequence
    with open(file_path, "r") as f:
        for line in f:
            # Parse annotation line
            frame, obj_id, class_id, height, width, rle = line.strip().split(" ", 5)
            frame_id = int(frame)
            obj_id = int(obj_id)
            class_id = int(class_id)  # Class ID (1 for car, 2 for pedestrian)
            
            # Ignore class_id = 10
            if class_id == 10:
                continue
            
            # Map KITTI class ID to COCO class ID
            if class_id in KITTI_TO_COCO:
                class_id = KITTI_TO_COCO[class_id]
            else:
                continue  # Skip unknown classes
            
            # Create RLE object
            height, width = int(height), int(width)  # Corrected the position of height and width
            rle_obj = {'counts': rle, 'size': [height, width]}
            bbox = [int(coord) for coord in maskUtils.toBbox(rle_obj)]
            x, y, w, h = bbox

            # Normalize the bbox coordinates
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height

            # Prepare YOLO annotation (class_id , x_center, y_center, norm_w, norm_h)
            annotation = f"{class_id} {x_center} {y_center} {norm_w} {norm_h}"

            # Check if the image for the current frame exists
            image_file = f"{frame_id:06d}.png"
            image_path = os.path.join(sequence_image_dir, image_file)

            if os.path.exists(image_path):
                # Write YOLO annotations for this frame to a .txt file
                annotation_file = os.path.join(sequence_output_dir, f"{frame_id:06d}.txt")
                with open(annotation_file, "a") as ann_f:
                    ann_f.write(annotation + "\n")
            else:
                print(f"Image file {image_file} not found in sequence {seq_id:04d}. Skipping frame.")
