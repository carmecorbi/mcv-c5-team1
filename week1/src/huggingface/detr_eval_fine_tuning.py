import json
import glob

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# Directories containing ground truth and detection results
GT_FOLDER = "/ghome/c5mcv01/mcv-c5-team1/week1/src/huggingface/config/gt_annotations"
DT_FOLDER = "/ghome/c5mcv01/mcv-c5-team1/week1/src/huggingface/results_fine_tuning/results_txt"

# Categories of interest
CATEGORIES = {1: "person", 3: "car"}

def load_annotations(file_path):
    """Load annotations from a given file."""
    annotations = []
    with open(file_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split(",")))
            frame_id = int(values[0])
            annotations.append(tuple(values))
    return annotations

# Mapping classes from inference to match the ground truth (GT)
INFERENCE_TO_GT_CLASS_MAP = {0: 3, 1: 1}  # "car" → 3, "pedestrian" → 1

def convert_to_coco_format(detections, annotations):
    """Convert ground truth and detections into COCO format."""
    image_ids = sorted(set(ann[0] for ann in annotations))
    
    # Initialize COCO-format structures
    coco_gt = {
        "images": [{"id": img_id} for img_id in image_ids],
        "annotations": [],
        "categories": [{"id": cid, "name": cname} for cid, cname in CATEGORIES.items()],
    }
    coco_dt = []
    annotation_id = 1

    # Process ground truth annotations
    for ann in annotations:
        frame_id, _, class_id, x1, y1, x2, y2, _ = ann
        if class_id in CATEGORIES:
            coco_gt["annotations"].append({
                "id": annotation_id,
                "image_id": frame_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # Convert to COCO bbox format (x, y, width, height)
                "area": (x2 - x1) * (y2 - y1),  # Compute bounding box area
                "iscrowd": 0  # Assume all instances are not crowd annotations
            })
            annotation_id += 1

    # Process detection results
    for det in detections:
        frame_id, _, class_id, x1, y1, x2, y2, score = det
        # Convert inference class ID to the corresponding ground truth class ID
        class_id = INFERENCE_TO_GT_CLASS_MAP.get(class_id, None)
        
        if class_id in CATEGORIES and frame_id in image_ids:
            coco_dt.append({
                "image_id": frame_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # Convert to COCO bbox format
                "score": score  # Detection confidence score
            })

    return coco_gt, coco_dt


def evaluate_coco(coco_gt_data, coco_dt_data):
    """Evaluate object detection performance using COCO evaluation metrics."""
    # Save COCO-formatted ground truth and detections to JSON files
    with open("coco_gt.json", "w") as f:
        json.dump(coco_gt_data, f)
    with open("coco_dt.json", "w") as f:
        json.dump(coco_dt_data, f)

    # Load COCO ground truth and detection results
    coco_gt = COCO("coco_gt.json")
    coco_dt = coco_gt.loadRes("coco_dt.json")
    
    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats

# Load all ground truth annotations and detection results
all_gt_annotations = []
all_dt_detections = []

for gt_file in glob.glob(f"{GT_FOLDER}/*.txt"):
    dt_file = gt_file.replace(GT_FOLDER, DT_FOLDER)
    if not glob.os.path.exists(dt_file):
        continue

    all_gt_annotations.extend(load_annotations(gt_file))
    all_dt_detections.extend(load_annotations(dt_file))

# Convert to COCO format and evaluate
coco_gt_data, coco_dt_data = convert_to_coco_format(all_dt_detections, all_gt_annotations)
coco_metrics = evaluate_coco(coco_gt_data, coco_dt_data)

# Store final evaluation results
final_results = {
    "Mean AP": coco_metrics[0],
    "AP@0.50": coco_metrics[1],
    "AP@0.75": coco_metrics[2],
    "AP (Small)": coco_metrics[3],
    "AP (Medium)": coco_metrics[4],
    "AP (Large)": coco_metrics[5],
    "AR@1": coco_metrics[6],
    "AR@10": coco_metrics[7],
    "AR@100": coco_metrics[8],
    "AR (Small)": coco_metrics[9],
    "AR (Medium)": coco_metrics[10],
    "AR (Large)": coco_metrics[11]
}

# Print the evaluation results in JSON format
print(json.dumps(final_results, indent=4))




