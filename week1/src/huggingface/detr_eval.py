import json
import numpy as np
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_FOLDER = "/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/gt_annotations"
DT_FOLDER = "/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/results_txt"

CATEGORIES = {1: "person", 3: "car"}

def load_annotations(file_path):
    annotations = []
    with open(file_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split(",")))
            frame_id = int(values[0])
            annotations.append(tuple(values))
    return annotations

def convert_to_coco_format(detections, annotations):
    image_ids = sorted(set(ann[0] for ann in annotations))
    coco_gt = {
        "images": [{"id": img_id} for img_id in image_ids],
        "annotations": [],
        "categories": [{"id": cid, "name": cname} for cid, cname in CATEGORIES.items()],
    }
    coco_dt = []
    annotation_id = 1

    for ann in annotations:
        frame_id, _, class_id, x1, y1, x2, y2, _ = ann
        if class_id in CATEGORIES:
            coco_gt["annotations"].append({
                "id": annotation_id,
                "image_id": frame_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0
            })
            annotation_id += 1

    for det in detections:
        frame_id, _, class_id, x1, y1, x2, y2, score = det
        if class_id in CATEGORIES and frame_id in image_ids:
            coco_dt.append({
                "image_id": frame_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score
            })

    return coco_gt, coco_dt

def evaluate_coco(coco_gt_data, coco_dt_data):
    with open("coco_gt.json", "w") as f:
        json.dump(coco_gt_data, f)
    with open("coco_dt.json", "w") as f:
        json.dump(coco_dt_data, f)

    coco_gt = COCO("coco_gt.json")
    coco_dt = coco_gt.loadRes("coco_dt.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval

def compute_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

all_gt_annotations = []
all_dt_detections = []

for gt_file in glob.glob(f"{GT_FOLDER}/*.txt"):
    dt_file = gt_file.replace(GT_FOLDER, DT_FOLDER)
    if not glob.os.path.exists(dt_file):
        continue

    all_gt_annotations.extend(load_annotations(gt_file))
    all_dt_detections.extend(load_annotations(dt_file))

coco_gt_data, coco_dt_data = convert_to_coco_format(all_dt_detections, all_gt_annotations)
coco_eval = evaluate_coco(coco_gt_data, coco_dt_data)

# Mapeo de categorías en COCO eval
cat_id_map = {cat_id: idx for idx, cat_id in enumerate(CATEGORIES.keys())}

results = {}
for cat_id, cat_name in CATEGORIES.items():
    if cat_id in cat_id_map:
        cat_index = cat_id_map[cat_id]
        precision = coco_eval.eval["precision"][0, :, cat_index, 0, 2].mean()
        recall = coco_eval.eval["recall"][0, cat_index, 2].mean()
        f1 = compute_f1(precision, recall)
        ap = coco_eval.stats[1 + cat_index]
        ap_50 = coco_eval.stats[3 + cat_index]
        results[cat_name] = {
            "AP": ap,
            "AP@0.50": ap_50,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        }

map_50 = coco_eval.stats[1]
map_75 = coco_eval.stats[2]
mAP = coco_eval.stats[0]
mean_precision = np.mean([res["Precision"] for res in results.values()])
mean_recall = np.mean([res["Recall"] for res in results.values()])

final_results = {
    "Mean AP": mAP,
    "Mean AP@0.50": map_50,
    "Mean AP@0.75": map_75,
    "Mean Precision": mean_precision,
    "Mean Recall": mean_recall,
    **results
}

print(json.dumps(final_results, indent=4))





