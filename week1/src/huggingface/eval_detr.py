from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os

# Path to the ground truth and predictions
GT_ANNOTATIONS_PATH = "/ghome/c3mcv02/mcv-c5-team1/data/training/val/annotations/instances_val.json"
PREDICTIONS_DIR = "/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/results_inference/coco_predictions"

# Load COCO ground truth and predictions
coco_gt = COCO(GT_ANNOTATIONS_PATH)

# Create COCO results from the predictions
coco_dt = coco_gt.loadRes([json.load(open(os.path.join(PREDICTIONS_DIR, f))) for f in os.listdir(PREDICTIONS_DIR)])

# Initialize COCOeval object
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

# Run the evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Print specific metrics
print("Mean Average Precision (mAP):", coco_eval.stats[0])  # mAP at IoU 0.50:0.95
print("Average Precision at IoU=0.50:", coco_eval.stats[1])  # AP at IoU 0.50
print("Average Precision at IoU=0.75:", coco_eval.stats[2])  # AP at IoU 0.75
print("Average Recall at IoU=0.50:", coco_eval.stats[3])  # AR at IoU 0.50
