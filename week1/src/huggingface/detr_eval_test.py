import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def convert_to_coco_format(detections, annotations, image_ids):
    """Converts the detection and ground truth data into COCO format."""
    coco_gt = {
        "images": [{"id": img_id} for img_id in image_ids],
        "annotations": [],
        "categories": [{"id": 3, "name": "car"}],
    }
    
    coco_dt = []
    annotation_id = 1
    
    for ann in annotations:
        frame_id, _, class_id, x1, y1, x2, y2, _ = ann
        if class_id == 3:  # Filtramos solo "car"
            coco_gt["annotations"].append({
                "id": annotation_id,
                "image_id": frame_id,
                "category_id": 3,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0
            })
            annotation_id += 1
    
    for det in detections:
        frame_id, _, class_id, x1, y1, x2, y2, score = det
        if class_id == 3:
            coco_dt.append({
                "image_id": frame_id,
                "category_id": 3,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score
            })
    
    return coco_gt, coco_dt

# Datos
annotations = [
    (0, 1000, 3, 288, 188, 524, 288, 1.0),
    (1, 1000, 3, 220, 190, 479, 299, 1.0),
    (2, 1000, 3, 138, 192, 427, 309, 1.0),
    (3, 1000, 3, 42, 194, 368, 324, 1.0),
    (4, 1000, 3, 0, 197, 309, 342, 1.0),
    (4, 1001, 3, 1219, 196, 1242, 217, 1.0),
    (5, 1000, 3, 0, 199, 233, 328, 1.0),
    (5, 1001, 3, 1180, 188, 1242, 217, 1.0),
    (6, 1000, 3, 0, 203, 145, 334, 1.0),
    (6, 1001, 3, 1145, 183, 1242, 218, 1.0),
    (7, 1000, 3, 0, 231, 37, 350, 1.0),
    (7, 1001, 3, 1108, 182, 1242, 219, 1.0),
    (8, 1001, 3, 1077, 182, 1222, 219, 1.0),
    (9, 1001, 3, 1052, 181, 1185, 216, 1.0),
    (10, 1001, 3, 1016, 180, 1132, 217, 1.0)
]

detections = [
    (0, -1, 3, 292, 187, 522, 289, 0.9986),
    (1, -1, 3, 220, 188, 474, 296, 0.9994),
    (2, -1, 3, 146, 187, 425, 308, 0.9986),
    (3, -1, 3, 43, 192, 371, 325, 0.9992),
    (4, -1, 3, 0, 193, 305, 342, 0.9991),
    (5, -1, 3, 1181, 185, 1241, 218, 0.9827),
    (5, -1, 3, 0, 197, 236, 335, 0.9990),
    (6, -1, 3, 1152, 180, 1241, 218, 0.9888),
    (6, -1, 3, 0, 205, 147, 337, 0.9983),
    (7, -1, 3, 0, 224, 38, 348, 0.9745),
    (7, -1, 3, 1126, 181, 1239, 219, 0.9886),
    (8, -1, 3, 1084, 181, 1209, 217, 0.9949),
    (9, -1, 3, 1063, 180, 1184, 216, 0.9957),
    (10, -1, 3, 1027, 180, 1138, 215, 0.9928)
]

# Extraer IDs de imágenes
image_ids = sorted(set(ann[0] for ann in annotations))

# Convertir a formato COCO
coco_gt_data, coco_dt_data = convert_to_coco_format(detections, annotations, image_ids)

# Guardar JSON temporalmente
with open("coco_gt.json", "w") as f:
    json.dump(coco_gt_data, f)
with open("coco_dt.json", "w") as f:
    json.dump(coco_dt_data, f)

# Evaluación con COCO
coco_gt = COCO("coco_gt.json")
coco_dt = coco_gt.loadRes("coco_dt.json")

coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.params.iouThrs = np.array([0.50])  # IoU=0.50
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Imprimir AP@0.50
print(f"AP@0.50: {coco_eval.stats[1]:.4f}")

