

from tqdm import tqdm
from ultralytics import SAM
from ultralytics.data import YOLODataset
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy
from pathlib import Path
import cv2
import os


def yolo_bbox2segment(im_dir, save_dir=None, sam_model="sam_b.pt"):
    """
    Converts existing object detection dataset (bounding boxes) to segmentation dataset in YOLO format,
    preserving the original train/val directory structure.
    
    Args:
        im_dir (str | Path): Path to image directory to convert.
        save_dir (str | Path): Path to save the generated labels, maintaining the folder structure.
        sam_model (str): Segmentation model to use for intermediate segmentation data.
    """
    im_dir = Path(im_dir)
    save_dir = Path(save_dir) if save_dir else im_dir.parent / "seg_labels"
    sam_model = SAM(sam_model)
    
    for split in ["train", "val"]:
        split_dir = im_dir / split
        if not split_dir.exists():
            continue
        
        for seq_folder in sorted(split_dir.iterdir()):
            if not seq_folder.is_dir():
                continue
            
            dataset = YOLODataset(seq_folder, data=dict(names=list(range(1000))))
            if len(dataset.labels[0]["segments"]) > 0:
                LOGGER.info(f"Segmentation labels detected in {seq_folder}, skipping!")
                continue
            
            LOGGER.info(f"Processing {seq_folder}, generating segment labels with SAM!")
            seg_save_dir = save_dir / split / seq_folder.name
            seg_save_dir.mkdir(parents=True, exist_ok=True)
            
            for l in tqdm(dataset.labels, total=len(dataset.labels), desc=f"Processing {seq_folder.name}"):
                h, w = l["shape"]
                boxes = l["bboxes"]
                if len(boxes) == 0:
                    continue
                boxes[:, [0, 2]] *= w
                boxes[:, [1, 3]] *= h
                im = cv2.imread(l["im_file"])
                sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False)
                l["segments"] = sam_results[0].masks.xyn
            
            for l in dataset.labels:
                texts = []
                lb_name = Path(l["im_file"]).with_suffix(".txt").name
                txt_file = seg_save_dir / lb_name
                cls = l["cls"]
                for i, s in enumerate(l["segments"]):
                    line = (int(cls[i]), *s.reshape(-1))
                    texts.append(("%g " * len(line)).rstrip() % line)
                if texts:
                    with open(txt_file, "a") as f:
                        f.writelines(text + "\n" for text in texts)
            
    LOGGER.info(f"Generated segment labels saved in {save_dir}")
    
# Example usage
yolo_bbox2segment('/ghome/c5mcv01/mcv-c5-team1/week2/src/ultralytics/data/images', 
                  save_dir='/ghome/c5mcv01/mcv-c5-team1/week2/src/ultralytics/data/seg_labels', 
                  sam_model='/ghome/c5mcv01/mcv-c5-team1/week2/src/ultralytics/sam_b.pt')