import os
import cv2
from detectron2.structures import BoxMode

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import json, random, torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from typing import Literal
from PIL import Image
from detectron2.data import DatasetMapper, detection_utils
from pycocotools import mask as mask_utils

import os
import xml.etree.ElementTree as ET


def get_YOLOData_dicts(data_dir):
    """
    Loads a dataset with YOLO-format annotations.

    Args:
        data_dir (str): Path to the dataset.
    
    Returns:
        list[dict]: List of annotations in Detectron2's dataset format.
    """
    dataset_dicts = []
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")
    
    for idx, file_name in enumerate(sorted(os.listdir(image_dir))):
        if not file_name.lower().endswith((".jpg", ".png")):
            continue
        
        record = {}
        image_path = os.path.join(image_dir, file_name)
        record["file_name"] = image_path
        
        im = cv2.imread(image_path)
        if im is None:
            continue
        height, width = im.shape[:2]
        record["height"] = height
        record["width"] = width
        record["image_id"] = idx
        
        # Corresponding labels. Each annotation file should have one line per object:
        # class_id x_center y_center width height
        label_file = os.path.join(label_dir, os.path.splitext(file_name)[0] + ".txt")
        objs = []
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    # YOLO format: class_id, x_center, y_center, w, h (all normalized)
                    class_id, x_center, y_center, w, h = parts
                    class_id = int(class_id)
                    x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
                    
                    # Convert normalized coordinates to absolute pixel values
                    x_center_abs = x_center * width
                    y_center_abs = y_center * height
                    w_abs = w * width
                    h_abs = h * height
                    
                    # Convert center coordinates to [x_min, y_min, x_max, y_max]
                    x_min = max(int(x_center_abs - w_abs / 2), 0)
                    y_min = max(int(y_center_abs - h_abs / 2), 0)
                    x_max = min(int(x_center_abs + w_abs / 2), width)
                    y_max = min(int(y_center_abs + h_abs / 2), height)
                    
                    obj = {
                        "bbox": [x_min, y_min, x_max, y_max],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": class_id,
                    }
                    objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


class AlbumentationsMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True, augmentations=None):
        """Initializes albumentations mapper.

        Args:
            cfg (Any): Configuration for the model.
            is_train (bool, optional): Whether is train dataset. Defaults to True.
            augmentations (Any, optional): Augmentations from albumentations to apply. Defaults to None.
        """
        super().__init__(cfg, is_train)
        self.augmentations = augmentations

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = cv2.imread(dataset_dict["file_name"]) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_train and "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            bboxes = [obj["bbox"] for obj in annotations]
            category_ids = [obj["category_id"] for obj in annotations]

            
            transformed = self.augmentations(image=image, bboxes=bboxes, category_ids=category_ids)
            image = transformed["image"]
            
            # Update the bounding boxes with transformed coordinates
            for i, annotation in enumerate(annotations):
                if i < len(transformed["bboxes"]):
                    annotation["bbox"] = transformed["bboxes"][i]
            
            # Convert to Instances format for Detectron2
            annos = []
            for annotation in annotations:
                obj = {
                    "bbox": annotation["bbox"],
                    "bbox_mode": annotation.get("bbox_mode", BoxMode.XYWH_ABS),
                    "category_id": annotation["category_id"]
                }
                annos.append(obj)
            
            # Create Instances object with the correct image size
            instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = instances
        
        # Convert to CHW format
        image = image.transpose(2, 0, 1)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image))
        
        return dataset_dict

# Example usage to verify the correct format conversion (Aquarium)
'''
if __name__ == "__main__":
    for d in ["train", "val", "test"]:
        DatasetCatalog.register("AquariumDataCots_" + d, lambda d=d: get_YOLOData_dicts("aquarium-data-cots/" + d))
        MetadataCatalog.get("AquariumDataCots_" + d).set(thing_classes=['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'])
    metadata = MetadataCatalog.get("AquariumDataCots_train")

    dataset_dicts = get_YOLOData_dicts("aquarium-data-cots/train")
    output_dir = "output_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        # Define the output file path and save the image
        output_path = os.path.join(output_dir, f"visualized_image_{d['image_id']}.png")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
        print(f"Saved visualization to {output_path}")
'''
# Example usage to verify the correct format conversion (GlobalWheatHead2020)
'''
if __name__ == "__main__":
    for d in ["train", "val", "test"]:
        DatasetCatalog.register("GlobalWheat_" + d, lambda d=d: get_YOLOData_dicts("GlobalWheat/" + d))
        MetadataCatalog.get("GlobalWheat_" + d).set(thing_classes=['wheat_head'])
    metadata = MetadataCatalog.get("GlobalWheat_train")

    dataset_dicts = get_YOLOData_dicts("GlobalWheat/train")
    output_dir = "output_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        # Define the output file path and save the image
        output_path = os.path.join(output_dir, f"visualized_image_{d['image_id']}.png")
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
        print(f"Saved visualization to {output_path}")
'''