from typing import Literal
from PIL import Image
from detectron2.structures import BoxMode
from pycocotools import mask as mask_utils
import os


class CustomKittiMotsDataset:
    def __init__(self, data_dir, use_coco_ids = True, split: str = Literal['val', 'train']):
        """
        Custom dataset for KITTI-MOTS format
        
        Args:
            data_dir (str): Base directory containing the dataset
            split (str, optional): Either "train", "val", or None for all data
            use_coco_ids (bool, optional): Whether to use COCO class IDs (True) or KITTI-MOTS IDs (False)
        """
        self.data_dir = data_dir
        self.instances_dir = os.path.join(data_dir, 'instances_txt')
        self.image_dir = os.path.join(data_dir, 'training', split)
        
        # Class mapping (original KITTI-MOTS uses integers)
        self.classes = ["car", "pedestrian"]
        
        if use_coco_ids:
            self.class_map = {
                1: 2, # Map to COCO classes (3 -> cars, 1 -> pedestrian)
                2: 0
            }
        else:
            self.class_map = {
                1: 0, # Map to KITTI-MOTS classes
                2: 1
            }
        
        # Find all image paths and build an index
        self.samples = []
        
        # Get all sequence directories
        sequence_dirs = sorted([d for d in os.listdir(self.image_dir) 
                               if os.path.isdir(os.path.join(self.image_dir, d))])
        
        for sequence_dir in sequence_dirs:
            sequence_path = os.path.join(self.image_dir, sequence_dir)
            sequence_id = sequence_dir
            
            # Get all image files in the sequence
            image_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.png')])
            
            for image_file in image_files:
                frame_id = os.path.splitext(image_file)[0]
                image_path = os.path.join(sequence_path, image_file)
                self.samples.append({
                    'sequence_id': sequence_id,
                    'frame_id': frame_id,
                    'image_path': image_path
                })
        
        # Use all data
        self.indices = list(range(len(self.samples)))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the real index
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]
        
        sequence_id = sample['sequence_id']
        frame_id = sample['frame_id']
        image_path = sample['image_path']
        
        # Load image
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Create a unique image ID
        image_id = f"{sequence_id}_{frame_id}"
        
        # Look for the annotation file
        ann_file = f"{sequence_id}.txt"
        ann_path = os.path.join(self.instances_dir, ann_file)
        
        annotations = []
        
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                lines = f.readlines()
            
            # Filter annotations for this frame
            frame_annotations = [line for line in lines if int(line.strip().split()[0]) == int(frame_id)]
            
            for line in frame_annotations:
                parts = line.strip().split()
                if len(parts) >= 5:
                    frame_id_ann = int(parts[0])
                    instance_full_id = int(parts[1])
                    
                    # Extract class ID and instance ID
                    class_id = instance_full_id // 1000
                    
                    # Only process valid class IDs
                    if class_id not in self.class_map:
                        continue
                    
                    # Extract RLE from the last part
                    rle_str = parts[5]
                    h, w = int(parts[3]), int(parts[4])
                    rle_obj = {'counts': rle_str, 'size': [h, w]}
                    
                    # Convert RLE to bbox
                    bbox = [int(coord) for coord in mask_utils.toBbox(rle_obj)]
                    x, y, w, h = bbox
                    bbox_xyxy = [x, y, x + w, y + h]
                    
                    # Create annotation dictionary
                    annotation = {
                        "bbox": bbox_xyxy,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": self.class_map[class_id],
                    }
                    annotations.append(annotation)
        
        record = {
            "file_name": image_path,
            "height": img_height,
            "width": img_width,
            "image_id": image_id,
            "annotations": annotations,
        }
        
        return record
    