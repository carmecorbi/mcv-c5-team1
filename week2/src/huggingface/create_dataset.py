import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from datasets import Image as DatasetImage
from huggingface_hub import login

# Run this and it will prompt for your token
login()

# Dataset paths
data_dir = "/ghome/c5mcv01/mcv-c5-team1/data/"
instances_dir = os.path.join(data_dir, "instances")
train_img_dir = os.path.join(data_dir, "training/train")
val_img_dir = os.path.join(data_dir, "training/val")

label2id = {
    "background": 0,
    "car": 1,
    "person": 2,
}

# Function to read and parse annotations
def parse_annotations(seq_id: int, img_id: int) -> str:
    """Reads the annotation file and converts it into a structured dictionary.

    Args:
        seq_id (int): Sequence ID in integer format.
        img_id (int): Image ID in integer format.

    Returns:
        str: Path to the mask PNG file.
    """
    mask_file = os.path.join(instances_dir, f"{seq_id:04d}", f"{img_id:06d}.png")
    if not os.path.exists(mask_file):
        return None
    return mask_file

# Function to process a single example
def process_example(img_path: str, seq_folder: str, img_id: int, image_id: int) -> dict:
    """Process a single example and return metadata (not the actual image)."""
    annotations = parse_annotations(int(seq_folder), int(img_id))
    
    # Get image dimensions without fully loading it
    with Image.open(img_path) as img:
        width, height = img.size
    
    sample = {
        "image_id": image_id,
        "image_path": img_path,
        "seq_folder": seq_folder,
        "img_id": img_id,
        "width": width,
        "height": height,
        "annotations": annotations
    } 
    return sample

def convert_annotation(kitti_annotation):
        # Original KITTI-MOTS format
        obj_ids = np.unique(kitti_annotation)
        
        # Create empty 3-channel annotation
        height, width = kitti_annotation.shape
        new_annotation = np.zeros((height, width, 3), dtype=np.uint8)
        
        for obj_id in obj_ids:
            # Skip background (0)
            if obj_id == 0:
                continue
                
            # Extract class_id and instance_id
            class_id = obj_id // 1000
            instance_id = obj_id % 1000
            
            # Skip class_id 10 as requested
            if class_id == 10:
                continue
                
            # Create mask for this object
            mask = (kitti_annotation == obj_id)
            
            # Set channel 1 to class_id
            new_annotation[:, :, 0][mask] = class_id
            
            # Set channel 2 to instance_id (ensure each instance has unique ID)
            new_annotation[:, :, 1][mask] = instance_id
            
            # Channel 3 remains zeros
        
        # Convert to PIL Image
        return Image.fromarray(new_annotation)

def fill_data_splits(train_img_dir, val_img_dir, instances_dir):
    """
    Fill the train_split and validation_split dictionaries with images and their annotations.
    
    Args:
        train_img_dir (str): Path to the training images directory.
        val_img_dir (str): Path to the validation images directory.
        instances_dir (str): Path to the instances (annotations) directory.
        
    Returns:
        tuple: (train_split, validation_split) dictionaries with 'image' and 'annotation' lists.
    """
    train_split = {"image": [], "annotation": []}
    validation_split = {"image": [], "annotation": []}
    
    # Helper function to process a directory and fill the appropriate split
    def process_directory(img_dir, split_dict):
        print(f"Processing {img_dir}...")
        
        # Get all sequence directories
        seq_dirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
        
        for seq_id in seq_dirs:
            # Get all images in this sequence
            seq_path = os.path.join(img_dir, seq_id)
            image_files = [f for f in os.listdir(seq_path) if f.endswith('.png')]
            
            for img_file in tqdm(image_files, desc=f"Sequence {seq_id}"):
                frame_id = img_file.split('.')[0]
                
                # Construct paths
                img_path = os.path.join(seq_path, img_file)
                # Make sure we're using the same format for the annotation path
                mask_path = os.path.join(instances_dir, seq_id, f"{frame_id}.png")
                
                # Skip if annotations don't exist
                if not os.path.exists(mask_path):
                    continue
                    
                try:
                    # Load images
                    image = Image.open(img_path)
                    kitti_annotation = np.array(Image.open(mask_path))
                    annotation = convert_annotation(kitti_annotation)
                    
                    # Add to split dictionary
                    split_dict["image"].append(image)
                    split_dict["annotation"].append(annotation)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Process training and validation directories
    process_directory(train_img_dir, train_split)
    process_directory(val_img_dir, validation_split)
    
    print(f"Loaded {len(train_split['image'])} training images and {len(validation_split['image'])} validation images")
    
    return train_split, validation_split

train_split, validation_split = fill_data_splits(train_img_dir, val_img_dir, instances_dir)

def create_instance_segmentation_dataset(label2id, **splits):
    dataset_dict = {}
    for split_name, split in splits.items():
        split["semantic_class_to_id"] = [label2id] * len(split["image"])
        dataset_split = (
            Dataset.from_dict(split)
            .cast_column("image", DatasetImage())
            .cast_column("annotation", DatasetImage())
        )
        dataset_dict[split_name] = dataset_split
    return DatasetDict(dataset_dict)

dataset = create_instance_segmentation_dataset(label2id, train=train_split, validation=validation_split)
dataset.push_to_hub("yeray142/kitti-mots-instance")