import os
import torch
import albumentations as A
import numpy as np

from datasets import Dataset, DatasetDict
from pycocotools import mask as mask_utils
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer
from tqdm import tqdm
from functools import partial
from metrics import compute_metrics


# Dataset paths
data_dir = "/ghome/c5mcv01/mcv-c5-team1/data/"
annotations_dir = os.path.join(data_dir, "instances_txt")
train_img_dir = os.path.join(data_dir, "training/train")
val_img_dir = os.path.join(data_dir, "training/val")

# Valid classes of kitti mots gt
valid_classes = {1: "car", 2: "pedestrian"}

id2label = {
    0: "car",
    1: "pedestrian",
}
label2id = {v: k for k, v in id2label.items()}

# Function to read and parse annotations
def parse_annotations(seq_id):
    """Reads the annotation file and converts it into a structured dictionary."""
    txt_file = os.path.join(annotations_dir, f"{seq_id}.txt")
    if not os.path.exists(txt_file):
        return {}

    annotations = {}
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue  # Skip malformed lines
            
            frame = parts[0].zfill(6)  # Normalize the frame to 6 digits
            track_id, class_id = parts[1], int(parts[2])
            height, width = int(parts[3]), int(parts[4])
            rle_mask = parts[5]

            if class_id in valid_classes:
                if frame not in annotations:
                    annotations[frame] = []
                annotations[frame].append({
                    "track_id": int(track_id),
                    "class_id": class_id - 1, # To use class 0 and 1
                    "height": height,
                    "width": width,
                    "mask": rle_mask
                })
    return annotations

# Function to load data
def load_data(img_dir):
    """Loads image paths and their annotations into a list."""
    data = []
    image_id_counter = 0

    for seq_folder in tqdm(sorted(os.listdir(img_dir)), desc="Loading dataset..."):
        seq_path = os.path.join(img_dir, seq_folder)
        if not os.path.isdir(seq_path):
            continue  # Skip non-directories

        # Parse annotations once per sequence
        annotations = parse_annotations(seq_folder)
        
        for img_name in sorted(os.listdir(seq_path)):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue  # Skip invalid files

            img_path = os.path.join(seq_path, img_name)
            img_id = os.path.splitext(img_name)[0]  # ID without extension

            # Only process images with annotations
            if img_id in annotations:
                # Get image dimensions
                from PIL import Image
                with Image.open(img_path) as img:
                    width, height = img.size
                
                    objects = {
                        "id": [],
                        "area": [],
                        "bbox": [],
                        "category": []
                    }

                    for ann in annotations[img_id]:
                        track_id = ann["track_id"]
                        class_id = ann["class_id"]
                        
                        # Convert RLE to bbox
                        rle_obj = {'counts': ann["mask"], 'size': [ann["height"], ann["width"]]}
                        bbox = [int(coord) for coord in mask_utils.toBbox(rle_obj)]
                        area = bbox[2] * bbox[3]

                        objects["id"].append(track_id)
                        objects["area"].append(area)
                        objects["bbox"].append(bbox)
                        objects["category"].append(class_id)

                    data.append({
                        "image_id": image_id_counter,
                        "image": img,
                        "width": width,
                        "height": height,
                        "objects": objects
                    })
                    image_id_counter += 1
    return data

# Load training and validation data
print("Loading data...")
train_data = load_data(train_img_dir)
val_data = load_data(val_img_dir)

# Convert to Hugging Face Dataset
print("Creating HuggingFace datasets...")
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
dataset_dict = DatasetDict({"train": train_dataset, "val": val_dataset})
print(f"Example data point: {dataset_dict["train"][0]}")

# Train augmentation
train_augment_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
)
# Validation augmentation
validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
)

# Get the image processor
checkpoint = "facebook/detr-resnet-50-dc5"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# Taken from: https://huggingface.co/docs/transformers/tasks/object_detection
def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }

# Taken from: https://huggingface.co/docs/transformers/tasks/object_detection 
def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)
    return result


train_transform_batch = partial(
    augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
)
validation_transform_batch = partial(
    augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
)

# Apply transformations
dataset_dict["train"] = dataset_dict["train"].with_transform(train_transform_batch)
dataset_dict["val"] = dataset_dict["val"].with_transform(validation_transform_batch)
# print(f"Dataset with transforms: {dataset_dict["train"][0]}")

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data

eval_compute_metrics_fn = partial(
    compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
)

# Load model for object detection from same checkpoint
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Set the training arguments
training_args = TrainingArguments(
    output_dir="detr_finetuned_kitti",
    num_train_epochs=30,
    fp16=False,
    per_device_train_batch_size=10, # Don't change this (adjusted for RTX 3090 24GB)
    dataloader_num_workers=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    weight_decay=1e-4,
    max_grad_norm=0.01,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    push_to_hub=False,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["val"],
    processing_class=image_processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)

# Start training process
trainer.train()