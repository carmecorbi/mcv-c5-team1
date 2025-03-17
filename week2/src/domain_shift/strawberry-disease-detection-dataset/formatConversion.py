import os
import json
import glob
import argparse
import numpy as np
from PIL import Image
from labelme import utils
from datasets import Dataset, DatasetDict
from datasets import Image as DatasetImage

CLASS_LABELS = ["Angular Leafspot", "Anthracnose Fruit Rot", "Blossom Blight", "Gray Mold", "Leaf Spot", "Powdery Mildew Fruit", "Powdery Mildew Leaf"]
DATASET_PATH = '/home/usuaris/imatge/judit.salavedra/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/'
label_name_to_value = {
    "Background": 0,
    "Angular Leafspot": 1,
    "Anthracnose Fruit Rot": 2,
    "Blossom Blight": 3,
    "Gray Mold": 4,
    "Leaf Spot": 5,
    "Powdery Mildew Fruit": 6,
    "Powdery Mildew Leaf": 7
}

def json_to_mask(json_file, out_dir):
    """
    Converts a LabelMe JSON file to a segmentation mask image.
        output mask: 1-channel .png (each pixel corresponds to a class ID)
    """
    # Load the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Get image dimensions from the JSON metadata (if available)
    height = data.get("imageHeight")
    width = data.get("imageWidth")
    
    # Convert shapes (polygons) to a label image.
    # This function returns a label image and a list of label names.
    cls_mask, ins_mask = utils.shapes_to_label(img_shape=(height, width), shapes=data["shapes"], label_name_to_value=label_name_to_value)
    
    # Save the class mask image as a PNG
    out_file_cls = os.path.join(out_dir, os.path.splitext(os.path.basename(json_file))[0] + "_class_mask.png")
    class_mask_img = Image.fromarray(cls_mask.astype(np.uint8))
    class_mask_img.save(out_file_cls)
    print(f"Saved class mask: {out_file_cls}")

    # Save the instance mask image as a PNG
    out_file_ins = os.path.join(out_dir, os.path.splitext(os.path.basename(json_file))[0] + "_instance_mask.png")
    instance_mask_img = Image.fromarray(ins_mask.astype(np.uint8))
    instance_mask_img.save(out_file_ins)
    print(f"Saved instance mask: {out_file_ins}")


def json_to_HFmask(json_file, out_dir):
    """
    Converts a LabelMe JSON file to a segmentation mask image.
    Output mask: 3-channel .png:
      - 1st channel: Class IDs
      - 2nd channel: Instance mask
      - 3rd channel: Empty (all zeros)
    """
    # Load the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Get image dimensions from the JSON metadata (if available)
    height = data.get("imageHeight")
    width = data.get("imageWidth")

    # Convert shapes (polygons) to a label image
    cls_mask, ins_mask = utils.shapes_to_label(img_shape=(height, width), shapes=data["shapes"], label_name_to_value=label_name_to_value)

    # Create a 3-channel mask
    mask_3channel = np.zeros((height, width, 3), dtype=np.uint8)
    mask_3channel[..., 0] = cls_mask  # Class IDs
    mask_3channel[..., 1] = ins_mask  # Instance mask
    mask_3channel[..., 2] = 0         # Empty channel (all zeros)

    # Save the combined 3-channel mask as a PNG
    out_file_combined = os.path.join(out_dir, os.path.splitext(os.path.basename(json_file))[0] + ".png")
    combined_mask_img = Image.fromarray(mask_3channel)
    combined_mask_img.save(out_file_combined)
    print(f"Saved combined mask: {out_file_combined}")

def get_HF_annotations():
    for d in ["val"]:
        # Create output directory if it doesn't exist
        output_dir = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/"+d+"/annotations"
        os.makedirs(output_dir, exist_ok=True)
        input_dir = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/"+d+"/labels"
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        if not json_files:
            print("No JSON files found in the specified input directory.")
    
        # Convert each JSON file to a mask image
        for json_file in json_files:
            json_to_HFmask(json_file, output_dir)


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

def load_data(image_dir, mask_dir):
    images = []
    annotations = []
    print
    for img_name in sorted(os.listdir(image_dir)):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(image_dir, img_name)
            mask_name = img_name.replace(".jpg", ".png")  # Change extension for mask
            mask_path = os.path.join(mask_dir, mask_name)
            
            if os.path.exists(mask_path):  # Ensures the mask exists
                images.append(Image.open(img_path).convert("RGB"))
                annotations.append(Image.open(mask_path))
    
    return {
        "image": images,
        "annotation": annotations,
    }



if __name__ == "__main__":    
    
    #get_HF_annotations()

    parser = argparse.ArgumentParser(description='KittiMots Dataset')
    parser.add_argument('-t', '--token', help="Token to upload dataset", required=False)
    args = parser.parse_args()
    token = args.token

    train_dir = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/train/"
    val_dir = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/val/"
    # Load train and validation data
    train_data = load_data(train_dir+"images", train_dir+"annotations")
    val_data = load_data(val_dir+"images", val_dir+"annotations")

    label2id = {
        "Background": 0,
        "Angular Leafspot": 1,
        "Anthracnose Fruit Rot": 2,
        "Blossom Blight": 3,
        "Gray Mold": 4,
        "Leaf Spot": 5,
        "Powdery Mildew Fruit": 6,
        "Powdery Mildew Leaf": 7
    }


    dataset = create_instance_segmentation_dataset(label2id, train=train_data, validation=val_data)
    dataset.push_to_hub("jsalavedra/strawberry_disease", token=token)
    