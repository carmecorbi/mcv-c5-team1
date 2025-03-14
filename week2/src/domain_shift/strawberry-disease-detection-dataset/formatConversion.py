import os
import json
import glob
import argparse
import numpy as np
from PIL import Image
from labelme import utils

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
    label_img, label_names = utils.shapes_to_label(img_shape=(height, width), shapes=data["shapes"], label_name_to_value=label_name_to_value)
    
    # Optionally, you can save the label names (e.g., for later mapping)
    labels_file = os.path.join(out_dir, os.path.splitext(os.path.basename(json_file))[0] + "_labels.txt")
    with open(labels_file, "w") as f:
        for label in label_names:
            f.write(str(label) + "\n")
    
    # Save the label image (mask) as a PNG
    out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(json_file))[0] + "_mask.png")
    mask = Image.fromarray(label_img.astype(np.uint8))
    mask.save(out_file)
    print(f"Saved mask: {out_file}")


if __name__ == "__main__":    
    
    # Get all JSON files in the input directory
    for d in ["train"]:
        # Create output directory if it doesn't exist
        output_dir = "/home/usuaris/imatge/judit.salavedra/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/"+d+"/labels"
        os.makedirs(output_dir, exist_ok=True)
        input_dir = "/home/usuaris/imatge/judit.salavedra/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/"+d+"/labels"
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        if not json_files:
            print("No JSON files found in the specified input directory.")
    
        # Convert each JSON file to a mask image
        for json_file in json_files:
            json_to_mask(json_file, output_dir)