import argparse
import pandas as pd
import ast
import os
import numpy as np

from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# Supress warnings from YOLOv8
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='ultralytics')


def filter_images_without_people(df, images_dir):
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
    
    # Iterate through each row in the dataframe
    filtered_df = df.copy()
    indices_to_drop = []
    for idx, row in tqdm(df.iterrows()):
        image_path = os.path.join(images_dir, row['Image_Name'] + '.jpg')
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found")
            raise FileNotFoundError(f"Image {image_path} not found")
        try:
            # Load image with PIL
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert PIL image to numpy array (what YOLO expects)
            img_array = np.array(img)
            results = model(img_array, conf=0.75)
            
            # Check if any person is detected
            # Class 0 is 'person' in COCO dataset used by YOLO
            for result in results:
                if 0 in result.boxes.cls:
                    print(f"Person detected in {row['Image_Name']}, marking for removal")
                    indices_to_drop.append(idx)
                    break
                    
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Remove all rows containing people at once
    filtered_df = filtered_df.drop(indices_to_drop)
    print(f"Removed {len(indices_to_drop)} images containing people")
    return filtered_df


def main():
    parser = argparse.ArgumentParser(description="Clean and process image dataset.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the cleaned CSV file.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing images.")
    args = parser.parse_args()

    # Load data from CSV
    data = pd.read_csv(args.csv_path)
    data['Title'] = data['Title'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [x] if isinstance(x, str) else x)
    
    # Filter out images with people
    cleaned_data = filter_images_without_people(data, args.images_dir)
    
    # Save cleaned data to CSV
    cleaned_data.to_csv(args.output_path, index=False)
    print(f"Cleaned data saved to {args.output_path}")
    
if __name__ == "__main__":
    main()