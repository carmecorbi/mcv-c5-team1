import os
import json
import pickle
import pandas as pd
from PIL import Image
import easyocr
import shutil

def load_data(csv_path):
    """Load data from .csv file and preprocess it"""

    data = pd.read_csv(csv_path)
    df = data[['Image_Name', 'Title']]
    
    # Remove rows with NaN values in either column
    df = df.dropna(subset=['Image_Name', 'Title'])
    
    # Remove rows with empty strings in either column
    df = df[(df['Image_Name'] != '') & (df['Title'] != '')]
    
    # Filter out rows with '#NAME?'
    df = df[df['Image_Name'] != '#NAME?']  # Filter out rows with '#NAME?'
    
    # Set title to string type
    df['Title'] = df['Title'].astype(str)
    return df



csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/raw_data.csv'
df = load_data(csv_path)

## Step 1: Filter images with text

# Define the directory where the images are located

img_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/images'  # Modify this path as needed

reader = easyocr.Reader(['en'])

# Function to detect text in an image using EasyOCR
def contains_text(image_path):
    try:
        # Apply OCR to extract text from the image
        result = reader.readtext(image_path)

        # If text is detected, return True
        return bool(result)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

# Filter the DataFrame to include only images that contain text
df_with_text_images = df[df['Image_Name'].apply(lambda img_name: contains_text(f'{img_path}/{img_name}.jpg'))]

# Save the names of images containing text
image_names_with_text = df_with_text_images['Image_Name'].tolist()

# Print the names of images containing text
print("Images with text:")
for img_name in image_names_with_text:
    print(img_name)

# Create a directory to store images with text
output_dir = 'images_with_text'
os.makedirs(output_dir, exist_ok=True)

# Copy images with text to the new directory
for img_name in image_names_with_text:
    img_path_full = f"{img_path}/{img_name}.jpg"
    if os.path.exists(img_path_full):
        shutil.copy(img_path_full, f"{output_dir}/{img_name}.jpg")
    else:
        print(f"Image not found: {img_path_full}")

# Once I have manually selected the ones to remove, create a new DataFrame --> cleaned.csv

images_with_text_dir = '/ghome/c5mcv01/mcv-c5-team1/week4/src/dataset/images_with_text'

# Get the list of image names in the "images_with_text" folder
image_files = [f for f in os.listdir(images_with_text_dir) if f.endswith('.jpg')]

# Extract only the image names (without extensions)
image_names_with_text = [os.path.splitext(f)[0] for f in image_files]

# Filter the DataFrame to remove rows with images in "images_with_text"
df_cleaned = df[~df['Image_Name'].isin(image_names_with_text)]

# Save the new DataFrame to a CSV file
cleaned_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week4/src/dataset/cleaned.csv'
df_cleaned.to_csv(cleaned_csv_path, index=False)

# Print the size of the original and cleaned DataFrame
print(f"Original DataFrame size: {df.shape}") 
print(f"Cleaned DataFrame size: {df_cleaned.shape}")  



