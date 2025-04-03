import pandas as pd
import os
import hashlib


# Load the cleaned CSV file
cleaned_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week4/src/dataset/cleaned.csv'
df = pd.read_csv(cleaned_csv_path)

# Define the image directory
img_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/images'

# Function to compute the hash of an image file
def get_image_hash(image_path):
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()  # Compute MD5 hash
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Create a dictionary to store image hashes and corresponding DataFrame indexes
image_hashes = {}
rows_to_merge = {}

# Iterate through DataFrame rows
for index, row in df.iterrows():
    img_name = row['Image_Name']
    image_path = f'{img_path}/{img_name}.jpg'

    if os.path.exists(image_path):
        img_hash = get_image_hash(image_path)

        if img_hash:
            if img_hash in image_hashes:
                # If hash already exists, store indexes of duplicate images
                existing_index = image_hashes[img_hash]
                if existing_index not in rows_to_merge:
                    rows_to_merge[existing_index] = [existing_index]  # Start with the first occurrence
                rows_to_merge[existing_index].append(index)  # Add duplicate row
            else:
                image_hashes[img_hash] = index

# Merge duplicate rows
for main_index, duplicate_indexes in rows_to_merge.items():
    # Combine Titles
    unique_titles = set()
    for idx in duplicate_indexes:
        unique_titles.add(df.at[idx, 'Title'])
    df.at[main_index, 'Title'] = list(unique_titles)

    # Print the titles for the merged images
    print(f"Image {df.at[main_index, 'Image_Name']} has the following titles: {','.join(unique_titles)}")

# Drop duplicate rows, keeping the first occurrence
df = df.drop(index=[idx for indexes in rows_to_merge.values() for idx in indexes[1:]])

# Ensure all Title entries are lists
for index in df.index:
    current_value = df.at[index, 'Title']
    
    # Check if current value is not already a list
    if isinstance(current_value, str):
        df.at[index, 'Title'] = [current_value]

# Save the updated DataFrame
updated_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week4/src/dataset/cleaned_merged.csv'
df.to_csv(updated_csv_path, index=False)

print(f"Updated DataFrame saved to {updated_csv_path}")
print(f"Final DataFrame size: {df.shape}")
