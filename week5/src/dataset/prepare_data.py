import os
import json
import pandas as pd
import ast
import csv

from sklearn.model_selection import train_test_split


def load_data(csv_path):
    """Load data from .csv file and preprocess it"""

    data = pd.read_csv(csv_path)
    data['Title'] = data['Title'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [x] if isinstance(x, str) else x)
    return data

def split_data(df, partitions_dir: str, test_size=0.2, val_size=0.5):
    """Split dataset into training, validation, and test sets"""
    if not os.path.exists(partitions_dir):
        os.makedirs(partitions_dir)

    # Split the dataset into train, validation, and test sets
    print("Splitting dataset into partitions...")
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=42, shuffle=True)

    # Create the paths for the CSV files
    train_csv = os.path.join(partitions_dir, 'train.csv')
    val_csv   = os.path.join(partitions_dir, 'val.csv')
    test_csv  = os.path.join(partitions_dir, 'test.csv')

    # Save the partitions to CSV files
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return train_df, val_df, test_df

def load_partitions_from_csv(train_csv: str, val_csv: str, test_csv: str, partitions_path: str):
    """Load precomputed sets (.csv files) and return a dictionary of indices or identifiers"""

    print("Loading partitions from precomputed CSV files...")
    # Assume each CSV has a column 'id' or use the DataFrame index if not
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    partitions = {
        'train': list(train_df.index),
        'val': list(val_df.index),
        'test': list(test_df.index)
    }

    with open(partitions_path, "w") as f:
        json.dump(partitions, f)
    return partitions

def clean_dataset_captions(csv_path, out_csv_path):

    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Collapse multiple spaces into one
    df['Title'] = df['Title'].apply(lambda x: ' '.join(str(x).split()).lower())

    # Save the cleaned DataFrame back to a new CSV
    df.to_csv(out_csv_path, index=False)

    print(f"Cleaned CSV saved to: {out_csv_path}")

def create_csv_from_synthetic(out_file_path: str, csv_file_path: str):

    # Initialize list to store data
    data = []  
    current_title = None

    path = "/ghome/c5mcv01/mcv-c5-team1/week5"
    # For job 72945 the variations were not finished for the last image-title pair
    stop_caption = "Generating variations for caption: Pizza with Eggs, Roasted Red Peppers, Olives and Arugula"
    stop_job_path = '/ghome/c5mcv01/mcv-c5-team1/week5/logs/job_test_generate_c5mcv01_72945.out'

    # Read the .out file line-by-line.
    with open(out_file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Stop processing if the stop caption is encountered
            if line.startswith(stop_caption) and out_file_path == stop_job_path:
                print(f"Stop caption found: {line}")
                break

            # Check if the line is for a caption variation title.
            if line.startswith("Generating image for variation:"):
                # Splits the string on the first occurrence of ":" into a list of 2 parts, 
                # keep the 2nd element, and remove any leading or trailing whitespace.
                current_title = line.split(":", 1)[1].strip()

                # If the title starts whith "-" remove it and leading spaces
                if current_title.startswith("-"):
                    current_title = current_title[1:].strip()
                
                # Convert to lowercase & remove extra spaces
                current_title = ' '.join(current_title.split()).lower()

            # Check if the line is for the image path.
            elif line.startswith("Image saved as:"):
                # Splits the string on the first occurrence of ":" into a list of 2 parts, 
                # keep the 2nd element, and remove any leading or trailing whitespace.
                image_path_raw = line.split(":", 1)[1].strip()
                basename = os.path.basename(image_path_raw)
                image_name = os.path.splitext(basename)[0]

                # Append the record to our data list.
                data.append({
                    "Image_Name": image_name,
                    "Title": str([current_title])
                })

    # Write the extracted data to a CSV file.
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Image_Name', 'Title']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)
    
    print(f"CSV file created at: {csv_file_path}")

def merge_csv_files(csv_files: str, csv_file_path: str):

    # Read and concatenate them
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # Save the merged result
    merged_df.to_csv(csv_file_path, index=False)

    print(f"Merge complete! Saved at {csv_file_path}")

def split_data_with_synthetic(original_csv: str, synthetic_csv: str, partitions_dir: str, test_size=0.1, val_size=0.1):
    """
    Split image-title data into Train/Val/Test partitions:
        Val: 10% total, 50% original data & 50% synthetic data
        Test: 10% total, 100% original data
        Train: 80% total, remaining samples
    """

    os.makedirs(partitions_dir, exist_ok=True)

    # Load datasets & compute the total number of image-title-pairs
    original_df = pd.read_csv(original_csv)
    synthetic_df = pd.read_csv(synthetic_csv)
    total_samples = len(original_df) + len(synthetic_df)
    print(f"Total number of samples: {total_samples}")

    # Compute val/test sizes
    num_test = int(test_size * total_samples)     
    num_val = int(val_size * total_samples)   
    print(f"Expected Val Samples: {num_val}")
    print(f"Expected Test Samples: {num_test}")         

    # 1) Split original dataset into Test and the rest
    test_set, original_remaining = train_test_split(
        original_df, 
        test_size=len(original_df) - num_test, 
        random_state=42,
        shuffle=True
    )

    # 2) Split datasets to generate Val set and rest
    num_val_orig = num_val // 2
    num_val_syn = num_val - num_val_orig
    print(f"Expected Val Samples from Original dataset: {num_val_orig}")  
    print(f"Expected Val Samples from Synthetic dataset: {num_val_syn}")  

    # Data from original dataset
    val_orig, train_orig = train_test_split(
        original_remaining, 
        test_size=len(original_remaining) - num_val_orig,
        random_state=42,
        shuffle=True
    )
    
    # Data from synthetic dataset
    val_syn, train_syn = train_test_split(
        synthetic_df, 
        test_size=len(synthetic_df) - num_val_syn,
        random_state=42,
        shuffle=True
    )

    # Combined Val set
    val_set = pd.concat([val_orig, val_syn]).sample(frac=1, random_state=42).reset_index(drop=True)

    # 3) Get Train set
    train_set = pd.concat([train_orig, train_syn]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save Partitions
    train_set.to_csv(os.path.join(partitions_dir, 'train.csv'), index=False)
    val_set.to_csv(os.path.join(partitions_dir, 'val.csv'), index=False)
    test_set.to_csv(os.path.join(partitions_dir, 'test.csv'), index=False)

    print(f"Train samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    print(f"Test samples: {len(test_set)}")
    print(f"Splits saved to {partitions_dir}")

def remove_duplicated_samples(csv_path: str, out_csv: str):
    # Read the csv file
    df = pd.read_csv(csv_path)
    original_count = len(df)

    # remove duplicates based on both columns
    df_clean = df.drop_duplicates()
    unique_count = len(df_clean)

    # Save the cleaned DataFrame to a new csv
    df_clean.to_csv(out_csv, index=False)

    print(f"Original count: {original_count} | Unique rows: {unique_count} | Duplicates removed: {original_count - unique_count}")
    print(f"Duplicates removed! Cleaned CSV saved at: {out_csv}")

def main_create_synthetic_csv():
    # Job from Fri to Mon
    out_file_path = '/ghome/c5mcv01/mcv-c5-team1/week5/logs/job_test_generate_c5mcv01_72945.out'
    csv_file_path = '/ghome/c5mcv01/mcv-c5-team1/week5/data/synthetic_1.csv'
    # Job from Mon to Mon
    #out_file_path = '/ghome/c5mcv01/mcv-c5-team1/week5/logs/job_test_generate_c5mcv01_73392.out'
    #csv_file_path = '/ghome/c5mcv01/mcv-c5-team1/week5/data/synthetic_2.csv'
    # Job from Mon to Tue
    #out_file_path = '/ghome/c5mcv01/mcv-c5-team1/week5/logs/job_test_generate_c5mcv01_73431.out'
    #csv_file_path = '/ghome/c5mcv01/mcv-c5-team1/week5/data/synthetic_3.csv'

    create_csv_from_synthetic(out_file_path, csv_file_path)

def main_prepare_data_split():
    csv_path = '/ghome/c5mcv01/mcv-c5-team1/week4/data/final.csv'
    partitions_dir = '/ghome/c5mcv01/mcv-c5-team1/week4/data'
    
    df = load_data(csv_path)
    _ = split_data(df, partitions_dir, test_size=0.2, val_size=0.5)

    total_images = df.shape[0]
    print("Total images:", total_images) 


if __name__ == "__main__":

    csv_files = [
        '/ghome/c5mcv01/mcv-c5-team1/week5/data/synthetic_1.csv',
        '/ghome/c5mcv01/mcv-c5-team1/week5/data/synthetic_2.csv',
        '/ghome/c5mcv01/mcv-c5-team1/week5/data/synthetic_3.csv'
    ]

    data_folder = '/ghome/c5mcv01/mcv-c5-team1/week5/data'
    csv_merged_synthetic = '/ghome/c5mcv01/mcv-c5-team1/week5/data/synthetic.csv'
    csv_merged_synthetic_cleaned = '/ghome/c5mcv01/mcv-c5-team1/week5/data/synthetic_cleaned.csv'
    csv_original = '/ghome/c5mcv01/mcv-c5-team1/week4/data/final.csv'
    csv_original_cleaned = '/ghome/c5mcv01/mcv-c5-team1/week5/data/original.csv'

    #main_create_synthetic_csv()
    merge_csv_files(csv_files, csv_merged_synthetic)
    remove_duplicated_samples(csv_merged_synthetic, csv_merged_synthetic_cleaned)
    #clean_dataset_captions(csv_original, csv_original_cleaned)
    split_data_with_synthetic(csv_original_cleaned, csv_merged_synthetic_cleaned, data_folder, test_size=0.1, val_size=0.1)















