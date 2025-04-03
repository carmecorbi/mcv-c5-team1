import os
import json
import pandas as pd
import ast

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

def main_prepare_data_split():
    csv_path = '/ghome/c5mcv01/mcv-c5-team1/week4/data/final.csv'
    partitions_dir = '/ghome/c5mcv01/mcv-c5-team1/week4/data'
    
    df = load_data(csv_path)
    _ = split_data(df, partitions_dir, test_size=0.2, val_size=0.5)

    total_images = df.shape[0]
    print("Total images:", total_images) 

if __name__ == "__main__":
    main_prepare_data_split()















