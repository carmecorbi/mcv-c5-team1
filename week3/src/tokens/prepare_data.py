import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader


def load_data(csv_path):
    """Load data from .csv file and preprocess it"""

    data = pd.read_csv(csv_path)
    df = data[['Image_Name', 'Title']]
    df = df[df['Image_Name'] != '#NAME?']  # Filter out rows with '#NAME?'
    df['Title'] = df['Title'].fillna('').astype(str)
    return df

def extract_characters(df, char_set_path: str):
    """Extract unique characters from 'Title' column and save it"""

    special_chars = ['<SOS>', '<EOS>', '<PAD>']
    all_chars = set()

    # Collect unique characters from titles
    for caption in df['Title']:
        all_chars.update(caption)  

    all_chars_list = special_chars + list(all_chars)

    # Save to file
    with open(char_set_path, "wb") as f:
        pickle.dump(all_chars_list, f)

    return all_chars_list

def split_data(df, partitions_dir: str, test_size=0.2, val_size=0.5):
    """Split dataset into training, validation, and test sets"""

    if not os.path.exists(partitions_dir):
        os.makedirs(partitions_dir)

    print("Splitting dataset into partitions...")
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=42, shuffle=True)

    train_csv = os.path.join(partitions_dir, 'train.csv')
    val_csv   = os.path.join(partitions_dir, 'val.csv')
    test_csv  = os.path.join(partitions_dir, 'test.csv')

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
    char_set_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/char_set.pkl'

    csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/raw_data.csv'
    
    df = load_data(csv_path)
    #chars_set = extract_characters(df, char_set_path)
    #train, val, test = split_data(df, partitions_dir, test_size=0.2, val_size=0.5)


    total_images = df.shape[0]
    print("Total images:", total_images)

    all_chars = extract_characters(df, char_set_path)  


    print("Character Set Loaded:", all_chars)

    # Verify the dictionaries
    NUM_CHAR = len(char2idx)   

#if __name__ == "__main__":

    














