from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTImageProcessor
import pandas as pd
from torch.utils.data import DataLoader
from src.tokenizer import Tokenizer

import random
import torch


class Data(Dataset):
    def __init__(self, data, partition, img_path: str, tokenizer: Tokenizer, image_processor: ViTImageProcessor, text_max_len: int = 201, return_path=False):
        self.data = data
        self.img_path = img_path
        self.partition = partition
        
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
        self.num_captions = 1
        self.max_len = text_max_len
        self.return_path = return_path

    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        real_idx = self.num_captions * self.partition[idx]
        item = self.data.iloc[real_idx: real_idx+self.num_captions]
        
        # Image processing
        try:
            img_name = item["Image_Name"].reset_index(drop=True)[0]
            img = Image.open(f'{self.img_path}/{img_name}.jpg').convert('RGB')
        except FileNotFoundError:
            print(f"Error loading image {img_name}")
        img_batch = self.image_processor(images=[img], return_tensors="pt").pixel_values
        img = img_batch[0]
    
        # Caption processing
        caption = item["Title"].reset_index(drop=True)[random.choice(list(range(self.num_captions)))]
        cap_idx = self.tokenizer.encode(caption)
        
        if not self.return_path:
            return img, torch.tensor(cap_idx)
        return img, torch.tensor(cap_idx), f'{self.img_path}/{img_name}.jpg'
    
    
# Example of usage
if __name__ == "__main__":
    # Paths
    img_path = "/ghome/c5mcv01/mcv-c5-team1/week3/data/images"
    train_csv_path = "/ghome/c5mcv01/mcv-c5-team1/week3/data/train.csv"
    
    # Load tokenizer
    tokenizer = Tokenizer()
    
    # Load image processor
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    # Load the dataset
    train_df = pd.read_csv(train_csv_path)
    partitions = {
        'train': list(train_df.index)
    }

    # Print dataset sizes
    print(f"Training set: {len(train_df)} samples")

    # Load data
    data_train = Data(train_df, partitions['train'], img_path=img_path, tokenizer=tokenizer, image_processor=image_processor)

    # Create dataloaders
    dataloader_train = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    # Get a batch
    for batch in dataloader_train:
        print(batch)
        break