from torch import nn
from sklearn.model_selection import train_test_split
from src.dataset.data import Data
from torch.utils.data import DataLoader
from src.models.baseline import Model
from src.trainer import Trainer
from torch.optim import Adam

import torch
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 10


# TODO: Change these paths
csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/raw_data.csv'
img_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/images'
data = pd.read_csv(csv_path)

df = data[['Image_Name', 'Title']]

df['Title'] = df['Title'].fillna('').astype(str)

special_chars = ['<SOS>', '<EOS>', '<PAD>']

# Extract unique characters from the 'Title' column
all_chars = set()

# Loop over each caption in the 'Title' column
for caption in df['Title']:
    all_chars.update(caption)  # Add each unique character to the set

# Add special characters to the set of characters
all_chars.update(special_chars)
all_chars_list = list(all_chars)

# Create the char2idx and idx2char dictionaries
char2idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
idx2char = {idx: char for char, idx in char2idx.items()}

# Verify the dictionaries
NUM_CHAR = len(char2idx)   
print(all_chars)  

# First, split into 80% train and 20% temp (validation + test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

partitions = {
    'train': list(train_df.index),
    'val': list(val_df.index),
    'test': list(test_df.index)
}

# Print dataset sizes
print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

'''
# A simple example to calculate loss of a single batch (size 2)
dataset = Data(df, partitions['train'], img_path=img_path, char2idx=char2idx, chars=all_chars)

img1, caption1 = next(iter(dataset))
img2, caption2 = next(iter(dataset))
caption1 = torch.tensor(caption1)
caption2 = torch.tensor(caption2)
img = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)))
caption = torch.cat((caption1.unsqueeze(0), caption2.unsqueeze(0)))
img, caption = img.to(DEVICE), caption.to(DEVICE)
model = Model(num_char=NUM_CHAR, char2idx=char2idx).to(DEVICE)
pred = model(img)
crit = nn.CrossEntropyLoss()
loss = crit(pred, caption)
print(loss)
'''

data_train = Data(data, partitions['train'],img_path=img_path, char2idx=char2idx, chars=all_chars_list)
data_valid = Data(data, partitions['val'],img_path=img_path, char2idx=char2idx, chars=all_chars_list)
data_test = Data(data, partitions['test'],img_path=img_path, char2idx=char2idx, chars=all_chars_list)

dataloader_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_valid = DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=False)
dataloader_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)

model = Model(num_char=NUM_CHAR, char2idx=char2idx).to(DEVICE)
args = {
    'epochs': EPOCHS,
}

trainer = Trainer(model, Adam(model.parameters(), lr=1e-4), nn.CrossEntropyLoss(), DEVICE, kwargs=args)
trainer.train(dataloader_train, dataloader_valid, dataloader_test)
