from torch import nn
from src.dataset.data import Data
from torch.utils.data import DataLoader
from src.models.baseline import Model
from src.trainer import Trainer
from torch.optim import Adam
from src.dataset.prepare_data import load_data
from src.tokens.char import CharTokenizer

import torch
import pandas as pd


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 60
EPOCHS = 10


# TODO: Change these paths
csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/raw_data.csv'
img_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/images'
train_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/train.csv'
val_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/val.csv'
test_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/test.csv'

# Load data
df = load_data(csv_path)
special_chars = ['<SOS>', '<EOS>', '<PAD>']

# Extract unique characters from the 'Title' column
all_chars = set()

# Loop over each caption in the 'Title' column
for caption in df['Title']:
    all_chars.update(caption)
all_chars_list = special_chars + list(all_chars)

# Create the char2idx and idx2char dictionaries
char2idx = {char: idx for idx, char in enumerate(sorted(all_chars_list))}
idx2char = {idx: char for char, idx in char2idx.items()}

# Get the tokenizer
char_tokenizer = CharTokenizer(all_chars_list, special_chars=special_chars, max_len=201)

# Load the dataset
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
test_df = pd.read_csv(test_csv_path)
partitions = {
    'train': list(train_df.index),
    'val': list(val_df.index),
    'test': list(test_df.index)
}

# Print dataset sizes
print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

data_train = Data(df, partitions['train'], img_path=img_path, char2idx=char2idx, chars=all_chars_list, tokenizer=char_tokenizer)
data_valid = Data(df, partitions['val'], img_path=img_path, char2idx=char2idx, chars=all_chars_list, tokenizer=char_tokenizer)
data_test = Data(df, partitions['test'], img_path=img_path, char2idx=char2idx, chars=all_chars_list, tokenizer=char_tokenizer)

dataloader_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_valid = DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=False)
dataloader_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)

model = Model(num_char=len(char_tokenizer), char2idx=char2idx).to(DEVICE)
args = {
    'epochs': EPOCHS,
}
trainer = Trainer(model, Adam(model.parameters(), lr=1e-2), nn.CrossEntropyLoss(), char_tokenizer, DEVICE, **args)
trainer.train(dataloader_train, dataloader_valid, dataloader_test)
