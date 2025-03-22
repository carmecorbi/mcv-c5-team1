from torch import nn
from src.dataset.data import Data
from torch.utils.data import DataLoader
from src.models.lstm import Model
from src.tokens.char import get_vocabulary
from src.tokens.bert import BertTokenizer
from src.lightning_trainer import train_with_lightning

import torch
import pandas as pd

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 60
EPOCHS = 100

# Model parameters
NUM_LAYERS = 2
FREEZE_BACKBONE = True
LEARNING_RATE = 1e-3

# This is for BertTokenizer
TEXT_MAX_LEN = 60


# TODO: Change these paths
csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/raw_data.csv'
img_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/images'
train_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/train.csv'
val_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/val.csv'
test_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/test.csv'

# Get char2idx
_, char2idx, _ = get_vocabulary(csv_path)

# Get the tokenizer
bert_tokenizer = BertTokenizer(max_length=TEXT_MAX_LEN)

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

# Load data
data_train = Data(train_df, partitions['train'], img_path=img_path, tokenizer=bert_tokenizer)
data_valid = Data(val_df, partitions['val'], img_path=img_path, tokenizer=bert_tokenizer)
data_test = Data(test_df, partitions['test'], img_path=img_path, tokenizer=bert_tokenizer)

# Create dataloaders
dataloader_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
dataloader_valid = DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
dataloader_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Create the model
model = Model(
    num_char=len(bert_tokenizer), 
    char2idx=char2idx, 
    text_max_len=TEXT_MAX_LEN,
    num_layers=NUM_LAYERS,
    freeze_backbone=FREEZE_BACKBONE).to(DEVICE)

# Lightning training
lightning_model, trainer = train_with_lightning(
    model=model,
    criterion=nn.CrossEntropyLoss(),
    tokenizer=bert_tokenizer,
    train_loader=dataloader_train,
    val_loader=dataloader_valid,
    test_loader=dataloader_test,
    max_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    exp_name="bert_tokenizer_lstm_backbone_frozen"
)

