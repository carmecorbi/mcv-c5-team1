from torch import nn
from src.dataset.data import Data
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
import re
from src.models.baseline import Model
from src.dataset.prepare_data import load_data
from src.tokens.char import CharTokenizer
from src.tokens.bert import BertTokenizer
from src.tokens.word import WordTokenizer
from src.lightning_trainer import train_with_lightning

import torch
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 60
EPOCHS = 100
TEXT_MAX_LEN = 25


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

vocab = set()

for caption in df['Title']:
        # Everything is included (punctuation, uppercase, lowercase)
    segm_tokens = re.split(r'(\s+)', caption)
    word_list = []
    for segm in segm_tokens:
        if segm.isspace():
            word_list.append(segm)
        else:
            word_list.extend(word_tokenize(segm))
    vocab.update(word_list)

vocab_list = special_chars + list(vocab)
# Get the tokenizer
#char_tokenizer = CharTokenizer(all_chars_list, special_chars=special_chars, max_len=201)
#bert_tokenizer = BertTokenizer(max_length=TEXT_MAX_LEN)
word_tokenizer = WordTokenizer(vocab=vocab_list,special_chars=special_chars,max_len=TEXT_MAX_LEN)

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

data_train = Data(df, partitions['train'], img_path=img_path, tokenizer=word_tokenizer)
data_valid = Data(df, partitions['val'], img_path=img_path, tokenizer=word_tokenizer)
data_test = Data(df, partitions['test'], img_path=img_path, tokenizer=word_tokenizer)

dataloader_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
dataloader_valid = DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
dataloader_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

model = Model(num_char=len(word_tokenizer), char2idx=char2idx, text_max_len=TEXT_MAX_LEN).to(DEVICE)

# Lightning training
lightning_model, trainer = train_with_lightning(
    model=model,
    criterion=nn.CrossEntropyLoss(),
    tokenizer=word_tokenizer,
    train_loader=dataloader_train,
    val_loader=dataloader_valid,
    test_loader=dataloader_test,
    max_epochs=EPOCHS,
    learning_rate=1e-3,
    exp_name="char_tokenizer_baseline"
)

