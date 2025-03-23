import re
import torch
import pandas as pd
import pytorch_lightning as pl
from src.tokens.char import get_vocabulary

from src.models.baseline import Model
from src.dataset.prepare_data import load_data
from src.tokens.word import WordTokenizer
from src.tokens.bert import BertTokenizer
from src.lightning_trainer import LightningTrainer
from torch import nn
from src.dataset.data import Data
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from PIL import Image

import torchvision.transforms as transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 60

# This is for WordTokenizer
#TEXT_MAX_LEN = 25
# This is for BertTokenizer
TEXT_MAX_LEN = 60

csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/raw_data.csv'
img_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/images'
train_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/train.csv'
val_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/val.csv'
test_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/test.csv'

# Load data
df = load_data(csv_path)
special_chars = ['<SOS>', '<EOS>', '<PAD>']
'''
# Extract unique characters from the 'Title' column
all_chars = set()
for caption in df['Title']:
    all_chars.update(caption)
all_chars_list = special_chars + list(all_chars)

# Create the char2idx and idx2char dictionaries
char2idx = {char: idx for idx, char in enumerate(sorted(all_chars_list))}
idx2char = {idx: char for char, idx in char2idx.items()}
'''
_, char2idx, _ = get_vocabulary(csv_path)

vocab = []
seen = set()
for caption in df['Title']:
    segm_tokens = re.split(r'(\s+)', caption)
    for segm in segm_tokens:
        tokens = [segm] if segm.isspace() else word_tokenize(segm)
        for token in tokens:
            if token not in seen:
                seen.add(token)
                vocab.append(token)  # Keep order of first appearance

vocab_list = special_chars + vocab

# Get the tokenizer
#word_tokenizer = WordTokenizer(vocab=vocab_list,special_chars=special_chars,max_len=TEXT_MAX_LEN)
bert_tokenizer = BertTokenizer(max_length=TEXT_MAX_LEN)

# Load the datasets
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

data_train = Data(train_df, partitions['train'], img_path=img_path, tokenizer=bert_tokenizer)
data_valid = Data(val_df, partitions['val'], img_path=img_path, tokenizer=bert_tokenizer)
data_test = Data(test_df, partitions['test'], img_path=img_path, tokenizer=bert_tokenizer)

dataloader_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
dataloader_valid = DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
dataloader_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Set the model
# hardocde char2idx for SOS, EOS, PAD
char2idx = {
    "<SOS>": 0,
    "<EOS>": 1,
    "<PAD>": 2
}
model = Model(num_char=len(bert_tokenizer), char2idx=char2idx, text_max_len=TEXT_MAX_LEN).to(DEVICE)

# Cargar el modelo desde el checkpoint
checkpoint_path = "/ghome/c5mcv01/mcv-c5-team1/week3/results/bert_tokenizer_baseline/version_9/checkpoints/epoch=7-step=1440.ckpt"
lightning_model = LightningTrainer.load_from_checkpoint(
    checkpoint_path,
    model=model,
    criterion=nn.CrossEntropyLoss(),
    tokenizer=bert_tokenizer
)

sample_batch = next(iter(dataloader_test))
sample_image, sample_caption = sample_batch  
caption_text = sample_caption.cpu().numpy()[0]
gt_caption = bert_tokenizer.decode(caption_text)

sample_image = sample_image.to(DEVICE)
matching_row = test_df[test_df["Title"] == gt_caption]

#image_name = matching_row.iloc[0]["Image_Name"]  # Obtenim el nom de la imatge
#image_path = f"/ghome/c5mcv01/mcv-c5-team1/week3/data/images/{image_name}.jpg"
#original_image = Image.open(image_path).convert("RGB")

# Guardar la imatge amb el seu nom original
#save_path = f"{image_name}_output.jpg"
#original_image.save(save_path)

lightning_model.eval()
with torch.no_grad():
    output = lightning_model(sample_image)
predicted_tokens = output.argmax(dim=1).cpu().numpy()[0]
print(predicted_tokens)
predicted_caption = bert_tokenizer.decode(predicted_tokens)  

# Mostrem la predicció i el nom de la imatge
print(f"GT Title: {gt_caption}")
print(f"Prediction Title: {predicted_caption}")