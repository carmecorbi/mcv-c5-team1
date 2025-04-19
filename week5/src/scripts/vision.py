import torch
import ast
import random

import numpy as np
import pandas as pd

from transformers import ViTImageProcessor, ViTForImageClassification, VisionEncoderDecoderModel
from torch import nn
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig
from PIL import Image


class ImageDataset(Dataset):
    """
    ImageDataset: A custom dataset to load images and corresponding text labels from a Hugging Face dataset.

    Attributes:
    - dataset: Hugging Face dataset split.
    - processor: Preprocessing function for images.
    """

    def __init__(self, data: pd.DataFrame, img_path: str, processor: ViTImageProcessor, is_test=False):
        """
        Initializes the dataset by loading a Hugging Face dataset and configuring an image processor.
        
        Parameters:
        - dataset_name: str, name of the Hugging Face dataset.
        - processor: Callable, processes image data into a format suitable for the model.
        """
        data['Title'] = data['Title'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [x] if isinstance(x, str) else x)
        self.data = data
        self.img_path = img_path
        
        self.is_test = is_test
        
        # Processor for image preprocessing
        self.processor = processor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a single data item, including the processed image and corresponding text label.

        Parameters:
        - idx: int, index of the item to retrieve.

        Returns:
        - dict: Contains processed image and its text label.
        """
        # Get the image and text data
        item = self.data.iloc[idx]
        img_name = item["Image_Name"]
        image = Image.open(f'{self.img_path}/{img_name}.jpg').convert('RGB')
        
        # Load the image using PIL
        if not isinstance(image, Image.Image):
            image = Image.open(image.tobytesio())

        # Ensure the image is in RGB format
        image = image.convert('RGB')
        
        # get the rgb value of the image as np after resizing to 224x224
        rgb_val = np.array(image.resize((224, 224), Image.BICUBIC))
        image = self.processor(image, return_tensors="pt")
        image = {key: val.squeeze(0) for key, val in image.items()}  # Remove batch dimension

        return {
            'input': image,
            'text': random.choice(item["Title"]) if not self.is_test else item["Title"],
            'image': rgb_val
        }

class Projector(nn.Module):
    """
    Projector: A feedforward neural network for projecting feature embeddings to a target dimension.

    Attributes:
    - inp_layer: Input linear layer.
    - layers: Sequence of hidden layers.
    - dropout: Dropout applied between layers.
    - out_layer: Output linear layer.
    """

    def __init__(self, in_features, out_features, num_hidden=2):
        """
        Initializes the Projector.

        Parameters:
        - in_features: int, size of the input feature vector.
        - out_features: int, size of the output feature vector.
        - num_hidden: int, number of hidden layers (default: 2).
        """
        super(Projector, self).__init__()
        self.inp_layer = nn.Linear(in_features, out_features)
        self.layers = nn.ModuleList([nn.Linear(out_features, out_features) for _ in range(num_hidden)])
        self.dropout = nn.Dropout(0.1)
        self.out_layer = nn.Linear(out_features, out_features)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x: torch.Tensor, input tensor.

        Returns:
        - torch.Tensor, output tensor.
        """
        x = self.inp_layer(x)
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        return x

class VisionEncoder(nn.Module):
    """
    VisionEncoder: Wraps a vision model to extract hidden states as feature embeddings.

    Attributes:
    - model: Pre-trained vision model.
    - device: Torch device (GPU/CPU).
    """

    def __init__(self, model):
        """
        Initializes the VisionEncoder.

        Parameters:
        - model: nn.Module, pre-trained vision model.
        """
        super(VisionEncoder, self).__init__()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, inputs):
        """
        Forward pass to obtain feature embeddings.

        Parameters:
        - inputs: dict, preprocessed inputs compatible with the vision model.

        Returns:
        - torch.Tensor, last hidden state of the vision model.
        """
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]  # Extract last hidden state
        # return outputs.last_hidden_state

def get_image_encoder(model_name, image_processor_name=None, load_vit_gpt2=False, use_peft=False):
    """
    Loads a vision model and its processor, optionally applying Parameter-Efficient Fine-Tuning (PEFT).

    Parameters:
    - model_name: str, name of the pre-trained vision model.
    - use_peft: bool, whether to apply PEFT (default: False).

    Returns:
    - processor: Image processor for pre-processing.
    - model: Pre-trained vision model.
    - hidden_size: int, size of the model's hidden layer.
    """
    image_processor_name = image_processor_name if image_processor_name else model_name
    processor = ViTImageProcessor.from_pretrained(image_processor_name)
    
    if load_vit_gpt2:
        # Load the VisionEncoderDecoderModel for image captioning tasks
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Load the encoder and decoder separately
        encoder = model.encoder
        hidden_size = encoder.config.hidden_size
    else:
        model = ViTForImageClassification.from_pretrained(model_name)
        hidden_size = model.config.hidden_size
        
        if use_peft:
            peft_config = LoraConfig(
                task_type=None, 
                inference_mode=False, 
                r=16, 
                lora_alpha=32, 
                lora_dropout=0.1, 
                target_modules=['dense']
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            for param in model.parameters():
                param.requires_grad = False

    return processor, model, hidden_size

if __name__ == '__main__':
    processor, model, hidden_size = get_image_encoder('google/vit-base-patch16-224')
    
    # Paths
    img_path = "/ghome/c5mcv01/mcv-c5-team1/week3/data/images"
    train_csv_path = "/ghome/c5mcv01/mcv-c5-team1/week3/data/train.csv"
    valid_csv_path = "/ghome/c5mcv01/mcv-c5-team1/week3/data/val.csv"

    # Load the dataset
    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)

    # Create datasets
    train_dataset = ImageDataset(train_df, img_path, processor)
    val_dataset = ImageDataset(valid_df, img_path, processor)
    
    # Print dataset sizes
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")

    # DataLoader setup
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize encoder and projector
    vision_encoder = VisionEncoder(model)
    vision_projector = Projector(hidden_size, 768)

    for batch in train_loader:
        vision_embeddings = vision_encoder(batch['input'])
        print(vision_embeddings.shape)
        vision_tokens = vision_projector(vision_embeddings)
        print(vision_tokens.shape)
        break
