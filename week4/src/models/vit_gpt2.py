from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from src.metrics.metrics import Metric
from src.tokenizer import Tokenizer
from src.dataset.data import Data

import pandas as pd
import torch
import numpy as np
import os
import argparse

class Vit_GPT2():
    def __init__(self, 
                 model_path: str = "nlpconnect/vit-gpt2-image-captioning", 
                 img_processor_path: str = "nlpconnect/vit-gpt2-image-captioning", 
                 tokenizer_path: str = "nlpconnect/vit-gpt2-image-captioning",
                 max_len: int = 16,
                 num_beams: int = 4):
        """Initialize the model.

        Args:
            model_path (str, optional): Model path from Hub or local path. Defaults to "nlpconnect/vit-gpt2-image-captioning".
            img_processor_path (str, optional): Image processor from Hub or local path. Defaults to "nlpconnect/vit-gpt2-image-captioning".
            tokenizer_path (str, optional): Tokenizer path from Hub or local path. Defaults to "nlpconnect/vit-gpt2-image-captioning".
            max_len (int, optional): Max length for the generated caption. Defaults to 16.
            num_beams (int, optional): Number of beams to be used by the generator. Defaults to 4.
        """
        # Load model, image processor and tokenizer
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.feature_extractor = ViTImageProcessor.from_pretrained(img_processor_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.compute_metric = Metric()
        
        # Set model to use available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device")
        self.model.to(self.device)
        
        # Set parameters
        self.max_len = max_len
        self.num_beams = num_beams

    def inference(self, inputs: Image.Image | list[Image.Image] | torch.Tensor | list[torch.Tensor]) -> str | list[str]:
        """Generate captions for the input images.

        Args:
            inputs (PIL.Image | list[PIL.Image] | torch.Tensor | list[torch.Tensor]): Input image(s).

        Returns:
            str | list[str]: Generated caption(s).
        """
        gen_kwargs = {"max_length": self.max_len, "num_beams": self.num_beams}
        
        # Extract pixel values from images
        if isinstance(inputs, torch.Tensor):
            pixel_values = inputs
        else:
            images = []
            for image in inputs:
                if isinstance(image, Image.Image):
                    if image.mode != "RGB":
                        image = image.convert(mode="RGB")
                    images.append(image)
                elif isinstance(image, torch.Tensor):
                    images.append(image)
            pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        
        # Move pixel values to device
        pixel_values = pixel_values.to(self.device)
        
        # Generate captions
        output_ids = self.model.generate(pixel_values, **gen_kwargs)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
    
    def evaluate(self, loader: torch.utils.data.DataLoader) -> dict:
        """Evaluate the model on a dataset.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset. Images must be processed using the image processor

        Returns:
            dict: Evaluation metrics averaged across all batches.
        """
        all_metrics = []
        
        self.model.eval()
        for batch in tqdm(loader):
            images, captions = batch
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Run inference
            preds = self.inference(images)
            preds = [pred.strip() for pred in preds]
            
            # Compute metrics
            captions_decoded = self.tokenizer.batch_decode(captions, skip_special_tokens=True)
            captions_decoded = np.array([[text] for text in captions_decoded], dtype=object)
            metrics = self.compute_metric(preds, captions_decoded)
            all_metrics.append(metrics)
        
        # Average all metrics in the dictionaries
        avg_metrics = {}
        if all_metrics:
            # Get all metric keys from the first dictionary
            metric_keys = all_metrics[0].keys()
            
            # Calculate average for each metric across all batches
            for key in metric_keys:
                avg_metrics[key] = np.mean([metrics[key] for metrics in all_metrics])
        return avg_metrics
    

    def train(self, 
              train_set: torch.utils.data.Dataset, 
              val_set: torch.utils.data.Dataset, 
              model_name: str = "vit-gpt2-image-captioning", 
              freeze_vit: bool=False, freeze_gpt2: bool=False, epochs: int = 1, output_dir: str = None):
        """Train the model on a dataset.

        Args:
            model_name (str, optional): Name of the model to define the output_dir.
            dataset (torch.utils.data.Dataset): Dataset to train on.
            freeze_vit (bool, optional): Boolean setting wether the ViT encoder is fine-tuned or frozen.
            freeze_gpt2 (bool, optional): Boolean setting wether the GPT2 decoder is fine-tuned or frozen.
            epochs (int, optional): Number of epochs to train. Defaults to 1.
        """
        # Fine-tuning strategies:
        if freeze_vit:
            for name, param in self.model.encoder.named_parameters():
                print(f"Freezing encoder param: {name}")
                param.requires_grad = False

        if freeze_gpt2:
            for name, param in self.model.decoder.named_parameters():
                print(f"Freezing decoder param: {name}")
                param.requires_grad = False

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{output_dir}/{model_name}",
            learning_rate=5e-5,
            num_train_epochs=epochs,
            fp16=True,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=2,
            save_total_limit=3,
            eval_strategy="epoch",
            eval_steps=1,
            save_strategy="epoch",
            save_steps=1,
            logging_strategy="epoch",
            logging_steps=1,
            remove_unused_columns=True,
            push_to_hub=False,
            label_names=["labels"],
            load_best_model_at_end=True,
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            # compute_metrics=custom_compute_metrics,
            data_collator=custom_data_collator
        )
        trainer.train()

def custom_data_collator(features):
    """
    Custom data collator for ViT-GPT2 model that expects a list of tuples (image_tensor, caption_tensor).
    
    Args:
        features: List of tuples where each tuple contains (image_tensor, caption_tensor)
        
    Returns:
        dict: Dictionary with pixel_values and labels keys for the model
    """
    # Unzip the list of tuples into separate lists
    pixel_values, input_ids = zip(*features)
    
    # Stack the tensors
    pixel_values = torch.stack(pixel_values)
    input_ids = torch.stack(input_ids)
    
    # Return the dictionary with the required keys for the model
    return {
        "pixel_values": pixel_values,
        "labels": input_ids
    }


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='ViT-GPT2')
    parser.add_argument('-d', '--data_dir', help="Path to dataset", required=True)
    parser.add_argument('-i', '--infer_image', help="Path to image to infer", required=False)
    parser.add_argument('-o', '--output_dir', help="Output directory for the model", default=None, required=False)
    parser.add_argument('-t', '--task', help="Task to perform: inference, evaluation or train", required=True, choices=["inference", "evaluation", "train"])
    parser.add_argument('-m', '--model_file', default="nlpconnect/vit-gpt2-image-captioning", help="Path to the pretrained model .")
    parser.add_argument('-tk', '--tokenizer_path', default="nlpconnect/vit-gpt2-image-captioning", help="Path to the tokenizer.")
    parser.add_argument('-fe', '--freeze_encoder', help="Wether to freeze or not the encoder (ViT)", action="store_true")
    parser.add_argument('-fd', '--freeze_decoder', help="Wether to freeze or not the decoder (GPT2)", action="store_true")
    parser.add_argument('--num_workers', required=False, default=4, type=int, help="Number of workers to load dataset.")
    parser.add_argument('--num_epochs', help="Number of epochs to train", default=1, type=int)
    parser.add_argument('--eval_set', help="Evaluation set to use", default="test", choices=["train", "test", "val"])
    parser.add_argument('--model_name', help="Name of the model to save", default="vit-gpt2-image-captioning")
    parser.add_argument('--max_len', help="Max length for the generated caption", default=16, type=int)
    args = parser.parse_args()
    
    model = Vit_GPT2(model_path=args.model_file, tokenizer_path=args.tokenizer_path, max_len=args.max_len)
    train_csv_path = os.path.join(args.data_dir, "train.csv")
    val_csv_path = os.path.join(args.data_dir, "val.csv")
    test_csv_path = os.path.join(args.data_dir, "test.csv")
    img_path = os.path.join(args.data_dir, "images")
    tokenizer = Tokenizer(tokenizer_path=args.tokenizer_path)
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    if args.task == "evaluation":        
        # Evaluate the model
        df = pd.read_csv(test_csv_path)
        if args.eval_set == "train":
            df = pd.read_csv(train_csv_path)
        elif args.eval_set == "val":
            df = pd.read_csv(val_csv_path)
        
        # Load datasets
        partitions = {
            'partition': list(df.index)
        }
        print(f"Training set: {len(df)} samples")
        
        # Evaluate the model
        data_train = Data(df, partitions['partition'], img_path=img_path, tokenizer=tokenizer, image_processor=image_processor)
        dataloader_train = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        metrics = model.evaluate(dataloader_train)
        print(metrics)
    elif args.task == "inference":
        assert args.infer_image is not None, "Please provide an image to infer using (-i, --infer_image)"
        
        # Test the model
        image = Image.open(args.infer_image)
        caption = model.inference([image])
        print(caption)
    elif args.task == "train":
        assert args.output_dir is not None, "Please provide an output directory to save the model using (-o, --output_dir)"
                
        # Load datasets
        train_df, val_df, test_df = pd.read_csv(train_csv_path), pd.read_csv(val_csv_path), pd.read_csv(test_csv_path)
        partitions = {
            'train': list(train_df.index),
            'val': list(val_df.index),
            'test': list(test_df.index)
        }
        print(f"Training set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        
        # Train the model
        data_train = Data(train_df, partitions['train'], img_path=img_path, tokenizer=tokenizer, image_processor=image_processor)
        data_val = Data(val_df, partitions['val'], img_path=img_path, tokenizer=tokenizer, image_processor=image_processor)
        model.train(
            train_set=data_train, 
            val_set=data_val, 
            model_name=args.model_name,
            freeze_vit=args.freeze_encoder, 
            freeze_gpt2=args.freeze_decoder, 
            epochs=args.num_epochs,
            output_dir=args.output_dir)

