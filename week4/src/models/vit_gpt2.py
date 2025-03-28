from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
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
                 num_beams: int = 4,
                 attn_pdrop: int = 0.1,
                 embd_pdrop: int = 0.1,
                 resid_pdrop: int = 0.1,
                 attention_probs_dropout_prob: int = 0,
                 hidden_dropout_prob: int = 0,
                 freeze_vit: bool=False, freeze_gpt2: bool=False):
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
        print(f"model params: {self.model.config.decoder}")
        self.model.config.decoder.attn_pdrop = attn_pdrop
        self.model.config.decoder.embd_pdrop = embd_pdrop
        self.model.config.decoder.resid_pdrop = resid_pdrop

        self.model.config.encoder.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.model.config.encoder.hidden_dropout_prob = hidden_dropout_prob

        self.feature_extractor = ViTImageProcessor.from_pretrained(img_processor_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.compute_metric = Metric()
        
        # Set model to use available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device")
        self.model.to(self.device)
        
        # Freeze ViT of GPT2 models
        if freeze_vit:
            for name, param in self.model.encoder.named_parameters():
                print(f"Freezing encoder param: {name}")
                param.requires_grad = False

        if freeze_gpt2:
            for name, param in self.model.decoder.named_parameters():
                print(f"Freezing decoder param: {name}")
                param.requires_grad = False
        
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
              freeze_vit: bool=False, freeze_gpt2: bool=False, epochs: int = 1, output_dir: str = None, **kwargs):
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

        lr_scheduler = kwargs.get("lr_scheduler", "inverse_sqrt")
        assert lr_scheduler in ["linear", "inverse_sqrt", "cosine_with_min_lr", "warmup_stable_decay"], "LR scheduler type not valid"
        
        lr_scheduler_kwargs = None
        if lr_scheduler == "cosine_with_min_lr":
            lr_scheduler_kwargs = {
                "min_lr_rate": 0.2
            }
        elif lr_scheduler == "warmup_stable_decay":
            lr_scheduler_kwargs = {
                "num_decay_steps": 300,
                "num_stable_steps": 8400 - kwargs.get("warmup_steps",0) - 3000
            }

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{output_dir}/{model_name}",
            learning_rate=kwargs.get("learning_rate", 0.001),
            num_train_epochs=epochs,
            fp16=True,
            per_device_train_batch_size=kwargs.get("batch_size", 64),
            per_device_eval_batch_size=kwargs.get("batch_size", 64),
            gradient_accumulation_steps=2,
            weight_decay=kwargs.get("weight_decay", 0.0001),
            warmup_steps=kwargs.get("warmup_steps", 0), 
            lr_scheduler_kwargs = lr_scheduler_kwargs,
            lr_scheduler_type=lr_scheduler,
            max_grad_norm=kwargs.get("gradient_clip_val", 0.0001),                 
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
            data_collator=custom_data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0)]
        )
        trainer.train()

        # Get the best validation loss from the logs
        best_eval_loss = min(trainer.state.log_history, key=lambda x: x.get("eval_loss", float("inf"))).get("eval_loss")

        print(f"Best Validation Loss: {best_eval_loss}")

        # Return the best validation loss
        return best_eval_loss
    
    def print_parameters(self) -> None:
        """Print the number of total and trainable parameters for the entire model, encoder and decoder.
        """
        def count_parameters(model: torch.nn.Module):
            """Helper function to count total and trainable parameters."""
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_params, trainable_params
        
        # Count parameters for the entire model
        total_model_params, trainable_model_params = count_parameters(self.model)
        print("Entire Model:")
        print(f"Total Parameters: {total_model_params:,}")
        print(f"Trainable Parameters: {trainable_model_params:,}")
        print(f"Percentage of Trainable Parameters: {trainable_model_params/total_model_params*100:.2f}%\n")

        # Count parameters for the encoder
        total_encoder_params, trainable_encoder_params = count_parameters(self.model.encoder)
        print("Encoder:")
        print(f"Total Parameters: {total_encoder_params:,}")
        print(f"Trainable Parameters: {trainable_encoder_params:,}")
        print(f"Percentage of Trainable Parameters: {trainable_encoder_params/total_encoder_params*100:.2f}%\n")

        # Count parameters for the decoder
        total_decoder_params, trainable_decoder_params = count_parameters(self.model.decoder)
        print("Decoder:")
        print(f"Total Parameters: {total_decoder_params:,}")
        print(f"Trainable Parameters: {trainable_decoder_params:,}")
        print(f"Percentage of Trainable Parameters: {trainable_decoder_params/total_decoder_params*100:.2f}%")


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
    parser.add_argument('-d', '--data_dir', help="Path to dataset", required=False, default="/ghome/c5mcv01/mcv-c5-team1/week3/data")
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

