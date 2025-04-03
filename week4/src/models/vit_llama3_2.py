import schedulefree
import torch
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import src.scripts.llm as llm
import src.scripts.anymodal as anymodal
import src.scripts.vision as vision

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import GradScaler

from PIL import Image
from src.metrics.metrics import Metric
from src.tokenizer import Tokenizer


def train(token: str, kwargs):
    # Load language model and tokenizer
    llm_tokenizer, llm_model = llm.get_llm(
        kwargs.model_name, 
        access_token=token
    )
    llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)
    
    # Load vision model components
    image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=True)
    
    # TODO: Uncomment the following line if you want to load a specific checkpoint (to be fixed)
    #image_processor, vision_model, vision_hidden_size = vision.get_image_encoder(
    #    '/ghome/c5mcv01/mcv-c5-team1/week4/results/vit_gpt2_fully_unfrozen/checkpoint-8400',
    #    image_processor_name='google/vit-base-patch16-224', 
    #    use_peft=False, load_vit_gpt2=True)
    
    # This part should load the dataset using the appropriate ImageDataset (see vision.py for more details)
    img_path = kwargs.image_path
    train_df = pd.read_csv(kwargs.train_csv_path)
    valid_df = pd.read_csv(kwargs.val_csv_path)
    train_dataset = vision.ImageDataset(train_df, img_path, image_processor)
    val_dataset = vision.ImageDataset(valid_df, img_path, image_processor)
    
    # Get the size of the dataset
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    
    # Dataloader configuration
    batch_size = kwargs.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get the size of the dataloaders
    train_size = len(train_loader)
    val_size = len(val_loader)
    print(f"Train loader size: {train_size}, Validation loader size: {val_size}")
    
    # Initialize vision tokenizer and encoder
    vision_encoder = vision.VisionEncoder(vision_model)
    vision_tokenizer = vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1)
    
    # Initialize MultiModalModel
    multimodal_model = anymodal.MultiModalModel(
        input_processor=None,
        input_encoder=vision_encoder,
        input_tokenizer=vision_tokenizer,
        language_tokenizer=llm_tokenizer,
        language_model=llm_model,
        lm_peft = llm.add_peft,
        prompt_text="The description of the given image is: ")
    
    # Use LoraConfig for PEFT if needed
    multimodal_model.language_model = llm.add_peft(multimodal_model.language_model)
    
    # Training configuration
    num_epochs = kwargs.num_epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Number of epochs: {num_epochs}")
    multimodal_model = multimodal_model.to(device)
    multimodal_model.train()

    # Optimizer
    optimizer = schedulefree.AdamWScheduleFree(multimodal_model.parameters(), lr=3e-4)
    optimizer.train()

    # Scheduler
    scaler = GradScaler()
    os.makedirs(f"{kwargs.output_dir}/checkpoints", exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        training_losses = []
        for _, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1} Training", leave=False):
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, loss = multimodal_model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            training_losses.append(loss.item())
        
        avg_train_loss = sum(training_losses) / len(training_losses)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        multimodal_model.eval()
        validation_losses = []
        best_val_loss = float('inf')
        
        with torch.no_grad():
            for _, batch in tqdm(enumerate(val_loader), desc=f"Epoch {epoch+1} Validation", leave=False):
                _, loss = multimodal_model(batch)
                validation_losses.append(loss.item())
            
            # Calculate average validation loss
            avg_val_loss = sum(validation_losses) / len(validation_losses)
            print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
            
            # Save the best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"Saving best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
                
                # Save the best model
                os.makedirs(os.path.dirname(f"{kwargs.output_dir}/checkpoints/best_model.pt"), exist_ok=True)
                multimodal_model._save_model(f"{kwargs.output_dir}/checkpoints/best_model.pt")

            # Decode a random validation sample
            for _ in range(5):
                sample_idx = np.random.randint(len(val_dataset))
                sample = val_dataset[sample_idx]
                print("Actual Text: ", sample['text'])
                print("Generated Text: ", multimodal_model.generate(sample['input'], max_new_tokens=120))
        multimodal_model.train()
        
        # Save the model checkpoint
        if epoch % 5 == 0:
            os.makedirs(os.path.dirname(f"{kwargs.output_dir}/checkpoints/epoch_{epoch+1}.pt"), exist_ok=True)
            multimodal_model._save_model(f"{kwargs.output_dir}/checkpoints/epoch_{epoch+1}.pt")
        
        # Print training and validation loss
        print(f"Epoch {epoch+1} completed. Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print("-" * 50)
        
    # Evaluate the model
    multimodal_model.eval()
    for _ in range(5):
        sample_idx = np.random.randint(len(val_dataset))
        sample = val_dataset[sample_idx]
        
        # save the image with the caption and the generated caption
        image = sample['image']
        caption = sample['text']
        generated_caption = multimodal_model.generate(sample['input'], max_new_tokens=120)

        # Save the image and captions
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(f"{kwargs.output_dir}/image_{sample_idx}.png")
        
        # Save the captions to a text file
        with open(f"{kwargs.output_dir}/image_{sample_idx}_caption.txt", "w") as f:
            f.write(f"Actual Possible Captions: {caption}\n")
            f.write(f"Generated Caption: {generated_caption}\n")
            
def get_pretrained_model(token: str, model_name):
    llm_tokenizer, llm_model = llm.get_llm(
        model_name, 
        access_token=token
    )

    llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)
    image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=True)
    
    # Initialize vision tokenizer and encoder
    vision_encoder = vision.VisionEncoder(vision_model)
    vision_tokenizer = vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1)

    multimodal_model = anymodal.MultiModalModel(
            input_processor=None,
            input_encoder=vision_encoder,
            input_tokenizer=vision_tokenizer,
            language_tokenizer=llm_tokenizer,
            language_model=llm_model,
            lm_peft = llm.add_peft,
            prompt_text="The description of the given image is: ")
    multimodal_model.language_model = llm.add_peft(multimodal_model.language_model)
    multimodal_model._load_model(args.model_file)

    multimodal_model.eval()
    
    return image_processor, multimodal_model, llm_tokenizer
    
def inference(token: str, kwargs):
    image_processor, multimodal_model = get_pretrained_model(token, kwargs.model_name)
    
    # Inference of the image
    img = Image.open(kwargs.infer_image_path).convert("RGB")

    # Process the image with the model processor
    image = image_processor(img, return_tensors="pt")
    image = {key: val.squeeze(0) for key, val in image.items()}  #

    # Caption generation
    generated_caption = multimodal_model.generate(image, max_new_tokens=120)
    print(f"Generated Caption: {generated_caption}\n")
    

def evaluate(dataset: torch.utils.data.Dataset, tokenizer: Tokenizer, model: anymodal.MultiModalModel) -> list:
    """
    Avalua el model sobre un conjunt complet de dades.

    Args:
        dataset (torch.utils.data.Dataset): Dataset on cada mostra és un diccionari que conté:
            - 'input': la imatge processada per al model.
            - 'text': la descripció veritable de la imatge (si està disponible).
            - 'image': la imatge en RGB per si cal realitzar alguna operació addicional.
        tokenizer (Tokenizer): Tokenizer per desxifrar les etiquetes o descripcions.
        model (anymodal.MultiModalModel): El model multimodal per generar les descripcions.

    Retorna:
        List[dict]: Una llista de diccionaris, cada un conté:
            - 'image_path': el camí de la imatge.
            - 'generated_caption': la descripció generada per `model.generate`.
            - 'ground_truth': la descripció veritable si està disponible.
    """
    metric = Metric()  # Assumim que tens una funció per a les mètriques
    metrics_sum = {}
    num_samples = 0

    # Iterem pel dataset
    for idx in tqdm(range(len(dataset))):
        # Obtenir la mostra del dataset
        item = dataset[idx]
        image_input = item['input']
        ground_truth = item['text']

        # Generar la descripció amb el model multimodal
        generated_caption = model.generate(image_input, max_new_tokens=120)

        # Càlcul de les mètriques (com comparant la descripció generada amb la veritable)
        result = metric([generated_caption], [ground_truth])

        # Acumular les mètriques
        for key, value in result.items():
            metrics_sum[key] = metrics_sum.get(key, 0) + value
        num_samples += 1

    # Promediar les mètriques
    averaged_metrics = {key: value / num_samples for key, value in metrics_sum.items()}
    return averaged_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Task and model configuration
    parser.add_argument('-m', '--model_file', default=None, help="Path to the pretrained model .")
    parser.add_argument('-t', '--task', help="Task to perform: inference, evaluation or train", required=True, choices=["infer", "eval", "train"], default="train")
    parser.add_argument('--hf_token', help="Hugging Face token for accessing models and datasets.", required=True)
    parser.add_argument('--model_name', default="meta-llama/Llama-3.2-1B", help="Name of the model.", choices=["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"])
    
    # Paths and configurations
    parser.add_argument('--image_path', help="Path to the images directory.", default="/ghome/c5mcv01/mcv-c5-team1/week3/data/images")
    parser.add_argument('--train_csv_path', help="Path to the training CSV file.", default="/ghome/c5mcv01/mcv-c5-team1/week4/data/train.csv")
    parser.add_argument('--val_csv_path', help="Path to the validation CSV file.", default="/ghome/c5mcv01/mcv-c5-team1/week4/data/val.csv")
    parser.add_argument('--test_csv_path', help="Path to the test CSV file.", default="/ghome/c5mcv01/mcv-c5-team1/week4/data/test.csv")
    parser.add_argument('--output_dir', help="Directory to save the model and results.", default="results/vit_llama3_2_1B")
    
    # Training and evaluation configurations
    parser.add_argument('--infer_image_path', help="Path to the images directory.")
    parser.add_argument('--batch_size', type=int, help="Batch size for training and validation.", default=12)
    parser.add_argument('--num_epochs', type=int, help="Number of epochs for training.", default=10)
    parser.add_argument('--eval_set', help="Evaluation set to use", default="test", choices=["train", "test", "val"])
    args = parser.parse_args()
    
    # Get the Hugging Face token from the command line arguments
    hf_token = args.hf_token
    
    # Call the get_model function with the provided token
    if args.task == "train":
        train(hf_token, args)
    elif args.task == "infer":
        assert args.infer_image_path is not None, "Image path is required for inference (--infer_image_path)"
        inference(hf_token, args)
    elif args.task == "eval":
        assert args.model_file is not None, "Model file path is required for evaluation (--model_file)"
        image_processor, multimodal_model, llm_tokenizer = get_pretrained_model(hf_token)
        
        # Load datasets
        train_df, val_df, test_df = pd.read_csv(args.train_csv_path), pd.read_csv(args.val_csv_path), pd.read_csv(args.test_csv_path)

        # Create datasets
        test_dataset = vision.ImageDataset(test_df, "/ghome/c5mcv01/mcv-c5-team1/week3/data/images", image_processor, is_test=True)
        train_dataset = vision.ImageDataset(train_df, "/ghome/c5mcv01/mcv-c5-team1/week3/data/images", image_processor, is_test=True)
        val_dataset = vision.ImageDataset(val_df, "/ghome/c5mcv01/mcv-c5-team1/week3/data/images", image_processor, is_test=True)
        
        # Evaluate the model
        if args.eval_set == "train" :
            averaged_metrics = evaluate(train_dataset, llm_tokenizer, multimodal_model)
        elif args.eval_set == "val" :
            averaged_metrics = evaluate(val_dataset, llm_tokenizer, multimodal_model)
        elif args.eval_set == "test" :
            averaged_metrics = evaluate(test_dataset, llm_tokenizer, multimodal_model)
        print(f"Averaged Metrics test: {averaged_metrics}")
