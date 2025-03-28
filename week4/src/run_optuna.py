import optuna
import torch
import pandas as pd
import os
import joblib

from optuna.trial import Trial
from datetime import datetime
from src.dataset.data import Data
from src.models.vit_gpt2 import Vit_GPT2
from transformers import ViTImageProcessor
from src.tokenizer import Tokenizer

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TRIALS = 50

# Create a directory for saving study results
STUDY_DIR = 'optuna_studies_task1'
os.makedirs(STUDY_DIR, exist_ok=True)

def objective(trial: Trial):

    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    scheduler_type = "linear"
    
    if use_scheduler:
        scheduler_type = trial.suggest_categorical("scheduler_type", ["inverse_sqrt", "cosine_with_min_lr", "warmup_stable_decay"])

    gradient_clip_val=0
    use_gracient_clip = trial.suggest_categorical("use_gradient_clip", [True, False])
    if use_gracient_clip:
        gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.1, 10.0, log=True)

    hyperparams = {
        # Learning rate and scheduler
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "lr_scheduler": scheduler_type,
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        
        # Whether to clip gradients or not
        "gradient_clip_val": gradient_clip_val,
        
        # Scheduler params
        "warmup_steps": trial.suggest_int("warmup_steps", 500, 2e3)   
    }

    # Hyperparameters to tune
    # 1. Dropout rate

    attn_pdrop = trial.suggest_float("attn_pdrop", 0.1, 0.5)
    embd_pdrop = trial.suggest_float("embd_pdrop", 0.1, 0.5)
    resid_pdrop = trial.suggest_float("resid_pdrop", 0.1, 0.5)
    attention_probs_dropout_prob = trial.suggest_float("attention_probs_dropout_prob", 0.1, 0.5)
    hidden_dropout_prob = trial.suggest_float("hidden_dropout_prob", 0.1, 0.5)
    
    model = Vit_GPT2(model_path="nlpconnect/vit-gpt2-image-captioning", tokenizer_path="nlpconnect/vit-gpt2-image-captioning", max_len=16, attn_pdrop=attn_pdrop, embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_dropout_prob=hidden_dropout_prob)

    data_dir = "/ghome/c5mcv01/mcv-c5-team1/week3/data"
    train_csv_path = os.path.join(data_dir, "train.csv")
    val_csv_path = os.path.join(data_dir, "val.csv")
    test_csv_path = os.path.join(data_dir, "test.csv")
    img_path = os.path.join(data_dir, "images")
    tokenizer = Tokenizer(tokenizer_path="nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    output_dir = f"/ghome/c5mcv01/mcv-c5-team1/week4/{STUDY_DIR}"
                
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
    exp_name = f"optuna_task1_trial_{trial.number}_FullyUnfrozen_dropout"
    data_train = Data(train_df, partitions['train'], img_path=img_path, tokenizer=tokenizer, image_processor=image_processor)
    data_val = Data(val_df, partitions['val'], img_path=img_path, tokenizer=tokenizer, image_processor=image_processor)
    best_eval_loss = model.train(
        train_set=data_train, 
        val_set=data_val, 
        model_name=exp_name,
        freeze_vit=False, 
        freeze_gpt2=False, 
        epochs=100,
        output_dir=output_dir,
        **hyperparams
        )

    return best_eval_loss

def main():
    # Create a timestamp for the study
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"ocr_optimization_{timestamp}"
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, timeout=None)
    
    # Print optimization results
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save study results
    study_path = os.path.join(STUDY_DIR, f"{study_name}.pkl")
    joblib.dump(study, study_path)
    print(f"Study saved to {study_path}")
    
    # Save best parameters
    best_params_path = os.path.join(STUDY_DIR, f"{study_name}_best_params.txt")
    with open(best_params_path, "w") as f:
        f.write(f"Best trial value: {best_trial.value}\n")
        f.write("Best parameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"{key}: {value}\n")
    print(f"Best parameters saved to {best_params_path}")

if __name__ == "__main__":
    main()