import optuna
from optuna.trial import Trial
import torch
import pandas as pd
import torch.nn as nn
import os
import joblib
from datetime import datetime

from src.dataset.data import Data
from torch.utils.data import DataLoader
from src.models.lstm import Model
from src.tokens.char import get_vocabulary
from src.tokens.bert import BertTokenizer
from src.lightning_trainer import train_with_lightning

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 60
MAX_EPOCHS = 100
TEXT_MAX_LEN = 60
N_TRIALS = 20

# Paths
csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/raw_data.csv'
img_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/images'
train_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/train.csv'
val_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/val.csv'
test_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/test.csv'

# Create a directory for saving study results
STUDY_DIR = 'optuna_studies'
os.makedirs(STUDY_DIR, exist_ok=True)

def objective(trial: Trial):
    # Hyperparameters to tune
    # 1. Dropout rate
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    embedding_dropout_rate = trial.suggest_float("embedding_dropout_rate", 0.1, 0.5)
    
    # 2. Learning rate and optimization parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    use_gracient_clip = trial.suggest_categorical("use_gradient_clip", [True, False])
    if use_gracient_clip:
        gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.1, 10.0, log=True)
    freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, False])
    
    # Scheduler parameters
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    scheduler_type = None
    scheduler_params = {}
    
    if use_scheduler:
        scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "linear", "step"])
        if scheduler_type == "step":
            scheduler_params = {
                "step_size": trial.suggest_int("step_size", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.1, 0.9)
            }
        elif scheduler_type == "cosine":
            scheduler_params = {
                "T_max": trial.suggest_int("t_max", 5, MAX_EPOCHS)
            }
    
    # 3. Teacher forcing
    use_teacher_forcing = trial.suggest_categorical("use_teacher_forcing", [True, False])
    teacher_forcing_ratio = 0.0
    if use_teacher_forcing:
        teacher_forcing_ratio = trial.suggest_float("teacher_forcing_ratio", 0.1, 0.9)
    
    # 4. Attention mechanism
    use_attention = trial.suggest_categorical("use_attention", [True, False])
    
    # Number of layers
    num_layers = trial.suggest_int("num_layers", 1, 3)
    
    # Get char2idx
    _, char2idx, _ = get_vocabulary(csv_path)
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
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        embedding_dropout=embedding_dropout_rate,
        freeze_backbone=freeze_backbone,
        use_attention=use_attention
    ).to(DEVICE)
    
    # Build experiment name
    exp_name = f"optuna_trial_{trial.number}_lr{learning_rate:.6f}_dropout{dropout_rate:.2f}"
    _, trainer = train_with_lightning(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        tokenizer=bert_tokenizer,
        train_loader=dataloader_train,
        val_loader=dataloader_valid,
        test_loader=dataloader_test,
        max_epochs=MAX_EPOCHS,
        learning_rate=learning_rate,
        use_teacher_forcing=use_teacher_forcing,
        teacher_forcing_ratio=teacher_forcing_ratio,
        gradient_clip_val=gradient_clip_val if use_gracient_clip else None,
        scheduler_type=scheduler_type,
        scheduler_params=scheduler_params,
        early_stopping_criteria="val_loss",
        exp_name=exp_name
    )
    
    # TODO: Change all of this to use the actual metric from the model
    val_metrics = trainer.callback_metrics
    print(f"Trial {trial.number} - Validation loss: {val_metrics['val_loss'].item()}")
    return val_metrics["val_loss"].item()

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