import optuna
from ultralytics import YOLO
import time
from ultralytics.data.augment import Albumentations
import albumentations as A
   
    
def objective(trial):
    # Parameters to optimize
    mixup = trial.suggest_float('mixup', 0.0, 0.5)  # MixUp between 0.0 and 0.3
    dropout = trial.suggest_float('dropout', 0.0, 0.5)  # Dropout between 0 and 0.5
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.01)  # Weight decay between 0 and 0.001
    optimizer = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW'])  # Optimizer choice
    augmentations = Albumentations(p=0.5)
    # Initialize model
    model = YOLO('/ghome/c3mcv02/mcv-c5-team1/week1/checkpoints/yolo/yolo11n.pt')

    # Training parameters
    start_time = time.time()
    results = model.train(
        data='/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/data/data.yaml',
        epochs=50,
        batch=8,
        imgsz=640,
        device='cuda',
        patience=20,  # Early stopping patience
        project='optuna_finetune_unfrozen_albumentations',  # Project name
        freeze=10,
        classes=[0, 2],
        mixup=mixup,
        dropout=dropout,
        weight_decay=weight_decay,
        optimizer='auto',
        augment=True,
        verbose=False  # Set to True for more training details
    )

    # Retrieve the mAP at IoU=0.5
    mAP_50 = results.box.map50 

    # Training time to monitor duration in the optimization
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")

    return mAP_50  # Objective: maximize the mAP at IoU=0.5

# Create an Optuna study
study = optuna.create_study(direction='maximize')  # Maximize the mAP

# Optimize the objective
study.optimize(objective, n_trials=25)  # Run 10 trials to explore parameter combinations

# Final results
print("Best trial:", study.best_trial)
print("Best parameters:", study.best_trial.params)
print("Best mAP at IoU=0.5:", study.best_value)
