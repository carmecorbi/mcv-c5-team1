import optuna
from ultralytics import YOLO
import time

# Optimization function to train the model
def objective(trial):
    # Parameters to optimize
    mixup = trial.suggest_float('mixup', 0.0, 0.5)  # MixUp between 0.0 and 0.3
    dropout = trial.suggest_float('dropout', 0.0, 0.5)  # Dropout between 0 and 0.5
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.001)  # Weight decay between 0 and 0.001
    optimizer = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW'])  # Optimizer choice
    degrees = trial.suggest_float('degrees', 0.0, 90.0)  # Rotation degrees
    scale = trial.suggest_float('scale', 0.2, 1.0)  # Scaling factor

    # Initialize model
    model = YOLO('/ghome/c3mcv02/mcv-c5-team1/week1/checkpoints/yolo/yolo11n.pt')

    # Training parameters
    start_time = time.time()
    results = model.train(
        data='/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/data/data.yaml',
        epochs=30,
        batch=8,
        imgsz=640,
        device='cuda',
        patience=20,  # Early stopping patience
        project='optuna_finetune',  # Project name
        freeze=0,
        classes=[0, 2],
        mixup=mixup,
        dropout=dropout,
        weight_decay=weight_decay,
        optimizer=optimizer,
        degrees=degrees,
        scale=scale,
        # Other augmentations like 'hsv', 'flip', etc., can be fixed or optimized as well
        iou=0.5,  # mAP IoU=0.5
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
study.optimize(objective, n_trials=10)  # Run 10 trials to explore parameter combinations

# Final results
print("Best trial:", study.best_trial)
print("Best parameters:", study.best_trial.params)
print("Best mAP at IoU=0.5:", study.best_value)
