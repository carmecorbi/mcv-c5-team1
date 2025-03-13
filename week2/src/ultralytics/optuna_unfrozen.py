import optuna
from ultralytics import YOLO
import time

# Optimization function to train the model
def objective(trial):
    # Parameters to optimize
    dropout = trial.suggest_float('dropout', 0.0, 0.5)  # Dropout between 0 and 0.5
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.01)  # Weight decay between 0 and 0.001
    degrees = trial.suggest_float('degrees', -5, 5)  # Rotation degrees
    batch = trial.suggest_categorical('batch', [4, 8, 16, 32]) 
    hsv_h = trial.suggest_float('mixup', 0.0, 0.3)
    hsv_v = trial.suggest_float('dropout', 0.0, 0.5) 

    # Initialize model
    model = YOLO('/ghome/c5mcv01/mcv-c5-team1/week2/src/ultralytics/yolo11n-seg.pt')

    # Training parameters
    start_time = time.time()
    results = model.train(
        data='/ghome/c5mcv01/mcv-c5-team1/week2/src/ultralytics/data/data.yaml',
        epochs=50,
        batch=batch,
        imgsz=1024,
        device='cuda',
        patience=20,  # Early stopping patience
        project='optuna_all_unfrozen',  # Project name
        freeze=0,
        classes=[0, 2],
        mixup=0.0,
        hsv_h = hsv_h,
        hsv_v = hsv_v,
        dropout=dropout,
        weight_decay=weight_decay,
        optimizer='auto',
        degrees=degrees,
        mosaic=0.0,
        scale=0.0,
        fliplr=0.0,
        erasing=0.0,
        bgr=0.0,
        translate=0.0,
        crop_fraction=0.1,
        verbose=False  # Set to True for more training details
    )

    # Retrieve the mAP at IoU=0.5
    mAP_50 = results.seg.map50 

    # Training time to monitor duration in the optimization
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")

    return mAP_50  # Objective: maximize the mAP at IoU=0.5

# Create an Optuna study
study = optuna.create_study(direction='maximize')  # Maximize the mAP

# Optimize the objective
study.optimize(objective, n_trials=20)  

# Final results
print("Best trial:", study.best_trial)
print("Best parameters:", study.best_trial.params)
print("Best mAP at IoU=0.5:", study.best_value)