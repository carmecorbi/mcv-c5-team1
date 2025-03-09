import argparse
import optuna

from faster_rcnn import FasterRCNN
from functools import partial


def objective(trial, model_trainer: FasterRCNN, data_dir: str, dataset_name: str, num_classes: int, output_dir: str):
    # Hyperparameters to optimize
    hyperparams = {
        # Model hyperparameters
        "batch_size_per_image": trial.suggest_categorical("batch_size_per_image", [64, 128, 256]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
        
        # Learning rate and scheduler
        "base_lr": trial.suggest_float("base_lr", 1e-5, 1e-2, log=True),
        "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["WarmupMultiStepLR", "WarmupCosineLR"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        
        # Whether to clip gradients or not
        "clip_gradients": trial.suggest_categorical("clip_gradients", [True, False])
    }
    
    # Train with current hyperparameters
    output_dir = f"{output_dir}/trial_{trial.number}"
    results = model_trainer.train_model(
        data_dir=data_dir,
        dataset_name=dataset_name,
        num_classes=num_classes,
        output_dir=output_dir,
        **hyperparams
    )
    
    # Extract metrics (assuming COCO mAP as primary metric)
    return results['bbox']['AP']


if __name__ == '__main__':

    
    # Get the model
    model = FasterRCNN("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", num_frozen_layers=2)
    
    
    # Create study and optimize
    study = optuna.create_study(direction="maximize")
    data_dir = "/ghome/c5mcv01/mcv-c5-team1/"
    output_dir = "./output/optuna/GlobalWheat/train_Freeze2_1"
    datasetName = "GlobalWheat" #"AquariumDataCots"
    study.optimize(
        partial(objective, model_trainer=model, data_dir=data_dir, dataset_name=datasetName, num_classes=1, output_dir=output_dir),
        n_trials=10
    )