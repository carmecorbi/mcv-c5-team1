import argparse
import optuna

from mask_rcnn_optuna import MaskRCNN
from functools import partial


def objective(trial, model_trainer: MaskRCNN, data_dir: str, dataset_name: str, output_dir: str, freeze_backbone: bool = False):
    # Hyperparameters to optimize
    hyperparams = {
        # Model hyperparameters
        "batch_size_per_image": trial.suggest_categorical("batch_size_per_image", [64, 128, 256]),
        "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
        
        # Learning rate and scheduler
        "base_lr": trial.suggest_float("base_lr", 3e-4, 1e-3, log=True),
        "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["WarmupMultiStepLR", "WarmupCosineLR"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        
        # Whether to clip gradients or not
        "clip_gradients": trial.suggest_categorical("clip_gradients", [True, False]),
        #"clip_gradients": False,
        
        # Wether to freeze or not the backbone
        "freeze_backbone": freeze_backbone
    }
    
    # Train with current hyperparameters
    output_dir = f"{output_dir}/trial_{trial.number}"
    results = model_trainer.train_model(
        data_dir=data_dir,
        dataset_name=dataset_name,
        output_dir=output_dir,
        **hyperparams
    )
    
    # Extract metrics (assuming COCO mAP as primary metric)
    return results['segm']['AP']


if __name__ == '__main__':

    # Get the arguments from CLI
    data_dir = '/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/'
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    weights_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    score_threshold = 0.5
    num_workers = 2
    output_dir = "./output/optuna/mask_rcnn_allunfrozen/"
    freeze_backbone = 5
    
    # Get the model
    model = MaskRCNN(config_file, weights_file, score_threshold, num_workers)
    
    # Create study and optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(
        partial(objective, model_trainer=model, data_dir=data_dir, dataset_name="strawberry", output_dir=output_dir, freeze_backbone=freeze_backbone),
        n_trials=15
    )