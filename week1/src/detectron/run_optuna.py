import argparse
import optuna

from week1.src.detectron.faster_rcnn import FasterRCNN
from functools import partial


def objective(trial, model_trainer: FasterRCNN, data_dir: str, dataset_name: str, output_dir: str, freeze_backbone: bool = False):
    # Hyperparameters to optimize
    hyperparams = {
        # Model hyperparameters
        "batch_size_per_image": trial.suggest_categorical("batch_size_per_image", [64, 128, 256, 512]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
        
        # Learning rate and scheduler
        "base_lr": trial.suggest_float("base_lr", 1e-4, 1e-2, log=True),
        "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["WarmupMultiStepLR", "WarmupCosineLR"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        
        # Whether to clip gradients or not
        #"clip_gradients": trial.suggest_categorical("clip_gradients", [True, False]),
        "clip_gradients": False,
        
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
    return results['bbox']['AP']


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='KittiMots Dataset')
    parser.add_argument('-d', '--data_dir', help="Path to validation dataset", required=True)
    parser.add_argument('-c', '--config_file', default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help="Path to the model config yaml from model zoo.")
    parser.add_argument('-w', '--weights_file', default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help="Path to the weights file or model zoo config yaml.")
    parser.add_argument('-s', '--score_threshold', type=float, default=0.5, help="Score threshold for predictions.")
    parser.add_argument('-o', '--output_dir', help="Output directory for the model", default=None, required=True)
    parser.add_argument('-fb', '--freeze_backbone', help="Wether to freeze or not the backbone", action="store_true")
    parser.add_argument('--num_workers', required=False, default=4, type=int, help="Number of workers to load dataset.")
    args = parser.parse_args()
    
    # Get the arguments from CLI
    data_dir = args.data_dir
    config_file = args.config_file
    weights_file = args.weights_file
    score_threshold = args.score_threshold
    num_workers = args.num_workers
    output_dir = args.output_dir
    freeze_backbone = args.freeze_backbone
    
    # Get the model
    model = FasterRCNN(config_file, weights_file, score_threshold, num_workers)
    
    # Create study and optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(
        partial(objective, model_trainer=model, data_dir=data_dir, dataset_name="kitti-mots", output_dir=output_dir, freeze_backbone=freeze_backbone),
        n_trials=20
    )