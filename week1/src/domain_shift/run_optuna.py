import argparse
import optuna

from faster_rcnn import FasterRCNN
from functools import partial


def objective(trial, model_trainer: FasterRCNN, data_dir: str, dataset_name: str, num_classes: int, class_names: str, output_dir: str):
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
    results = model_trainer.train_model_optuna(
        data_dir=data_dir,
        dataset_name=dataset_name,
        num_classes=num_classes,
        class_labels=class_labels,
        output_dir=output_dir,
        **hyperparams
    )
    
    # Extract metrics (assuming COCO mAP as primary metric)
    return results['bbox']['AP']


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Domain Shift')
    parser.add_argument('-d', '--data_dir', help="Path to dataset", required=False)
    parser.add_argument('-dt', '--dataset', help="Dataset (Aquarium, GlobalWheatHead)", required=True)
    parser.add_argument('-c', '--config_file', default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help="Path to the model config yaml from model zoo.")
    parser.add_argument('-w', '--weights_file', default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help="Path to the weights file or model zoo config yaml.")
    parser.add_argument('-s', '--score_threshold', type=float, default=0.5, help="Score threshold for predictions.")
    parser.add_argument('-o', '--output_dir', help="Output directory for the model", default=None)
    parser.add_argument('--n_trials', required=False, default=10, type=int, help="Number of trials.")
    parser.add_argument('--num_frozen_blocks', required=False, default=0, type=int, help="Number of backbone frozen blocks.")
    args = parser.parse_args()
    
    # Get the arguments from CLI
    data_dir = args.data_dir
    dataset = args.dataset
    config_file = args.config_file
    weights_file = args.weights_file
    num_frozen_layers = args.num_frozen_blocks
    output_dir = args.output_dir
    n_trials = args.n_trials
    # Get the model
    model = FasterRCNN(config_file, weights_file, num_frozen_layers=num_frozen_layers)
    
    if dataset == "Aquarium":
        num_classes = 7
        class_labels = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
    elif dataset == "GlobalWheatHead":
        num_classes = 1
        class_labels = ["wheat_head"]
    else:
        raise ValueError("Unsupported dataset")
    # Create study and optimize
    study = optuna.create_study(direction="maximize")
    #data_dir = "/ghome/c5mcv01/mcv-c5-team1/"
    #output_dir = "./output/optuna/GlobalWheat/train_Freeze2_1"
    study.optimize(
        partial(objective, model_trainer=model, data_dir=data_dir, dataset_name=dataset, num_classes=num_classes, class_names=class_labels, output_dir=output_dir),
        n_trials=n_trials
    )