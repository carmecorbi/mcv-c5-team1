from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset
from dataset import get_YOLOData_dicts
from trainer import CustomTrainer

#from detectron.dataset import CustomKittiMotsDataset

import numpy as np
import cv2
import os
import argparse


class FasterRCNN:
    def __init__(self, config_file: str, weights_file: str, score_threshold: float=0.5, num_workers: int=4, num_frozen_layers: int=0):
        """Initializes the FasterRCNN model

        Args:
            config_file (str): Path to the model configuration file. Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
            weights_file (str): Path to the model weights file. Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well.
            score_threshold (float, optional): Confidence score threshold. Defaults to 0.5.
            num_workers (int, optional): Number of workers for the dataset loading.
        """
        # Load the model configuration and weights
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))

        # Freeze backbone layers (ResNet-50)
        # 5: backbone frozen
        self.cfg.MODEL.RESNETS.FREEZE_AT = num_frozen_layers
        # Get the weights from model_zoo if it is not find in the filesystem
        if os.path.isfile(weights_file):
            print(f"Loading weights from a file: {weights_file}")
            self.cfg.MODEL.WEIGHTS = weights_file
            self.is_coco = False
        else:
            print("Loading weights from model zoo...")
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
            self.is_coco = True
        
        # Set the threshold for scoring
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        
    def run_inference(self, image: np.ndarray, num_classes: int = 2) -> dict:
        """Run inference on the input image or batch of images

        Args:
            image (np.ndarray): OpenCV image or batch of images. In [B, C, H, W] format.
            num_classes (int, optional): Number of classes in the model.

        Returns:
            dict: Dictionary containing the predictions. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        """
        if not self.is_coco:
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        
        predictor = DefaultPredictor(self.cfg)
        
        # Make it compatible with batches of images
        predictions = []
        if len(image.shape) == 3:
            predictions = predictor(image)
        elif len(image.shape) == 4:
            for img in image:
                predictions.append(predictor(img))
        else:
            raise ValueError("Unsupported image shape")
        return predictions
    
    def visualize_predictions(self, image: np.ndarray, predictions: dict, class_names: list[str] = ["car", "pedestrian"]) -> np.ndarray:
        """Visualize the predictions on the input image

        Args:
            image (np.ndarray): OpenCV image. In BGR format.
            predictions (dict): Dictionary containing the predictions. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
            class_names (list[str], optional): List of class names. Default to ["car", "pedestrian"]

        Returns:
            np.ndarray: Visualized image. In BGR format.
        """
        if not self.is_coco:
            assert class_names is not None, "Class names must be defined if using non-COCO dataset."
            # Register a temporary metadata catalog with the provided class names
            temp_metadata_name = "temp_visualization_metadata"
            if temp_metadata_name in MetadataCatalog:
                MetadataCatalog.remove(temp_metadata_name)
            MetadataCatalog.get(temp_metadata_name).set(thing_classes=class_names)
            metadata = MetadataCatalog.get(temp_metadata_name)
        else:
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        
        v = Visualizer(image[:, :, ::-1], metadata, scale=10)
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]
    
    
    def train_model(self, data_dir: str, dataset_name: str = "kitti-mots", num_classes: int = 2, class_labels: str = "wheat_head", output_dir: str = "./output/train", **kwargs) -> dict:
        """Train a model to a given dataset.

        Args:
            data_dir (str): Dataset directory.
            dataset_name (str, optional): Dataset name. Defaults to "kitti-mots".
            num_classes (int, optional): Number of classes in the dataset. Defaults to 2.
            class_labels (list of str): Labels of the possible classes. Deafults to "wheat_head"
            output_dir (str, optional): Output directory for the training process. Defaults to "./output/train".

        Returns:
            dict: Results of the training process.
        """
        # Register dataset and metadata for training and validation
       
        try:
            for split in ["train", "val"]:
                DatasetCatalog.register(dataset_name + "_train", lambda: get_YOLOData_dicts(data_dir+"/train"))
                DatasetCatalog.register(dataset_name + "_val", lambda: get_YOLOData_dicts(data_dir+"/valid"))
                MetadataCatalog.get(dataset_name + "_train").set(thing_classes=class_labels) 
                MetadataCatalog.get(dataset_name + "_val").set(thing_classes=class_labels) 
        except AssertionError:
            print("Dataset already registered, continuing...")
        

        # Set the number of classes
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        
        # TRAINING SPECIFIC CONFIGURATION
        self.cfg.DATASETS.TRAIN = (dataset_name + "_train",)
        self.cfg.DATASETS.TEST = (dataset_name + "_val",)

        # Solver parameters
        
        self.cfg.SOLVER.IMS_PER_BATCH = 16  # Batch size (adjust based on GPU memory)
        self.cfg.SOLVER.BASE_LR = 0.0016872479672742837  # Learning rate
        self.cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
        self.cfg.SOLVER.MAX_ITER = 2000    # Maximum iterations
        self.cfg.SOLVER.WEIGHT_DECAY = 0.00021293027823732804
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoint every 1000 iterations
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Faster, and good enough for COCO pre-trained models
        self.cfg.TEST.EVAL_PERIOD = 500
        self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False
        
        # Output directory for training logs and checkpoints
        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        # Create trainer and start training
        
        print("Starting training...")
        trainer = CustomTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        print("-" * 50)
        print("Starting final evaluation on validation")
        print("Creating predictor...")
        eval_cfg = self.cfg.clone()
        eval_cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        predictor = DefaultPredictor(eval_cfg)
        
        # Evaluate on the test dataset using COCO evaluator
        print("Performing evaluation...")
        validation_output_dir = os.path.join(output_dir, "final_validation")
        os.makedirs(validation_output_dir, exist_ok=True)
        evaluator = COCOEvaluator(dataset_name + "_val", eval_cfg, False, output_dir=validation_output_dir)
        val_loader = build_detection_test_loader(eval_cfg, dataset_name + "_val")
        
        # Run inference and return results
        print("Running inference on the test dataset...")
        return inference_on_dataset(predictor.model, val_loader, evaluator)
    
    def train_model_optuna(self, data_dir: str, dataset_name: str = "kitti-mots", num_classes: int = 2, class_labels: str = "wheat_head", output_dir: str = "./output/train", **kwargs) -> dict:
        """Train a model to a given dataset.

        Args:
            data_dir (str): Dataset directory.
            dataset_name (str, optional): Dataset name. Defaults to "kitti-mots".
            num_classes (int, optional): Number of classes in the dataset. Defaults to 2.
            class_labels (list of str): Labels of the possible classes. Deafults to "wheat_head".
            output_dir (str, optional): Output directory for the training process. Defaults to "./output/train".

        Returns:
            dict: Results of the training process.
        """
        # Register dataset and metadata for training and validation
        try:
            for split in ["train", "val"]:
                DatasetCatalog.register(dataset_name + "_train", lambda: get_YOLOData_dicts(data_dir+"/train"))
                DatasetCatalog.register(dataset_name + "_val", lambda: get_YOLOData_dicts(data_dir+"/valid"))
                MetadataCatalog.get(dataset_name + "_train").set(thing_classes=class_labels) 
                MetadataCatalog.get(dataset_name + "_val").set(thing_classes=class_labels) 
        except AssertionError:
            print("Dataset already registered, continuing...")
        

        # Set the number of classes
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        
        # TRAINING SPECIFIC CONFIGURATION
        self.cfg.DATASETS.TRAIN = (dataset_name + "_train",)
        self.cfg.DATASETS.TEST = (dataset_name + "_val",)

        # Solver parameters (optimizer)
        lr_scheduler = kwargs.get("lr_scheduler", "WarmupMultiStepLR")
        assert lr_scheduler in ["WarmupMultiStepLR", "WarmupCosineLR"], "LR scheduler must be WarmupMultiStepLR or WarmupCosineLR"
        self.cfg.SOLVER.LR_SCHEDULER_NAME = lr_scheduler # Learning rate scheduler
        self.cfg.SOLVER.IMS_PER_BATCH = kwargs.get("batch_size", 4) # Batch size
        self.cfg.SOLVER.BASE_LR = kwargs.get("base_lr", 0.00025)  # Learning rate
        self.cfg.SOLVER.MOMENTUM = kwargs.get("momentum", 0.9)  # Momentum
        self.cfg.SOLVER.NESTEROV = kwargs.get("nesterov", False)  # Nesterov momentum
        self.cfg.SOLVER.WEIGHT_DECAY = kwargs.get("weight_decay", 0.0001) # Weight decay
        self.cfg.SOLVER.GAMMA = kwargs.get("gamma", 0.1) # Learning rate decay factor
        self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED = kwargs.get("clip_gradients", False) # Clip gradients
        self.cfg.SOLVER.STEPS = kwargs.get("steps", (1000, 4500)) # Steps for learning rate decay

        # Fixed parameters (do not change)
        self.cfg.SOLVER.MAX_ITER = kwargs.get("max_iter", 2000)
        self.cfg.SOLVER.CHECKPOINT_PERIOD = kwargs.get("checkpoint_period", 1000)

        # Test parameters (evaluation)
        self.cfg.TEST.EVAL_PERIOD = kwargs.get("eval_period", 500)
        
        # Output directory for training logs and checkpoints
        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        # Create trainer and start training
        print("Starting training...")
        trainer = CustomTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        print("-" * 50)
        print("Starting final evaluation on validation")
        print("Creating predictor...")
        eval_cfg = self.cfg.clone()
        eval_cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        predictor = DefaultPredictor(eval_cfg)
        
        # Evaluate on the test dataset using COCO evaluator
        print("Performing evaluation...")
        validation_output_dir = os.path.join(output_dir, "final_validation")
        os.makedirs(validation_output_dir, exist_ok=True)
        evaluator = COCOEvaluator(dataset_name + "_val", eval_cfg, False, output_dir=validation_output_dir)
        val_loader = build_detection_test_loader(eval_cfg, dataset_name + "_val")

        # Run inference and return results
        print("Running inference on the test dataset...")
        return inference_on_dataset(predictor.model, val_loader, evaluator)
        
        
    
    
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Domain Shift')
    parser.add_argument('-d', '--data_dir', help="Path to dataset", required=False)
    parser.add_argument('-t', '--task', help="Task to do (infer, train)", required=True)
    parser.add_argument('-dt', '--dataset', help="Dataset (Aquarium, GlobalWheatHead)", required=True)
    parser.add_argument('-i', '--input_image', help="Input image to infer on (only for infer task)", type=str, required=False)
    parser.add_argument('-c', '--config_file', default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help="Path to the model config yaml from model zoo.")
    parser.add_argument('-w', '--weights_file', default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help="Path to the weights file or model zoo config yaml.")
    parser.add_argument('-s', '--score_threshold', type=float, default=0.5, help="Score threshold for predictions.")
    parser.add_argument('-o', '--output_dir', help="Output directory for the model", default=None)
    parser.add_argument('--num_workers', required=False, default=4, type=int, help="Number of workers to load dataset.")
    parser.add_argument('--num_frozen_blocks', required=False, default=0, type=int, help="Number of backbone frozen blocks.")
    args = parser.parse_args()
    
    # Get the arguments from CLI
    data_dir = args.data_dir
    dataset = args.dataset
    config_file = args.config_file
    weights_file = args.weights_file
    score_threshold = args.score_threshold
    num_workers = args.num_workers
    num_frozen_layers = args.num_frozen_blocks
    task = args.task
    output_dir = args.output_dir
    
    # Get the model
    model = FasterRCNN(config_file, weights_file, score_threshold, num_workers, num_frozen_layers)

    # Set the number of classes
    if dataset == "Aquarium":
        num_classes = 7
        class_labels = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
    elif dataset == "GlobalWheatHead":
        num_classes = 1
        class_labels = ["wheat_head"]
    else:
        raise ValueError("Unsupported dataset")

    
    if task == "train":
        assert data_dir, "Data directory must be specified for train task (use -d <DATA_DIRECTORY>)"
        results = model.train_model(data_dir=data_dir, dataset_name=dataset, num_classes=num_classes, class_labels=class_labels, output_dir=output_dir)
        print("Results: ", results)

    elif task == 'infer':
        assert args.input_image, "You should include an input image for infer task (use --input_image <PATH_TO_IMAGE>)"
        input_image = args.input_image
        
        # Get the image
        image = cv2.imread(input_image)
        print(f"Image shape: {image.shape}")
        
        # Get the predictions
        predictions = model.run_inference(image, num_classes=num_classes)
        visualized_image = model.visualize_predictions(image, predictions, class_names=class_labels)
        
        # Save image for processing
        print(f"Visualized image shape: {visualized_image.shape}")
        cv2.imwrite(f"{output_dir}/visualized_image_finetuned.png", visualized_image)

