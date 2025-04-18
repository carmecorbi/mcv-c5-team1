import numpy as np 
import os

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from week2.src.detectron.dataset import CustomKittiMotsDataset
from week2.src.detectron.trainer import CustomTrainer

# To test if it works
from detectron2.engine import DefaultTrainer


class MaskRCNN:
    def __init__(self, config_file: str, weights_file: str, score_threshold: float = 0.5, num_workers: int = 4):
        """Initializes the MaskRCNN model

        Args:
            config_file (str): Path to the model configuration file.
            weights_file (str): Path to the model weights file. Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well.
            score_threshold (float, optional): onfidence score threshold. Defaults to 0.5.
            num_workers (int, optional): Number of workers for the dataset loading. Defaults to 4.
        """
        # Load the model configuration and weights
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        
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
            pred = predictor(image)

            # Filter instances for cars and pedestrians
            instances = pred['instances']
            if self.is_coco:
                keep_mask = (instances.pred_classes == 0) | (instances.pred_classes == 2)
            else:
                keep_mask = (instances.pred_classes == 0) | (instances.pred_classes == 1)

            pred['instances'] = instances[keep_mask]
            predictions = pred
        elif len(image.shape) == 4:
            for img in image:
                pred = predictor(img)

                # Filter instances for cars and pedestrians
                instances = pred['instances']
                if self.is_coco:
                    keep_mask = (instances.pred_classes == 0) | (instances.pred_classes == 2)
                else:
                    keep_mask = (instances.pred_classes == 0) | (instances.pred_classes == 1)

                pred['instances'] = instances[keep_mask]
                predictions.append(pred)
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
        
        v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]
    
    def evaluate_model(self, data_dir: str, dataset_name: str = "kitti-mots", num_classes: int = 2, output_dir: str = "./output/eval") -> dict:
        """Evaluates the model into a custom dataset.

        Args:
            data_dir (str): Dataset directory.
            dataset_name (str, optional): Dataset name. Defaults to "kitti-mots".
            num_classes (int, optional): Number of classes in the dataset. Defaults to 2.
            output_dir (str, optional): Output directory for the evaluation. Defaults to "./output/eval".

        Returns:
            dict: Inference results
        """
        # Register dataset
        # TODO: Update this dataset to include the mask loading.
        DatasetCatalog.register(dataset_name, lambda: CustomKittiMotsDataset(data_dir, use_coco_ids=self.is_coco, split="val"))
        
        if not self.is_coco:
            print("Evaluating with custom class IDs...")
            MetadataCatalog.get(dataset_name).set(thing_classes=["0", "1"])
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        else:
            print("Evaluating with COCO class IDs...")
            coco_classes = [""] * 81
            coco_classes[0] = "person"
            coco_classes[2] = "car"
            MetadataCatalog.get(dataset_name).set(thing_classes=coco_classes)
            
        # Set input mask as bitmask
        self.cfg.INPUT.MASK_FORMAT = "bitmask" 
        
        # Create a predictor for inference
        print("Creating predictor...")
        predictor = DefaultPredictor(self.cfg)
        
        # Evaluate on the test dataset using COCO evaluator
        print("Performing evaluation...")
        evaluator = COCOEvaluator(dataset_name, self.cfg, False, output_dir=output_dir)
        val_loader = build_detection_test_loader(self.cfg, dataset_name)
        
        # Run inference and return results
        print("Running inference on the test dataset...")
        return inference_on_dataset(predictor.model, val_loader, evaluator)
    
    def train_model(self, data_dir: str, dataset_name: str = "kitti-mots", num_classes: int = 2, output_dir: str = "./output/train", **kwargs) -> dict:
        """Train a model to a given dataset.

        Args:
            data_dir (str): Dataset directory.
            dataset_name (str, optional): Dataset name. Defaults to "kitti-mots".
            num_classes (int, optional): Number of classes in the dataset. Defaults to 2.
            output_dir (str, optional): Output directory for the training process. Defaults to "./output/train".
            **kwargs: Additional arguments for the training process.

        Returns:
            dict: Results of the training process.
        """
        # Set input mask as bitmask
        self.cfg.INPUT.MASK_FORMAT = "bitmask"
        
        # Register dataset and metadata for training and validation
        try:
            for split in ["train", "val"]:
                DatasetCatalog.register(dataset_name + "_" + split, lambda: CustomKittiMotsDataset(data_dir, use_coco_ids=False, split=split))
                MetadataCatalog.get(dataset_name + "_" + split).set(thing_classes=["0", "1"])
        except AssertionError:
            print("Dataset already registered, continuing...")
        
        # Model parameters
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = kwargs.get("batch_size_per_image", 128) # 128 is faster and enough for this dataset
        self.cfg.MODEL.BACKBONE.FREEZE_AT = 5 if kwargs.get("freeze_backbone", False) else 0
        
        # Datasets parameters (train and test)
        self.cfg.DATASETS.TRAIN = (dataset_name + "_train",)
        self.cfg.DATASETS.TEST = (dataset_name + "_val",)

        # Solver parameters (optimizer)
        lr_scheduler = kwargs.get("lr_scheduler", "WarmupMultiStepLR")
        assert lr_scheduler in ["WarmupMultiStepLR", "WarmupCosineLR"], "LR scheduler must be WarmupMultiStepLR or WarmupCosineLR"
        self.cfg.SOLVER.LR_SCHEDULER_NAME = lr_scheduler # Learning rate scheduler
        self.cfg.SOLVER.IMS_PER_BATCH = kwargs.get("batch_size", 8) # Batch size
        print(f"BATCH SIZE: {self.cfg.SOLVER.IMS_PER_BATCH}")
        self.cfg.SOLVER.BASE_LR = kwargs.get("base_lr", 0.001)  # Learning rate
        self.cfg.SOLVER.MOMENTUM = kwargs.get("momentum", 0.9)  # Momentum
        self.cfg.SOLVER.NESTEROV = kwargs.get("nesterov", False)  # Nesterov momentum
        self.cfg.SOLVER.WEIGHT_DECAY = kwargs.get("weight_decay", 0.0001) # Weight decay
        self.cfg.SOLVER.GAMMA = kwargs.get("gamma", 0.1) # Learning rate decay factor
        self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED = kwargs.get("clip_gradients", False) # Clip gradients
        self.cfg.SOLVER.STEPS = kwargs.get("steps", (1000, 2000)) # Steps for learning rate decay

        # Fixed parameters (do not change)
        self.cfg.SOLVER.MAX_ITER = kwargs.get("max_iter", 3000)
        self.cfg.SOLVER.CHECKPOINT_PERIOD = kwargs.get("checkpoint_period", 1000)

        # Test parameters (evaluation)
        self.cfg.TEST.EVAL_PERIOD = kwargs.get("eval_period", 1000)

        # Output directory for training logs and checkpoints
        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        # Create trainer and start training
        print("Starting training...")
        trainer = CustomTrainer(self.cfg)
        # trainer = DefaultTrainer(self.cfg)
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
    