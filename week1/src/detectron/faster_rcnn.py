from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset

from detectron.dataset import CustomKittiMotsDataset

import numpy as np
import cv2
import os


class FasterRCNN:
    def __init__(self, config_file: str, weights_file: str, score_threshold: float=0.5, num_workers: int=4):
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
    
    def train_model(self, data_dir: str, dataset_name: str = "kitti-mots", num_classes: int = 2, output_dir: str = "./output/train") -> dict:
        """Train a model to a given dataset.

        Args:
            data_dir (str): Dataset directory.
            dataset_name (str, optional): Dataset name. Defaults to "kitti-mots".
            num_classes (int, optional): Number of classes in the dataset. Defaults to 2.
            output_dir (str, optional): Output directory for the training process. Defaults to "./output/train".

        Returns:
            dict: Results of the training process.
        """
        # Register dataset
        DatasetCatalog.register(dataset_name + "_train", lambda: CustomKittiMotsDataset(data_dir, use_coco_ids=False, split="train"))
        DatasetCatalog.register(dataset_name + "_val", lambda: CustomKittiMotsDataset(data_dir, use_coco_ids=False, split="val"))
        MetadataCatalog.get(dataset_name + "_train").set(thing_classes=["0", "1"]) 
        MetadataCatalog.get(dataset_name + "_val").set(thing_classes=["0", "1"]) 
        
        # Set the number of classes
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        
        # TRAINING SPECIFIC CONFIGURATION
        self.cfg.DATASETS.TRAIN = (dataset_name + "_train",)
        self.cfg.DATASETS.TEST = (dataset_name + "_val",)

        # Solver parameters
        self.cfg.SOLVER.IMS_PER_BATCH = 4  # Batch size (adjust based on GPU memory)
        self.cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
        self.cfg.SOLVER.MAX_ITER = 5000    # Maximum iterations
        self.cfg.SOLVER.STEPS = (3000, 4500)  # Learning rate decay steps
        self.cfg.SOLVER.GAMMA = 0.1        # Learning rate decay factor
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoint every 1000 iterations
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Faster, and good enough for COCO pre-trained models
        self.cfg.TEST.EVAL_PERIOD = 500

        # Output directory for training logs and checkpoints
        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        # Create trainer and start training
        print("Starting training...")
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        return trainer.train()
    
    
# Example usage
if __name__ == "__main__":
    """
    # Task c: Run inference on single image
    image = cv2.imread("/ghome/c3mcv02/mcv-c5-team1/data/testing/0000/000000.png")
    print(f"Image shape: {image.shape}")
    predictions = model.run_inference(image)
    visualized_image = model.visualize_predictions(image, predictions)
    print(f"Visualized image shape: {visualized_image.shape}")
    cv2.imwrite("visualized_image.png", visualized_image)
    """
    
    """
    # Task d: Evaluate on dataset
    model = FasterRCNN("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    results = model.evaluate_model("/ghome/c3mcv02/mcv-c5-team1/data", output_dir="./output/eval_pretrained")
    print(results)
    """
    
    # Task e: Train on custom dataset
    model = FasterRCNN("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    results = model.train_model("/ghome/c3mcv02/mcv-c5-team1/data", num_classes=2, output_dir="./output/train")
    
    # Task f_1: Evaluate fine-tuned
    #model = FasterRCNN("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "/ghome/c3mcv02/mcv-c5-team1/week1/src/detectron/output/train/model_final.pth")
    #results = model.evaluate_model("/ghome/c3mcv02/mcv-c5-team1/data", num_classes=2, output_dir="./output/eval_finetuned")
    
    """
    # Task f_2: Run inference on single image with finetuned
    model = FasterRCNN("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "/ghome/c3mcv02/mcv-c5-team1/week1/src/detectron/output/train/model_final.pth")
    image = cv2.imread("/ghome/c3mcv02/mcv-c5-team1/data/testing/0000/000000.png")
    print(f"Image shape: {image.shape}")
    predictions = model.run_inference(image)
    visualized_image = model.visualize_predictions(image, predictions)
    print(f"Visualized image shape: {visualized_image.shape}")
    cv2.imwrite("visualized_image_finetuned.png", visualized_image)
    """