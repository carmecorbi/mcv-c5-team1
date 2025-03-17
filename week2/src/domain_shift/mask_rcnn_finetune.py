import numpy as np 
import os
import detectron2

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode

import matplotlib.pyplot as plt
import numpy as np
import torch, os, json, cv2, random
import glob

from typing import Literal
from PIL import Image
from detectron2.data import DatasetMapper, detection_utils
from pycocotools import mask as mask_utils

import albumentations as A

from detectron2.data import build_detection_train_loader


CLASS_LABELS = ["Angular Leafspot", "Anthracnose Fruit Rot", "Blossom Blight", "Gray Mold", "Leaf Spot", "Powdery Mildew Fruit", "Powdery Mildew Leaf"]
dataset_path = '/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/'

def get_augmentations() -> A.Compose:
	"""Get the augmentations to apply.

	Returns:
		A.Compose: Compose of augmentations from albumentations.
	"""
	return A.Compose([
		A.MotionBlur(p=0.2, blur_limit=(3, 8)),
  		A.Illumination(p=0.2, intensity_range=(0.1, 0.2)),
        A.ColorJitter(p=0.2, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        A.Rotate(p=0.2, limit=(-45, 45)),
        A.Sharpen(p=0.2, alpha=(0,1), lightness=(0.75, 2))
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.1, label_fields=['category_ids']))

def get_ds_dicts(dataset_path):
        # List all JSON files in the dataset directory.
        json_files = glob.glob(os.path.join(dataset_path,"labels", "*.json"))
        print(f"json_files: ")
        dataset_dicts = []
        image_id = 0
        
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
                
            # Assume each JSON file contains annotations for one image.
            record = {}
            filename = os.path.join(dataset_path,"images", data["imagePath"])
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = height
            record["width"] = width

            annos = data["shapes"]
            objs = []
            
            for anno in annos:
                px = [a[0] for a in anno['points']]
                py = [a[1] for a in anno['points']]
                poly = [(x, y) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [max(np.min(px),0), max(np.min(py),0), min(np.max(px), width), min(np.max(py), height)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": CLASS_LABELS.index(anno['label']),
                    "iscrowd": 0
                }
                objs.append(obj)
            
            record["annotations"] = objs
            dataset_dicts.append(record)
            image_id += 1
        
        return dataset_dicts

class CustomTrainer(DefaultTrainer):
	@classmethod
	def build_evaluator(cls, cfg, dataset_name, output_folder=None):
		if output_folder is None:
			output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
		return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)
	
	@classmethod
	def build_train_loader(cls, cfg):
		mapper = AlbumentationsMapper(cfg, is_train=True, augmentations=get_augmentations())
		return build_detection_train_loader(cfg, mapper=mapper)
		# return build_detection_train_loader(cfg)

class AlbumentationsMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True, augmentations=None):
        """Initializes albumentations mapper.

        Args:
            cfg (Any): Configuration for the model.
            is_train (bool, optional): Whether is train dataset. Defaults to True.
            augmentations (Any, optional): Augmentations from albumentations to apply. Defaults to None.
        """
        super().__init__(cfg, is_train)
        self.augmentations = augmentations

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = cv2.imread(dataset_dict["file_name"]) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_train and "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            bboxes = [obj["bbox"] for obj in annotations]
            category_ids = [obj["category_id"] for obj in annotations]
            
            transformed = self.augmentations(image=image, bboxes=bboxes, category_ids=category_ids)
            image = transformed["image"]
            
            # Update the bounding boxes with transformed coordinates
            for i, annotation in enumerate(annotations):
                if i < len(transformed["bboxes"]):
                    annotation["bbox"] = transformed["bboxes"][i]
            
            # Convert to Instances format for Detectron2
            annos = []
            for annotation in annotations:
                obj = {
                    "bbox": annotation["bbox"],
                    "bbox_mode": annotation.get("bbox_mode", BoxMode.XYWH_ABS),
                    "segmentation": annotation.get("segmentation"),
                    "category_id": annotation["category_id"],
                    "iscrowd": annotation.get("iscrowd", 0),
                }
                annos.append(obj)
            
            # Create Instances object with the correct image size
            instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = instances
        
        # Convert to CHW format
        image = image.transpose(2, 0, 1)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image))
        
        return dataset_dict


class MaskRCNN:
    def __init__(self, config_file: str, weights_file: str, score_threshold: float = 0.7, num_workers: int = 2):
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
        
        v = Visualizer(image[:, :, ::-1], metadata, scale=1.8)
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]
    
    def run_inference(self, image: np.ndarray, num_classes: int = 7) -> dict:
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
            pred['instances'] = instances
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

    def evaluate_model(self, data_dir: str, dataset_name: str = "strawberry", num_classes: int = 7, output_dir: str = "./output/eval") -> dict:
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
        DatasetCatalog.register(dataset_name, lambda: get_ds_dicts(data_dir, use_coco_ids=self.is_coco, split="val"))
        
        if not self.is_coco:
            print("Evaluating with custom class IDs...")
            MetadataCatalog.get(dataset_name).set(thing_classes=CLASS_LABELS)
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


    def train_model(self, data_dir: str, dataset_name: str = "kitti-mots", num_classes: int = 7, output_dir: str = "./output/train", **kwargs) -> dict:
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
        
        # Register dataset and metadata for training and validation
        try:
            for d in ["train", "val"]:
                print(f"path: {dataset_path + d}")
                DatasetCatalog.register("strawberry_" + d, lambda d=d: get_ds_dicts(dataset_path + d))
                MetadataCatalog.get("strawberry_" + d).set(thing_classes=CLASS_LABELS)
        
        except AssertionError:
            print("Dataset already registered, continuing...")

        # Model parameters 
        self.cfg.DATASETS.TRAIN = ("strawberry_train",)
        self.cfg.DATASETS.TEST = ("strawberry_val",)
        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.BASE_LR = 0.0009943827408717774
        self.cfg.SOLVER.LR_SCHEDULER_NAME = 'WarmupCosineLR'
        self.cfg.SOLVER.WEIGHT_DECAY = 4.369992210931218e-05
        self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False
        self.cfg.SOLVER.MAX_ITER = 4000
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64 # faster, and good enough for this toy dataset
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # only has one class (ballon)
        self.cfg.MODEL.BACKBONE.FREEZE_AT = 0 

        self.cfg.SOLVER.MOMENTUM = kwargs.get("momentum", 0.9)  # Momentum
        self.cfg.SOLVER.NESTEROV = kwargs.get("nesterov", False)  # Nesterov momentum
        self.cfg.SOLVER.GAMMA = kwargs.get("gamma", 0.1) # Learning rate decay factor
        self.cfg.SOLVER.STEPS = kwargs.get("steps", (100, 2000)) # Steps for learning rate decay

        # Fixed parameters (do not change)
        self.cfg.SOLVER.CHECKPOINT_PERIOD = kwargs.get("checkpoint_period", 1000)

        # Test parameters (evaluation)
        self.cfg.TEST.EVAL_PERIOD = kwargs.get("eval_period", 1000)

        # Output directory for training logs and checkpoints
        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        # Create trainer and start training
        print("Starting training...")
        #trainer = CustomTrainer(self.cfg)
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
        evaluator = COCOEvaluator("strawberry_val", eval_cfg, False, output_dir=validation_output_dir)
        val_loader = build_detection_test_loader(eval_cfg, "strawberry_val")
        
        # Run inference and return results
        print("Running inference on the test dataset...")
        return inference_on_dataset(predictor.model, val_loader, evaluator)


if __name__ == '__main__':
    #Uncomment for training:
    '''
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    weights_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    score_threshold = 0.5
    num_workers = 2
    output_dir = "./output/mask_rcnn_allunfrozen_real/withAugmentations"
    freeze_backbone = 0
    
    # Get the model
    model = MaskRCNN(config_file, weights_file, score_threshold, num_workers)
    
    model.train_model(dataset_path, "strawberry-disease-dataset", num_classes=7, output_dir=output_dir)
    '''

    #Uncomment for inference:
    '''
    model_weights = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/output/mask_rcnn_allunfrozen_real/withAugmentations/model_final.pth"
    model = MaskRCNN("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", model_weights, 0.5, 2)
        
    input_image = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/train/images/angular_leafspot1.jpg"     
    output_dir = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/maskrcnn/output_visulaizations/powdery_mildew_leaf468.jpg"

    # Get the image
    image = cv2.imread(input_image)

    # Get the predictions
    predictions = model.run_inference(image, num_classes=7)
    visualized_image = model.visualize_predictions(image, predictions, class_names=CLASS_LABELS)
        
    cv2.imwrite(f"{output_dir}", visualized_image)
    '''