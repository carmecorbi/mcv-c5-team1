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
        DatasetCatalog.register(dataset_name, lambda: get_YOLOData_dicts(dataset_name+"/valid"))
        
        if not self.is_coco:
            print("Evaluating with custom class IDs...")
            MetadataCatalog.get(dataset_name).set(thing_classes=['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'])
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        else:
            print("!!!Evaluating with COCO class IDs...")
            #coco_classes = [""] * 81
            #coco_classes[0] = "person"
            #coco_classes[2] = "car"
            #MetadataCatalog.get(dataset_name).set(thing_classes=coco_classes)
        
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

        Returns:
            dict: Results of the training process.
        """
        # Register dataset and metadata for training and validation
       
        try:
            for split in ["train", "val"]:
                DatasetCatalog.register(dataset_name + "_train", lambda: get_YOLOData_dicts("aquarium-data-cots/train"))
                DatasetCatalog.register(dataset_name + "_val", lambda: get_YOLOData_dicts("aquarium-data-cots/valid"))
                #DatasetCatalog.register(dataset_name + "_test", lambda: get_YOLOData_dicts(dataset_name+"/test"))
                #MetadataCatalog.get(dataset_name + "_train").set(thing_classes=['wheat_head']) 
                #MetadataCatalog.get(dataset_name + "_val").set(thing_classes=['wheat_head']) 
                #MetadataCatalog.get(dataset_name + "_test").set(thing_classes=['wheat_head']) 
                MetadataCatalog.get(dataset_name + "_train").set(thing_classes=['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']) 
                MetadataCatalog.get(dataset_name + "_val").set(thing_classes=['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']) 
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
        
        ''' 
        # Uncoment for optuna
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
        '''
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
        #evaluator = COCOEvaluator(dataset_name + "_test", eval_cfg, False, output_dir=validation_output_dir)
        #val_loader = build_detection_test_loader(eval_cfg, dataset_name + "_train")
        
        # Run inference and return results
        print("Running inference on the test dataset...")
        return inference_on_dataset(predictor.model, val_loader, evaluator)
        
        
    
    
# Example usage

# Main for Aquarium dataset
if __name__ == "__main__":
    
    task = "infer"

    if task == "train":
        model = FasterRCNN("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", num_frozen_layers=5)
        print("Start Training")
        results = model.train_model("/ghome/c5mcv01/mcv-c5-team1/data", dataset_name= "AquariumDataCots", num_classes=7, output_dir="./output/Aquarium/training/trainF5_opt5_2000iter")
        print("Results: ", results)

    elif task == 'infer':
        model_weights = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/output/Aquarium/training/trainF5_opt5_2000iter/model_final.pth"
        model = FasterRCNN("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", model_weights, num_frozen_layers=5)
        
        input_image = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/aquarium-data-cots/test/images/IMG_2354_jpeg_jpg.rf.396e872c7fb0a95e911806986995ee7a.jpg"
        #input_image = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/aquarium-data-cots/test/images/IMG_2544_jpeg_jpg.rf.03f51bb9e1c57fb9cd62f8cbdca14e90.jpg"
        #input_image = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/aquarium-data-cots/test/images/IMG_2395_jpeg_jpg.rf.9f1503ad3b7a7c7938daed057cc4e9bc.jpg"
        #input_image = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/aquarium-data-cots/test/images/IMG_2465_jpeg_jpg.rf.7e699ec1d2e373d93dac32cd02db9438.jpg"
        #input_image = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/aquarium-data-cots/test/images/IMG_2496_jpeg_jpg.rf.3f91e7f18502074c89fa720a11926fab.jpg"
        
        output_dir = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/output_visualizations/Aquarium"

        # Get the image
        image = cv2.imread(input_image)
        print(f"Image shape: {image.shape}")
        
        # Get the predictions
        predictions = model.run_inference(image, num_classes=7)
        print(f"predictions: {predictions}")
        visualized_image = model.visualize_predictions(image, predictions, class_names=['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'])
        
        # Save image for processing
        print(f"Visualized image shape: {visualized_image.shape}")
        cv2.imwrite(f"{output_dir}/INFER_F5_visualized_image_finetuned5.png", visualized_image)

'''
# Main for GlobalWheatHead2020 Dataset
if __name__ == "__main__":
    
    task = "train"

    if task == "train":
        model = FasterRCNN("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", num_frozen_layers=5)
        print("Start Training")
        results = model.train_model("/ghome/c5mcv01/mcv-c5-team1/data", dataset_name= "GlobalWheat", num_classes=1, output_dir="./output/GlobalWheat/training/ppt/trainF5_opt5_3000iter_v2")
        print("Results: ", results)

    elif task == "test":
        model_weights = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/output/GlobalWheat/training/ppt/trainF0_opt0_3000iter/model_final.pth"
        model = FasterRCNN("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", model_weights, num_frozen_layers=0)
        print("Start Testing F=0, optuna0")
        results = model.train_model("/ghome/c5mcv01/mcv-c5-team1/data", dataset_name= "GlobalWheat", num_classes=1, output_dir="./output/GlobalWheat/training/ppt/trainF0_opt0_3000iter")
        print("Results: ", results)

    elif task == 'infer':
        model_weights = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/output/GlobalWheat/training/ppt/trainF0_opt0_3000iter/model_final.pth"
        model = FasterRCNN("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", model_weights, num_frozen_layers=0)
        #input_image = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/GlobalWheat/test/images/063b1190-8045-4b6b-ae85-731475d82ca0.png"
        #input_image = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/GlobalWheat/test/images/0bd71b05-0f8c-43f7-84c5-6033b0885141.png"
        input_image = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/GlobalWheat/test/images/32aa6056-9afd-4ef7-bc02-1defb76c20a6.png"
        #input_image = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/GlobalWheat/test/images/143ad1ee-1728-4154-aeae-14d10bc04c2c.png"
        #input_image = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/GlobalWheat/test/images/0a8b589a-a972-43cc-87e0-91d4afe7fe71.png"
        
        output_dir = "/ghome/c5mcv01/mcv-c5-team1/week1/src/domain_shift/output_visualizations"

        # Get the image
        image = cv2.imread(input_image)
        print(f"Image shape: {image.shape}")
        
        # Get the predictions
        predictions = model.run_inference(image, num_classes=1)
        print(f"predictions: {predictions}")
        visualized_image = model.visualize_predictions(image, predictions, class_names=["wheat_head"])
        
        # Save image for processing
        print(f"Visualized image shape: {visualized_image.shape}")
        cv2.imwrite(f"{output_dir}/INFER_F2_visualized_image_finetuned5__1.png", visualized_image)
'''