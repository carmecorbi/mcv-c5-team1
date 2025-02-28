from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import numpy as np 
import cv2


class FasterRCNN:
    def __init__(self, config_file: str, weights_file: str, score_threshold: float=0.5):
        """Initializes the FasterRCNN model

        Args:
            config_file (str): Path to the model configuration file. Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
            weights_file (str): Path to the model weights file. Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well.
            score_threshold (float, optional): Confidence score threshold. Defaults to 0.5.
        """
        # Load the model configuration and weights
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        
        # Create the predictor
        self.predictor = DefaultPredictor(self.cfg)
        
        pass
        
    def run_inference(self, image: np.ndarray) -> dict:
        """Run inference on the input image or batch of images

        Args:
            image (np.ndarray): OpenCV image or batch of images. In BGR format.

        Returns:
            dict: Dictionary containing the predictions. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        """
        return self.predictor(image)
    
    def visualize_predictions(self, image: np.ndarray, predictions: dict) -> np.ndarray:
        """Visualize the predictions on the input image

        Args:
            image (np.ndarray): OpenCV image. In BGR format.
            predictions (dict): Dictionary containing the predictions. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification

        Returns:
            np.ndarray: Visualized image. In BGR format.
        """
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]
    
    
# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = FasterRCNN("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    # Run inference
    image = cv2.imread("/ghome/c3mcv02/mcv-c5-team1/data/testing/0000/000000.png")
    predictions = model.run_inference(image)
    
    # Visualize the predictions
    visualized_image = model.visualize_predictions(image, predictions)
    
    # Save the image 
    cv2.imwrite("visualized_image.png", visualized_image)
    
    # Test with a batch of images
    #batch_images = np.stack([image, image], axis=0)
    #predictions = model.run_inference(batch_images)