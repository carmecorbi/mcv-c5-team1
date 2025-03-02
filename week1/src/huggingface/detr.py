from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import numpy as np
import cv2

class DeTR:
    def __init__(self, model_name: str = "facebook/detr-resnet-50", score_threshold: float = 0.5):
        """Initializes the DeTR model"""
        # Load processor and model
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.score_threshold = score_threshold

    def run_inference(self, image: np.ndarray) -> dict:
        """Run inference on the input image"""
        # Convert image to RGB and preprocess
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        
        # Run model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs

    def visualize_predictions(self, image: np.ndarray, outputs) -> np.ndarray:
        """Visualize the predictions on the input image"""
        scores = outputs.logits.softmax(-1)[0, :, :-1].max(-1)[0]
        keep = scores > self.score_threshold
        boxes = outputs.pred_boxes[0, keep].detach().cpu().numpy()
        
        # Draw boxes on image
        for box in boxes:
            x_min, y_min, width, height = box
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_min + width), int(y_min + height)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return image

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = DeTR()
    
    # Load image
    image = cv2.imread("/ghome/c3mcv02/mcv-c5-team1/data/testing/0000/000000.png")
    print(f"Image shape: {image.shape}")
    
    # Run inference
    predictions = model.run_inference(image)
    
    # Visualize predictions
    visualized_image = model.visualize_predictions(image, predictions)
    print(f"Visualized image shape: {visualized_image.shape}")
    
    # Save the image
    cv2.imwrite("visualized_detr.png", visualized_image)
    
    # Test with a batch of images
    batch_images = np.stack([image, image], axis=0)
    predictions = model.run_inference(batch_images)
    
    # Visualize the predictions
    visualized_images = []
    for i, img in enumerate(batch_images):
        visualized_image = model.visualize_predictions(img, predictions[i])
        visualized_images.append(visualized_image)
    print(f"Visualized images shape: {visualized_images[0].shape}")
    cv2.imwrite("visualized_image_batch.png", visualized_images[0])
    cv2.imwrite("visualized_image_batch2.png", visualized_images[1])
