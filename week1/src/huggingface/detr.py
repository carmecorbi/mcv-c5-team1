from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import numpy as np
import cv2

class DeTR: 
    def __init__(self, model_name: str = "facebook/detr-resnet-50", model_path: str = None, score_threshold: float = 0.5):
        """
        Initializes the DeTR model. 
        If model_path is provided, loads fine-tuned weights; otherwise, it uses the pretrained model.
        """
        if model_path:  # Load fine-tuned model if a path is provided
            print(f"Loading fine-tuned model from {model_path}")
            self.processor = DetrImageProcessor.from_pretrained(model_path)
            self.model = DetrForObjectDetection.from_pretrained(model_path)
        else:  # Use Facebook's pretrained model
            print(f"Loading pretrained model: {model_name}")
            self.processor = DetrImageProcessor.from_pretrained(model_name)
            self.model = DetrForObjectDetection.from_pretrained(model_name)

        self.score_threshold = score_threshold
        self.model.eval()  # Set model to evaluation mode for inference


    def run_inference(self, image: np.ndarray) -> list:
        """Run inference on the input image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        scores = outputs.logits.softmax(-1)[0, :, :-1].max(-1)[0]
        keep = scores > self.score_threshold

        h, w, _ = image.shape
        boxes = outputs.pred_boxes[0, keep].detach().cpu().numpy()
        labels = outputs.logits.argmax(-1)[0, keep].detach().cpu().numpy()

        results = []
        for i in range(len(boxes)):
            results.append({
                "class_id": int(labels[i]),  
                "bbox": boxes[i].tolist(),
                "score": float(scores[keep][i])  
            })
        
        return results


    def visualize_predictions(self, image: np.ndarray, outputs: list, finetuning = False) -> tuple:
        """Visualize predictions for 'person' and 'car' only, with custom colors.
        
        Returns:
            - image (np.ndarray): Image with visualized detections
            - bboxes (list): List of bounding boxes in pixel coordinates [(x_min, y_min, x_max, y_max, class_name, score), ...]
        """

        if not outputs:
            print("No objects detected with the required confidence")
            return image, []

        h, w, _ = image.shape
        bboxes = []

        # Define colors
        if finetuning == False:
            color_map = {"person": (255, 0, 0), "car": (255, 0, 255)}  # Blue for person, pink for car
        else:
            color_map = {"pedestrian": (255, 0, 0), "car": (255, 0, 255)}  # Blue for person, pink for car

        for obj in outputs:
            class_id = obj["class_id"]
            bbox = obj["bbox"]
            score = obj["score"]

            # Get class name
            class_name = self.model.config.id2label[class_id]

            # Filter only "person" and "car"
            if class_name not in color_map:
                continue  

            # Convert bbox (center_x, center_y, width, height) → (x_min, y_min, x_max, y_max)
            x_min = int((bbox[0] - bbox[2] / 2) * w)
            y_min = int((bbox[1] - bbox[3] / 2) * h)
            x_max = int((bbox[0] + bbox[2] / 2) * w)
            y_max = int((bbox[1] + bbox[3] / 2) * h)

            # Store the box in the list
            bboxes.append((x_min, y_min, x_max, y_max, class_name, score))

            # Draw bounding box
            color = color_map[class_name]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)

            # Draw text background
            text = f"{class_name} {score * 100:.1f}%"
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x, text_y = x_min, y_min - 5
            cv2.rectangle(image, (text_x, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), (255, 255, 255), -1)

            # Draw label with confidence score
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        return image, bboxes




# Example usage
if __name__ == "__main__":
    # Initialize model
    model = DeTR()
    
    # Task c: Run inference on single image
    image = cv2.imread("/ghome/c3mcv02/mcv-c5-team1/data/testing/0000/000000.png")
    if image is None:
        raise ValueError("Error: Unable to load the image. Check the path.")
    
    print(f"Image shape: {image.shape}")
    predictions = model.run_inference(image)
    visualized_image, _ = model.visualize_predictions(image, predictions)
    print(f"Visualized image shape: {visualized_image.shape}")
    cv2.imwrite("visualized_image.png", visualized_image)

