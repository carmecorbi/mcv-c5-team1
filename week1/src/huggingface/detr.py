from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import numpy as np
import cv2

class DeTR:
    def __init__(self, model_name: str = "facebook/detr-resnet-50", score_threshold: float = 0.5):
        """Initializes the DeTR model"""
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.score_threshold = score_threshold

    def run_inference(self, image: np.ndarray) -> dict:
        """Run inference on the input image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def visualize_predictions(self, image: np.ndarray, outputs) -> np.ndarray:
        """Visualize predictions for 'person' and 'car' only, with custom colors"""
        scores = outputs.logits.softmax(-1)[0, :, :-1].max(-1)[0]
        keep = scores > self.score_threshold

        if keep.sum() == 0:
            print("No objects detected with the required confidence")
            return image

        h, w, _ = image.shape
        boxes = outputs.pred_boxes[0, keep].detach().cpu().numpy()
        labels = outputs.logits.argmax(-1)[0, keep].detach().cpu().numpy()

        # Convert box format: (center_x, center_y, width, height) -> (x_min, y_min, x_max, y_max)
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # Class names and filtering
        category_names = [self.model.config.id2label[label] for label in labels]
        filtered_indices = [i for i, name in enumerate(category_names) if name in ["person", "car"]]

        # Define colors
        color_map = {"person": (255, 0, 0), "car": (255, 0, 255)}  # Blue for person, pink for car

        # Draw only filtered objects
        for i in filtered_indices:
            box = boxes[i]
            class_name = category_names[i]
            confidence = f"{scores[keep][i] * 100:.1f}%"
            x_min, y_min, x_max, y_max = map(int, box)

            # Draw bounding box (thinner lines)
            color = color_map[class_name]  
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)  # Line thickness = 1

            # Draw background for text
            text = f"{class_name} {confidence}"
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x, text_y = x_min, y_min - 5
            cv2.rectangle(image, (text_x, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), (255, 255, 255), -1)

            # Draw label with confidence score
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        return image


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
    visualized_image = model.visualize_predictions(image, predictions)
    print(f"Visualized image shape: {visualized_image.shape}")
    cv2.imwrite("visualized_image.png", visualized_image)

