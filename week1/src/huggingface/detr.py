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
                "class_id": int(labels[i]),  # Convierte a int explícitamente
                "bbox": boxes[i].tolist(),
                "score": float(scores[keep][i])  # Convierte a float
            })
        
        return results


    def visualize_predictions(self, image: np.ndarray, outputs: list) -> tuple:
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
        color_map = {"person": (255, 0, 0), "car": (255, 0, 255)}  # Blue for person, pink for car

        for obj in outputs:
            class_id = obj["class_id"]
            bbox = obj["bbox"]
            score = obj["score"]

            # Obtener el nombre de la clase
            class_name = self.model.config.id2label[class_id]

            # Filtrar solo "person" y "car"
            if class_name not in color_map:
                continue  

            # Convertir bbox (center_x, center_y, width, height) → (x_min, y_min, x_max, y_max)
            x_min = int((bbox[0] - bbox[2] / 2) * w)
            y_min = int((bbox[1] - bbox[3] / 2) * h)
            x_max = int((bbox[0] + bbox[2] / 2) * w)
            y_max = int((bbox[1] + bbox[3] / 2) * h)

            # Guardar la caja en la lista
            bboxes.append((x_min, y_min, x_max, y_max, class_name, score))

            # Dibujar bounding box
            color = color_map[class_name]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)

            # Dibujar el fondo del texto
            text = f"{class_name} {score * 100:.1f}%"
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x, text_y = x_min, y_min - 5
            cv2.rectangle(image, (text_x, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), (255, 255, 255), -1)

            # Dibujar la etiqueta con la confianza
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

