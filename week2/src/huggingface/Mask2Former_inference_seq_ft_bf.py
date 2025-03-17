import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import cv2

# Determine whether to use GPU (CUDA) or fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_segmentation(image, segmentation, segments_info, id2label, alpha=0.5):
    """
    Overlays the segmentation mask on the original image with colors, contours, and class labels.
    
    Parameters:
        - image: Original PIL image
        - segmentation: Segmentation mask (numpy array)
        - segments_info: List of segment dictionaries containing {'id', 'label_id', 'score'}
        - id2label: Dictionary mapping label IDs to class names
        - alpha: Transparency level of the overlay (0=transparent, 1=opaque)
    """
    # Convert PIL image to NumPy
    image = np.array(image.convert("RGB"))
    
    # Create mask overlays
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    contour_overlay = np.zeros_like(image, dtype=np.uint8)
    
    # Define fixed colors for specific classes
    fixed_colors = {
        1: np.array([0, 0, 255], dtype=np.uint8),   # Blue for label_id = 0 --> car
        0: np.array([225, 0, 255], dtype=np.uint8)  # Magenta for label_id = 1 --> person
    }
    
    # Generate colors for segments
    np.random.seed(42)
    colors = {segment["id"]: fixed_colors.get(segment["label_id"], np.random.randint(0, 255, size=(3,), dtype=np.uint8)) 
                for segment in segments_info}
    
    # Overlay masks and prepare for label positioning
    for segment in segments_info:
        segment_id = segment["id"]
        label_id = segment["label_id"]
        label = id2label.get(label_id, f"Unknown-{label_id}")
        score = segment["score"]
        
        # Extract mask for this segment
        mask = segmentation == segment_id
        
        # Apply color to mask
        mask_overlay[mask] = colors[segment_id]
        
        # Add white contours for better segmentation visibility
        if np.any(mask):
            binary_mask = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_overlay, contours, -1, (255, 255, 255), 2)
        
        # Place labels on the mask (adjusting for smaller or crowded masks)
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            # Compute the centroid of the mask
            y_mean, x_mean = int(np.mean(y_indices)), int(np.mean(x_indices))
            
            # Adjust label placement to avoid clashing
            label_text = f"{label} ({score:.2f})"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x, text_y = x_mean, y_mean - 10  # Place text slightly above the centroid
            
            # Adjust to ensure the text doesn't go out of frame bounds
            text_x = max(0, min(text_x, image.shape[1] - text_size[0]))
            text_y = max(15, text_y)  # Keep some margin at the top
            
            # Draw label background for better readability
            cv2.rectangle(
                image, 
                (text_x, text_y - text_size[1] - 2), 
                (text_x + text_size[0], text_y + 2), 
                (0, 0, 0),  # Black background
                -1  # Filled rectangle
            )
            
            # Overlay the label text
            cv2.putText(
                image,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                lineType=cv2.LINE_AA
            )
    
    # Create a boolean mask for blending
    mask_boolean = np.any(mask_overlay > 0, axis=-1)
    
    # Blend original image with the mask overlay
    blended = image.copy()
    blended[mask_boolean] = (image[mask_boolean] * (1 - alpha) + mask_overlay[mask_boolean] * alpha).astype(np.uint8)
    
    # Apply contour overlay with high opacity for better visibility
    contour_boolean = np.any(contour_overlay > 0, axis=-1)
    blended[contour_boolean] = (blended[contour_boolean] * 0.3 + contour_overlay[contour_boolean] * 0.7).astype(np.uint8)
    
    return blended


def run_inference(sequence_id):
    """
    Runs instance segmentation inference on a sequence of images.

    Args:
        sequence_id (str): The sequence ID corresponding to a folder of images.

    Saves the segmented images to an output directory.
    """
    input_dir = f"/ghome/c5mcv01/mcv-c5-team1/data/training/val/{sequence_id}"
    output_dir = f"/ghome/c5mcv01/mcv-c5-team1/week2/src/huggingface/results/inference_ft_bf/{sequence_id}"
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Load the Mask2Former model and image processor
    processor = AutoImageProcessor.from_pretrained("/ghome/c5mcv01/mcv-c5-team1/week2/src/huggingface/finetune-instance-segmentation-ade20k-mini-mask2former_backbone_frozen_1/checkpoint-945")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("/ghome/c5mcv01/mcv-c5-team1/week2/src/huggingface/finetune-instance-segmentation-ade20k-mini-mask2former_backbone_frozen_1/checkpoint-945")
    model.to(device)  # Move model to the selected device (CPU or GPU)
    
    print("Clases disponibles en el modelo:")
    for class_id, class_name in model.config.id2label.items():
        print(f"ID: {class_id}, Clase: {class_name}")
        
    # Get a sorted list of all PNG images in the input directory
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path)  # Load the image
        inputs = processor(images=image, return_tensors="pt")  # Preprocess the image
        inputs.to(device)  # Move input tensors to the selected device
        
        # Perform inference without tracking gradients
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process the segmentation results, keeping only detections with score ≥ 0.9
        results = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]], threshold=0.5)[0]
        
        for segment in results["segments_info"]:
            print(f"ID: {segment['id']}, Label: {model.config.id2label.get(segment['label_id'], 'Unknown')}, Label_id: {segment['label_id']}, Score: {segment['score']:.2f}")

        # Visualize and save the segmentation result
        image_with_masks = visualize_segmentation(image, results['segmentation'].numpy(), results["segments_info"], model.config.id2label, alpha=0.5)
        save_path = os.path.join(output_dir, image_file)
        Image.fromarray(image_with_masks).save(save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    """
    Entry point of the script. Takes a sequence ID as a command-line argument.
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <sequence_id>")
        sys.exit(1)
    
    sequence_id = sys.argv[1]  # Get the sequence ID from command-line arguments
    run_inference(sequence_id)  # Run inference on the given sequence
