import torch
import numpy as np
import cv2

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image


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
        0: np.array([0, 0, 255], dtype=np.uint8),   # Blue for label_id = 0
        2: np.array([225, 0, 255], dtype=np.uint8)  # Magenta for label_id = 2
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



if __name__ == "__main__":
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")

    # Load image and prepare input
    image = Image.open("/ghome/c5mcv01/mcv-c5-team1/data/training/val/0002/000206.png")
    inputs = processor(images=image, return_tensors="pt")
    for k,v in inputs.items():
        print(k,v.shape)
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    print(outputs.keys())

    # Post-process segmentation
    results = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]], threshold=0.5)[0]
    print(results.keys())
    for segment in results['segments_info']:
        print(segment)
    
    # Get the segment label
    segment_to_label = {segment['id']: segment['label_id'] for segment in results["segments_info"]}
    print(segment_to_label)
    
    #print("Clases disponibles en el modelo:")
    #for class_id, class_name in model.config.id2label.items():
    #    print(f"ID: {class_id}, Clase: {class_name}")
        
    #for segment in results["segments_info"]:
    #    print(f"ID: {segment['id']}, Label: {model.config.id2label[segment['label_id']]}, Score: {segment['score']:.2f}")

    filtered_segments = [segment for segment in results["segments_info"] if segment["label_id"] in {0, 2}]
    print(filtered_segments)

    # Visualize results
    image_with_masks = visualize_segmentation(image, results['segmentation'].numpy(), filtered_segments, model.config.id2label, alpha=0.5)
    save_path = "/ghome/c5mcv01/mcv-c5-team1/week2/src/huggingface/results/mask_segment_2.png"
    Image.fromarray(image_with_masks).save(save_path)
    