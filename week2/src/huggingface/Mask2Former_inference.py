import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image

def visualize_segmentation(image, segmentation, segments_info, id2label, alpha=0.5):
    """
    Overlays the segmentation mask on the original image with colors and class labels.
    
    Parameters:
        - image: Original PIL image
        - segmentation: Segmentation mask (numpy array)
        - segments_info: List of segment dictionaries containing {'id', 'label_id', 'score'}
        - id2label: Dictionary mapping label IDs to class names
        - alpha: Transparency level of the overlay (0=transparent, 1=opaque)
    """
    image = np.array(image.convert("RGB"))  # Convert PIL image to NumPy array
    mask_overlay = np.zeros_like(image, dtype=np.uint8)  # Create an empty mask overlay

    # Define fixed colors for specific labels
    fixed_colors = {
        0: np.array([0, 0, 255], dtype=np.uint8),   # Blue for label_id = 0
        2: np.array([225, 0, 255], dtype=np.uint8) # Magenta   for label_id = 2
    }

    # Generate random colors for other segments
    np.random.seed(42)
    colors = {segment["id"]: fixed_colors.get(segment["label_id"], np.random.randint(0, 255, size=(3,), dtype=np.uint8)) for segment in segments_info}

    # Apply masks to overlay
    for segment in segments_info:
        segment_id = segment["id"]
        label = id2label.get(segment["label_id"], "Unknown")  # Get class name
        score = segment["score"]

        # Create a mask for the current segment
        mask = segmentation == segment_id
        mask_overlay[mask] = colors[segment_id]

        # Find mask centroid to place the label
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            y_mean, x_mean = np.mean(y_indices).astype(int), np.mean(x_indices).astype(int)
            plt.text(x_mean, y_mean, f"{label} ({score:.2f})", color="white", fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    # Blend original image with the mask overlay
    blended = (image * (1 - alpha) + mask_overlay * alpha).astype(np.uint8)

    return blended

if __name__ == "__main__":
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance")

    # Load image
    image = Image.open("/ghome/c5mcv01/mcv-c5-team1/data/training/val/0010/000000.png")

    # Forward pass
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process segmentation
    results = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]], threshold=0.3)[0]
    
    segment_to_label = {segment['id']: segment['label_id'] for segment in results["segments_info"]}
    print(segment_to_label)

    # Visualize results
    image_with_masks = visualize_segmentation(image, results['segmentation'].numpy(), results['segments_info'], model.config.id2label, alpha=0.5)
    
    save_path = "/ghome/c5mcv01/mcv-c5-team1/week2/src/huggingface/results/mask_segment_1.png"
    Image.fromarray(image_with_masks).save(save_path)



    
    
    

    





