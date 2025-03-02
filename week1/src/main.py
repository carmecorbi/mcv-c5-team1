from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch

import numpy as np
from detectron.faster_rcnn import FasterRCNN

import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm


class KittiMotsDataset(Dataset):
    def __init__(self, root: str | Path, transform: any=None):
        """Create a dataset class for KittiMots dataset.

        Args:
            root (str | Path): Path to the dataset.
            transform (any, optional): Transformations to apply to the dataset. Defaults to None.
        """
        self.root = root
        self.transform = transform
        self.data = datasets.ImageFolder(root, transform=transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.data[idx]
    
    
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='KittiMots Dataset')
    parser.add_argument('-v', '--val', help="Path to validation dataset", required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-c', '--config_file', default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument('-w', '--weights_file', default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument('-t', '--score_threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    # Load validation dataset
    val_transforms = transforms.Compose([
        transforms.Resize((375, 1242)),
        transforms.ToTensor(),
    ])
    val_dataset = KittiMotsDataset(args.val, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Define model
    model = FasterRCNN(config_file=args.config_file, weights_file=args.weights_file, score_threshold=args.score_threshold)
    
    print("-"*50)
    # Run inference
    output_images = []
    preds = []
    for images, _ in tqdm(val_loader):
        images = images.permute(0, 2, 3, 1)  # Change shape from (B,C,H,W) to (B,H,W,C)
        images = (images.numpy() * 255).astype(np.uint8)  # Now shape will be (B,H,W,C)
        
        # Debug: Check image values
        print(f"Image min/max values: {images.min()}, {images.max()}")
        print(f"Images shape: {images.shape}")
        
        predictions = model.run_inference(images)
        
        print(f"Predictions shape: {len(predictions)}")
        print(f"Number of instances: {len(predictions[0]['instances'])}")
        
        for i, image in enumerate(images):
            # Debug: Check if image is valid
            if image.max() == 0:
                print("Warning: Image is all black!")
            visualized_image = model.visualize_predictions(image, predictions[i])
            output_images.append(visualized_image)
        preds.append(predictions)
        print("-"*50)
        
    # Show some images in matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))  # Larger figure size
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Add spacing between subplots
    for idx, ax in enumerate(axes.flat):
        if idx < len(output_images):
            ax.imshow(output_images[idx])
            ax.set_title(f'Detection {idx+1}')
            ax.axis('off')  # Hide axes
        else:
            ax.axis('off')  # Hide empty subplots
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("output.png", bbox_inches='tight', dpi=300)
    plt.close()
    
