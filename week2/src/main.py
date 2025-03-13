import argparse
import cv2
import os

from week2.src.detectron.mask_rcnn import MaskRCNN
    
    
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='KittiMots Dataset')
    parser.add_argument('-d', '--data_dir', help="Path to validation dataset", required=False)
    parser.add_argument('-t', '--task', help="Task to do (infer, train, eval)", required=True)
    parser.add_argument('-i', '--input_image', help="Input image to infer on (only for infer task)", type=str, required=False)
    parser.add_argument('-c', '--config_file', default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="Path to the model config yaml from model zoo.")
    parser.add_argument('-w', '--weights_file', default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="Path to the weights file or model zoo config yaml.")
    parser.add_argument('-s', '--score_threshold', type=float, default=0.5, help="Score threshold for predictions.")
    parser.add_argument('-o', '--output_dir', help="Output directory for the model", default=None)
    parser.add_argument('--num_workers', required=False, default=4, type=int, help="Number of workers to load dataset.")
    args = parser.parse_args()
    
    # Get the arguments from CLI
    data_dir = args.data_dir
    config_file = args.config_file
    weights_file = args.weights_file
    score_threshold = args.score_threshold
    num_workers = args.num_workers
    task = args.task
    output_dir = args.output_dir
    
    # Get the model
    model = MaskRCNN(config_file, weights_file, score_threshold, num_workers)
    
    if task == "train":
        assert data_dir, "Data directory must be specified for eval task (use -d <DATA_DIRECTORY>)"
        raise NotImplementedError
        hyperparams = {
            # Model hyperparameters
            "batch_size_per_image": 256,
            "batch_size": 8,
            
            # Learning rate and scheduler
            "base_lr": 7.7e-3,
            "lr_scheduler": "WarmupCosineLR",
            "weight_decay": 1.2e-5,
            
            # Whether to clip gradients or not
            "clip_gradients": False,
            
            # Wether to freeze or not the backbone
            "freeze_backbone": False
        }
        
        results = model.train_model(data_dir=data_dir, output_dir=output_dir, **hyperparams)
        print("Results: ", results)
    elif task == "eval":
        assert data_dir, "Data directory must be specified for eval task (use -d <DATA_DIRECTORY>)"
        print(model.evaluate_model(data_dir=data_dir, output_dir=output_dir))
    elif task == 'infer':
        assert args.input_image, "You should include an input image for infer task (use --input_image <PATH_TO_IMAGE>)"
        input_image = args.input_image
        
        # Get the image
        image = cv2.imread(input_image)
        print(f"Image shape: {image.shape}")
        
        # Get the predictions
        predictions = model.run_inference(image)
        visualized_image = model.visualize_predictions(image, predictions)
        
        # Save image for processing
        print(f"Visualized image shape: {visualized_image.shape}")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print(f"Directory created successfully at {output_dir}")
        cv2.imwrite(f"{output_dir}/visualized_image_finetuned.png", visualized_image)
    
