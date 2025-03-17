import json
import torch
import numpy as np
import pycocotools.mask as mask_utils
import argparse

from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--ground_truth", help="Path to the JSON GT annotations in COCO Format", default="/data/cvcqml/common/ycordero/mcv-c5-team1/week2/detectron/output/train_no_augment_fully_unfrozen/inference/kitti-mots_val_coco_format.json")
    parser.add_argument("-ck", "--checkpoint_path_or_name", help="Path to the checkpoint file or name of the checkpoint in HF Hub.", default="yeray142/finetune-instance-segmentation-ade20k-mini-mask2former_augmentation")
    parser.add_argument("--token", help="Token for Hugging Face Hub", required=False, default=None)
    args = parser.parse_args()
    
    # Paths (adjust these to your dataset)
    gt_annotation_file = args.ground_truth

    # Set up device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #checkpoint = "facebook/mask2former-swin-tiny-coco-instance"
    checkpoint = args.checkpoint_path_or_name

    # Load the processor and model
    processor = AutoImageProcessor.from_pretrained(checkpoint, token=args.token, force_download=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, token=args.token, force_download=True)
    model.to(device)
    model.eval()  # set model to evaluation mode

    # Load COCO ground truth
    coco_gt = COCO(gt_annotation_file)

    predictions = []

    # Iterate over all images in the COCO ground truth
    img_ids = coco_gt.getImgIds()
    for image_info in tqdm(coco_gt.loadImgs(img_ids), desc="Evaluating"):
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_path = file_name
        
        # Load image in RGB
        image = Image.open(image_path).convert("RGB")
        
        # Process image and run inference
        inputs = processor(images=image, return_tensors="pt")
        
        # Move tensors to the device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process to obtain instance segmentation results.
        results = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]], threshold=0.9)[0]
        seg_map = results["segmentation"].numpy()
        for segment in results["segments_info"]:
            seg_id = segment["id"]
            category_id = segment["label_id"]
            score = segment["score"]
            
            # Create a binary mask for this instance
            binary_mask = (seg_map == seg_id).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue
            
            # Encode the mask using pycocotools (ensure Fortran order)
            encoded_mask = mask_utils.encode(np.asfortranarray(binary_mask))
            
            # COCO expects the 'counts' field as a string, not bytes
            encoded_mask["counts"] = encoded_mask["counts"].decode("utf-8")
            
            if model.config.id2label[category_id] not in ["car", "person"]:
                continue
            
            model_person_category = model.config.label2id["person"]
            model_car_category = model.config.label2id["car"]

            category_map = {
                model_car_category: 0,
                model_person_category: 1
            }

            prediction = {
                "image_id": image_id,
                "category_id": category_map[category_id],
                "segmentation": encoded_mask,
                "score": score
            }
            predictions.append(prediction)

    # Save predictions to a JSON file
    pred_file = "predictions.json"
    with open(pred_file, "w") as f:
        json.dump(predictions, f)

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()