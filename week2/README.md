<h2 align="center">WEEK 2: Tasks</h2>

## Table of Contents

- [Project Structure W1](#project-structure-w1)
- [Task A: Run inference and evaluate with pre-trained Faster Mask R-CNN, Mask2Former, and YOLO11n-seg on KITTI-MOTS dataset](#task-a-run-inference-and-evaluate-with-pre-trained-faster-mask-r-cnn-mask2former-and-yolo11n-seg-on-kitti-mots-dataset)
- [Task B: Fine-tune Mask R-CNN, Mask2Former, and YOLO11n-seg on KITTI-MOTS (Similar Domain)](#task-b-fine-tune-mask-r-cnn-mask2former-and-yolo11n-seg-on-kitti-mots-similar-domain)
- [Task C: Fine-tune Mask2Former on Different Dataset (Domain shift)](#task-c-fine-tune-mask2former-on-different-dataset-domain-shift)
- [Task D: Analyse the difference among the different object detector models](#task-d-analyse-the-difference-among-the-different-object-detector-models)
- [Optional Task: Get Familiar with Segment Anything Model (SAM)](#optional-task-get-familiar-with-segment-anything-model-sam)

## Project Structure W2

## Task A: Run inference and evaluate with pre-trained Faster Mask R-CNN, Mask2Former, and YOLO11n-seg on KITTI-MOTS dataset

### Mask2Former
Mask2Former is a state-of-the-art universal segmentation model capable of instance, panoptic, and semantic segmentation. It is based on a transformer-based architecture, enabling robust and accurate object segmentation. In this project, we use the pre-trained `facebook/mask2former-swin-tiny-coco-instance` model from Hugging Face's transformers library to perform instance segmentation on the KITTI-MOTS dataset.

#### Inference
The inference pipeline utilizes the Hugging Face `AutoImageProcessor` to preprocess images before passing them to the `Mask2FormerForUniversalSegmentation` model. The model generates segmentation masks, which are post-processed to retain only two classes: person (ID 0) and car (ID 2). Additionally, a **threshold of 0.5** is applied to filter detections based on their confidence scores. The masks are visualized with fixed colors (blue for persons and pink for cars) and overlaid on the original KITTI-MOTS images.

To run inference on a specific sequence, use the following command:

```bash
python3 Mask2Former_inference_seq.py <sequence_id>
```

Below are examples from sequence **0016**, showing the output after inference:

| 000014.png | 000201.png |
|---------------------------------------|---------------------------------------|
| ![000014](https://github.com/user-attachments/assets/7b602490-f220-487c-bd9a-9ddfea511cbd) | ![000201](https://github.com/user-attachments/assets/f3c52b95-4d54-4946-9e1e-d1282914924f) |

#### Evaluation
To evaluate the pre-train model, simply run the following command:
```bash
python3 Mask2Former_eval.py
```

The results are:
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.047
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.078
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.048
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.008
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.061
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.095
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.011
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.047
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.060
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.028
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.075
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.108
```

### YOLO11n-seg 
YOLO11n-seg is a lightweight object detection and segmentation model based on the YOLO family. In this project, we use the Ultralytics implementation to evaluate performance on the KITTI-MOTS dataset.

#### Inference
To execute the inference on a specific sequence, run the following command:

```bash
python ultralytics/inference.py --seq <sequence_number> --out <output_directory>
```
The inference will only segment objects of interest: person and car. This is specified by the classes parameter in the code, which is set to segment only class IDs corresponding to **person** (ID 0) and **car** (ID 2).

Here are some examples from the sequence 0016 that show the output after running the inference:

| 000014.png | 000201.png |
|---------------------------------------|---------------------------------------|
|  ![000014](https://github.com/user-attachments/assets/06ed630a-3d56-4e2a-ab9c-d70cda9d32de) |![000201](https://github.com/user-attachments/assets/c73dce54-157f-40bb-b574-a2b7f791c367) |

#### Conversion Process
To convert object detection annotations (last week) into instance segmentation format, we leveraged the **Segment Anything Model (SAM)** from Ultralytics. The original dataset contained bounding box annotations in YOLO format. Using the provided Ultralytics conversion script, we applied SAM to generate segmentation masks, and saved them in a new directory. This approach efficiently transformed detection labels into polygonal-based segmentation annotations, enabling instance segmentation evaluation while preserving the YOLO format. The implemenntation was inspired by this [article](https://medium.com/@dhanyasethumadhavan/yolo-v8-how-to-convert-custom-object-detection-datasets-to-segmentation-datasets-using-sam-models-d80eb9abea61). 

To run the conversion, execute the following command:
```bash
python ultralytics/bbtoseg.py
```
#### Evaluation
To evaluate the pre-train model, simply run the following command:
```bash
python ultralytics/evaluation.py --m <model_path>
```
| Metric                                             | Value                                      |
|----------------------------------------------------|--------------------------------------------|
| **Average precision**                              | 0.30 (person), 0.47 (car)           |
| **Mean average precision at IoU=0.50**            | 0.69                              |
| **Mean average precision at IoU=0.75**            | 0.38                                   |
| **Precision**                                      | 0.61 (person), 0.87 (car)           |
| **Recall**                                         | 0.65 (person), 0.63 (car)           |


## Task B: Fine-tune Mask R-CNN, Mask2Former, and YOLO11n-seg on KITTI-MOTS (Similar Domain)

### YOLO11n-seg 
We fine-tune YOLO11n-seg on the KITT-MOTS dataset using two different fine-tuning strategies:
1. Fully Unfrozen Model
2. Backbone Frozen

The goal is to optimize the model's performance by tuning hyperparameters using Optuna, maximizing the mAP at IoU=0.5.

We conducted hyperparameter optimization considering the following parameters:
- **Dropout:** [0.0, 0.5] (regularization technique to prevent overfitting)
- **Weight Decay:** [0.0, 0.01] (L2 regularization to improve generalization)
- **Optimizer:** [SGD, Adam, AdamW] (different optimization algorithms)
- **Rotation Degrees:** [-5, 5] (image rotation augmentation)
- **Hsv Hue and Saturation**: [0,0.3] , [0,0.5] (adjusting the Hue and Saturation of images for color space augmentation)
- **IoU Threshold (NMS)**: [0.5,0.6,0.7] (Intersection over Union threshold used in Non-Maximum Suppression to filter overalpping bounding boxes)

Training Parameters
- **Dataset:** KITTI-MOTS 
- **Epochs:** 50
- **Device:** GPU (CUDA)
- **Early Stopping Patience:** 5
- **Classes Trained:** 0 (Car), 2 (Pedestrian)

Each trial was evaluated based on the mAP at IoU=0.5, aiming to maximize performance. Total trials: 20.

For the first strategy, run the following Python script:
```bash
python ultralytics/optuna_unfrozen.py
```

For the second strategy, run the following Python script:
```bash
python ultralytics/optuna_backbone.py
```

#### Fine-tuning Results

| Finetune Strategy    | Optimizer | Regularization                    | Augmentation                                  | mAP@0.5 | mAP@0.75 | AP (class)                           |
|----------------------|-----------|------------------------------------|-----------------------------------------------|---------|----------|--------------------------------------|
| **Backbone Frozen**   | AdamW     | D(0.08), L2(7e-3)                 | Degrees(4.93), HSV_h(0.27), HSV_s(0.08)          | 0.76    | 0.53     | 0.39 (pedestrian), 0.58 (car)         |
| **Fully Unfrozen**    | SGD     | D(6e-3), L2(4.5e-3)                 | Degrees(3.55), HSV_h(0.16), HSV_s(0.005)          | **0.77**   | **0.54**    | **0.41 (pedestrian), 0.56 (car)**      |

## Task C: Fine-tune Mask2Former on Different Dataset (Domain shift)

## Task D: Analyse the difference among the different object detector models

## Optional Task: Get Familiar with Segment Anything Model (SAM)


