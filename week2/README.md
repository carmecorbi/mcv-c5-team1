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

### Mask R-CNN
#### Running Inference
To perform inference on a single image using the Mask R-CNN model, use the following command:

```bash
python -m week2.src.main.py -t infer -i /path/to/your/image.jpg -o /path/to/output/directory
```

##### Required Arguments
- `-t <infer, train, eval>`: Specifies task (for inference use `infer`)
- `-i`: Path to input image

##### Optional Arguments
- `-o`: Output directory for saving visualization (default: None)
- `-c`: Path to model config file (default: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
- `-w`: Path to weights file (default: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
- `-s`: Detection confidence threshold (default: 0.5)
- `--num_workers`: Number of workers for data loading (default: 4)

The script will process the image and save the visualization with detected objects in the specified output directory as `visualized_image_finetuned.png`.

Below you can find some examples of applying Mask R-CNN to some images:
| Example 1 | Example 2 |
|---------------------------------------|---------------------------------------|
| ![image](https://github.com/user-attachments/assets/618aa76e-e642-4c14-9d89-516f66d67805) | ![image](https://github.com/user-attachments/assets/bf45bd15-e45c-4a7a-86c9-f464f5694902) |

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

### Mask R-CNN
#### Usage
To fine-tune the Mask R-CNN model on your custom dataset, use the following command:

```bash
python -m src.week2.main.py -t train -d /path/to/dataset -o /path/to/output_directory -c /path/to/model_config --num_workers 4
```

#### Process Overview
The training process begins by registering the KITTI-MOTS dataset splits (train and validation) in the Detectron2 catalog. The model configuration is then automatically set up with optimized hyperparameters for fine-tuning. The training runs for 3000 iterations by default, after which a final evaluation is performed on the validation set using the COCO evaluation metrics. All training logs, checkpoints, and the final model weights are saved in the specified output directory.

#### Optional Parameters
- `--num_workers`: Number of data loading workers (default: 4)
- `-c`: Path to custom model configuration
- `-w`: Path to initial weights file
- `-s`: Detection confidence threshold (default: 0.5)

The trained model can then be used for inference or evaluation tasks using the same script with different task arguments (`-t infer` or `-t eval`). 

### Mask2Former

For fine-tuning Mask2Former, we follow the script provided by [Hugging Face Transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation).  

We experiment with two training strategies:  

### 1️. Fully Unfrozen Model  
In this setup, we fine-tune all layers of the model.  

```bash
python run_instance_segmentation.py \
    --model_name_or_path facebook/mask2former-swin-tiny-coco-instance \
    --output_dir finetune-instance-segmentation-ade20k-mini-mask2former_augmentation \
    --dataset_name yeray142/kitti-mots-instance \
    --do_reduce_labels \
    --image_height 621 \
    --image_width 187 \
    --do_train \
    --fp16 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers \
    --dataloader_prefetch_factor 4 \
    --do_eval \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub
```

### 2. Backbone Frozen  
In this setup, we freeze the backbone (Swin Transformer) and fine-tune only the higher-level layers.  

```bash
python run_instance_segmentation_bf.py \
    --model_name_or_path facebook/mask2former-swin-tiny-coco-instance \
    --output_dir finetune-instance-segmentation-ade20k-mini-mask2former_backbone_frozen_1 \
    --dataset_name yeray142/kitti-mots-instance \
    --do_reduce_labels \
    --image_height 621 \
    --image_width 187 \
    --do_train \
    --fp16 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers \
    --dataloader_prefetch_factor 4 \
    --do_eval \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub
```

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

### Strawberry Disease: Description

The Strawberry Disease Detection dataset is designed to support DL models addressing instance segmentation tasks to accurately perform strawberries disease diagnosis under complex environmental conditions. This dataset includes a large amount of images collected from multiple greenhouses in South Korea that capture the natural variability of agricultural environments (e.g., background variability, natural illumination conditions). It enables the detection and instance segmentation of the following 7 classes of strawberry diseases: [Angular Leafspot, Anthracnose Fruit Rot, Blossom Blight, Gray Mold, Leaf Spot, Powdery Mildew Fruit, Powdery Mildew Leaf]. 

The dataset is divided intro three subsets (annotations fro train and validation sets are provided in 'Labelme' format):

| Subset    | Number of Images | 
|----------------------|-----------|
| **Train**   | 1450    | 
| **Validation**    | 307     | 
| **Test**    | 743     |

The details of the dataset are documented in its official paper, which can be found here: [An Instance Segmentation Model for Strawberry Diseases Based on Mask R-CNN](https://www.mdpi.com/1424-8220/21/19/6565). The dataset can be downloaded from Kaggle [Strawberry Disease Detection Dataset](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset).

### Fine-tuning Strategies

We fine-tuned 2 models: `Mask R-CNN` and `Mask2Former`. Initially, we had to fine-tune `Mask2Former` on the new dataset (domain shift). However, we had some problems with this model during inference and fine-tuning (similar domain) tasks. For this reason, we decided to also fine-tune `Mask R-CNN` to make sure we could report results on the domain shift task. In both cases, the two fine-tuning strategies explored include:
1. Backbone Frozen
2. Fully Unfrozen

Since the downloaded dataset provides annotations as .json files in Labelme format, we have had to convert them to the correct format for each case:
- **Mask R-CNN:**  Labels are expected to be .json files providing image metadata and segmentation information in COCO format (`get_ds_dicts(dataset_path)` in `mask_rcnn_finetune.py`).
- **Mask2Former:** Annotations are expected to be 3-channel masks images (1st channel: class_id mask, 2nd channel: instance_id mask, 3rd channel: all 0s). We used the `utils.shapes_to_label(img_shape, shapes, label_name_to_value)` function from [labelme GitHub Repository](https://github.com/wkentaro/labelme/blob/main/labelme/utils/shape.py) to get the class and instances GT masks from the .json annotations. We get 1 `.png` mask for each image in the dataset (uploaded train/val sets at HuggingFace (jsalavedra/strawberry_disease) following the HuggingFace scripts provided in this [GitHub](https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation). 


### Usage

#### Mask R-CNN

We started using Optuna to find the optimum set of hyperparameters. To perform hyperparameter optimization for Mask R-CNN training on the Strawberry Disease dataset, use the following command:
```bash
python3 run_optuna.py -d /path/to/data -c COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml -w COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml -s score_threshold -o /path/to/output_directory --num_workers num_workers --n_trials number_of_trials
```
Each trial was evaluated based on the mAP at IoU=0.5, aiming to maximize performance. The optimization was performed over 15 trials. The optimal hyperparameter values for each strategy can be found in the slides linked at the beginning of this README.md file.
To set whether the backbone is frozen or trainable, you can modify the value assigned to `self.cfg.MODEL.BACKBONE.FREEZE_AT = value` in `mask_rcnn_optuna.py` (Strategy A: backbone frozen with value=5, Strategy B: backbone unfrozen with value=0).

Once the hyperparameters are set (values must be set on the `mask_rcnn_finetune.py` script at `train_model()` function), the Mask R-CNN model can be fine-tuned on your custom dataset using the following command (Uncomment tha code lines in main() that are marked for training):
```bash
python3 mask_rcnn_finetune.py 
```
The same command can be used to perform inference on the trained model given an image (Uncomment the code lines in main() that are marked for inference).

#### Mask2Former

Due to time limitations, we were unable to optimize hyperparameters using Optuna. We trained the Mask2Former model on the Strawberry Disease Dataset with the two fine-tuning strategies defined above. To train, use the following command:
```bash
python run_instance_segmentation.py \
    --model_name_or_path facebook/mask2former-swin-tiny-coco-instance \
    --output_dir finetune-instance-segmentation-mini-mask2former_augmentation_default_backboneFrozen \
    --dataset_name jsalavedra/strawberry_disease \
    --do_reduce_labels \
    --image_height 419 \
    --image_width 419 \
    --do_train \
    --fp16 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers \
    --dataloader_prefetch_factor 4 \
    --do_eval \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub
```
In order to set whether the backbone (Swin Transformer) is freezed or unfreezed, uncomment or comment, repsectively the following lines in the `run_instance_segmentation.py` script:
```
for param in model.model.pixel_level_module.encoder.parameters():
        param.requires_grad = False
```

To perform inference on the trained model given an image, use the following command (set the `image_path` in `main()` and the `output_image_name` in `run_inference()` function):
```bash
python3 Mask2Former_inference_seq_ft.py
```

## Task D: Analyse the difference among the different object detector models

## Optional Task: Get Familiar with Segment Anything Model (SAM)

### Finetuning the Decoder of SAM

The Segment Anything Model (SAM) decoder can be fine-tuned for domain-specific segmentation tasks while keeping the image encoder frozen. This process involves adapting the lightweight mask decoder to new domains without retraining the computationally expensive encoder, significantly reducing computational requirements. The fine-tuning pipeline typically includes preparing a custom dataset, selecting appropriate hyperparameters, and implementing training loops that focus on optimizing the decoder weights for specialized segmentation tasks. 

For a complete implementation walkthrough, refer to the `fine-tuning.ipynb` notebook in the `sam/` folder, which provides detailed code examples and explanations of the entire fine-tuning process.

### Inference with Point and Bounding Box Prompting

SAM's interactive segmentation capabilities allow for inference using sparse prompts such as points and bounding boxes to generate precise masks. Points can be provided as either foreground or background indicators, guiding the model to segment specific objects, while bounding boxes help constrain the region of interest for more accurate segmentation results. The model processes these prompts alongside the encoded image features to predict high-quality segmentation masks in real-time, enabling interactive applications. 

The complete inference pipeline with various prompting strategies is demonstrated in the `inference.ipynb` notebook in the `sam/` folder, which covers loading pretrained models, processing prompts, and visualizing the resulting segmentation masks. 


