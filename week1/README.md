<h2 align="center">WEEK 1: Tasks</h2>

## Table of Contents

- [Project Structure W1](#project-structure-w1)
- [Task C: Run inference with pre-trained Faster R-CNN, DeTR, and YOLOv11n on KITTI-MOTS dataset](#task-c-run-inference-with-pre-trained-faster-r-cnn-detr-and-yolov11n-on-kitti-mots-dataset)
- [Task D: Evaluate pre-trained Faster R-CNN, DeTR, and YOLOv11n on KITTI-MOTS dataset](#task-d-evaluate-pre-trained-faster-r-cnn-detr-and-yolov11n-on-kitti-mots-dataset)
- [Task E: Fine-tune Faster R-CNN, DeTR, and YOLO on KITTI-MOTS (Similar Domain)](#task-e-fine-tune-faster-r-cnn-detr-and-yolo-on-kitti-mots-similar-domain)
- [Task F: Fine-tune Faster R-CNN on Different Dataset (Domain shift)](#task-f-fine-tune-faster-r-cnn-on-different-dataset-domain-shift)
- [Task G: Analyse the difference among the different object detector methods models](#task-g-analyse-the-difference-among-the-different-object-detector-methods-models)


### Project Structure W1


### Task C: Run inference with pre-trained Faster R-CNN, DeTR and YOLOv11n on KITTI-MOTS dataset.

#### Dataset description
The KITTI Multi-Object Tracking and Segmentation (KITTI-MOTS) dataset is an extension of the KITTI dataset, designed for multi-object tracking and instance segmentation in autonomous driving scenarios. It contains annotated sequences of real-world traffic scenes, focusing on pedestrians and vehicles.

The dataset is divided into three subsets:

| Subset      | Number of Sequences |
|------------|--------------------|
| **Train**  | 12                |
| **Validation** | 9 |
| **Test**   | 29                 |

The **Validation** set was created by taking 9 sequences from the original training set, which had 21 sequences in total. This approach is documented in the official paper of the dataset, which can be found here: [KITTI-MOTS Paper](https://arxiv.org/pdf/1902.03604).

#### Faster R-CNN
##### Running Inference
To perform inference on a single image using the Faster R-CNN model, use the following command:

```bash
python main.py -t infer -i /path/to/your/image.jpg -o /path/to/output/directory
```

###### Required Arguments
- `-t <infer, train, eval>`: Specifies task (for inference use `infer`)
- `-i`: Path to input image

###### Optional Arguments
- `-o`: Output directory for saving visualization (default: None)
- `-c`: Path to model config file (default: COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml)
- `-w`: Path to weights file (default: COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml)
- `-s`: Detection confidence threshold (default: 0.5)
- `--num_workers`: Number of workers for data loading (default: 4)

The script will process the image and save the visualization with detected objects in the specified output directory as `visualized_image_finetuned.png`.

##### Sample results

| 0016/000109.png | 0008/000168.png |
|---------------------------------------|---------------------------------------|
| ![image](https://github.com/user-attachments/assets/ac17b5f4-7193-4047-9d52-70592624d603) | ![image](https://github.com/user-attachments/assets/a70e6f52-748b-4dff-b710-bb351d5d6191) |



#### DeTR
DeTR (DEtection TRansformer) is an object detection model based on Transformers, developed by Facebook AI. Unlike traditional methods such as Faster R-CNN and YOLO, DeTR replaces conventional detection components with an attention-based architecture, allowing it to capture long-range relationships within an image without the need for predefined anchors.

For this task, we used the **Hugging Face Transformers** implementation with the pre-trained model **facebook/detr-resnet-50**. This model takes an input image, processes it using `DetrImageProcessor`, and generates object predictions with `DetrForObjectDetection`.

DeTR directly predicts bounding boxes and object classes in a single pass, eliminating the need for steps such as region proposal generation or Non-Maximum Suppression (NMS). However, its inference is generally slower compared to YOLO due to its Transformer-based architecture.

The code assumes the following directory structure: 

    mcv-c5-team1/
    │── data/
    │   ├── instances_txt/  # Ground truth annotations
    │   │   ├── 0000.txt
    │   │   └── ...
    │   ├── training/
    │   │   ├── val/  # Folder containing input images
    │   │   │   ├── 0002/
    │   │   │   ├── 0006/
    │   │   │   └── ...
    │── week1/
    │   ├── src/
    │   │   ├── huggingface/
    │   │   │   ├── config/
    │   │   │   │   ├── gt_annotations/  # Ground truth annotations converted
    │   │   │   ├── results/
    │   │   │   │   ├── results_inference/  # Folder for saving visualized images
    │   │   │   │   ├── results_txt/  # Folder for saving inference results in txt format
    │   │   │   ├── detr.py  # DeTR model class
    │   │   │   ├── inference_detr_txt_files.py  # Inference script

To run the inference en the validation dataset, use the following command:

```bash
python3 inference_detr_txt_files.py
```

The inference is limited to detecting **person** and **car**, as defined in the **CLASS_MAP** parameter. These class IDs follow the **COCO dataset** format, where **person** is **1** and **car** is **3**, which is the standard used by DeTR.

The script will:

- Save the visualized images with bounding boxes in the `results_inference` folder.
- Store the detection results in a `.txt` file inside the `results_txt` folder.

Each detection in the `.txt` file follows this format:

```bash
frame_id, -1, class_id, x_min, y_min, x_max, y_max, confidence_score
```

Where:
- **frame_id**: Index of the image in the sequence.
- **-1**: Placeholder for `object_id` (not used in this case).
- **class_id**: Object class (1 = person, 3 = car).
- **x_min, y_min, x_max, y_max**: Bounding box coordinates in pixels.
- **confidence_score**: Model confidence score.

Here are some example images from the sequence 0014 that show the output after running the inference:

| 000037.png | 000101.png |
|---------------------------------------|---------------------------------------|
| ![000037](https://github.com/user-attachments/assets/b19a19f0-4ca3-496f-b5df-6bbf96f49f85) | ![000101](https://github.com/user-attachments/assets/68864b3c-4ab1-4261-be44-472bc91af3d7)|


#### YOLOv11n
YOLOv11n (You Only Look Once) is a real-time object detection model that is part of the YOLO family, known for its speed and efficiency in detecting objects. For this task, we used the **Ultralytics implementation** of YOLOv11n, which is optimized to provide high accuracy and fast inference times. YOLOv11n works by dividing the input image into a grid and predicting bounding boxes and class probabilities directly from each grid cell. 

The code assumes the following directory structure: 
    
    week1/
    │── checkpoints/
    │   └── yolo/
    │       └── yolo11n.pt  # Pre-trained model
    │── src/
    │   └── ultralytics/
    │       ├── data/
    │       │   ├── images/val/  # Folder containing input images
    │       │   │   ├── 0002/
    │       │   │   ├── 0006/
    │       │   │   └── ...
    │       ├── results_inference/  # Folder for saving results
    │       └── inference.py  # Inference script

To execute the inference on a specific sequence, run the following command from the terminal:

```bash
python inference.py --seq <sequence_number> --out <output_directory>
```
The inference will only detect objects of interest: person and car. This is specified by the **classes** parameter in the code, which is set to detect only class IDs corresponding to **person** (ID 0) and **car** (ID 2). 

Here are some example images from the sequence 0014 that show the output after running the inference:

| 000037.png          | 000101.png       |
|---------------------------------------|---------------------------------------|
| ![000037](https://github.com/user-attachments/assets/24638324-6819-4d08-914f-24484e012a99) | ![000101](https://github.com/user-attachments/assets/2111ff43-98a5-43b6-9a85-e3d8322a76e8) |

### Task D: Evaluate pre-trained Faster R-CNN, DeTR, and YOLOv11n on KITTI-MOTS dataset

#### Faster R-CNN
To evaluate the model performance on a dataset, run the script in evaluation mode with the dataset directory provided:

```bash
python main.py -t eval -d /path/to/your/dataset -o /path/to/output/directory
```

This command uses the evaluation task (`-t eval`) to load the specified dataset directory (`-d`) and generate evaluation results, which will be printed to the console and optionally saved in the provided output directory (`-o`). The evaluation leverages the COCO evaluation tool integrated with detectron2 to assess performance on the KITTI-MOTS dataset.

##### Evaluation Process

The evaluation process follows these steps:

1. **Dataset Registration:**  
   The custom dataset is registered under a configurable name (default "kitti-mots"). Depending on the model configuration (`self.is_coco`), the dataset is set up either with custom or COCO class IDs.

2. **Configuration Update:**  
   The number of classes is set in the model configuration to match the dataset. For COCO evaluations, the appropriate class names are provided.

3. **Predictor Initialization:**  
   A predictor instance is created using the updated configuration, which prepares the model for inference.

4. **Evaluation Setup and Execution:**  
   With a COCO evaluator and a detection test loader built for the registered dataset, the model runs inference on the test dataset. The results are then aggregated and returned in a dictionary, containing evaluation metrics computed by the COCO evaluation tool.

##### Evaluation Results

The table below summarizes the evaluation results for the Faster R-CNN model (Detectron2) on the validation set:

| Metric                                                                 | Value  |
|------------------------------------------------------------------------|--------|
| Average Precision (AP) @[ IoU=0.50:0.95, area=all, maxDets=100 ]         | 0.570  |
| Average Precision (AP) @[ IoU=0.50, area=all, maxDets=100 ]              | 0.799  |
| Average Precision (AP) @[ IoU=0.75, area=all, maxDets=100 ]              | 0.651  |
| Average Precision (AP) @[ IoU=0.50:0.95, area=small, maxDets=100 ]       | 0.300  |
| Average Precision (AP) @[ IoU=0.50:0.95, area=medium, maxDets=100 ]      | 0.620  |
| Average Precision (AP) @[ IoU=0.50:0.95, area=large, maxDets=100 ]       | 0.715  |
| Average Recall (AR) @[ IoU=0.50:0.95, area=all, maxDets=1 ]              | 0.200  |
| Average Recall (AR) @[ IoU=0.50:0.95, area=all, maxDets=10 ]             | 0.660  |
| Average Recall (AR) @[ IoU=0.50:0.95, area=all, maxDets=100 ]            | 0.672  |
| Average Recall (AR) @[ IoU=0.50:0.95, area=small, maxDets=100 ]          | 0.493  |
| Average Recall (AR) @[ IoU=0.50:0.95, area=medium, maxDets=100 ]         | 0.706  |
| Average Recall (AR) @[ IoU=0.50:0.95, area=large, maxDets=100 ]          | 0.797  |

#### DeTR

##### 1. Ground Truth Conversion

Before evaluating the object detection models, we need to convert the KITTI-MOTS ground truth annotations into the same format used for storing inference results. This allows for a direct comparison between predictions and ground truth data.

To achieve this, we use the script `convert_gt_for_detr.py`, which processes the original KITTI-MOTS annotations and converts them into the following format:

```bash
frame_id, object_id, class_id, x_min, y_min, x_max, y_max, confidence_score
```

where:
- **frame_id**: Image index in the sequence.
- **object_id**: Unique object identifier.
- **class_id**: Mapped class ID (1 = person, 3 = car following COCO format).
- **x_min, y_min, x_max, y_max**: Bounding box coordinates in pixels.
- **confidence_score**: Set to 1.0 for ground truth annotations.

###### Run Ground Truth Conversion
Execute the script with:

```bash
python3 convert_gt_for_detr.py
```

This will generate ground truth annotation files in:

```
/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/config/gt_annotations
```

##### 2. Evaluation Using COCO Metrics

Once the ground truth annotations are converted, we evaluate the performance of the object detection models using COCO metrics (AP, mAP, precision, recall, etc.).

We use the script `detr_eval.py`, which:
1. Loads the ground truth annotations and detection results.
2. Converts them into COCO format.
3. Evaluates the detection performance using `pycocotools`.
4. Outputs the evaluation results.

###### Run Evaluation
Execute the script with:

```bash
python3 detr_eval.py
```

The script reads ground truth annotations from:
```
/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/config/gt_annotations
```
and detection results from:
```
/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/results/results_txt
```

###### COCO Evaluation Output
The script prints the following metrics:

```batch
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.461
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.756
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.486
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.160
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.501
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.701
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.072
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.414
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.336
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.621
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.794
```

**Where:**  
- **AP (Average Precision)** evaluates detection performance by measuring precision across different IoU (Intersection over Union) thresholds:  
  - AP is computed at multiple IoU thresholds (0.50:0.95), including specific values like **0.50 (PASCAL VOC metric)** and **0.75 (stricter matching criteria)**.  
  - It is also computed separately for objects of different sizes: **small, medium, and large**.  

- **AR (Average Recall)** measures the ability to detect objects correctly at different recall levels:  
  - AR is evaluated by limiting the maximum number of detections per image (**1, 10, or 100 detections**).  
  - It is also analyzed based on object size (**small, medium, large**) to assess performance across different scales.  

#### YOLOv11n

##### Conversion Process
In this task, the KITTI-MOTS dataset annotations were converted into the format required for YOLOv11 evaluation. The conversion process involved mapping the KITTI-MOTS class IDs to COCO class IDs, extracting bounding boxes from RLE masks, and normalizing them for YOLO annotations.

YOLO annotations follow this format:

```bash
class_id   x_center  y_center  width  height
```
Where:

- **`class_id`**: The ID of the object class. For example, `0` represents a **person** and `2` represents a **car**.
- **`x_center`**: The **normalized** x-coordinate of the center of the bounding box (relative to the image width).
- **`y_center`**: The **normalized** y-coordinate of the center of the bounding box (relative to the image height).
- **`width`**: The **normalized** width of the bounding box (relative to the image width).
- **`height`**: The **normalized** height of the bounding box (relative to the image height).

Keys steps in the conversion process:

1. **Class Mapping**: 
   - KITTI class 1 (car) was mapped to class 2 (COCO car).
   - KITTI class 2 (pedestrian) was mapped to class 0 (COCO person).

2. **Bounding Box Calculation**: 
   - RLE segmentation masks were used to generate bounding boxes.
   - Bounding boxes were normalized to YOLO format: `class_id x_center y_center width height`.

3. **Output**: 
   - YOLO annotations were saved in `.txt` files for each frame, with one file per image in the sequence.
   - Annotations are saved in the `val/labels` directory under each sequence folder.

To perform this conversion, run the following command:

```bash
python convert_ultralytics.py
```

##### Evaluation
The evaluation process helps assess the performance of the YOLOv11 model by calculating key metrics that indicate how well the model performs in detecting objects across different classes.

#### 1. **Creating the `data.yaml` File**
The `data.yaml` file contains essential information required by the evaluation function:
- **Train and Val directories**: Paths to training and validation image datasets.
- **Class names**: Mapping of class indices to class names (e.g., `0: person`, `2: car`).

#### 2. **Model Evaluation**
The model is evaluated using the `.val()` method, which computes several important metrics:
- **Average Precision (AP)**: Measures the detection accuracy for each class.
- **Mean Average Precision (mAP)**: The average AP across all classes.
- **Precision**: The fraction of correct predictions out of all predicted instances.
- **Recall**: The fraction of correct predictions out of all actual objects in the dataset.
- **F1 Score**: The harmonic mean of precision and recall.

#### 3. **Running the Evaluation**
To evaluate the model, simply run the following command:
```bash
python evaluation.py --m <model_path>
```
| Metric                                             | Value                                      |
|----------------------------------------------------|--------------------------------------------|
| **Average precision**                              | 0.41 (person), 0.55 (car)           |
| **Average precision at IoU=0.50**                  | 0.67 (person), 0.77 (car)           |
| **F1 score**                                       | 0.67 (person), 0.72 (car)            |
| **Mean average precision**                        | 0.48                                    |
| **Mean average precision at IoU=0.50**            | 0.72                                    |
| **Mean average precision at IoU=0.75**            | 0.52                                   |
| **Mean precision**                                 | 0.78                                    |
| **Mean recall**                                    | 0.63                                    |
| **Precision**                                      | 0.69 (person), 0.87 (car)           |
| **Recall**                                         | 0.64 (person), 0.62 (car)           |


### Task E: Fine-tune Faster R-CNN, DeTR, and YOLO on KITTI-MOTS (Similar Domain)

#### Faster R-CNN
##### Usage
To fine-tune the Faster R-CNN model on your custom dataset, use the following command:

```bash
python main.py -t train -d /path/to/dataset -o /path/to/output_directory -c COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --num_workers 4
```

##### Process Overview
The training process begins by registering the KITTI-MOTS dataset splits (train and validation) in the Detectron2 catalog. The model configuration is then automatically set up with optimized hyperparameters for fine-tuning, including:

- Learning rate scheduler with warm-up period
- Batch size of 8 images per iteration
- Base learning rate of 0.001
- SGD optimizer with momentum 0.9
- Weight decay of 0.0001
- Training checkpoints saved every 1000 iterations
- Validation performed every 1000 iterations

The training runs for 3000 iterations by default, after which a final evaluation is performed on the validation set using the COCO evaluation metrics. All training logs, checkpoints, and the final model weights are saved in the specified output directory.

##### Optional Parameters
- `--num_workers`: Number of data loading workers (default: 4)
- `-c`: Path to custom model configuration
- `-w`: Path to initial weights file
- `-s`: Detection confidence threshold (default: 0.5)

The trained model can then be used for inference or evaluation tasks using the same script with different task arguments (`-t infer` or `-t eval`). 

#### YOLO11n
We fine-tune YOLOv11n on the KITT-MOTS dataset using two different fine-tuning strategies:
1. Fully Unfrozen Model
2. Backbone Frozen

The goal is to optimize the model's performance by tuning hyperparameters using Optuna, maximizing the mAP at IoU=0.5 

We conducted hyperparameter optimization considering the following parameters:
- **Mixup:** [0.0, 0.5] (data augmentation technique for blending images)
- **Dropout:** [0.0, 0.5] (regularization technique to prevent overfitting)
- **Weight Decay:** [0.0, 0.001] (L2 regularization to improve generalization)
- **Optimizer:** [SGD, Adam, AdamW] (different optimization algorithms)
- **Rotation Degrees:** [0.0, 90.0] (image rotation augmentation)
- **Scale:** [0.2, 1.0] (image scaling augmentation)
- **Batch Size:** [4,8,16,32]

Training Parameters
- **Dataset:** KITTI-MOTS 
- **Epochs:** 50
- **Device:** GPU (CUDA)
- **Early Stopping Patience:** 50 epochs
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

**Fine-tuning Results**:

| Finetune Strategy    | Optimizer | Regularization                    | Augmentation                                  | mAP@0.5 | mAP@0.75 | AP (class)                           |
|----------------------|-----------|------------------------------------|-----------------------------------------------|---------|----------|--------------------------------------|
| **Backbone Frozen**   | SGD     | D(0.14), L2(5.7e-3)                 | MP(0.35), Degrees(7.6), Scale(0.54)          | 0.84    | 0.69     | 0.5 (pedestrian), 0.71 (car)         |
| **Fully Unfrozen**    | AdamW     | D(0.17), L2(4e-4)                 | MP(5e-2), Degrees(18.53), Scale(0.82)          | 0.8    | 0.67     | 0.45 (pedestrian), 0.72 (car)        |

#### DeTR

##### Strategy A: Fully unfrozen

```bash
python3 detr_train.py
```

```bash
python3 inference_detr_txt_files_fine_tuning.py
```

```bash
python3 detr_eval_fine_tuning.py
```

##### Strategy B: Backbone frozen

```bash
python3 detr_train_backbone_frozen.py
```

```bash
python3 inference_detr_txt_files_fine_tuning_backbone_frozen.py
```

```bash
python3 detr_eval_fine_tuning_bf.py
```


### Task F: Fine-tune Faster R-CNN on Different Dataset (Domain shift)

#### GlobalWheatHeadDetection2020: Dataset description
The GlobalWheatHeadDetection2020 dataset is designed for wheat phenotyping and crop management tasks. Accurate wheat head detection is required to assess crop health based on wheat head density and size. This dataset includes a large amount of images (according to the research field) that capture the natural variability of wheat head appearance of cultivars from different continents to enable single-class object detection research on wheat heads.  

The dataset is divided into three subsets (labels for train and validation sets are provided in YOLO format):

| Subset      | Number of Images | Sets of images |
|------------|--------------------|-----------------|
| **Train**  | 2675                | arvalis_1, arvalis_2, arvalis_3, rres_1, inrae_1, usask_1 |
| **Validation** | 747 | ethz_1 |
| **Test**   | 1276                 | utokio_1, utokio_2, nau_1, uq_1|

The details of the dataset are documented in its official paper, which can be found here: [GlobalWheatHeadDetector2020 Paper](https://spj.science.org/doi/full/10.34133/2020/3521852?adobe_mc=MCMID%3D13000678418609464879081490540568399952%7CMCORGID%3D242B6472541199F70A4C98A6%2540AdobeOrg%7CTS%3D1670889600).

#### Aquarium: Dataset description
The Aquarium dataset is designed for underwater creatures detection tasks for underwater life monitoring. The small set of images provided were collected from 2 aquariums in the US and provide a significant domain shift w.r.t. COCO and ImageNet. It enables the detection of the following 7 classes of underwater species: ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'].  

The dataset is divided into three subsets (labels for train and validation sets are provided in YOLO format):

| Subset      | Number of Images |
|------------|--------------------|
| **Train**  | 448                | 
| **Validation** | 127 | ethz_1 |
| **Test**   | 63                 | 

The dataset can be found here: [Aquarium Dataset](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots).

#### Fine-tuning Strategies

We fine-tuned the `Faster R-CNN` model on our (domain shift) selected datasets. The three fine-tuning strategies explored include:
1. Fully Unfrozen Model (`--num_frozen_blocks=0`)
2. The 2 first blocks of the backbone (ResNet-50 pretrained on ImageNet) frozen (`--num_frozen_blocks=2`)
3. Backbone frozen (`--num_frozen_blocks=5`)

##### Usage
We started using Optuna to find the optimum set of hyperparameters. To perform hyperparameter optimization for Faster R-CNN training on any of the previous dataset, use the following command:

```bash
python3 run_optuna.py -d /path/to/dataset -dt DatasetName -c COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml -w /path/to/initial/weights/file -s score_threshold -o /path/to/output_directory --n_trials number_of_trials --num_frozen_blocks num_frozen_backbone_blocks
```

According to the two datasets we use, the `DatasetName` can be `GlobalWheatHead` for the GlobalWheatHeadDetector2020 dataset or `Aquarium` for the Aquarium dataset. 

Each trial was evaluated based on the mAP at IoU=0.5, aiming to maximize performance. The optimization was performed over ~3 trials due to the saturation of the cluster. The optimal hyperparameter values for each strategy (and for each dataset) can be found in the slides linked at the beggining of this README.md file.

Once the hyperparameters are set (values must be set on the faster_rcnn.py script at train() function). To fine-tune the Faster R-CNN model (`task=train`) on your custom dataset (`DatasetName`), use the following command:

```bash
python3 faster_rcnn.py -d /path/to/dataset -t train -dt DatasetName -c COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml -w /path/to/initial/weights/file -s score_threshold -o /path/to/output_directory --n_trials number_of_trials --num_workers 4 --num_frozen_blocks num_frozen_backbone_blocks
```

In order to perform inference on the trained model given an image, use the following command:

```bash
python3 faster_rcnn.py -t infer -dt DatasetName -i /path/to/chosen/image -c COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml -w /path/to/weights/file -s score_threshold -o /path/to/output_directory --n_trials number_of_trials --num_workers 4 --num_frozen_blocks num_frozen_backbone_blocks
```

### Task G: Analyse the difference among the different object detector methods models
