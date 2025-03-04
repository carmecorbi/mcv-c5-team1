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

### Task F: Fine-tune Faster R-CNN on Different Dataset (Domain shift)

### Task G: Analyse the difference among the different object detector methods models
