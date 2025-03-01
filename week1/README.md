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
  │       └── inference_c.py  # Inference script

To execute the inference on a specific sequence, run the following command from the terminal:

```bash
python inference_c.py <numbersequence>
```
The inference will only detect objects of interest: person and car. This is specified by the **classes** parameter in the code, which is set to detect only class IDs corresponding to **person** (ID 0) and **car** (ID 2). 

Here are some example images from the sequence 0014 that show the output after running the inference:

| ![000037](https://github.com/user-attachments/assets/24638324-6819-4d08-914f-24484e012a99) | ![000101](https://github.com/user-attachments/assets/2111ff43-98a5-43b6-9a85-e3d8322a76e8) |
|---------------------------------------|---------------------------------------|
| 000037.png          | 000101.png       |



### Task D: Evaluate pre-trained Faster R-CNN, DeTR, and YOLOv11n on KITTI-MOTS dataset

### Task E: Fine-tune Faster R-CNN, DeTR, and YOLO on KITTI-MOTS (Similar Domain)

### Task F: Fine-tune Faster R-CNN on Different Dataset (Domain shift)

### Task G: Analyse the difference among the different object detector methods models
