<h2 align="center">WEEK 1: Tasks</h2>

## Table of Contents

- [Project Structure W1](#project-structure-w1)
- [Task C: Run inference with pre-trained Faster R-CNN, DeTR, and YOLOv11n on KITTI-MOTS dataset](#task-c-run-inference-with-pre-trained-faster-r-cnn-detr-and-yolov11n-on-kitti-mots-dataset)
- [Task D:  Evaluate pre-trained Faster R-CNN, DeTR and YOLOv11n on KITTI-MOTS dataset.]
- [Task E:  Fine-tune Faster R-CNN, DeTR and YOLO on KITTI-MOTS (Similar Domain)]
- [Task F: Fine-tune Faster R-CNN on Different Dataset (Domain shift)]
- [Task G: Analyse the difference among the different object detector methods models]


### Project Structure W1

week1/
    ├── checkpoints/
    ├── src/
    │   ├── detectron/
    │   ├── ultralytics/
    │   ├── ...

### Task C: Run inference with pre-trained Faster R-CNN, DeTR and YOLOv11n on KITTI-MOTS dataset.

#### Dataset description
The KITTI Multi-Object Tracking and Segmentation (KITTI-MOTS) dataset is an extension of the KITTI dataset, designed for multi-object tracking and instance segmentation in autonomous driving scenarios. It contains annotated sequences of real-world traffic scenes, focusing on pedestrians and vehicles.

The dataset is divided into three subsets:

| Subset      | Number of Sequences |
|------------|--------------------|
| **Train**  | 12                |
| **Validation** | 9 |
| **Test**   | 29                 |
