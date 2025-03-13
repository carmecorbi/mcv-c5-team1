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

### YOLO11n-seg 
YOLO11n-seg is a lightweight object detection and segmentation model based on the YOLO family. In this project, we use the Ultralytics implementation to evaluate performance on the KITTI-MOTS dataset.

#### Inference
To execute the inference on a specific sequence, run the following command:

```bash
python inference.py --seq <sequence_number> --out <output_directory>
```
The inference will only segment objects of interest: person and car. This is specified by the classes parameter in the code, which is set to segment only class IDs corresponding to **person** (ID 0) and **car** (ID 2).

Here are some examples from the sequeence 0018 that show the output after running the inference:

| 000014.png | 000201.png |
|---------------------------------------|---------------------------------------|
|  ![000014](https://github.com/user-attachments/assets/06ed630a-3d56-4e2a-ab9c-d70cda9d32de) |![000201](https://github.com/user-attachments/assets/c73dce54-157f-40bb-b574-a2b7f791c367) |

#### Conversion Process
To convert object detection annotations (last week) into instance segmentation format, we leveraged the **Segment Anything Model (SAM)** from Ultralytics. The original dataset contained bounding box annotations in YOLO format. Using the provided Ultralytics conversion script, we applied SAM to generate segmentation masks, and saved them in a new directory. This approach efficiently transformed detection labels into polygonal-based segmentation annotations, enabling instance segmentation evaluation while preserving the YOLO format. The implemenntation was inspired by this article[!](https://medium.com/@dhanyasethumadhavan/yolo-v8-how-to-convert-custom-object-detection-datasets-to-segmentation-datasets-using-sam-models-d80eb9abea61)




## Task B: Fine-tune Mask R-CNN, Mask2Former, and YOLO11n-seg on KITTI-MOTS (Similar Domain)

## Task C: Fine-tune Mask2Former on Different Dataset (Domain shift)

## Task D: Analyse the difference among the different object detector models

## Optional Task: Get Familiar with Segment Anything Model (SAM)


