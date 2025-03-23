<h2 align="center">WEEK 3: Tasks</h2>

## Table of Contents

- [Train and evaluate the baseline model with the food dataset](#train-and-evaluate-the-baseline-model-with-the-food-dataset)
- [Change the text representation level among wordpiece-level and word-level](#change-the-text-representation-level-among-wordpiece-level-and-word-level)
- [Change the encoder of the baseline model to VGG-16](#change-the-encoder-of-the-baseline-model-to-vgg-16)
- [Change the decoder of the baseline model to LSTM](#change-the-decoder-of-the-baseline-model-to-lstm)


## Train and evaluate the baseline model with the food dataset

### Dataset overview
The dataset consists of **13,466 images** of various food dishes. It has been split into training, validation, and test sets as follows:

- **Training Set (80%)**: 10,772 images
- **Validation Set (10%)**: 1,347 images
- **Test Set (10%)**: 1,347 images

The objective of this dataset is to **predict food dish titles** based on images of dishes. The dataset is available from Kaggle and was originally expected to contain 13,582 images and their corresponding annotations (5 columns). However, after cleaning the data, **13,501 images** were actually available for further use.

**Download the dataset**: Run the following command to download the dataset.
   ```bash
   python src/dataset/download.py
```
### Dataset Cleaning Process
Before proceeding with training the model, the dataset underwent a cleaning process:

1. **Removed Rows with Empty or Missing Data:**
   - Rows with empty strings in the `Image_Name` or `Title` columns were removed.
   - Rows with `NaN` values in the `Image_Name` or `Title` columns were removed.
   
2. **Filtered Out Invalid Image Names:**
   - Rows with the value `#NAME?` in the `Image_Name` column were removed.

After cleaning, the dataset was reduced to **13,446 images**.

**Clean and split the dataset** Run the following command:
```bash
python src/dataset/prepare_data.py
```
### Baseline Model Architecture

In the baseline model, the task is to predict the food dish title from an image. The model architecture consists of two major parts:

1. **ResNet-18 (Encoder)**: A pre-trained ResNet-18 model, widely used for image classification tasks, is used here as the encoder. 
   
2. **GRU (Decoder)**: The GRU network is used as the decoder, which generates the sequence of characters (the food dish title) based on the features provided by the ResNet-18 encoder. 

The process follows the character-level representation approach, meaning that we treat each word in the dish title as a sequence of characters, not words. 

## Change the text representation level among wordpiece-level and word-level. 

## Change the encoder of the baseline model: VGG-16

## Change the decoder of the baseline model: LSTM



