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

## Change the text representation level among wordpiece-level and word-level. 

## Change the encoder of the baseline model: VGG-16

## Change the decoder of the baseline model: LSTM



