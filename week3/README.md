<h2 align="center">WEEK 3: Multimodal Recognition (Image Captioning)</h2>

## Table of Contents

- [Project Structure W2](#project-structure-w2)
- [Task 1: Download the dataset](#task-1-download-the-dataset-food-ingredients-and-recipes-dataset-and-split-the-data-into-80-10-10-for-training-validation-and-test-sets)
- [Task 2: Use PyTorch to write your own dataloader](#task-2-use-pytorch-to-write-your-own-dataloader-with-image-input-data-and-its-recipe-title-groundtruth)
- [Task 4: Train and Evaluate the Baseline Model with the Food Dataset](#task-4-train-and-evaluate-the-baseline-model-with-the-food-dataset)
- [Tasks 6 & 7: Modify the Baseline Model, Train and Evaluate Custom Models](#tasks-6--7-modify-the-baseline-model-train-and-evaluate-custom-models)
  - [Change the Text Representation Level among Wordpiece-Level and Word-level](#change-the-text-representation-level-among-wordpiece-level-and-word-level)
  - [Change the Encoder of the Baseline Model to VGG-16](#change-the-encoder-of-the-baseline-model-to-vgg-16)
  - [Change the Decoder of the Baseline Model To LSTM](#change-the-decoder-of-the-baseline-model-to-lstm)

## Project Structure W2

## Task 1: Download the dataset “Food Ingredients and Recipes Dataset”, and split the data into 80-10-10 for training, validation and test sets

### Dataset overview
The dataset consists of **13,466 images** of various food dishes and it is available from Kaggle.

To **download the dataset**, run the following command:
   ```bash
   python src/dataset/download.py
```

### Dataset Cleaning Process

We have split the dataset into training, validation, and test sets as follows:

- **Training Set (80%)**: 10,772 images
- **Validation Set (10%)**: 1,347 images
- **Test Set (10%)**: 1,347 images

The dataset was originally expected to contain **13,582 images** and their corresponding annotations (5 columns). However, after cleaning the data, **13,501 images** were actually available for further use.

The data cleaning process involved the following steps:

1. **Removing Rows with Empty or Missing Data:**
   - Rows with empty strings in the `Image_Name` or `Title` columns were removed.
   - Rows with `NaN` values in the `Image_Name` or `Title` columns were removed.
   
2. **Filtering Out Invalid Image Names:**
   - Rows with the value `#NAME?` in the `Image_Name` column were removed.

After cleaning, the dataset was reduced to **13,446 images**.

**To clean and split the dataset**, run the following command:
```bash
python src/dataset/prepare_data.py
```

This will create CSV files (`train.csv`, `val.csv`, `test.csv`, `raw_data.csv`) in the `data/` directory.

## Task 2: Use PyTorch to write your own dataloader with image (input data) and its recipe title (groundtruth)

In this task, we have created a custom PyTorch Dataset class in the script `/mcv-c5-team1/week3/src/dataset/data.py`. 

This class loads and preprocesses images and their corresponding recipe titles, applying necessary transformations to the images and tokenizing the titles for model training.

## Task 4: Train and Evaluate the Baseline Model with the Food Dataset

### Baseline Model Architecture

In the baseline model, the task is to predict the food dish title from an image. The model architecture consists of two major parts:

1. **ResNet-18 (Encoder)**: A pre-trained ResNet-18 model, widely used for image classification tasks, is used here as the encoder. 
   
2. **GRU (Decoder)**: The GRU network is used as the decoder, which generates the sequence of characters (the food dish title) based on the features provided by the ResNet-18 encoder. 

The process follows the **character-level** representation approach, meaning that we treat each word in the dish title as a sequence of characters, not words. 

**To train the baseline model**, run the following command:
```bash
python src/train_example_char.py
```
### Evaluation

The following metrics were used to evaluate the model’s performance:

- **BLEU-1**: Measures the precision of 1-gram (unigram) matches between the predicted and reference captions, with a brevity penalty for shorter predictions.
- **BLEU-2**: Measures the precision of 2-gram (bigram) matches between the predicted and reference captions, with a brevity penalty.
-**ROUGE-L**: Measures the Longest Common Subsequence (LCS) between the predicted and reference captions.
- **METEOR**: Evaluates similarity based on unigram precision, recall, word matches, lengths, and word order.

#### Results

| **Metric**  | **Train**  | **Validation**  | **Test**  |
|-------------|------------|-----------------|-----------|
| **BLEU-1**  | 2.79e-3    | 0               | 0         |
| **BLEU-2**  | 7.07e-4    | 0               | 0         |
| **ROUGE-L** | 7.34e-2    | 0               | 8.07e-6   |
| **METEOR**  | 3.99e-2    | 0               | 6.40e-6   |
| **Loss**    | 0.38       | 0.65            | 0.67      |

## Tasks 6 & 7: Modify the Baseline Model, Train and Evaluate Custom Models

### Change the Text Representation Level among Wordpiece-Level and Word-level

#### Wordpiece-level Text Representation
In this approach, the model uses a **BERT-based tokenizer** from HuggingFace, which tokenizes text into subword units (wordpieces).
- **Encoding**: Convert a sequence of words into a list of token IDs using the BERT wordpiece ↔ idx mapping. Special tokens <CLS> (start) and <SEP> (end) are used, with padding added to ensure all input sequences are of equal length.
- **Decoding**: Convert token IDs back into word sequences using the tokenizer.

To train the baseline model with wordpiece-level representation, run the following command:
```bash
python src/train_example_bert.py
```
##### Results

| **Metric**  | **Train**  | **Validation**  | **Test**  |
|-------------|------------|-----------------|-----------|
| **BLEU-1**  | 0.64       | 0.08            | 0.07      |
| **BLEU-2**  | 0.53    | 4.78e-3               | 4.41e-3         |
| **ROUGE-L** | 0.67    | 0.06               | 0.06   |
| **METEOR**  | 0.63    | 0.04               | 0.03   |
| **Loss**    | 0.16       | 1.24            | 1.27      |

#### Word-level Text Representation
In this approach, text is tokenized at the word-level. Each word in the caption is treated as a separate token, and spaces are also included as individual tokens.
- **Encoding**: The caption is split into words, and tokens are mapped to indices. Special tokens like <SOS> (start), <EOS> (end), and <PAD> (padding) are added.
- **Decoding**: The token indices are converted back into a sequence of words. The <EOS> token marks the end of the caption, and the <SOS> token is removed if present at the start.

To train the baseline model with word-level representation, run the following command:
```bash
python src/train_example_word.py
```

##### Results
| **Metric**  | **Train**  | **Validation**  | **Test**  |
|-------------|------------|-----------------|-----------|
| **BLEU-1**  | 0.38       | 0.02            | 8.14e-3      |
| **BLEU-2**  | 0.31    | 0.01               | 4.84e-4         |
| **ROUGE-L** | 0.73    | 0.05               | 0.03   |
| **METEOR**  | 0.66    | 0.03               | 0.02   |
| **Loss**    | 0.26       | 3.98            | 1.85      |

#### Conclusions
Among the three tokenization approaches,**wordpiece-level tokenization (BERT)** clearly outperforms both character-level and word-level representations. However, all models suffered from severe overfitting, with the wordpiece model showing a large generalization gap despite achieving the best training metrics.

These findings suggest that while the choice of tokenization strategy significantly impacts model performance, additional regularization, model modifications, or pre-training may be necessary to address generalization issues.

### Change the Encoder of the Baseline Model to VGG-16
In this task, we replaced the ResNet-18 encoder with a **VGG-16 encoder**. VGG-16 is a convolutional neural network (CNN) architecture with 16 layers, often used for image classification tasks.

The **GRU decoder** remains unchanged from the baseline model, and the text representation level used is **wordpiece-level**.

**Train Encoder VGG-16**:

#### Results
| **Encoder** | **BLEU-1** | **BLEU-2** | **ROUGE-L** | **METEOR** | **Loss** |
|-------------|------------|------------|-------------|------------|----------|
| **Train**   |            |            |             |            |          |
| ResNet-18   | 0.64       | 0.53       | 0.67        | 0.63       | 0.16     |
| VGG-16      |            |            |             |            |          |
| **Val**     |            |            |             |            |          |
| ResNet-18   | 0.08       | 4.78e-3    | 0.06        | 0.04       | 1.24     |
| VGG-16      |            |            |             |            |          |
| **Test**    |            |            |             |            |          |
| ResNet-18   | 0.07       | 4.41e-3    | 0.06        | 0.03       | 1.27     |
| VGG-16      |            |            |             |            |          |

### Change the Decoder of the Baseline Model To LSTM
In this task, we replaced the **GRU decoder** with an **LSTM decoder**. LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) that is well-suited for sequence prediction tasks, especially those involving long-term dependencies.

The **ResNet-18 encoder** remains unchanged from the baseline model, and the text representation level used is **wordpiece-level**.

**Train Decoder LSTM**:

#### Results
| **Decoder**       | **BLEU-1** | **BLEU-2** | **ROUGE-L** | **METEOR** | **Loss** |
|-------------------|------------|------------|-------------|------------|----------|
| **Train**         |            |            |             |            |          |
| GRU               | 0.64       | 0.53       | 0.67        | 0.63       | 0.16     |
| LSTM (1 layer)    |            |            |             |            |          |
| LSTM (2 layers)   | 0.83       | 0.79       | 0.84        | 0.82       | 0.08     |
| **Val**           |            |            |             |            |          |
| GRU               | 0.08       | 4.78e-3    | 0.06        | 0.04       | 1.24     |
| LSTM (1 layer)    |            |            |             |            |          |
| LSTM (2 layers)   | 0.08       | 0.07       | 0.07        | 0.04       | 1.20     |
| **Test**          |            |            |             |            |          |
| GRU               | 0.07       | 4.41e-3    | 0.06        | 0.03       | 1.27     |
| LSTM (1 layer)    |            |            |             |            |          |
| LSTM (2 layers)   | 0.08       | 0.06       | 0.07        | 0.04       | 1.23     |




