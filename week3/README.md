<h2 align="center">WEEK 3: Tasks</h2>

## Table of Contents

- [Train and Evaluate the Baseline Model with the Food Dataset](#train-and-evaluate-the-baseline-model-with-the-food-dataset)
- [Change the Text Representation Level among Wordpiece-Level and Word-level](#change-the-text-representation-level-among-wordpiece-level-and-word-level)
- [Change the Encoder of the Baseline Model to VGG-16](#change-the-encoder-of-the-baseline-model-to-vgg-16)
- [Change the Decoder of the Baseline Model To LSTM](#change-the-decoder-of-the-baseline-model-to-lstm)


## Train and Evaluate the Baseline Model with the Food Dataset

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

**Clean and split the dataset**: Run the following command:
```bash
python src/dataset/prepare_data.py
```
### Baseline Model Architecture

In the baseline model, the task is to predict the food dish title from an image. The model architecture consists of two major parts:

1. **ResNet-18 (Encoder)**: A pre-trained ResNet-18 model, widely used for image classification tasks, is used here as the encoder. 
   
2. **GRU (Decoder)**: The GRU network is used as the decoder, which generates the sequence of characters (the food dish title) based on the features provided by the ResNet-18 encoder. 

The process follows the character-level representation approach, meaning that we treat each word in the dish title as a sequence of characters, not words. 

**Train baseline model**: Run the following command:
```bash
python src/train_example_char.py
```
### Evaluation

The following metrics were used to evaluate the model’s performance:

- BLEU-1: Measures the precision of 1-gram (unigram) matches between the predicted and reference captions, with a brevity penalty for shorter predictions.
- BLEU-2: Measures the precision of 2-gram (bigram) matches between the predicted and reference captions, with a brevity penalty.
- ROUGE-L: Measures the Longest Common Subsequence (LCS) between the predicted and reference captions.
- METEOR: Evaluates similarity based on unigram precision, recall, word matches, lengths, and word order.

#### Results

| **Metric**  | **Train**  | **Validation**  | **Test**  |
|-------------|------------|-----------------|-----------|
| **BLEU-1**  | 2.79e-3    | 0               | 0         |
| **BLEU-2**  | 7.07e-4    | 0               | 0         |
| **ROUGE-L** | 7.34e-2    | 0               | 8.07e-6   |
| **METEOR**  | 3.99e-2    | 0               | 6.40e-6   |
| **Loss**    | 0.38       | 0.65            | 0.67      |

## Change the Text Representation Level among Wordpiece-Level and Word-level

### Wordpiece-level Text Representation
In this approach, the model uses a BERT-based tokenizer from HuggingFace, which tokenizes text into subword units (wordpieces).
**Encoding**: Convert a sequence of words into a list of token IDs using the BERT wordpiece ↔ idx mapping. Special tokens <CLS> (start) and <SEP> (end) are used, with padding added to ensure all input sequences are of equal length.
**Decoding**: Convert token IDs back into word sequences using the tokenizer.

**Train baseline model + wordpiece-level**: Run the following command:
```bash
python src/train_example_bert.py
```
#### Results

| **Metric**  | **Train**  | **Validation**  | **Test**  |
|-------------|------------|-----------------|-----------|
| **BLEU-1**  | 0.64       | 0.08            | 0.07      |
| **BLEU-2**  | 0.53    | 4.78e-3               | 4.41e-3         |
| **ROUGE-L** | 0.67    | 0.06               | 0.06   |
| **METEOR**  | 0.63    | 0.04               | 0.03   |
| **Loss**    | 0.16       | 1.24            | 1.27      |

### Word-level Text Representation
In this approach, text is tokenized at the word-level. Each word in the caption is treated as a separate token, and spaces are also included as individual tokens.
**Encoding**: The caption is split into words, and tokens are mapped to indices. Special tokens like <SOS> (start), <EOS> (end), and <PAD> (padding) are added.
**Decoding**: The token indices are converted back into a sequence of words. The <EOS> token marks the end of the caption, and the <SOS> token is removed if present at the start.

**Train baseline model + word-level**: Run the following command:
```bash
python src/train_example_word.py
```

#### Results
| **Metric**  | **Train**  | **Validation**  | **Test**  |
|-------------|------------|-----------------|-----------|
| **BLEU-1**  | 0.38       | 0.02            | 8.14e-3      |
| **BLEU-2**  | 0.31    | 0.01               | 4.84e-4         |
| **ROUGE-L** | 0.73    | 0.05               | 0.03   |
| **METEOR**  | 0.66    | 0.03               | 0.02   |
| **Loss**    | 0.26       | 3.98            | 1.85      |

### Conclusions
Among the three tokenization approaches, word-piece (Bert) clearly outperforms both character-level (which failed completely) and word-level representations. All models suffer from severe overfitting, with the word-piece model showing a large generalization gap despite getting the best training metrics. 
These findings suggest that while tokenization strategy significantly impacts model capabilities, additional regularization, model modifications, or pre-training are necessary to address generalization issues.

## Change the Encoder of the Baseline Model to VGG-16
In this task, we replaced the ResNet-18 encoder with a VGG-16 encoder. The VGG-16 model is a convolutional neural network (CNN) architecture with 16 layers, which is commonly used in image classification tasks. 

The GRU decoder remains unchanged from the baseline model and the text representation level used is Wordpiece-level. 

**Train Encoder VGG-16**:

### Results
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

## Change the Decoder of the Baseline Model To LSTM
In this task, we replaced the GRU decoder with an LSTM decoder. LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) that is well-suited for sequence prediction tasks, especially those involving long-term dependencies.

The ResNet-18 encoder remains unchanged from the baseline model and the text representation level used is Wordpiece-level. 

**Train Decoder LSTM**:

### Results



