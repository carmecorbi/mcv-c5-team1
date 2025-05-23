<h1 align="center">WEEK 4: Multimodal Recognition (Image Captioning-2)</h1>

> [!IMPORTANT]
> The presentation of W4 from this group is available [here](https://docs.google.com/presentation/d/1oIzaA7k2VZU-GLhzZjniNisA11Gmp07b1mq10XYI3HM/edit?usp=sharing). The access link for our Overleaf project on image captioning for the third and fourth week is available [here](https://overleaf.cvc.uab.cat/read/fvdttsmgdtbs#c694bb). If for some reason you don't have permissions to access it, contact any of the administrators of this repository.

# Table of Contents

- [Project Structure W4](#project-structure-w4)
- [Task 1: Image Captioning using ViT-GPT2 architecture](#task-1-image-captioning-using-vit-gpt2-architecture)
  - [Task 1.1: Direct evaluation using pretrained weights from huggingface model "nlpconnect image-captioning"](#task-11-direct-evaluation-using-pretrained-weights-from-huggingface-model-nlpconnect-image-captioning)
  - [Task 1.2: Fine-tuning strategies](#task-12-fine-tuning-strategies)
    - [ViT (Fine-Tune), GPT2 (Frozen)](#vit-fine-tune-gpt2-frozen)
    - [ViT (Frozen), GPT2 (Fine-Tune)](#vit-frozen-gpt2-fine-tune)
    - [ViT (Fine-Tune), GPT2 (Fine-Tune)](#vit-fine-tune-gpt2-fine-tune)
  - [Task 1.3: Report a single table comparing the above methods using BLEU-1, BLEU-2, ROUGE-L, and METEOR](#task-13-report-a-single-table-comparing-the-above-methods-using-bleu-1-bleu-2-rouge-l-and-meteor)
- [Task 2: Image Captioning with LLMs](#task-2-image-captioning-with-llms)
  - [Task 2.1: Direct evaluation using Llama 3.2-11B model (multimodal)](#task-21-direct-evaluation-using-llama-32-11b-model-multimodal)
  - [Cleaning dataset](#cleaning-dataset)
  - [Task 2.2: Use your well trained ViT encoder as a frozen image feature extractor, and fine-tune decoders (Llama 3.2-1B and Llama 3.2-3B) using LoRA](#task-22-use-your-well-trained-vit-encoder-as-a-frozen-image-feature-extractor-and-fine-tune-decoders-llama-32-1b-and-llama-32-3b-using-lora)
  - [Task 2.3: Report a single table comparing the above methods using BLEU-1, BLEU-2, ROUGE-L, and METEOR](#task-23-report-a-single-table-comparing-the-above-methods-using-bleu-1-bleu-2-rouge-l-and-meteor)
  - [Fine-Tuning with Varying LoRA Parameters](#fine-tuning-with-varying-lora-parameters)


# Project Structure W4

# Task 1: Image Captioning using ViT-GPT2 architecture

## Task 1.1: Direct evaluation using pretrained weights from huggingface model "nlpconnect image-captioning"

To evaluate the pretrained model, we first perform inference on some test images.  

### Example inference command:

```bash
python3 -m src.models.vit_gpt2 --task inference --infer_image /ghome/c5mcv01/mcv-c5-team1/week3/data/images/-bloody-mary-tomato-toast-with-celery-and-horseradish-56389813.jpg
```

### Results:

| Image                         | Ground Truth Caption                          | Predicted Caption                          |
|--------------------------------|----------------------------------------------|--------------------------------------------|
| ![-bloody-mary-tomato-toast-with-celery-and-horseradish-56389813](https://github.com/user-attachments/assets/3d888b90-9f3c-48b6-b136-576338d88e08) | 'bloody mary tomato toast with celery and horseradish'           | 'two slices of pizza sitting on top of each other'        |
| ![salted-apple-pretzel-pie](https://github.com/user-attachments/assets/a05d0feb-1f72-4abb-8eb4-81488f69124a) | 'salted apple pretzel pie' | 'a plate of food on a table' |
| ![vanilla-cupcakes-353909](https://github.com/user-attachments/assets/ea2d4bb7-2ab2-4fb3-b658-e9bd5a8dfa1d) | 'vanilla cupcakes' | 'a table topped with a bunch of cupcakes' |

These results show how the model generates image descriptions based on its pretrained knowledge. 

### Evaluation:

To evaluate the pretrained model, we can use the following command:

```bash
python3 -m src.models.vit_gpt2 -t evaluation \
    -d /ghome/c5mcv01/mcv-c5-team1/week3/data \
    --eval_set test
```

The evaluation is performed using the following metrics:
- **BLEU-1**: Measures the precision of unigrams (individual words).
- **BLEU-2**: Measures the precision of bigrams (pairs of words).
- **ROUGE-L**: Assesses the longest common subsequence between the predicted and ground truth captions.
- **METEOR**: Evaluates the predictions based on exact, stem, synonym, and paraphrase matches.

#### Evaluation Results:

| Set   | BLEU-1 | BLEU-2  | ROUGE-L | METEOR |
|-------|--------|---------|---------|--------|
| Train | 0.03   | 4.54e-4 | 0.05    | 0.04   |
| Val   | 0.04   | 9.58e-4 | 0.05    | 0.04   |
| Test  | 0.04   | 3.74e-4 | 0.06    | 0.04   |

Metrics are really low for this model, suggesting it is not able to correctly identify different dishes and probably predicting generic sentences. The model was pretrained on a different domain dataset, so in here the difference of train, val and test is just for the purpose of comparison with fine-tuned versions.

## Task 1.2: Fine-tuning strategies

### ViT (Fine-Tune), GPT2 (Frozen)

```bash
python3 -m src.models.vit_gpt2 -d /ghome/c5mcv01/mcv-c5-team1/week3/data \
                            -o  /ghome/c5mcv01/mcv-c5-team1/week4/results \
                            -t train \
                            -fd --num_epochs=100 --model_name=vit_gpt2_forzen_decoder
```


### ViT (Frozen), GPT2 (Fine-Tune)

```bash
python3 -m src.models.vit_gpt2 -d /ghome/c5mcv01/mcv-c5-team1/week3/data \
                            -o  /ghome/c5mcv01/mcv-c5-team1/week4/results \
                            -t train \
                            -fe --num_epochs=100 --model_name=vit_gpt2_forzen_encoder
```

### ViT (Fine-Tune), GPT2 (Fine-Tune)

```bash
python3 -m src.models.vit_gpt2 -d /ghome/c5mcv01/mcv-c5-team1/week3/data \
                            -o  /ghome/c5mcv01/mcv-c5-team1/week4/results \
                            -t train \
                            --num_epochs=100 --model_name=vit_gpt2_fully_unfrozen
```

## Task 1.3: Report a single table comparing the above methods using BLEU-1, BLEU-2, ROUGE-L, and METEOR

### Quantitative Results:

To evaluate the fine-tuned models, we can use the following command:

```bash
python3 -m src.models.vit_gpt2 -t evaluation \
    -d /ghome/c5mcv01/mcv-c5-team1/week3/data \
    -m /ghome/c5mcv01/mcv-c5-team1/week4/results/vit_gpt2_fully_unfrozen/checkpoint-8400 --eval_set test \
```

<table>
  <thead>
    <tr>
      <th>Strategy</th>
      <th>Set</th>
      <th>BLEU-1</th>
      <th>BLEU-2</th>
      <th>ROUGE-L</th>
      <th>METEOR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>Frozen Decoder</b></td>
      <td>Train</td><td>0.30</td><td>0.20</td><td>0.35</td><td>0.31</td>
    </tr>
    <tr>
      <td>Val</td><td>0.08</td><td>0.01</td><td>0.07</td><td>0.05</td>
    </tr>
    <tr>
      <td>Test</td><td>0.08</td><td>0.01</td><td>0.08</td><td>0.05</td>
    </tr>
    <tr><td colspan="6"></td></tr>
    <tr>
      <td rowspan="3"><b>Frozen Encoder</b></td>
      <td>Train</td><td>0.47</td><td>0.43</td><td>0.47</td><td>0.45</td>
    </tr>
    <tr>
      <td>Val</td><td>0.14</td><td>0.05</td><td>0.13</td><td>0.09</td>
    </tr>
    <tr>
      <td>Test</td><td>0.14</td><td>0.05</td><td>0.13</td><td>0.09</td>
    </tr>
    <tr><td colspan="6"></td></tr>
    <tr>
      <td rowspan="3"><b>Fully Unfrozen (Best)</b></td>
      <td>Train</td><td>0.95</td><td>0.94</td><td>0.97</td><td>0.94</td>
    </tr>
    <tr>
      <td>Val</td><td>0.14</td><td>0.05</td><td>0.14</td><td>0.09</td>
    </tr>
    <tr>
      <td>Test</td><td>0.15</td><td>0.05</td><td>0.14</td><td>0.09</td>
    </tr>
  </tbody>
</table>

### Optuna with the best model:

After evaluating different configurations, we found that the best-performing model was the **Fully Unfrozen** one. Therefore, we performed hyperparameter optimization using Optuna to further enhance its performance.

**Key Hyperparameters Tuned:**

- **Learning Rate:** Adjusted logarithmically between 1e-5 and 1e-2.

- **Scheduler Type:** Chosen from linear, inverse_sqrt, cosine_with_min_lr, or warmup_stable_decay.

- **Gradient Clipping:** Optionally enabled with values in the range [0.1, 10.0].

- **Dropout Rates:** Tuned for attention, embedding, residual, and hidden layers in the range [0.1, 0.5].

- **Weight Decay:** Regularization parameter between 1e-5 and 1e-3.

- **Warmup Steps:** Number of steps for learning rate warmup, ranging from 500 to 2000.

Run this command to use optuna:

```bash
python3 -m src.run_optuna
```

Result with the Best Validation Loss:

`Best is trial 34 with value: 0.4573763906955719.`

Now we evaluate the model obtained with trial 34 with this command:

```bash
python3 -m src.models.vit_gpt2 -t evaluation  \
    -d /ghome/c5mcv01/mcv-c5-team1/week3/data \
    -m /ghome/c5mcv01/mcv-c5-team1/week4/optuna_studies_task1/optuna_task1_trial_34_FullyUnfrozen_dropout/checkpoint-1183 --eval_set test \
```

| Set   | BLEU-1 | BLEU-2  | ROUGE-L | METEOR |
|-------|--------|---------|---------|--------|
| Train | 0.30   | 0.23    | 0.32    | 0.29   |
| Val   | 0.11   | 0.03    | 0.11    | 0.07   |
| Test  | 0.11   | 0.04    | 0.11    | 0.08   |

Despite performing hyperparameter tuning with Optuna, we did not achieve better results compared to the original fine-tuned model. Therefore, we use the original fine-tuned model for inference.

### Qualitative Results:

| Image      | Set                   | Ground Truth Caption                          | Predicted Caption with Pretrained Model    | Predicted Caption with Fully Unfrozen   |
|--------------------------------|----------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|
| ![mochi-covered-strawberries-56389993](https://github.com/user-attachments/assets/4829ddef-2153-4b04-aa1a-4b05bfa8906c) | Test | 'mochi covered strawberries'           | 'a white plate topped with strawberries and blueberries'        | 'Strawberry-Miso Tofu Balls' |
| ![nutter-butter-cookies](https://github.com/user-attachments/assets/69c0aec6-abdf-48b6-8491-679086e52bdc) | Test | 'nutter butter cookies' | 'a table topped with lots of doughnuts' | 'S'mores Cheesecake' |
| ![fried-egg-and-sausage-ciabatta-breakfast-pizzas-241096](https://github.com/user-attachments/assets/d1f090c4-86f9-43e2-81e6-d52a245ed885) | Test | 'fried egg and sausage ciabatta breakfast pizzas' | 'a white plate topped with a piece of bread' | 'Eggs Toast with Fried Eggs and Toasted Brioche' |


# Task 2: Image Captioning with LLMs

## Task 2.1: Direct evaluation using Llama 3.2-11B model (multimodal)

As the Llama 3.2-11B model is not available, we use the Gemma 4B model for this task: [Gemma 3-4B IT](https://huggingface.co/google/gemma-3-4b-it).

### Model Input Structure

The model processes input through a structured message format that defines roles in the conversation. The input follows this structure:

```python
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You have to do image captioning with the images I provide to you. Only do the image captioning as an expert in dishes. \nBe as specific as possible with the dish, only provide the caption, nothing more, nothing less."}]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "path": image_path
            }
        ]
    }
]
```

- The **system role** sets the model's behavior, instructing it to act as an expert in dish recognition and generate captions with high specificity.
- The **user role** provides the image to be processed, ensuring that the model focuses on generating a relevant caption without additional responses.

### Inference:

To run the model and generate image captions, execute the following command:

```bash
python3 -m src.models.llm -i /ghome/c5mcv01/mcv-c5-team1/week3/data/images/nutter-butter-cookies.jpg -t infer
```

**Example Output:**

```
Generated caption: Dutch peanut cookies with vanilla buttercream.
```

### Qualitative Results:

| Image | Ground Truth Caption                              | Predicted Caption with Pretrained Model Gemma 4B |
| ----- | ------------------------------------------------- | ------------------------------------------------ |
| ![mochi-covered-strawberries-56389993](https://github.com/user-attachments/assets/4829ddef-2153-4b04-aa1a-4b05bfa8906c)  | 'mochi covered strawberries'                      | 'Chocolate Mousse Mochi with Fresh Strawberries' |
| ![nutter-butter-cookies](https://github.com/user-attachments/assets/69c0aec6-abdf-48b6-8491-679086e52bdc) | 'nutter butter cookies'                           | 'Dutch peanut cookies with vanilla buttercream.' |
| ![fried-egg-and-sausage-ciabatta-breakfast-pizzas-241096](https://github.com/user-attachments/assets/d1f090c4-86f9-43e2-81e6-d52a245ed885) | 'fried egg and sausage ciabatta breakfast pizzas' | 'Irish Breakfast Toast with Fried Egg and Bacon' |

The results highlight that the model successfully generates relevant dish captions but sometimes introduces regional variations or slight misinterpretations of the dish composition.

### Quantitative Results:

To evaluate the model, we can use the following command:

```bash
python3 -m src.models.llm -d /ghome/c5mcv01/mcv-c5-team1/week3/data -t eval --eval_set test
```

Results:

| Set   | BLEU-1 | BLEU-2  | ROUGE-L | METEOR |
|-------|--------|---------|---------|--------|
| Train | 0.11   | 0.02    | 0.23    | 0.20   |
| Val   | 0.10   | 0.01    | 0.23    | 0.20   |
| Test  | 0.10   | 0.01    | 0.22    | 0.20   |

## Cleaning dataset

Based on the poor performance observed in Task 1, we decide to clean the dataset. The dataset includes many images and captions that could hinder model learning, such as images containing only text, duplicated visuals with different captions, or images with people.

This cleaning process is split into three main steps:

### Step 1: Cleaning Images Containing Only Text

- **Model Used**: [EasyOCR](https://github.com/JaidedAI/EasyOCR)  
  A Python OCR library used for text detection (English only).
  
- **Process**:
  1. Scan all images using OCR.
  2. Review images that contain only text.
  3. Keep a few relevant ones (e.g., *Chicken Lettuce Cups*).
  4. Remove the rest from the dataset `.csv`.

- **Command**:
  ```bash
  python3 cleaning_text.py
  ```

- **Dataset Overview**:
  - Before: `13,466` images  
  - After: `13,264` images  
  - Removed: `202` images

---

### Step 2: Removing Duplicate Images with Different Captions

- **Method Used**: [hashlib MD5](https://docs.python.org/3/library/hashlib.html)  
  Used to compute a unique hash for each image and detect duplicates.

- **Process**:
  1. Generate MD5 hash for each image.
  2. Identify duplicates (same image, different captions).
  3. Merge entries, keeping the first image name and combining unique captions.

- **Command**:
  ```bash
  python3 cleaning_images.py
  ```

- **Dataset Overview**:
  - Before: `13,264` images  
  - After: `12,972` images  
  - Removed: `292` duplicates

---

### Step 3: Removing Images with People

- **Model Used**: [YOLOv8n](https://docs.ultralytics.com/es/models/yolov8/#performance-metrics)  
  For object detection with a confidence threshold of `0.75`.

- **Process**:
  1. Apply person detection to all images.
  2. Remove images where at least one person is detected (with ≥ 75% confidence).

- **Command**:
  ```bash
  python3 cleaning_persons.py \
    --csv_path cleaned_merged.csv \
    --output_path final.csv \
    --images_dir /ghome/c5mcv01/mcv-c5-team1/week3/data/images
  ```

- **Dataset Overview**:
  - Before: `12,972` images  
  - After: `12,934` images  
  - Removed: `38` images


## Task 2.2: Use your well trained ViT encoder as a frozen image feature extractor, and fine-tune decoders (Llama 3.2-1B and Llama 3.2-3B) using LoRA

### Llama 3.2-1B Fine-Tuning:

```bash
python3 -m src.models.vit_llama3_2 -t train --model_name meta-llama/Llama-3.2-1B \
    --hf_token 'hugging face access token' --num_epochs 15 \
    --output_dir results/vit_llama3_2_1B_cleaned
```

#### Inference:

```bash
python3 -m src.models.vit_llama3_2 -t infer --model_name meta-llama/Llama-3.2-1B \
    --hf_token 'hugging face access token' \
    --model_file results/vit_llama3_2_1B_cleaned/checkpoints/best_model.pt \
    --infer_image_path /ghome/c5mcv01/mcv-c5-team1/week3/data/images/milk-chocolate-peanut-butter-sandwich-cookies-233945.jpg
```

### Llama 3.2-3B Fine-Tuning:

```bash
python3 -m src.models.vit_llama3_2 -t train --model_name meta-llama/Llama-3.2-3B \
    --hf_token 'hugging face access token' --num_epochs 15  --batch_size 2 \
    --output_dir results/vit_llama3_2_3B_cleaned
```

#### Inference:


```bash
python3 -m src.models.vit_llama3_2 -t infer --model_name meta-llama/Llama-3.2-3B \
    --hf_token 'hugging face access token' \
    --model_file results/vit_llama3_2_3B_cleaned/checkpoints/best_model.pt \
    --infer_image_path /ghome/c5mcv01/mcv-c5-team1/week3/data/images/milk-chocolate-peanut-butter-sandwich-cookies-233945.jpg
```


### Qualitative Results:

| Image | Ground Truth Caption                              | Predicted Caption with Llama 3.2-1B Fine-Tuning | Predicted Caption with Llama 3.2-3B Fine-Tuning |
| ----- | ------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| ![butter-cookies-with-raisins-5316](https://github.com/user-attachments/assets/c24984e4-ea55-4cc6-881c-87cff75bcbec) | 'butter cookies with raisins'                      | 'Spiced Sesame Balls' | 'Pistachio Cardamom Crescents' |
| ![spinach-gnocchi-51262540](https://github.com/user-attachments/assets/7eb02af3-71bc-4628-8108-55c8276204b3) | 'spinach gnocchi' | 'Christmas Tree Fritters' | 'Fried Green Olives with Shrimp and Oregano' |
| ![milk-chocolate-peanut-butter-sandwich-cookies-233945](https://github.com/user-attachments/assets/25c7ce80-abab-47fa-b3df-e70992bbb08b) | 'milk chocolate peanut butter sandwich cookies' | 'Cinnamon Apple Pie with Cheddar Crust' | 'Caramel Madness' |

## Task 2.3: Report a single table comparing the above methods using BLEU-1, BLEU-2, ROUGE-L, and METEOR

For evaluation run:

```bash
python3 -m src.models.vit_llama3_2 -t eval --eval_set test --model_name meta-llama/Llama-3.2-1B \
    --hf_token 'hugging face access token' \
    --model_file results/vit_llama3_2_1B_cleaned/checkpoints/best_model.pt \
```

```bash
python3 -m src.models.vit_llama3_2 -t eval --eval_set test --model_name meta-llama/Llama-3.2-3B \
    --hf_token 'hugging face access token' \
    --model_file results/vit_llama3_2_3B_cleaned/checkpoints/best_model.pt \
```

<table>
  <thead>
    <tr>
      <th>Strategy</th>
      <th>Set</th>
      <th>BLEU-1</th>
      <th>BLEU-2</th>
      <th>ROUGE-L</th>
      <th>METEOR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>Llama 3.2 1B</b></td>
      <td>Train</td><td>0.99</td><td>0.97</td><td>0.99</td><td>0.97</td>
    </tr>
    <tr>
      <td>Val</td><td>0.10</td><td>0.02</td><td>0.13</td><td>0.09</td>
    </tr>
    <tr>
      <td>Test</td><td>0.11</td><td>0.02</td><td>0.13</td><td>0.10</td>
    </tr>
    <tr><td colspan="6"></td></tr>
    <tr>
      <td rowspan="3"><b>Llama 3.2 3B</b></td>
      <td>Train</td><td>0.29</td><td>0.21</td><td>0.32</td><td>0.28</td>
    </tr>
    <tr>
      <td>Val</td><td>0.10</td><td>0.02</td><td>0.12</td><td>0.09</td>
    </tr>
    <tr>
      <td>Test</td><td>0.11</td><td>0.02</td><td>0.13</td><td>0.09</td>
    </tr>
  </tbody>
</table>

Results Analysis: The Llama 3.2 1B model shows strong overfitting, with extremely high performance on the training set but poor generalization to validation and test sets. In contrast, Llama 3.2 3B, while performing worse on the training set, demonstrates better generalization across validation and test sets. 

## Fine-Tuning with Varying LoRA Parameters

We conducted additional experiments fine-tuning the ViT + LLaMA 3.2 1B model using different LoRA parameter configurations. In previous experiments, we observed that both the 1B and 3B versions of LLaMA 3.2 achieved similar performance, with both models showing signs of overfitting. As a result, we continued experimenting with the 1B model due to its faster training time.

In this section, we isolate the effect of three key LoRA hyperparameters by modifying **one parameter at a time** while keeping the others at their default values. To reduce overfitting, the number of training epochs was decreased from **15 to 10** in all cases.

### LoRA Parameters Investigated:
- **Alpha** (LoRA scaling): `[16, 32 (default), 64]`
- **Dropout**: `[0.1 (default), 0.25, 0.5]`
- **R** (Attention Dimension / Rank): `[4, 8 (default), 16]`

### Training Commands Used:
```bash
# Varying LoRA Rank (r)
python3 -m src.models.vit_llama3_2 -t train --hf_token 'hugging face access token' --num_epochs 10 --output_dir results/vit_llama3_2_1B_cleaned_r4 --lora_r 4
python3 -m src.models.vit_llama3_2 -t train --hf_token 'hugging face access token' --num_epochs 10 --output_dir results/vit_llama3_2_1B_cleaned_r16 --lora_r 16

# Varying Dropout
python3 -m src.models.vit_llama3_2 -t train --hf_token 'hugging face access token' --num_epochs 10 --output_dir results/vit_llama3_2_1B_cleaned_dropout25 --lora_dropout 0.25
python3 -m src.models.vit_llama3_2 -t train --hf_token 'hugging face access token' --num_epochs 10 --output_dir results/vit_llama3_2_1B_cleaned_dropout50 --lora_dropout 0.50

# Varying Alpha
python3 -m src.models.vit_llama3_2 -t train --hf_token 'hugging face access token' --num_epochs 10 --output_dir results/vit_llama3_2_1B_cleaned_alpha16 --lora_alpha 16
python3 -m src.models.vit_llama3_2 -t train --hf_token 'hugging face access token' --num_epochs 10 --output_dir results/vit_llama3_2_1B_cleaned_alpha64 --lora_alpha 64
```

---

### Quantitative Results: LoRA Alpha Scaling

<table>
  <thead>
    <tr>
      <th>Alpha</th>
      <th>Set</th>
      <th>BLEU-1</th>
      <th>BLEU-2</th>
      <th>ROUGE-L</th>
      <th>METEOR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">16</td>
      <td>Val</td><td>0.03</td><td>0.002</td><td>0.04</td><td>0.02</td>
    </tr>
    <tr>
      <td>Test</td><td>0.03</td><td>0.0007</td><td>0.04</td><td>0.03</td>
    </tr>
    <tr>
      <td rowspan="2"><b>32</b></td>
      <td>Val</td><td><b>0.10</b></td><td><b>0.02</b></td><td><b>0.13</b></td><td><b>0.09</b></td>
    </tr>
    <tr>
      <td>Test</td><td><b>0.11</b></td><td><b>0.02</b></td><td><b>0.13</b></td><td><b>0.10</b></td>
    </tr>
    <tr>
      <td rowspan="2">64</td>
      <td>Val</td><td>0.08</td><td>0.02</td><td>0.10</td><td>0.08</td>
    </tr>
    <tr>
      <td>Test</td><td>0.07</td><td>0.02</td><td>0.10</td><td>0.07</td>
    </tr>
  </tbody>
</table>


---

### Quantitative Results: LoRA Dropout

<table>
  <thead>
    <tr>
      <th>Dropout</th>
      <th>Set</th>
      <th>BLEU-1</th>
      <th>BLEU-2</th>
      <th>ROUGE-L</th>
      <th>METEOR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">0.1</td>
      <td>Val</td><td>0.10</td><td>0.02</td><td>0.13</td><td>0.09</td>
    </tr>
    <tr>
      <td>Test</td><td>0.11</td><td>0.02</td><td>0.13</td><td>0.10</td>
    </tr>
    <tr>
      <td rowspan="2"><b>0.25</b></td>
      <td>Val</td><td><b>0.11</b></td><td><b>0.03</b></td><td><b>0.13</b></td><td><b>0.10</b></td>
    </tr>
    <tr>
      <td>Test</td><td><b>0.11</b></td><td><b>0.03</b></td><td><b>0.13</b></td><td><b>0.09</b></td>
    </tr>
    <tr>
      <td rowspan="2">0.5</td>
      <td>Val</td><td>0.11</td><td>0.02</td><td>0.14</td><td>0.10</td>
    </tr>
    <tr>
      <td>Test</td><td>0.11</td><td>0.02</td><td>0.14</td><td>0.10</td>
    </tr>
  </tbody>
</table>

---

### Quantitative Results: LoRA Rank (r)

<table>
  <thead>
    <tr>
      <th>r</th>
      <th>Set</th>
      <th>BLEU-1</th>
      <th>BLEU-2</th>
      <th>ROUGE-L</th>
      <th>METEOR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">4</td>
      <td>Val</td><td>0.11</td><td>0.02</td><td>0.13</td><td>0.10</td>
    </tr>
    <tr>
      <td>Test</td><td>0.10</td><td>0.02</td><td>0.13</td><td>0.09</td>
    </tr>
    <tr>
      <td rowspan="2"><b>8</b></td>
      <td>Val</td><td><b>0.10</b></td><td><b>0.02</b></td><td><b>0.13</b></td><td><b>0.09</b></td>
    </tr>
    <tr>
      <td>Test</td><td><b>0.11</b></td><td><b>0.02</b></td><td><b>0.13</b></td><td><b>0.10</b></td>
    </tr>
    <tr>
      <td rowspan="2">16</td>
      <td>Val</td><td>0.10</td><td>0.02</td><td>0.13</td><td>0.08</td>
    </tr>
    <tr>
      <td>Test</td><td>0.10</td><td>0.02</td><td>0.13</td><td>0.09</td>
    </tr>
  </tbody>
</table>

Among all the LoRA parameter variations, the only configuration that consistently improved performance was increasing the **dropout** to 0.25, which led to slight gains across all evaluation metrics. Other changes to **alpha** and **r** did not result in clear improvements, suggesting that regularization via dropout was more effective than adjusting model capacity in this context.

