<h1 align="center">WEEK 4: Multimodal Recognition (Image Captioning-2)</h1>

# Table of Contents

- [Project Structure W4](#project-structure-w4)
- [Task 1: Image Captioning using ViT-GPT2 architecture](#task-1-image-captioning-using-vit-gpt2-architecture)
  - [Task 1.1: Direct evaluation using pretrained weights from huggingface model "nlpconnect image-captioning"](#task-11-direct-evaluation-using-pretrained-weights-from-huggingface-model-nlpconnect-image-captioning)
  - [Task 1.2: Fine-tuning strategies](#task-12-fine-tuning-strategies)
    - [ViT (Fine-Tune), GPT2 (Frozen)](#vit-fine-tune-gpt2-frozen)
    - [ViT (Frozen), GPT2 (Fine-Tune)](#vit-frozen-gpt2-fine-tune)
    - [ViT (Fine-Tune), GPT2 (Fine-Tune)](#vit-fine-tune-gpt2-fine-tune)
  - [Task 1.3: Report a single table comparing the above methods using BLEU-1, BLEU-2, ROUGE-L, and METEOR](#task-13-report-a-single-table-comparing-the-above-methods-using-bleu-1-bleu-2-rouge-l-and-meteor)
  - [Task 1.4: Compare and discuss your results against those obtained using last week's methods](#task-14-compare-and-discuss-your-results-against-those-obtained-using-last-weeks-methods)
- [Task 2: Image Captioning with LLMs](#task-2-image-captioning-with-llms)
  - [Task 2.1: Direct evaluation using Llama 3.2-11B model (multimodal)](#task-21-direct-evaluation-using-llama-32-11b-model-multimodal)
  - [Task 2.2: Use your well trained ViT encoder as a frozen image feature extractor, and fine-tune decoders (Llama 3.2-1B and Llama 3.2-3B) using LoRA](#task-22-use-your-well-trained-vit-encoder-as-a-frozen-image-feature-extractor-and-fine-tune-decoders-llama-32-1b-and-llama-32-3b-using-lora)
  - [Task 2.3: Report a single table comparing the above methods using BLEU-1, BLEU-2, ROUGE-L, and METEOR](#task-23-report-a-single-table-comparing-the-above-methods-using-bleu-1-bleu-2-rouge-l-and-meteor)


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

## Task 1.4: Compare and discuss your results against those obtained using last week's methods

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
    --infer_image_path /ghome/c5mcv01/mcv-c5-team1/week3/data/images/nutter-butter-cookies.jpg
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
    --infer_image_path /ghome/c5mcv01/mcv-c5-team1/week3/data/images/nutter-butter-cookies.jpg
```


### Qualitative Results:

| Image | Ground Truth Caption                              | Predicted Caption with Llama 3.2-1B Fine-Tuning | Predicted Caption with Llama 3.2-3B Fine-Tuning |
| ----- | ------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| ![mochi-covered-strawberries-56389993](https://github.com/user-attachments/assets/4829ddef-2153-4b04-aa1a-4b05bfa8906c)  | 'mochi covered strawberries'                      | 'Matcha-Dipped Salmon with Asparagus and Mint' | 'Strawberries with Berries and Yogurt' |
| ![nutter-butter-cookies](https://github.com/user-attachments/assets/69c0aec6-abdf-48b6-8491-679086e52bdc) | 'nutter butter cookies'                           | 'Rosemary Orange Turnovers' | '3-Ingredient Coconut Cardamom Cookies' |
| ![fried-egg-and-sausage-ciabatta-breakfast-pizzas-241096](https://github.com/user-attachments/assets/d1f090c4-86f9-43e2-81e6-d52a245ed885) | 'fried egg and sausage ciabatta breakfast pizzas' | 'Poached Eggs and Spinach on Toast with Vinegar' | 'Poached Egg on Toast' |

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
      <td>Train</td><td>0.96</td><td>0.94</td><td>0.97</td><td>0.94</td>
    </tr>
    <tr>
      <td>Val</td><td>0.10</td><td>0.02</td><td>0.13</td><td>0.09</td>
    </tr>
    <tr>
      <td>Test</td><td>0.11</td><td>0.02</td><td>0.13</td><td>0.09</td>
    </tr>
    <tr><td colspan="6"></td></tr>
    <tr>
      <td rowspan="3"><b>Llama 3.2 3B</b></td>
      <td>Train</td><td>0.31</td><td>0.16</td><td>0.36</td><td>0.28</td>
    </tr>
    <tr>
      <td>Val</td><td>0.12</td><td>0.03</td><td>0.16</td><td>0.10</td>
    </tr>
    <tr>
      <td>Test</td><td>0.12</td><td>0.02</td><td>0.15</td><td>0.10</td>
    </tr>
  </tbody>
</table>

# Cleaning dataset

## Step 1: Cleaning Images containing only TEXT

```bash
python3 cleaning_text.py
```

## Step 2: Cleaning Duplicate Images with Different Captions

```bash
python3 cleaning_images.py
```
## Step 3: Cleaning Person Images
```bash
python3 cleaning_persons.py --csv_path cleaned_merged.csv --output_path final.csv --images_dir /ghome/c5mcv01/mcv-c5-team1/week3/data/images
```
