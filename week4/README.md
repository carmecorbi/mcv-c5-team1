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
  - [Task 2.4: Compare and discuss the results obtained from all methods](#task-24-compare-and-discuss-the-results-obtained-from-all-methods)


# Project Structure W4

# Task 1: Image Captioning using ViT-GPT2 architecture

## Task 1.1: Direct evaluation using pretrained weights from huggingface model "nlpconnect image-captioning"

To evaluate the pretrained model, we first perform inference on some test images.  

### Example inference command:

```bash
python3 -m src.models.vit_gpt2 --task inference --infer_image /ghome/c5mcv01/mcv-c5-team1/week3/data/images/-bloody-mary-tomato-toast-with-celery-and-horseradish-56389813.jpg
```

### Results on test images:

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
      <td>Test</td><td>0.14</td><td>0.05</td><td>0.13</td><td>0.09</td>
    </tr>
  </tbody>
</table>

### Qualitative Results:

| Image                         | Ground Truth Caption                          | Predicted Caption with Pretrained Model    | Predicted Caption with Fully Unfrozen   |
|--------------------------------|----------------------------------------------|--------------------------------------------|--------------------------------------------|
| ![-bloody-mary-tomato-toast-with-celery-and-horseradish-56389813](https://github.com/user-attachments/assets/3d888b90-9f3c-48b6-b136-576338d88e08) | 'bloody mary tomato toast with celery and horseradish'           | 'two slices of pizza sitting on top of each other'        | 'Portobello Tomato Toast with Feta' |
| ![salted-apple-pretzel-pie](https://github.com/user-attachments/assets/a05d0feb-1f72-4abb-8eb4-81488f69124a) | 'salted apple pretzel pie' | 'a plate of food on a table' | 'Salted Apple Pretzel Pie' |
| ![vanilla-cupcakes-353909](https://github.com/user-attachments/assets/ea2d4bb7-2ab2-4fb3-b658-e9bd5a8dfa1d) | 'vanilla cupcakes' | 'a table topped with a bunch of cupcakes' | 'Vegan, Gluten-Free Vanilla Cupcakes' |

## Task 1.4: Compare and discuss your results against those obtained using last week's methods

# Task 2: Image Captioning with LLMs

## Task 2.1: Direct evaluation using Llama 3.2-11B model (multimodal)

## Task 2.2: Use your well trained ViT encoder as a frozen image feature extractor, and fine-tune decoders (Llama 3.2-1B and Llama 3.2-3B) using LoRA

## Task 2.3: Report a single table comparing the above methods using BLEU-1, BLEU-2, ROUGE-L, and METEOR

## Task 2.4: Compare and discuss the results obtained from all methods
