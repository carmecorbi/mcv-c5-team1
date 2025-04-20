<h1 align="center">WEEK 5: Diffusion Models</h1>

# Table of Contents

- [Task A: Install Stable Diffusion (SD), and play with 2.1, XL, Turbo...](#task-a-install-stable-diffusion-sd-and-play-with-21-xl-turbo)
- [Task B: Explore effects of ‘DDPM vs. DDIM’, ‘positive & negative prompting’, ‘strength of CFG’, ‘num. of denoising steps’, etc.](#task-b-explore-effects-of-ddpm-vs-ddim-positive--negative-prompting-strength-of-cfg-num-of-denoising-steps-etc)

---

## Task A: Install Stable Diffusion (SD), and play with 2.1, XL, Turbo...

The models used this week are:

### Latent Diffusion
- [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)  
- [Stable Diffusion 2.1 Turbo](https://huggingface.co/stabilityai/sd-turbo)  
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)  
- [Stable Diffusion XL Turbo](https://huggingface.co/stabilityai/sdxl-turbo)  

### Latent Diffusion + Flow Matching / Adversarial Distillation
- [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)  
- [Stable Diffusion 3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)  

---

To run inference on a specific prompt, use the following command:

```bash
python3 src/inference.py --prompt "Almond Horchata" --model_id stabilityai/stable-diffusion-2-1 --output_dir results/results_2_1
```

You can change the `--model_id` to one of the following options:

```text
choices = [
    "stabilityai/stable-diffusion-2-1", 
    "stabilityai/stable-diffusion-xl-base-1.0", 
    "stabilityai/sd-turbo", 
    "stabilityai/sdxl-turbo",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3.5-large-turbo"
]
```

## Task B: Explore effects of ‘DDPM vs. DDIM’, ‘positive & negative prompting’, ‘strength of CFG’, ‘num. of denoising steps’, etc.

### Effects of schedulers: DDPM vs. DDIM

```bash
python3 src/inference.py --prompt 'Fruit Salad with Fennel, Watercress, and Smoked Salt' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --output_dir results/results_2_1

python3 src/inference.py --prompt 'Fruit Salad with Fennel, Watercress, and Smoked Salt' --scheduler ddim --model_id stabilityai/stable-diffusion-2-1 --output_dir results/results_2_1
```

![image](https://github.com/user-attachments/assets/f20472a0-2e1e-4d92-ae56-c4d409a99fc6)

Note: These schedulers are NOT supported for the 3.5 versions of Stable Diffusion models out of the box.

### Effects of positive & negative prompting

```bash
python3 src/inference.py --prompt 'Fruit Salad with Fennel, Watercress, and Smoked Salt' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --output_dir results/results_2_1

python3 src/inference.py --prompt 'Fruit Salad with Fennel, Watercress, and Smoked Salt' --negative_prompt 'blurry,unrealistic,low resolution, overexposed, unnatural colors' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --output_dir results/results_2_1
```

![image](https://github.com/user-attachments/assets/7d50f6d3-d56f-4934-a451-bca535712638)


Note: SD 3.5 does NOT  use DDPM scheduler (it is incompatible). It uses Flow Match Euler Discrete Scheduler. Turbo versions do NOT make use of negative_prompt (it is disabled).

### Effects of CFG strength

- Stable diffusion 2.1 model with DDPM Scheduler

```bash
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 1 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 3 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 5 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 7.5 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 10 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 15 --output_dir results/results_2_1
```

![image](https://github.com/user-attachments/assets/614b01ab-bfed-481a-9060-d6b9ea891646)


- Stable diffusion 2.1 model with DDIM Scheduler

```bash
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddim --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 1 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddim --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 3 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddim --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 5 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddim --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 7.5 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddim --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 10 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --scheduler ddim --model_id stabilityai/stable-diffusion-2-1 --guidance_scale 15 --output_dir results/results_2_1
```

![image](https://github.com/user-attachments/assets/0cff02a3-6c6a-4a94-a118-96852113ef5c)

- Stable diffusion 3.5 model

```bash
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --guidance_scale 1
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --guidance_scale 3
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --guidance_scale 5
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --guidance_scale 7.5
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --guidance_scale 10
python3 src/inference.py --prompt 'Tuscan Porterhouse Steak with Red Wine-Peppercorn Jus' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --guidance_scale 15
```

![image](https://github.com/user-attachments/assets/a1675db2-dda5-4ba2-9fdd-c7525d4805a4)

### Number of denoising steps

- Stable diffusion 2.1 model with DDPM Scheduler

```bash
python3 src/inference.py --prompt 'Almond Horchata' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --number_inference_steps 5 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Almond Horchata' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --number_inference_steps 25 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Almond Horchata' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --number_inference_steps 50 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Almond Horchata' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --number_inference_steps 75 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Almond Horchata' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --number_inference_steps 100 --output_dir results/results_2_1
python3 src/inference.py --prompt 'Almond Horchata' --scheduler ddpm --model_id stabilityai/stable-diffusion-2-1 --number_inference_steps 150 --output_dir results/results_2_1
```

![image](https://github.com/user-attachments/assets/8112c48d-476d-4a25-ad21-7605e973abf4)

- Stable diffusion 3.5 model

```bash
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --number_inference_steps 5
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --number_inference_steps 25
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --number_inference_steps 50
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --number_inference_steps 75
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --number_inference_steps 100
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-medium --output_dir results/results_3_5 --token 'huggingface token' --number_inference_steps 150
```

![image](https://github.com/user-attachments/assets/2c66d41f-089c-4099-95d8-2bb95029a868)

- Stable diffusion 3.5 Large Turbo model

```bash
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-large-turbo --output_dir results/results_3_5_turbo --token 'huggingface token' --number_inference_steps 5
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-large-turbo --output_dir results/results_3_5_turbo --token 'huggingface token' --number_inference_steps 25
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-large-turbo --output_dir results/results_3_5_turbo --token 'huggingface token' --number_inference_steps 50
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-large-turbo --output_dir results/results_3_5_turbo --token 'huggingface token' --number_inference_steps 75
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-large-turbo --output_dir results/results_3_5_turbo --token 'huggingface token' --number_inference_steps 100
python3 src/inference.py --prompt 'Almond Horchata' --model_id stabilityai/stable-diffusion-3.5-large-turbo --output_dir results/results_3_5_turbo --token 'huggingface token' --number_inference_steps 150
```

![image](https://github.com/user-attachments/assets/55e890df-3436-4947-a17b-36e7dda74f59)

## Task C: Analysis and problem statement

### Problem 

- Overfitting on the training set due to captions being too specific to images. 
- Lack of diversity and minimal repeated patterns in the dataset. 
- Dataset is highly single-case specific, hindering proper generalization of the model.

### Research Question 
How can we leverage synthetic image generation via Stable Diffusion and automated prompt generation to create diverse, high-quality synthetic samples that improve generalization in image captioning tasks?

## Task D: Building a Complex Pipeline

### Step 1: Generating Synthetic Captions and Images

In this first part of the pipeline, we automatically generate diverse textual captions and corresponding synthetic images using a combination of a language model (Gemma 3B) and a Stable Diffusion model.

#### Process Overview

1. **Input**: A CSV file containing original dish or drink titles.
2. **Text Augmentation**:
   - For each original caption, 3 new variations are generated using a large language model (`google/gemma-3-1b-it`).
   - The prompt encourages variations that remain faithful to the original concept while introducing diversity via ingredients, styles, or preparation.
3. **Image Generation**:
   - Each of the 3 new captions is passed through a pre-trained Stable Diffusion model (`stabilityai/stable-diffusion-2-1`) to generate a corresponding synthetic image.
   - Positive and negative prompting is applied to ensure high-quality, realistic food/drink images.
4. **Output**:
   - Images are saved in a specified output directory.
   - A new CSV (`synthetic_captions.csv`) is created with the generated image names and their respective new captions.

#### Run the script

To execute this step, run:

```bash
python3 src/generate_data.py \
  --token 'huggingface token' \
  --output_dir results/synthetic_data_good 
```

This will save synthetic data (captions and images) under `results/synthetic_data_good`.

#### Models used:
- **Language model**: `google/gemma-3-1b-it` (via Hugging Face)
- **Diffusion model**: `stabilityai/stable-diffusion-2-1`




