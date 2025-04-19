<h1 align="center">WEEK 5: Diffusion Models</h1>

# Table of Contents

- [Task A: Install Stable Diffusion (SD), and play with 2.1, XL, Turbo...](#task-a-install-stable-diffusion-sd-and-play-with-21-xl-turbo)
- [Task B: Explore effects of DDPM vs. DDIM, positive & negative prompting, CFG strength, denoising steps, etc.](#task-b-explore-effects-of-ddpm-vs-ddim-positive--negative-prompting-cfg-strength-denoising-steps-etc)

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

