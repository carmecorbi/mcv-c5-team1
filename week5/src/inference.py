import argparse
import os
import re
import torch
import inspect

from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler, DiffusionPipeline, AutoPipelineForText2Image, StableDiffusion3Pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from transformers import T5EncoderModel

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Generate an image using Stable Diffusion.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the generated image")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1", 
                        choices=[
                            "stabilityai/stable-diffusion-2-1", 
                            "stabilityai/stable-diffusion-xl-base-1.0", 
                            "stabilityai/sd-turbo", 
                            "stabilityai/sdxl-turbo",
                            "stabilityai/stable-diffusion-3.5-medium",
                            "stabilityai/stable-diffusion-3.5-large-turbo"
                            ],
                        help="Model ID to load from Hugging Face")
    parser.add_argument("--height", type=int, default=169,
                        help="Height of the generated image")
    parser.add_argument("--width", type=int, default=274,
                        help="Width of the generated image")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to generate the image")
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim"],
                        help="Scheduler type: 'ddpm' or 'ddim'")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt to guide generation (e.g., 'blurry, low-res, bad anatomy')")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-Free Guidance scale (e.g., 1.1, 8, 12)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of inference steps for image generation")
    parser.add_argument("--token", type=str, required=False,
                        help="Hugging Face token for authentication")

    
    args = parser.parse_args()

    # Load the model
    if args.model_id == "stabilityai/stable-diffusion-2-1":
        pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
        unet_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
        print(f"UNet parameters: {unet_params / 1e6:.1f} M")
        text_encoder_params = sum(p.numel() for p in pipe.text_encoder.parameters() if p.requires_grad)
        print(f"Text encoder parameters: {text_encoder_params / 1e6:.1f} M")
        
    elif args.model_id == "stabilityai/stable-diffusion-xl-base-1.0":
        pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        unet_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
        print(f"UNet parameters: {unet_params / 1e6:.1f} M")
        text_encoder_params = sum(p.numel() for p in pipe.text_encoder.parameters() if p.requires_grad)
        print(f"Text encoder parameters: {text_encoder_params / 1e6:.1f} M")
        text_encoder_params2 = sum(p.numel() for p in pipe.text_encoder_2.parameters() if p.requires_grad)
        print(f"Text encoder parameters 2: {text_encoder_params2 / 1e6:.1f} M")
    elif args.model_id in ["stabilityai/sd-turbo", "stabilityai/sdxl-turbo"]:
        pipe = AutoPipelineForText2Image.from_pretrained(args.model_id, torch_dtype=torch.float16, variant="fp16")
        print(pipe)
        unet_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
        print(f"UNet parameters: {unet_params / 1e6:.1f} M")
        text_encoder_params = sum(p.numel() for p in pipe.text_encoder.parameters() if p.requires_grad)
        print(f"Text encoder parameters: {text_encoder_params / 1e6:.1f} M")
        #text_encoder_params2 = sum(p.numel() for p in pipe.text_encoder_2.parameters() if p.requires_grad)
        #print(f"Text encoder parameters 2: {text_encoder_params2 / 1e6:.1f} M")
    elif args.model_id in ["stabilityai/stable-diffusion-3.5-medium"]:
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, token = args.token)
        text_encoder_params = sum(p.numel() for p in pipe.text_encoder.parameters() if p.requires_grad)
        print(f"Text encoder parameters: {text_encoder_params / 1e6:.1f} M")
        text_encoder_params2 = sum(p.numel() for p in pipe.text_encoder_2.parameters() if p.requires_grad)
        print(f"Text encoder parameters 2: {text_encoder_params2 / 1e6:.1f} M")
        text_encoder_params3 = sum(p.numel() for p in pipe.text_encoder_3.parameters() if p.requires_grad)
        print(f"Text encoder parameters: {text_encoder_params3 / 1e6:.1f} M")
        transformer = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
        print(f"Transformer parameters: {transformer / 1e6:.1f} M")
    elif args.model_id in ["stabilityai/stable-diffusion-3.5-large-turbo"]:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            args.model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )

        t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.model_id, 
            transformer=model_nf4,
            text_encoder_3=t5_nf4,
            torch_dtype=torch.bfloat16,
            token = args.token
        )
        text_encoder_params = sum(p.numel() for p in pipeline.text_encoder.parameters() if p.requires_grad)
        print(f"Text encoder parameters: {text_encoder_params / 1e6:.1f} M")
        text_encoder_params2 = sum(p.numel() for p in pipeline.text_encoder_2.parameters() if p.requires_grad)
        print(f"Text encoder parameters 2: {text_encoder_params2 / 1e6:.1f} M")
        text_encoder_params3 = sum(p.numel() for p in pipeline.text_encoder_3.parameters() if p.requires_grad)
        print(f"Text encoder parameters: {text_encoder_params3 / 1e6:.1f} M")
        transformer = sum(p.numel() for p in pipeline.transformer.parameters() if p.requires_grad)
        print(f"Transformer parameters: {transformer / 1e6:.1f} M")
        pipeline.enable_model_cpu_offload()


    # Set the scheduler
    if args.model_id not in ["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3.5-large-turbo"]:
        scheduler_type = args.scheduler.lower()
        if scheduler_type == "ddpm":
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            print(pipe.scheduler)
        elif scheduler_type == "ddim":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            print(pipe.scheduler)

    # Generate the image
    if args.model_id not in ["stabilityai/stable-diffusion-3.5-large-turbo"]:
        print("\n🔍 Paràmetres disponibles a pipe(...):")
        signature = inspect.signature(pipe.__call__)
        for name, param in signature.parameters.items():
            default = param.default
            if default is inspect.Parameter.empty:
                print(f"- {name} (required)")
            else:
                print(f"- {name} (default = {default})")
        pipe = pipe.to(device)
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps
        ).images[0]
    else:
        print("\n🔍 Paràmetres disponibles a pipe(...):")
        signature = inspect.signature(pipeline.__call__)
        for name, param in signature.parameters.items():
            default = param.default
            if default is inspect.Parameter.empty:
                print(f"- {name} (required)")
            else:
                print(f"- {name} (default = {default})")
        image = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            max_sequence_length=512,
        ).images[0]
    
    image = image.resize((args.width, args.height))

    # Create results directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Clean up parts for filename
    def clean_filename(text):
        return re.sub(r"[^a-zA-Z0-9_-]", "_", text)[:80]

    safe_prompt = clean_filename(args.prompt)
    safe_model_id = clean_filename(args.model_id.split("/")[-1])
    if args.model_id not in ["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3.5-large-turbo"]:
        safe_scheduler = clean_filename(scheduler_type)
    safe_guidance = str(args.guidance_scale).replace(".", "_")
    safe_steps = f"steps{args.num_inference_steps}"

    # Filename now includes CFG and steps
    if args.model_id not in ["stabilityai/stable-diffusion-3.5-medium", "stabilityai/stable-diffusion-3.5-large-turbo"]:
        filename = f"{safe_prompt}_{safe_model_id}_{safe_scheduler}_cfg{safe_guidance}_{safe_steps}.png"
    else:
        filename = f"{safe_prompt}_{safe_model_id}_cfg{safe_guidance}_{safe_steps}.png"
    
    filepath = os.path.join(args.output_dir, filename)

    # Save the image
    image.save(filepath)
    print(f"Image saved as: {filepath}")
