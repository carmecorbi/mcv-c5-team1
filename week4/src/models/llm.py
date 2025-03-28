from transformers import AutoProcessor, Gemma3ForConditionalGeneration

import torch
import argparse

model_id = "google/gemma-3-4b-it"


def run_model(image_path: str) -> str:
    """Run the model with the given image path.

    Args:
        image_path (str): The path to the image.

    Returns:
        str: The generated caption.
    """
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You have to do image captioning with the images I provide to you. Only do the image captioning as an expert in dishes. \
                        Be as much specific with the dish, only provide the caption, nothing more, nothing less."}]
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

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=25, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with the given image path.")
    parser.add_argument("-i", "--image_path", type=str, help="The path to the input image.", required=True)
    args = parser.parse_args()
    
    # Run the model with the given image path
    caption = run_model(args.image_path)
    print("Generated caption:", caption)