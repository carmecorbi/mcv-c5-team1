import src.scripts.llm as llm
import src.scripts.anymodal as anymodal
import torch
import src.scripts.vision as vision

from PIL import Image


llm_tokenizer, llm_model = llm.get_llm(
        "meta-llama/Llama-3.2-3B", 
        access_token='HF_ACCESS_TOKEN'
    )

llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=True)

# Initialize vision tokenizer and encoder
vision_encoder = vision.VisionEncoder(vision_model)
vision_tokenizer = vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1)

# model_path = "/ghome/c5mcv01/mcv-c5-team1/week4/results/vit_llama3_2_1B/best_model.pt"
model_path = "/ghome/c5mcv01/mcv-c5-team1/week4/results/vit_llama3_2_3B/checkpoints/best_model.pt"


multimodal_model = anymodal.MultiModalModel(
        input_processor=None,
        input_encoder=vision_encoder,
        input_tokenizer=vision_tokenizer,
        language_tokenizer=llm_tokenizer,
        language_model=llm_model,
        lm_peft = llm.add_peft,
        prompt_text="The description of the given image is: ")
multimodal_model.language_model = llm.add_peft(multimodal_model.language_model)
multimodal_model._load_model(model_path)
multimodal_model.eval()

# Llista de rutes locals de les imatges
local_images = [
    {"path": "/ghome/c5mcv01/mcv-c5-team1/week3/data/images/asian-turkey-noodle-soup-with-ginger-and-chiles-231068.jpg"},
    {"path": "/ghome/c5mcv01/mcv-c5-team1/week3/data/images/cranberry-tangerine-conserve-350581.jpg"},
    {"path": "/ghome/c5mcv01/mcv-c5-team1/week3/data/images/roasted-hot-honey-shrimp-with-bok-choy-and-kimchi-rice-51261050.jpg"},
    {"path": "/ghome/c5mcv01/mcv-c5-team1/week3/data/images/perigord-walnut-tart-352880.jpg"}
]

# Processar cada imatge local
for idx, image_data in enumerate(local_images):
    # Obrir la imatge local
    img = Image.open(image_data["path"]).convert("RGB")
    print(f"[✅] Imatge {idx} carregada correctament: {image_data['path']}")

    # Processar la imatge amb el processador del model
    image = image_processor(img, return_tensors="pt")
    image = {key: val.squeeze(0) for key, val in image.items()}  # Treure dimensió batch

    # Generar el caption
    generated_caption = multimodal_model.generate(image, max_new_tokens=120)
    print(f"Generated Caption: {generated_caption}\n")
