import torch
import os
import pandas as pd

from src.tokenizer import Tokenizer
from src.scripts import vision, llm, anymodal
from src.metrics.metrics import Metric
from tqdm import tqdm



def evaluate(dataset: torch.utils.data.Dataset, tokenizer: Tokenizer, model: anymodal.MultiModalModel) -> list:
    """
    Avalua el model sobre un conjunt complet de dades.

    Args:
        dataset (torch.utils.data.Dataset): Dataset on cada mostra és un diccionari que conté:
            - 'input': la imatge processada per al model.
            - 'text': la descripció veritable de la imatge (si està disponible).
            - 'image': la imatge en RGB per si cal realitzar alguna operació addicional.
        tokenizer (Tokenizer): Tokenizer per desxifrar les etiquetes o descripcions.
        model (anymodal.MultiModalModel): El model multimodal per generar les descripcions.

    Retorna:
        List[dict]: Una llista de diccionaris, cada un conté:
            - 'image_path': el camí de la imatge.
            - 'generated_caption': la descripció generada per `model.generate`.
            - 'ground_truth': la descripció veritable si està disponible.
    """
    metric = Metric()  # Assumim que tens una funció per a les mètriques
    metrics_sum = {}
    num_samples = 0

    # Iterem pel dataset
    for idx in tqdm(range(len(dataset))):
        # Obtenir la mostra del dataset
        item = dataset[idx]
        image_input = item['input']  # La imatge processada
        ground_truth = item['text']  # La descripció veritable

        # Generar la descripció amb el model multimodal
        generated_caption = model.generate(image_input, max_new_tokens=120)

        # Càlcul de les mètriques (com comparant la descripció generada amb la veritable)
        result = metric([generated_caption], [[ground_truth]])

        # Acumular les mètriques
        for key, value in result.items():
            metrics_sum[key] = metrics_sum.get(key, 0) + value

        num_samples += 1

    # Promediar les mètriques
    averaged_metrics = {key: value / num_samples for key, value in metrics_sum.items()}
    return averaged_metrics

# Carregar el model i el tokenizer
llm_tokenizer, llm_model = llm.get_llm(
    "meta-llama/Llama-3.2-1B", 
    access_token='HF_ACCESS_TOKEN'
)

llm_hidden_size = llm.get_hidden_size(llm_tokenizer, llm_model)

# Processador d'imatges i model de visió
image_processor, vision_model, vision_hidden_size = vision.get_image_encoder('google/vit-base-patch16-224', use_peft=True)
vision_encoder = vision.VisionEncoder(vision_model)
vision_tokenizer = vision.Projector(vision_hidden_size, llm_hidden_size, num_hidden=1)

# Carregar el model multimodal
multimodal_model = anymodal.MultiModalModel(
    input_processor=None,
    input_encoder=vision_encoder,
    input_tokenizer=vision_tokenizer,
    language_tokenizer=llm_tokenizer,
    language_model=llm_model,
    lm_peft=llm.add_peft,
    prompt_text="The description of the given image is: "
)
multimodal_model.language_model = llm.add_peft(multimodal_model.language_model)
# Càrrega del model multimodal
model_path = "/ghome/c5mcv01/mcv-c5-team1/week4/results/vit_llama3_2_1B/checkpoints/best_model.pt"
#model_path = "/ghome/c5mcv01/mcv-c5-team1/week4/results/vit_llama3_2_3B/checkpoints/best_model.pt"
multimodal_model._load_model(model_path)
multimodal_model.eval()

# Llegir el dataset
data_dir = "/ghome/c5mcv01/mcv-c5-team1/week3/data"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

partitions = {
            'train': list(train_df.index),
            'val': list(val_df.index),
            'test': list(test_df.index)
        }


test_dataset = vision.ImageDataset(test_df, "/ghome/c5mcv01/mcv-c5-team1/week3/data/images", image_processor)
train_dataset = vision.ImageDataset(train_df, "/ghome/c5mcv01/mcv-c5-team1/week3/data/images", image_processor)
val_dataset = vision.ImageDataset(val_df, "/ghome/c5mcv01/mcv-c5-team1/week3/data/images", image_processor)
# Avaluar el model sobre el conjunt de test
averaged_metrics = evaluate(train_dataset, llm_tokenizer, multimodal_model)
print(f"Averaged Metrics train: {averaged_metrics}")
#averaged_metrics = evaluate(test_dataset, llm_tokenizer, multimodal_model)
#averaged_metrics2 = evaluate(val_dataset, llm_tokenizer, multimodal_model)
#print(f"Averaged Metrics test: {averaged_metrics}")
#print(f"Averaged Metrics validation: {averaged_metrics2}")