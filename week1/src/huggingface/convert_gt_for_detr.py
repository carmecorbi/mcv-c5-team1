import os
import re
from pycocotools import mask as maskUtils

# Directorios
data_dir = "/ghome/c3mcv02/mcv-c5-team1/data/instances_txt"
output_dir = "/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/gt_annotations"
os.makedirs(output_dir, exist_ok=True)

# Mapeo de clases KITTI MOTS -> COCO
class_mapping = {1: 3, 2: 1}  # 1 -> car (3 en COCO), 2 -> pedestrian (1 en COCO)
ignored_class = 10  # Clase a ignorar

# Expresión regular para extraer los valores
pattern = re.compile(r"(\d+) (\d+) (\d+) (\d+) (\d+) (.+)")

def process_file(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            match = pattern.match(line)
            if match:
                frame_id, object_id, class_id, height, width, rle = match.groups()
                frame_id, object_id, class_id = map(int, [frame_id, object_id, class_id])
                height, width = map(int, [height, width])
                
                # Ignorar clase 10
                if class_id == ignored_class:
                    continue
                
                # Mapear clase de KITTI MOTS a COCO
                new_class_id = class_mapping.get(class_id, class_id)
                
                # Decodificar RLE y obtener el bounding box
                rle_obj = {'counts': rle, 'size': [height, width]}
                bbox = maskUtils.toBbox(rle_obj)
                x, y, w, h = map(int, bbox)
                
                # Escribir en nuevo formato
                outfile.write(f"{frame_id}, {object_id}, {new_class_id}, {x}, {y}, {x + w}, {y + h}, 1.0\n")

# Procesar todos los archivos
def main():
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(data_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_file(input_path, output_path)
            print(f"Procesado: {filename}")
    print("Conversión completada.")

if __name__ == "__main__":
    main()
