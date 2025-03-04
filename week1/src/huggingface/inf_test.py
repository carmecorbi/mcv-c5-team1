import os
import cv2
import matplotlib.pyplot as plt

def draw_bboxes(image_path, annotation_path, output_path, target_frame=0):
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return
    
    # Leer las anotaciones
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
    
    # Dibujar los bounding boxes solo para el frame objetivo
    for line in annotations:
        parts = line.strip().split(', ')
        if len(parts) < 8:
            continue  # Evitar líneas incorrectas
        
        frame_id, _, class_id, x1, y1, x2, y2, _ = parts
        frame_id = int(frame_id)
        class_id = int(class_id)  # Convertir a entero para comparaciones
        print(class_id)
        
        if frame_id != target_frame or class_id == 10:
            continue  # Ignorar si no es el frame correcto o si es clase 10
        
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Asignar color según la clase (3 -> coche en verde, 1 -> peatón en rojo)
        if class_id == 3:
            color = (0, 255, 0)  # Verde para coches
        elif class_id == 1:
            color = (0, 0, 255)  # Rojo para peatones
        else:
            continue  # Si hay otra clase, ignorarla por seguridad
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Guardar la imagen con bounding boxes
    cv2.imwrite(output_path, image)
    
    # Mostrar la imagen
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Rutas de archivos
image_path = "/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/results_with_bboxes.png"
annotation_path = "/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/results_txt/0014.txt"
output_path = "/ghome/c3mcv02/mcv-c5-team1/week1/src/huggingface/results_with_bboxes_inf.png"

draw_bboxes(image_path, annotation_path, output_path, target_frame=0)


