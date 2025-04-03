import os
import pandas as pd


def load_data(csv_path):
    """Load data from .csv file and preprocess it"""
    data = pd.read_csv(csv_path)
    df = data[['Image_Name', 'Title']]
    
    # Remove rows with NaN values in either column
    df = df.dropna(subset=['Image_Name', 'Title'])
    
    # Remove rows with empty strings in either column
    df = df[(df['Image_Name'] != '') & (df['Title'] != '')]
    
    # Filter out rows with '#NAME?'
    df = df[df['Image_Name'] != '#NAME?']  # Filter out rows with '#NAME?'
    
    # Set title to string type
    df['Title'] = df['Title'].astype(str)
    return df


csv_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/raw_data.csv'
df = load_data(csv_path)

df2 = df.drop_duplicates(subset=['Title'], keep='first')
df3 = df.drop_duplicates(subset=['Image_Name'], keep='first')

images_with_text_dir = '/ghome/c5mcv01/mcv-c5-team1/week4/src/dataset/images_with_text'

# Obtenir la llista de noms d'imatges a la carpeta "images_with_text"
image_files = [f for f in os.listdir(images_with_text_dir) if f.endswith('.jpg')]

# Extreure només els noms de les imatges (sense extensió)
image_names_with_text = [os.path.splitext(f)[0] for f in image_files]

# Filtrar el DataFrame per eliminar les files que tenen imatges a "images_with_text"
df_cleaned = df[~df['Image_Name'].isin(image_names_with_text)]

# Guardar el nou DataFrame a un fitxer CSV
cleaned_csv_path = '/ghome/c5mcv01/mcv-c5-team1/week4/src/dataset/cleaned.csv'
df_cleaned.to_csv(cleaned_csv_path, index=False)

print(f"Mida del DataFrame original: {df.shape}") 
print(f"Mida del DataFrame netejat: {df_cleaned.shape}")
'''
duplicates = df[df.duplicated(subset=['Title'], keep=False)]
for title, group in duplicates.groupby('Title'):
    # Imprimir els 'image_name' de les files amb el mateix 'Title' en una mateixa línia
    image_names = group['Image_Name'].tolist()
    print(f"Title: {title} -> Image Names: {', '.join(image_names)}")
'''
'''
# Defineix el directori on es troben les imatges
img_path = '/ghome/c5mcv01/mcv-c5-team1/week3/data/images'  # Modifica aquesta ruta segons sigui necessari

reader = easyocr.Reader(['en'])

# Funció per detectar text a la imatge amb EasyOCR
def contains_text(image_path):
    try:
        # Construir el camí complet de la imatge
        
        
        # Aplicar OCR per extreure el text de la imatge
        result = reader.readtext(image_path)

        # Si hi ha text detectat, retornem True
        return bool(result)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

df_with_text_images = df[df['Image_Name'].apply(lambda img_name: contains_text(f'{img_path}/{img_name}.jpg'))]

# Guardar els noms de les imatges amb text
image_names_with_text = df_with_text_images['Image_Name'].tolist()

# Imprimir els noms de les imatges amb text
print("Imatges amb text:")
for img_name in image_names_with_text:
    print(img_name)

output_dir = 'images_with_text'
os.makedirs(output_dir, exist_ok=True)

for img_name in image_names_with_text:
    img_path_full = f"{img_path}/{img_name}.jpg"
    if os.path.exists(img_path_full):
        shutil.copy(img_path_full, f"{output_dir}/{img_name}.jpg")
    else:
        print(f"Imatge no trobada: {img_path_full}")

'''

