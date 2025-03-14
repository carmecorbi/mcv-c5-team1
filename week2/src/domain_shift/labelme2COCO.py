"""
This code converts instance segmentation datasets annotated with LabelMe to a per-image TXT file format.
Each TXT file (one per image) contains lines with:
    image_id instance_id class_id img_height img_width rle
The RLE is obtained from the pycocotools encoding.
"""

#!/usr/bin/env python
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import numpy as np
import PIL.Image
import labelme
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as maskUtils

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)



def visualize_encoded_mask(img, rle_str, output_path, alpha=0.5, cmap='jet'):
    """
    Visualizes an RLE-encoded mask over an image.

    Args:
        img (np.ndarray): The image as a NumPy array (H x W x 3).
        rle_str (str): The RLE encoded mask string (from pycocotools).
        alpha (float): Transparency of the mask overlay.
        cmap (str): Colormap used for the mask.
    
    Returns:
        None: Displays the image with the mask overlay.
    """
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Reassemble the RLE dictionary.
    # Ensure that counts is a bytes object.
    if isinstance(rle_str, str):
        counts = rle_str.encode('utf-8')
    else:
        counts = rle_str
    rle = {
        'counts': counts,
        'size': [height, width]
    }
    
    # Decode the RLE mask
    decoded_mask = maskUtils.decode(rle)
    print("Decoded mask shape:", decoded_mask.shape)
    print("Unique values in decoded mask:", np.unique(decoded_mask))
    
    # Visualize the image with the mask overlay
    plt.figure(figsize=(8, 8))
    #plt.imshow(img)
    plt.imshow(decoded_mask, alpha=alpha)
    plt.title("RLE Mask Visualization")
    plt.axis('off')
    
    # Save the visualization
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def labelme2RLE_Txt():
    # Define the sets you want to process.
    sets = ['val', 'test', 'train']
    output_dir = '/home/usuaris/imatge/judit.salavedra/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/gt'
    # Remove output directory if it exists.
    if osp.exists(output_dir):
        print('Output directory already exists:', output_dir)
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print('Creating dataset in:', output_dir)

    # Build a mapping from class names to IDs (skipping the background).
    class_name_to_id = {}
    labels_file = '/home/usuaris/imatge/judit.salavedra/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/versions/1/labels.txt'
    for i, line in enumerate(open(labels_file).readlines()):
        class_id = i  # Starting from 0, but you can adjust if necessary.
        class_name = line.strip()
        # Skip the background class (assumed to be '_background_')
        if class_id == 0:
            assert class_name == '_background_'
            continue
        class_name_to_id[class_name] = class_id

    # Process each set separately.
    for set in sets:
        # Input directory for a given set.
        input_dir = '/home/usuaris/imatge/judit.salavedra/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/versions/1/%s' % (set)
        # Create a subfolder in output for this set.
        output_set_dir = osp.join(output_dir, set)
        os.makedirs(output_set_dir, exist_ok=True)

        # Get all label JSON files from the 'label' subfolder.
        label_files = glob.glob(osp.join(input_dir, 'label', '*.json'))
        print(f"Processing set '{set}' with {len(label_files)} label files.")

        # Process each image (i.e. each label file).
        for image_idx, label_file in enumerate(label_files):
            with open(label_file) as f:
                label_data = json.load(f)
            # Get the image file name from label_data.
            path_parts = label_data['imagePath'].split("/")
            img_filename = path_parts[-1]
            # Construct full image path.
            img_file = '/home/usuaris/imatge/judit.salavedra/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/versions/1/%s/images/%s' % (set, img_filename)
            print(f"Processing image: {img_file}")
            try:
                img = np.asarray(PIL.Image.open(img_file))
            except Exception as e:
                print(f"Error opening image {img_file}: {e}")
                continue

            img_height, img_width = img.shape[:2]
            image_id = image_idx + 1  # Image id starting at 1.
            instance_counter = 1      # Instance id for each shape in the image.

            # Create an output TXT file for this image.
            # Here we use the image filename (without extension) for the TXT filename.
            base_filename = osp.splitext(img_filename)[0]
            out_txt_file = osp.join(output_set_dir, base_filename + '.txt')
            with open(out_txt_file, 'w') as fout:
                # Process each shape (each instance) in the label file.
                for shape in label_data['shapes']:
                    points = shape['points']
                    label = shape['label']
                    shape_type = shape.get('shape_type', None)
                    # Compute a binary mask for the instance.
                    mask = labelme.utils.shape_to_mask(img.shape[:2], points, shape_type)

                    # Get the class name (assume label might contain extra info separated by '-' and take the first part).
                    cls_name = label.split('-')[0]
                    if cls_name not in class_name_to_id:
                        print(f"Warning: class '{cls_name}' not in mapping, skipping instance.")
                        continue
                    cls_id = class_name_to_id[cls_name]

                    # Prepare mask for RLE encoding.
                    mask_for_encode = np.asfortranarray(mask.astype(np.uint8))
                    rle = pycocotools.mask.encode(mask_for_encode)
                    # pycocotools returns rle['counts'] as bytes; decode it if necessary.
                    if isinstance(rle['counts'], bytes):
                        rle_str = rle['counts'].decode('utf-8')
                    else:
                        rle_str = rle['counts']
                    # Write a line in the TXT file.
                    # Format: image_id instance_id class_id img_height img_width rle
                    line = f"{image_id} {instance_counter} {cls_id} {img_height} {img_width} {rle_str}\n"
                    fout.write(line)
                    instance_counter += 1

            print(f"Finished processing image {img_filename}, wrote {instance_counter - 1} instances to {out_txt_file}")
        print(f"Set '{set}' is done.")

if __name__ == '__main__':
    
    # Do the format conversion:
    #labelme2RLE_Txt()

    '''
    # Visualize the instance segm,entation mask over the image:
    # Load an example image (adjust the path as needed)
    img_path = '/home/usuaris/imatge/judit.salavedra/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/versions/1/train/images/anthracnose_fruit_rot50.jpg'
    img = np.asarray(PIL.Image.open(img_path))
    
    # Example RLE string (should come from your TXT file)
    # rle_str = "eVnX35b2N3M4M3N3M..."  # Replace with your actual RLE string
    rle_str = "Sbf58^;2L3O1O1O100000000O102N1O1O000000000000100O01O3M1O1O3L6KadU8"  # Placeholder
    output_path = "./rle_images/anthracnose_fruit_rot50.jpg"

    print(f"rle: {rle_str}")
    visualize_encoded_mask(img, rle_str[-1], output_path)
    '''
