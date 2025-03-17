import torch
import requests
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

# Load image
#image1 = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/test/images/angular_leafspot411.jpg"
image1 = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/test/images/anthracnose_fruit_rot104.jpg"
#image1 = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/test/images/blossom_blight190.jpg"
#image1 = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/test/images/gray_mold490.jpg"
#image1 = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/test/images/leaf_spot557.jpg"
#image1 = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/test/images/powdery_mildew_fruit165.jpg"
#image1 = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/strawberry-disease-detection-dataset/test/images/powdery_mildew_leaf468.jpg"
image = Image.open(image1)

# Load model and image processor
device = "cuda"
checkpoint = "/ghome/c5mcv01/mcv-c5-team1/week2/src/domain_shift/finetune-instance-segmentation-mini-mask2former_augmentation_default"

model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, device_map=device)
image_processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)

# Run inference on image
inputs = image_processor(images=[image], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Post-process outputs
outputs = image_processor.post_process_instance_segmentation(outputs, target_sizes=[(image.height, image.width)])

print("Mask shape: ", outputs[0]["segmentation"].shape)
print("Mask values: ", outputs[0]["segmentation"].unique())
for segment in outputs[0]["segments_info"]:
    print("Segment: ", segment)


segmentation = outputs[0]["segmentation"].numpy()

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(np.array(image))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(segmentation)
plt.axis("off")
plt.savefig("segmentation_plot2.png", bbox_inches='tight')
plt.close()