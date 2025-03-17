from transformers import Mask2FormerForUniversalSegmentation
from fvcore.nn import FlopCountAnalysis
import torch

model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params / 1e6:.1f}M")  # En millones

dummy_input = torch.randn(1, 3, 512, 512)  # Imagen de prueba
flops = FlopCountAnalysis(model, dummy_input)
print(f"Total FLOPs: {flops.total() / 1e9:.1f}B")  # En miles de millones
