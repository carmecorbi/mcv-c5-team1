from src.models.vit_gpt2 import Vit_GPT2

print("Fully unfrozen parameters:")
fully_unfrozen = Vit_GPT2()
fully_unfrozen.print_parameters()
print("-" * 50)

print("Frozen encoder parameters:")
frozen_encoder = Vit_GPT2(freeze_vit=True)
frozen_encoder.print_parameters()
print("-" * 50)

print("Frozen decoder parameters")
frozen_decoder = Vit_GPT2(freeze_gpt2=True)
frozen_decoder.print_parameters()
print("-" * 50)