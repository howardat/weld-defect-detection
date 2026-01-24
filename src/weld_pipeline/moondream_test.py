from transformers import AutoModelForCausalLM
from PIL import Image
import torch

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda", # "cuda" on Nvidia GPUs
)

# Load your image
image = Image.open("C:\\Users\\User\\Documents\\Olzhas\\weld-defect-detection\\data\\img\\pores2.jpg")

# Optionally set sampling settings
settings = {"temperature": 0.5, "max_tokens": 768, "top_p": 0.3}

# Generate a short caption
short_result = model.caption(
    image, 
    length="short", 
    settings=settings
)
print(short_result)