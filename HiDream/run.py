from diffusers import DiffusionPipeline
import torch

# Load HiDream Fast model from Hugging Face
model_id = "HiDream-ai/HiDream-I1-Fast"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)  # CPU only
pipe = pipe.to("cpu")

prompt = "a beautiful futuristic city at sunset, highly detailed, cinematic lighting"
image = pipe(prompt).images[0]

image.save("output.png")
print("Image saved as output.png")
