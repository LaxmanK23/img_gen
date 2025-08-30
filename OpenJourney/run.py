from diffusers import StableDiffusionPipeline
import torch

# Use OpenJourney model from Hugging Face
model_id = "Prompthero/openjourney-v4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")  # Ensure it's running on CPU

prompt = "A dreamy landscape with ethereal lighting and surreal colors, like MidJourney style"
image = pipe(prompt).images[0]

# Save the image to disk
image.save("output_openjourney.png")
print("Image saved as output_openjourney.png")
