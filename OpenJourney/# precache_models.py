# precache_models.py
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, CLIPTokenizerFast
import torch

MODEL_IDS = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "Prompthero/openjourney-v4",
    "dreamlike-art/dreamlike-photoreal-2.0",
]
FLAN = "google/flan-t5-small"
CLIP = "openai/clip-vit-base-patch32"

device = "cuda" if torch.cuda.is_available() else "cpu"

for mid in MODEL_IDS:
    print("Downloading model pipeline:", mid)
    # This will pull the weights & tokenizer into cache via diffusers
    try:
        pipe = StableDiffusionPipeline.from_pretrained(mid, torch_dtype=torch.float32)
    except Exception as e:
        print("Warning: failed to load pipeline directly (some models need extra params):", e)
    # delete to free memory if loaded
    try:
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
    except:
        pass

print("Downloading paraphraser and CLIP tokenizer...")
AutoModelForSeq2SeqLM.from_pretrained(FLAN)
AutoTokenizer.from_pretrained(FLAN)
CLIPTokenizerFast.from_pretrained(CLIP)

print("Done. Models are cached locally.")
