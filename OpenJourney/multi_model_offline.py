# multi_model_offline.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import CLIPTokenizerFast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paraphraser (optional) to shorten prompts for CLIP
FLAN_MODEL = "google/flan-t5-small"
CLIP_TOKENIZER = "openai/clip-vit-base-patch32"

# Models to run — change to any models you have cached locally
MODEL_IDS = [
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1",
    "Prompthero/openjourney-v4",
    "andite/anything-v4.0",
]

# Generation settings
MAX_CLIP_TOKENS = 77
LOCAL_ONLY = True   # <-- Set True for strict offline (will error if model not cached). Set False for first-run downloads.
OUTPUT_DIR = "multi_model_outputs"
INIT_IMAGE_PATH = None  # "input.png" to enable img2img; or None for txt2img
IMG2IMG_STRENGTH = 0.6
NUM_INFERENCE_STEPS = 28
NEGATIVE_PROMPT = "text, watermark, low-res, deformed, duplicate"
# ------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load paraphraser & CLIP tokenizer (for token counting) ---
print("Loading paraphraser and CLIP tokenizer (local_only=%s)..." % LOCAL_ONLY)
paraphraser_tok = AutoTokenizer.from_pretrained(FLAN_MODEL, local_files_only=LOCAL_ONLY)
paraphraser = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL, local_files_only=LOCAL_ONLY).to(DEVICE)
clip_tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_TOKENIZER, local_files_only=LOCAL_ONLY)

def clip_token_count(text: str) -> int:
    toks = clip_tokenizer(text, return_tensors="pt", padding=False, truncation=False)
    return toks.input_ids.shape[-1]

# Simple T5 paraphrase/shorten helper (deterministic)
def shorten_prompt(original_prompt: str, max_tokens: int = MAX_CLIP_TOKENS, max_passes: int = 2) -> str:
    from transformers import pipeline
    text2text = pipeline("text2text-generation", model=paraphraser, tokenizer=paraphraser_tok,
                         device=0 if DEVICE == "cuda" else -1)
    template = (
        f"Shorten the following image-generation prompt to fit within {max_tokens} CLIP tokens. "
        "Keep essential visual details (objects, placement, materials, style). Drop verbosity.\n\n"
        f"Original: '''{original_prompt}'''"
    )
    out = text2text(template, max_length=256, do_sample=False)[0]["generated_text"].strip()
    passes = 1
    while clip_token_count(out) > max_tokens and passes < max_passes:
        aggressive = (
            "Aggressively compress while preserving core visual elements (objects, placement, materials, style). "
            f"Prompt: '''{out}'''"
        )
        out = text2text(aggressive, max_length=200, do_sample=False)[0]["generated_text"].strip()
        passes += 1
    if clip_token_count(out) > max_tokens:
        toks = clip_tokenizer(out, return_tensors="pt")["input_ids"][0].tolist()
        out = clip_tokenizer.decode(toks[:max_tokens], skip_special_tokens=True).strip()
    return out

# --- Model generation function ---
def generate_for_model(model_id: str, prompt: str, out_name: str):
    print(f"\nLoading pipeline for model: {model_id} (local_only={LOCAL_ONLY})")
    try:
        if INIT_IMAGE_PATH:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=LOCAL_ONLY)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=LOCAL_ONLY)
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        return False

    pipe = pipe.to(DEVICE)
    # memory helpers
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_model_cpu_offload"):
        # optional: if installed with accelerate and available
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    gen_kwargs = dict(prompt=prompt, negative_prompt=NEGATIVE_PROMPT, num_inference_steps=NUM_INFERENCE_STEPS)
    if INIT_IMAGE_PATH:
        init_img = Image.open(INIT_IMAGE_PATH).convert("RGB")
        gen = pipe(init_image=init_img, strength=IMG2IMG_STRENGTH, **gen_kwargs)
    else:
        gen = pipe(**gen_kwargs)

    image = gen.images[0]
    save_path = os.path.join(OUTPUT_DIR, out_name)
    image.save(save_path)
    print("Saved ->", save_path)

    # free memory
    del pipe
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return True

if __name__ == "__main__":
    user_prompt = input("Enter user prompt: ").strip()
    print("Original CLIP token count:", clip_token_count(user_prompt))
    optimized = shorten_prompt(user_prompt)
    print("Optimized prompt:", optimized)
    print("Optimized CLIP token count:", clip_token_count(optimized))

    for idx, mid in enumerate(MODEL_IDS, start=1):
        safe_name = mid.replace("/", "_").replace(" ", "_")
        outfile = f"{idx:02d}_{safe_name}.png"
        ok = generate_for_model(mid, optimized, outfile)
        if not ok:
            print(f"Skipping model {mid} due to load error.")

    print("\nAll done. Outputs in:", os.path.abspath(OUTPUT_DIR))
