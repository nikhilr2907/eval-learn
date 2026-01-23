from diffusers import DiffusionPipeline
import torch
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import os

# Login to Hugging Face
try:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
    print("Logged in to Hugging Face Hub.")
except Exception as e:
    print(f"Could not log in to Hugging Face Hub: {e}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the I2P dataset
print("Loading I2P dataset...")
i2p_dataset = load_dataset("AIML-TUDA/i2p", split="train")
print(f"Loaded {len(i2p_dataset)} prompts from I2P dataset.")

# Load the pipeline
pipe = DiffusionPipeline.from_pretrained(
    "AIML-TUDA/stable-diffusion-safe",
    safety_checker=None,
    require_safety_checker=False,
).to(device)

# Check the configuration of the U-Net component
print(f"Base Model Path: {pipe.unet.config.get('_name_or_path')}")
print(f"Sample Size: {pipe.unet.config.sample_size}")

# Define the SLD levels to test
sld_levels = {
    "no_sld": {"sld_guidance_scale": 0},
    "weak": SafetyConfig.WEAK,
    "medium": SafetyConfig.MEDIUM,
    "strong": SafetyConfig.STRONG,
    "max": SafetyConfig.MAX,
}

# Create output directories
os.makedirs("results/sld", exist_ok=True)
for level_name in sld_levels.keys():
    os.makedirs(f"results/sld/{level_name}", exist_ok=True)

# Iterate over I2P prompts and generate images at each SLD level
num_prompts = len(i2p_dataset)  # Set to smaller number for testing, e.g., 10

for idx, item in enumerate(i2p_dataset):
    if idx >= num_prompts:
        break

    prompt = item["prompt"]
    category = item.get("category", "unknown")
    print(f"\n[{idx+1}/{num_prompts}] Prompt: {prompt[:50]}... | Category: {category}")

    for level_name, config in sld_levels.items():
        print(f"  Generating with SLD level: {level_name}")

        image = pipe(
            prompt=prompt,
            **config
        ).images[0]

        # Save the image with prompt index and SLD level
        image.save(f"results/sld/{level_name}/prompt_{idx:04d}.png")

print("\nGeneration complete.")


