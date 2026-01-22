from diffusers import DiffusionPipeline
import torch
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from dotenv import load_dotenv
from huggingface_hub import login
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

# Load the pipeline
pipe = DiffusionPipeline.from_pretrained(
    "AIML-TUDA/stable-diffusion-safe",
).to(device)


# Check the configuration of the U-Net component
print(f"Base Model Path: {pipe.unet.config.get('_name_or_path')}")

# Check the sample size (often differentiates v1 vs v2/XL)
print(f"Sample Size: {pipe.unet.config.sample_size}") 
# 64 usually indicates SD v1.4/v1.5 (64 * 8 = 512px)
# 96 usually indicates SD v2 (768px)


prompt = "astronaut riding a horse in space, digital art" 

# 1. Define the SLD levels you want to test
# "weka" is interpreted as WEAK. 
sld_levels = {
    "No SLD": {"sld_guidance_scale": 0}, # Disable SLD
    "Weak": SafetyConfig.WEAK,
    "Medium": SafetyConfig.MEDIUM,
    "Strong": SafetyConfig.STRONG,
    "Max": SafetyConfig.MAX,
}

# 2. Iterate and generate
for level_name, config in sld_levels.items():
    print(f"Generating for level: {level_name}")
    
    # We unpack the configuration dictionary (e.g., **SafetyConfig.WEAK) 
    # directly into the pipe call.
    image = pipe(
        prompt=prompt,
        **config 
    ).images[0]
    
    # Save the image
    image.save(f"results/sld/generated_image_{level_name.lower().replace(' ', '_')}.png")

print("Generation complete.")


