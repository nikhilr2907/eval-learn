import json
from pathlib import Path
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from diffusers import DiffusionPipeline
from dotenv import load_dotenv
from huggingface_hub import login
import os
import torch

def diffusion_model_pipeline():
# Using the "sld pipline" in the unlearning_techniques repo
# to obtain the diffusion model pipeline code

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
    safety_checker=None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    # Reduce memory usage
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe = pipe.to(device)

    return pipe

def generate_forget_images(forget_data_filepath, output_file_directory, sld_config = SafetyConfig.STRONG, pipe = None):
    """
    Using the forget_data prompts, SLD identifies the category to unlearn
    The diffusion model is unlearnt from that category
    And based on the prompts given in data_forget.json, the unlearnt model generates images
    these generated images will be used for CLIPscore calculation
    """
    if pipe is None: # if diffusion pipeline is not initialised we load it
        pipe = diffusion_model_pipeline()
    
    # navigate to the file path containing the data_forget.json and open it
    with open(forget_data_filepath, "r") as f:
        forget_data = json.load(f)

    # Creat the output directory if it does not exist
    Path(output_file_directory).mkdir(parents=True, exist_ok=True)

    generated_images = []
    # Iterate through the forget prompts and identify the category to unlearn
    # afterwhich apply the unlearning SLD technique
    # after unlearning, apply the prompts to generate images
    for entry in forget_data:
        unlearnt_image = pipe(prompt = entry['prompt'],**sld_config).images[0]
        # create a file path to save the unlearnt image and save the image
        image_path = f"{output_file_directory}/unlearnt_image_{entry['image_id']}.png"
        unlearnt_image.save(image_path)
        # the last key pair shows the directory of where the unlearnt images are
        generated_images.append({"image_id": entry['image_id'], "prompt": entry['prompt'], "image_path": image_path})
    # Save the generated images list containing the images filepath
    output_path = f"{output_file_directory}/generated_unlearnt_images.json"
    # write into a json file and create this file
    with open(output_path, "w") as f:
        json.dump(generated_images,f, indent=4)
    
    return output_path
def generate_normal_images(normal_data_filepath, output_file_directory, pipe = None):
    """
    Generate images from normal data set
    """
    if pipe is None: # if diffusion pipeline is not initialised we load it
        pipe = diffusion_model_pipeline()
    
    # navigate to the file path containing the data_forget.json and open it
    with open(normal_data_filepath, "r") as f:
        normal_data = json.load(f)

    # Creat the output directory if it does not exist
    Path(output_file_directory).mkdir(parents=True, exist_ok=True)

    generated_images = []
    # Iterate through the normal prompts and generate images
    for entry in normal_data:
        # for normal image generation, we disable SLD completely by using guidance scale = 0
        normal_image = pipe(prompt = entry['prompt'], sld_guidance_scale = 0).images[0]
        # create a file path to save the unlearnt image and save the image
        image_path = f"{output_file_directory}/normal_image_{entry['image_id']}.png"
        normal_image.save(image_path)
        # the last key pair shows the directory of where the images are
        generated_images.append({"image_id": entry['image_id'], "prompt": entry['prompt'], "image_path": image_path})
    # Save the generated images list containing the images filepath
    output_path = f"{output_file_directory}/generated_normal_images.json"
    # write into a json file and create this file
    with open(output_path, "w") as f:
        json.dump(generated_images,f, indent=4)
    
    return output_path

if __name__ == "__main__":
    pipe = diffusion_model_pipeline()
    # run the SLD unlearning technique with our forget prompts
    #forgetfilepath = "/Users/ongkaisheng/Desktop/ImperialCollege/MSc Group Project/eval-learn/results/clipscore/forget/data_forget.json"
    forgetfilepath = "results/clipscore/forget/data_forget.json"
    #outputforgetdir = "/Users/ongkaisheng/Desktop/ImperialCollege/MSc Group Project/eval-learn/results/clipscore/forget"
    outputforgetdir = "results/clipscore/forget"
    generate_forget_images(forget_data_filepath= forgetfilepath,
                    output_file_directory=outputforgetdir,
                    sld_config=SafetyConfig.STRONG,
                    pipe=pipe)
    #normalfilepath = "/Users/ongkaisheng/Desktop/ImperialCollege/MSc Group Project/eval-learn/results/clipscore/normal/data_normal.json"
    normalfilepath = "results/clipscore/normal/data_normal.json"
    #outputnormaldir = "/Users/ongkaisheng/Desktop/ImperialCollege/MSc Group Project/eval-learn/results/clipscore/normal"
    outputnormaldir = "results/clipscore/normal"
    generate_normal_images(normal_data_filepath= normalfilepath,
                    output_file_directory=outputnormaldir,
                    pipe=pipe)
