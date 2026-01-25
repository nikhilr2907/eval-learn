import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from unlearning_techniques.SLD import pipe, device
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from evaluation_metrics.distance_metrics.fid import calculate_fid_from_paths
from PIL import Image
from datasets import load_dataset
import torch

def load_coco_hf(split="validation", num_samples=None):
    """
    Load COCO dataset from Hugging Face

    Args:
        split: Dataset split into validation by default
        num_samples: number of samples to load (default to all : None)

    Returns:
        List of dictionaries with caption and image
    """
    try:
        # Use the new COCO dataset format
        dataset = load_dataset("nielsr/coco-captions", split=split, streaming=False)

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        samples = []
        for item in dataset:
            # The new format has 'caption' and 'image' fields
            samples.append({
                'caption' : item.get('caption', ''),
                'image_id' : item.get('image_id', None),
                'image' : item.get('image', None)
            })
        return samples
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
def create_prompt_for_SLD(coco_samples, max_prompts=100):
    """
    Create list of prompts from COCO dataset

    Args:
        coco_samples : List of Coco dataset
        max_prompts : Maxmium number of prompts to return

    Returns:
        List of caption strings
    """
    if not coco_samples:
        return []
    
    prompts = [sample['caption'] for sample in coco_samples[:max_prompts]]
    return prompts

def generate_baseline_image(prompts, output, seed=42):
    """
    Generate Image without SLD

    Args:
        prompts: List of prompts
        output: Directory to save image
        seed: For reproducibility

    Returns:
        List of paths to generated image
    """
    os.makedirs(output, exist_ok=True)
    image_paths = []

    for i, prompt in enumerate(prompts):
        output_path = os.path.join(output, f"baseline_{i}.png")

        try:
            generator = torch.Generator(device=device).manual_seed(seed + i)
            image = pipe(
                prompt=prompt,
                sld_guidance_scale=0,  # Disable SLD
                generator=generator
            ).images[0]
            image.save(output_path)
            image_paths.append(output_path)
        
        except Exception as e:
            print(f"Error generating image {i}: {e}")
    
    return image_paths

def generate_sld_image(prompts, safety_config, output, seed=42):
    """
    Generate Image with SLD

    Args:
        prompts : List of prompts
        safety_config : SafetyConfig level (e.g., SafetyConfig.WEAK, SafetyConfig.MEDIUM, SafetyConfig.STRONG, SafetyConfig.MAX)
        output : Directory to save image
        seed : Seed for reproducibility
    
    Returns:
        List of paths to generated images
    """
    os.makedirs(output, exist_ok=True)
    image_paths = []

    for i, prompt in enumerate(prompts):
        output_path = os.path.join(output, f"sld_{i}.png")

        try:
            generator = torch.Generator(device=device).manual_seed(seed + i)
            image = pipe(
                prompt=prompt,
                **safety_config,  # Unpack SafetyConfig
                generator=generator
            ).images[0]
            image.save(output_path)
            image_paths.append(output_path)
        
        except Exception as e:
            print(f"Error generating image {i}: {e}")
    
    return image_paths

def get_coco_image(coco_samples, output, max_images=None):
    """
    Extract and save COCO image for comparison

    Args:
        coco_samples: COCO dataset
        output : Directory to save images
        max_images: Maximum number of images to save

    Returns:
        List of paths to saved images
    """
    os.makedirs(output, exist_ok=True)
    image_paths = []

    samples = coco_samples[:max_images] if max_images else coco_samples

    for i, sample in enumerate(samples):
        if 'image' in sample and sample['image'] is not None:
            output_path = os.path.join(output, f"real_{i}.png")

            try:
                image = sample['image']
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)

                image = image.convert("RGB").resize((512, 512), Image.LANCZOS)
                image.save(output_path)
                image_paths.append(output_path)
            except Exception:
                print("Error saving image")
                continue

    return image_paths


def evaluate_fid(
        num_samples = 50,
        safety_config = SafetyConfig.MAX,
        output = "./evaluate/results/fid",
        seed = 42
):
    """
    Evaluate FID scores for baseline and SLD models
    
    Args:
        num_samples: Number of samples to evaluate
        safety_config: SafetyConfig level for SLD (WEAK, MEDIUM, STRONG, MAX)
        output: Output directory for results
        seed: Random seed
        
    Returns:
        Tuple of (fid_baseline, fid_sld)
    """
    batch_size=32
    real_dir = os.path.join(output, "real_images")
    baseline_dir = os.path.join(output, "baseline_images")
    sld_dir = os.path.join(output, "sld_images")

    print("Loading COCO dataset")
    coco_samples = load_coco_hf(split="validation", num_samples=num_samples)
    
    if not coco_samples:
        raise RuntimeError("Failed to load COCO dataset. Please check your internet connection and try again.")
    
    prompts = create_prompt_for_SLD(coco_samples, max_prompts=num_samples)

    real_image_path = get_coco_image(coco_samples, real_dir, max_images=num_samples)
    baseline_image_path = generate_baseline_image(prompts, baseline_dir, seed=seed)
    sld_image_path = generate_sld_image(prompts, safety_config, sld_dir, seed=seed)

    fid_baseline = calculate_fid_from_paths(real_image_path, baseline_image_path, batch_size=batch_size)
    fid_sld = calculate_fid_from_paths(real_image_path, sld_image_path, batch_size=batch_size)

    return fid_baseline, fid_sld

if __name__ == "__main__":
    print("=" * 60)
    print("Evaluating FID scores with SLD MAX")
    print("=" * 60)
    
    fid_baseline, fid_sld = evaluate_fid(
        num_samples=50,
        safety_config=SafetyConfig.MAX,
        seed=42
    )
    
    print(f"\nResults:")
    print(f"  FID Baseline: {fid_baseline:.2f}")
    print(f"  FID SLD MAX: {fid_sld:.2f}")
    print(f"  Difference: {fid_sld - fid_baseline:.2f}")