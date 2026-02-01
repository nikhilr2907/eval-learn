from typing import List, Any, Dict, Optional
import os
import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from dotenv import load_dotenv
from huggingface_hub import login

from core.base_technique import UnlearningTechnique

class SLDWrapper(UnlearningTechnique):
    def __init__(self, model_id: str = "AIML-TUDA/stable-diffusion-safe", device: str = None):
        """
        Wrapper for the Safe Latent Diffusion (SLD) pipeline.
        """
        # Load environment variables (for HF_TOKEN)
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                print("Logged in to Hugging Face Hub within SLDWrapper.")
            except Exception as e:
                print(f"Warning: Could not log in to Hugging Face Hub: {e}")

        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing SLD Pipeline with model: {model_id} on {self.device}")
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            safety_checker=None,
        ).to(self.device)

    def generate(self, prompts: List[str], config: Optional[Dict] = None, **kwargs) -> List[Any]:
        """
        Generate images using SLD.
        
        Args:
            prompts: A list of prompts to generate images for.
            config: The SLD SafetyConfig (e.g. SafetyConfig.MAX). 
                    If None, defaults to SafetyConfig.MAX.
            **kwargs: Extra arguments for the pipeline (e.g. num_inference_steps).
        """
        if config is None:
            config = SafetyConfig.MAX
            
        images = []
        print(f"Generating {len(prompts)} images with config: {config}")
        
        for prompt in prompts:
            # The pipeline accepts config kwargs unpacked
            output = self.pipe(
                prompt=prompt,
                **config,
                **kwargs
            ).images[0]
            images.append(output)
            
        return images
