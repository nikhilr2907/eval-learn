import torch
import os
from typing import List, Optional, Any
from PIL import Image
try:
    from PIL import Image
except ImportError:
    raise RuntimeError(
        "ConceptSteerersWrapper requires 'PIL'. Please install it via: pip install Pillow"
    )

# 1. Framework imports
from ...registry import register_technique
from ...logging_utils import get_logger
from .config import ConceptSteerersConfig

# 2. Technique-specific imports from your integrated folders
from .SDLens.hooked_sd_pipeline import HookedStableDiffusionPipeline
from .training.k_sparse_autoencoder import SparseAutoencoder
# Imports the hooks you just added to utils.py
from .utils import add_feature_on_text_prompt, minus_feature_on_text_prompt

logger = get_logger(__name__)

@register_technique("concept_steerers")
class ConceptSteerersWrapper:
    """
    Final Implementation of Concept Steerers (2025).
    Integrates SAE-based feature modulation into the Eval-Learn benchmarking framework.
    """
    def __init__(self, **kwargs):
        # Load typed configuration
        self.config = ConceptSteerersConfig.from_dict(kwargs)
        
        # Initialize the Hooked Pipeline (defaults to SD 1.4 for the 2025 paper setup)
        logger.info(f"Initializing Hooked SD Pipeline: {self.config.model_id}")
        self.pipe = HookedStableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            safety_checker=None,
            torch_dtype=torch.float32 # Matches research script dtype
        ).to(self.config.device)

        # Set the target steering layer (Layer 9 is the primary bottleneck for concept steering)
        self.target_block = 'text_encoder.text_model.encoder.layers.9'
        
        # Load the Sparse Autoencoder (SAE) dictionary from disk
        if self.config.sae_path and os.path.exists(self.config.sae_path):
            logger.info(f"Loading SAE weights from {self.config.sae_path}")
            self.sae = SparseAutoencoder.load_from_disk(self.config.sae_path).to(self.config.device)
        else:
            raise FileNotFoundError(f"Missing SAE checkpoint at: {self.config.sae_path}")

    def _get_steering_feature(self, concept_prompt: str, seed: int) -> torch.Tensor:
        """
        Extracts the semantic direction vector from the SAE for a given concept prompt.
        This reproduces the 'activation_modulation_across_prompt' logic.
        """
        # Cache activation for the concept prompt
        _, cache = self.pipe.run_with_cache(
            concept_prompt,
            positions_to_cache=[self.target_block],
            save_output=True,
            num_inference_steps=1,
            generator=torch.Generator(device="cpu").manual_seed(seed)
        )
        
        # Extract activations and encode via Sparse Autoencoder
        activations = cache['output'][self.target_block][:, 0, :].squeeze(0)
        with torch.no_grad():
            # Get the k-sparse latent representation
            activated = self.sae.encode_without_topk(activations)
        
        # Scale by multiplier and project back to hidden dimension using SAE decoder
        steering_feature = (activated * self.config.multiplier) @ self.sae.decoder.weight.T
        return steering_feature

    def _create_modulate_hook(self, steering_feature: torch.Tensor):
        """
        Constructs the toggle hook required for classifier-free guidance modulation.
        """
        call_counter = {"count": 0}
        
        def hook_function(*args, **kwargs):
            call_counter["count"] += 1
            # Logic from original repo:
            # First call (Conditional) -> Add the concept feature
            # Second call (Unconditional/Negative) -> Subtract the concept feature
            if call_counter["count"] == 1:
                return add_feature_on_text_prompt(self.sae, steering_feature, *args, **kwargs)
            else:
                return minus_feature_on_text_prompt(self.sae, steering_feature, *args, **kwargs)

        return hook_function

    def generate(self, prompts: List[str], seed: Optional[int] = None, **kwargs) -> List[Image.Image]:
        """
        Main generation entry point used by BenchmarkRunner.
        """
        images = []
        base_seed = seed if seed is not None else 42
        
        # Pre-calculate steering feature for the target concept (e.g., 'nudity')
        steering_feature = self._get_steering_feature(self.config.concept, base_seed)

        logger.info(f"Executing generation with Concept Steering (Strength: {self.config.multiplier})")
        
        for i, prompt in enumerate(prompts):
            # Run inference using SDLens' hook injection method
            output = self.pipe.run_with_hooks(
                prompt,
                position_hook_dict={
                    self.target_block: self._create_modulate_hook(steering_feature)
                },
                num_inference_steps=kwargs.get("num_inference_steps", 50),
                guidance_scale=kwargs.get("guidance_scale", 7.5),
                generator=torch.Generator(device="cpu").manual_seed(base_seed + i)
            )
            images.append(output.images[0])
                
        return images