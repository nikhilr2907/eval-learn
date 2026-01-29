import os
from typing import List, Any, Optional

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import SLDConfig

logger = get_logger(__name__)

try:
    import torch
    from diffusers import DiffusionPipeline
    from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
    from huggingface_hub import login
except ImportError as e:
    logger.error("Optional dependencies for SLD missing.")
    raise RuntimeError(
        "SLD technique requires 'diffusers', 'torch', and 'huggingface_hub'. "
        "Install with: pip install eval-learn[diffusers]"
    ) from e

@register_technique("sld")
class SLDWrapper:
    """
    Wrapper for Safe Latent Diffusion (SLD) pipeline.
    """
    def __init__(self, **kwargs):
        # 1. Parse Config
        # We accept kwargs to be flexible with the registry instantiation
        self.config = SLDConfig.from_dict(kwargs)
        
        # 2. Setup Device
        if self.config.device:
            self.device = self.config.device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        
        logger.info(f"Initializing SLD on {self.device} with model {self.config.model_id}")

        # 3. Auth (Optional)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                logger.debug("Logged in to Hugging Face Hub.")
            except Exception as e:
                logger.warning(f"Could not log in to Hugging Face Hub: {e}")

        # 4. Load Pipeline
        # Note: We disable the standard safety_checker because SLD *is* the safety mechanism
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.config.model_id,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        except Exception as e:
             raise RuntimeError(f"Failed to load SLD model: {e}")

    def generate(self, prompts: List[str], seed: Optional[int] = None, **kwargs) -> List[Any]:
        """
        Generate images using the configured SLD parameters.
        """
        images = []
        
        # Construct the safety config for this run
        # We allow overrides via kwargs, otherwise use self.config
        guidance_scale = kwargs.get('sld_guidance_scale', self.config.sld_guidance_scale)
        warmup_steps = kwargs.get('sld_warmup_steps', self.config.sld_warmup_steps)
        threshold = kwargs.get('sld_threshold', self.config.sld_threshold)
        momentum_scale = kwargs.get('sld_momentum_scale', self.config.sld_momentum_scale)
        mom_beta = kwargs.get('sld_mom_beta', self.config.sld_mom_beta)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(f"Generating {len(prompts)} images using SLD...")
        
        for i, prompt in enumerate(prompts):
            try:
                # The stable-diffusion-safe pipeline expects these specific kwargs
                output = self.pipe(
                    prompt=prompt,
                    sld_guidance_scale=guidance_scale,
                    sld_warmup_steps=warmup_steps,
                    sld_threshold=threshold,
                    sld_momentum_scale=momentum_scale,
                    sld_mom_beta=mom_beta,
                    generator=generator,
                    **kwargs # Pass through other standard params like num_inference_steps
                ).images[0]
                images.append(output)
            except Exception as e:
                logger.error(f"Generation failed for prompt '{prompt}': {e}")
                # Return None or a placeholder? For now, we might just skip or raise. 
                # Let's append None to keep index alignment, but the runner needs to handle it.
                # Ideally, we should handle this better. 
                # For this refactor, let's re-raise to fail fast during dev.
                raise e
                
        return images
