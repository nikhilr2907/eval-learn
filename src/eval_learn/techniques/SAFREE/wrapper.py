import os
from typing import List, Any, Optional

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import SAFREEConfig
from safree.pipeline import SAFREEPipeline
logger = get_logger(__name__)

try:
    import torch
    from huggingface_hub import login
except ImportError as e:
    logger.error("Optional dependencies for SLD missing.")
    raise RuntimeError(
        "SLD technique requires 'diffusers', 'torch', and 'huggingface_hub'. "
        "Install with: pip install eval-learn[diffusers]"
    ) from e



@register_technique("safree")
class SAFREETechnique():
    def __init__(self, **kwargs):
        self.config = SAFREEConfig.from_dict(kwargs)
        
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                logger.debug("Logged in to Hugging Face Hub.")
            except Exception as e:
                logger.warning(f"Could not log in to Hugging Face Hub: {e}")

        try:
            self.pipe = SAFREEPipeline.from_pretrained(
                self.config.model_id,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.config.device)
            
            # Register LRA hooks if enabled
            if self.config.enable_lra:
                self.pipe.enable_lra(
                    filter_type="high",
                    b1=self.config.freeu_b1,
                    b2=self.config.freeu_b2,
                    s1=self.config.freeu_s1,
                    s2=self.config.freeu_s2,
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to load SAFREE model: {e}")
    
    def generate(self, prompts: List[str], seed: Optional[int] = None, **kwargs) -> List[Any]:
        if seed is not None:
            generator = torch.Generator(self.config.device).manual_seed(seed)
        else:
            generator = None
            
        images = []
        for prompt in prompts:
            try:
                output = self.pipe(
                    prompt,
                    num_inference_steps=kwargs.get("num_inference_steps", 50),
                    guidance_scale=kwargs.get("guidance_scale", 7.5),
                    generator=generator,
                    # SAFREE params
                    unsafe_concepts=self.config.unsafe_concepts,
                    unsafe_category=self.config.concept_category,
                    enable_safree=True,
                    enable_svf=self.config.enable_svf,
                    enable_lra=self.config.enable_lra,
                    alpha=self.config.alpha,
                    upperbound_timestep=self.config.upperbound_timestep,
                )
                images.append(output.images[0])

            except Exception as e:
                logger.error(f"Generation failed for prompt '{prompt}': {e}")
                raise

        return images