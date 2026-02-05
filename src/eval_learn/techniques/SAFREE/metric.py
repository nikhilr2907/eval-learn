from eval_learn.registry.local import register_metric
import os
from typing import List, Any, Optional

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import SAFREEConfig
from safree.pipeline import SAFREEPipeline
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



@register_metric("safree")
class SAFREEMetric():
    def __init__(self, **kwargs):
        self.config = SAFREEConfig.from_dict(kwargs)    
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
            self.pipe = SAFREEPipeline.from_pretrained(
                self.config.model_id,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        except Exception as e:
             raise RuntimeError(f"Failed to load SAFREE model: {e}")
    
    def generate(self, prompts: List[str], seed: Optional[int] = None, **kwargs) -> List[Any]:

        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)
        else:
            generator = None
        images = []
        for i,prompt in enumerate(prompts):
            try:
                output = self.pipe(
                    prompt,
                    num_inference_steps=kwargs.get("num_inference_steps", 50),
                    guidance_scale=kwargs.get("guidance_scale", 7.5),
                    generator=generator,
                    safree_config=self.config,
                )
                images.append(output.images[0])

            except Exception as e:
                logger.error(f"Generation failed for prompt '{prompt}': {e}")
                raise

        return images