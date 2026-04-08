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
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )

        logger.info(
            f"Initializing SLD on {self.device} with model {self.config.model_id}"
        )

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
                self.config.model_id, safety_checker=None, requires_safety_checker=False
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load SLD model: {e}")

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Any]:
        """
        Generate images using the configured SLD parameters.

        SLD parameters (sld_guidance_scale, etc.) can be overridden per-call
        via *kwargs*; otherwise the values from self.config are used.
        All other kwargs (e.g. num_inference_steps) are forwarded to the pipeline.
        """
        # Extract SLD-specific params from kwargs so they don't get passed twice
        sld_params = {
            "sld_guidance_scale": kwargs.pop(
                "sld_guidance_scale", self.config.sld_guidance_scale
            ),
            "sld_warmup_steps": kwargs.pop(
                "sld_warmup_steps", self.config.sld_warmup_steps
            ),
            "sld_threshold": kwargs.pop("sld_threshold", self.config.sld_threshold),
            "sld_momentum_scale": kwargs.pop(
                "sld_momentum_scale", self.config.sld_momentum_scale
            ),
            "sld_mom_beta": kwargs.pop("sld_mom_beta", self.config.sld_mom_beta),
        }

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(f"Generating {len(prompts)} images using SLD...")

        images = []
        for i, prompt in enumerate(prompts):
            try:
                output = self.pipe(
                    prompt=prompt,
                    **sld_params,
                    generator=generator,
                    **kwargs,
                ).images[0]
                images.append(output)
            except Exception as e:
                logger.error(f"Generation failed for prompt '{prompt}': {e}")
                raise

        return images
