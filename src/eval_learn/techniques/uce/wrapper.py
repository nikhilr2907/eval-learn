from typing import List, Optional
from PIL import Image

# Import from external package
try:
    from uce import UCEPipeline
except ImportError:
    raise ImportError("UCEWrapper requires the 'uce' package. Package not installed.")

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import UCEConfig

logger = get_logger(__name__)


@register_technique("uce")
class UCEWrapper:
    """
    Thin wrapper around the external UCE package.

    This wrapper integrates the UCE (Unified Concept Editing) library into
    the Eval-Learn benchmarking framework via the standard technique interface.
    """

    def __init__(self, **kwargs):
        """Initialize wrapper by delegating to UCEPipeline."""
        self.config = UCEConfig.from_dict(kwargs)

        logger.info(f"Initializing UCE: {self.config.model_id}")

        # Delegate to external package
        self.pipeline = UCEPipeline(
            model_id=self.config.model_id,
            device=self.config.device,
            preset=self.config.preset,
            weights_path=self.config.uce_weights_path,
            use_fp16=self.config.use_fp16,
        )

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        """
        Generate images using UCE with concept erased.

        Args:
            prompts: List of text prompts.
            seed: Random seed for reproducibility.
            **kwargs: Additional generation parameters.

        Returns:
            List of PIL Images.
        """
        num_inference_steps = kwargs.pop(
            "num_inference_steps", self.config.num_inference_steps
        )
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        logger.info(f"Generating {len(prompts)} images with UCE")

        return self.pipeline.generate(
            prompts,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
