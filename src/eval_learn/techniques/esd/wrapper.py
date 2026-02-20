from typing import List, Optional
from PIL import Image

# Import from external package
try:
    from esd import ESDPipeline
except ImportError:
    raise ImportError(
        "ESDWrapper requires the 'esd' package. Package not installed."
    )

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import ESDConfig

logger = get_logger(__name__)


@register_technique("esd")
class ESDWrapper:
    """
    Thin wrapper around the external ESD package.

    This wrapper integrates the ESD (Erased Stable Diffusion) library into
    the Eval-Learn benchmarking framework via the standard technique interface.
    """

    def __init__(self, **kwargs):
        """Initialize wrapper by delegating to ESDPipeline."""
        self.config = ESDConfig.from_dict(kwargs)

        logger.info(f"Initializing ESD: {self.config.model_id}")

        # Delegate to external package
        self.pipeline = ESDPipeline(
            model_id=self.config.model_id,
            device=self.config.device,
            erase_concept=self.config.erase_concept,
            erase_from=self.config.erase_from,
            train_method=self.config.train_method,
            negative_guidance=self.config.negative_guidance,
            train_steps=self.config.train_steps,
            learning_rate=self.config.learning_rate,
            use_fp16=self.config.use_fp16,
            save_path=self.config.save_path,
        )

    def generate(self, prompts: List[str], seed: Optional[int] = None, **kwargs) -> List[Image.Image]:
        """
        Generate images using the concept-erased model.

        Args:
            prompts: List of text prompts.
            seed: Random seed for reproducibility.
            **kwargs: Additional generation parameters.

        Returns:
            List of PIL Images.
        """
        num_inference_steps = kwargs.pop("num_inference_steps", self.config.num_inference_steps)
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        logger.info(f"Generating {len(prompts)} images ('{self.config.erase_concept}' erased)")

        return self.pipeline.generate(
            prompts,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
