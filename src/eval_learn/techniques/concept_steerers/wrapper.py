from typing import List, Optional
from PIL import Image

# Import from external package
try:
    from concept_steerers import ConceptSteeringPipeline
except ImportError:
    raise ImportError(
        "ConceptSteerersWrapper requires the 'concept-steerers' package. Package not installed."
    )

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import ConceptSteerersConfig

logger = get_logger(__name__)


@register_technique("concept_steerers")
class ConceptSteerersWrapper:
    """
    Thin wrapper around the external concept-steerers package.

    This wrapper integrates the concept-steerers library into the
    Eval-Learn benchmarking framework via the standard technique interface.
    """

    def __init__(self, **kwargs):
        """Initialize wrapper by delegating to ConceptSteeringPipeline."""
        self.config = ConceptSteerersConfig.from_dict(kwargs)

        logger.info(f"Initializing Concept Steerers: {self.config.model_id}")

        # Delegate to external package
        self.pipeline = ConceptSteeringPipeline(
            model_id=self.config.model_id,
            device=self.config.device,
            sae_path=self.config.sae_path,
            concept=self.config.erase_concept,
            multiplier=self.config.multiplier,
        )

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        """
        Generate images using concept steering.

        Args:
            prompts: List of text prompts.
            seed: Random seed for reproducibility.
            **kwargs: Additional generation parameters.

        Returns:
            List of PIL Images.
        """
        logger.info(
            f"Generating {len(prompts)} images with concept steering (strength: {self.config.multiplier})"
        )
        return self.pipeline.generate(prompts, seed=seed, **kwargs)
