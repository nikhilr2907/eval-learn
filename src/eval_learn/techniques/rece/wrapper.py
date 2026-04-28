from typing import List, Optional
from PIL import Image

try:
    from rece import RECEPIPELINE
except ImportError:
    raise ImportError("RECETechnique requires the 'rece' package. Package not installed.")

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import RECEConfig

logger = get_logger(__name__)


@register_technique("rece")
class RECETechnique:
    """
    Thin wrapper around the external RECE package.

    Integrates Reliable and Efficient Concept Erasure (RECE) into the
    Eval-Learn benchmarking framework via the standard technique interface.
    """

    def __init__(self, **kwargs):
        self.config = RECEConfig.from_dict(kwargs)

        logger.info(
            f"Initializing RECE: model={self.config.model_id}, "
            f"emb_computing='{self.config.emb_computing}'"
        )

        self.pipeline = RECEPIPELINE(
            model_id=self.config.model_id,
            device=self.config.device,
            use_fp16=self.config.use_fp16,
            load_path=self.config.load_path,
            erase_concept=self.config.erase_concept,
            concept_type=self.config.concept_type,
            emb_computing=self.config.emb_computing,
            save_path=self.config.save_path,
        )

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        num_inference_steps = kwargs.pop(
            "num_inference_steps", self.config.num_inference_steps
        )
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        logger.info(f"Generating {len(prompts)} images with RECE")

        return self.pipeline.generate(
            prompts,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
