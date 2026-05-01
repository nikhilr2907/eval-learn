from typing import List, Optional
from PIL import Image

try:
    from salun_sd import SalUnPipeline
except ImportError:
    raise ImportError("SalUnTechnique requires the 'salun_sd' package. Package not installed.")

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import SalUnConfig

logger = get_logger(__name__)


@register_technique("salun")
class SalUnTechnique:
    """
    Thin wrapper around the external salun_sd package.

    Integrates SalUn (Saliency Unlearning) into the Eval-Learn benchmarking
    framework via the standard technique interface.

    SalUn builds a gradient-magnitude saliency mask over the UNet (Phase 1),
    then fine-tunes only the top-k% most concept-responsible weights to push
    the forget concept toward an anchor concept while preserving general
    generation quality (Phase 2). Training runs at construction time.
    """

    def __init__(self, **kwargs):
        self.config = SalUnConfig.from_dict(kwargs)

        logger.info(
            f"Initializing SalUn: model={self.config.model_id}, "
            f"erase='{self.config.erase_concept}', "
            f"anchor='{self.config.anchor_concept}', "
            f"threshold={self.config.threshold}, epochs={self.config.epochs}"
        )

        self.pipeline = SalUnPipeline(
            model_id=self.config.model_id,
            erase_concept=self.config.erase_concept,
            anchor_concept=self.config.anchor_concept,
            forget_data_path=self.config.forget_data_path,
            retain_data_path=self.config.retain_data_path,
            device=self.config.device,
            use_fp16=self.config.use_fp16,
            load_path=self.config.load_path,
            save_path=self.config.save_path,
            image_size=self.config.image_size,
            alpha=self.config.alpha,
            lr=self.config.lr,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            c_guidance=self.config.c_guidance,
            threshold=self.config.threshold,
            train_method=self.config.train_method,
        )

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        num_inference_steps = kwargs.pop(
            "num_inference_steps", self.config.num_inference_steps
        )
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        logger.info(
            f"Generating {len(prompts)} images ('{self.config.erase_concept}' unlearned)"
        )

        return self.pipeline.generate(
            prompts,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
