import os
from typing import Any, List, Optional

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import FreeRunConfig

logger = get_logger(__name__)

try:
    import torch
    from diffusers import AutoPipelineForText2Image
    from huggingface_hub import login
except ImportError as e:
    raise RuntimeError(
        "FreeRunTechnique requires 'diffusers', 'torch', and 'huggingface_hub'. "
        "Install with: pip install eval-learn[diffusers]"
    ) from e


@register_technique("free_run")
class FreeRunTechnique:
    """
    Runs any HF text-to-image model without any safety filtering.
    Specify the model via model_id in the config.
    """

    def __init__(self, **kwargs):
        self.config = FreeRunConfig.from_dict(kwargs)

        if self.config.device:
            self.device = self.config.device
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )

        logger.info(f"Initializing FreeRun on {self.device} with model {self.config.model_id}")

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                logger.debug("Logged in to Hugging Face Hub.")
            except Exception as e:
                logger.warning(f"Could not log in to Hugging Face Hub: {e}")

        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                self.config.model_id,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.config.model_id}': {e}") from e

    def generate(self, prompts: List[str], seed: Optional[int] = None, **kwargs) -> List[Any]:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(f"Generating {len(prompts)} images with {self.config.model_id}...")

        images = []
        for prompt in prompts:
            try:
                output = self.pipe(prompt=prompt, generator=generator, **kwargs)
                if not hasattr(output, "images") or not output.images:
                    raise RuntimeError(
                        f"Pipeline output has no images — is '{self.config.model_id}' a text-to-image model?"
                    )
                images.append(output.images[0])
            except Exception as e:
                logger.error(f"Generation failed for prompt '{prompt}': {e}")
                raise

        return images
