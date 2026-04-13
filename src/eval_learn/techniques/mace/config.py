from dataclasses import dataclass, field
from typing import List, Optional, Union
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class MACEConfig(BaseConfig):
    """
    Configuration for MACE (Mass Concept Erasure in Diffusion Models, CVPR 2024).

    MACE uses Closed-Form Refinement (CFR) to analytically update the K/V
    projection matrices in every cross-attention layer, mapping concept token
    representations to neutral/empty representations without any training loop.

    Key parameter:
    - lambda_cfr: Regularization strength. Higher = more conservative
      (better preservation of unrelated concepts, weaker erasure).
      Lower = more aggressive erasure, but may affect unrelated content.
    """

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"
    use_fp16: bool = True

    # Concept erasure settings — accepts a single string or a list of strings.
    # List example: ["nudity", "naked", "bare skin"] erases all synonyms at once.
    erase_concept: Union[str, List[str]] = "nudity"
    erase_from: Optional[Union[str, List[str]]] = (
        None  # Defaults to "" (fully erase to neutral)
    )

    # CFR regularization (core MACE hyperparameter)
    lambda_cfr: float = 0.1

    # Save modified UNet weights (optional, to avoid re-running CFR)
    save_path: Optional[str] = None

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

