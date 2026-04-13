from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class SSDConfig(BaseConfig):
    """
    Configuration for Selective Synaptic Dampening (SSD).

    SSD estimates diagonal Fisher Information for a forget concept and a
    neutral retain set, then dampens UNet parameters that are important for
    the forget concept but not for the retain set. No training loop is required.

    Key parameters:
    - alpha: Controls selectivity. Higher = only parameters that are strongly
      forget-specific get dampened. Lower = broader dampening.
      Typical range: 100–2000. Default: 200.
    - dampening_coeff: Global scale on the dampening strength. 1.0 applies the
      full ratio; values < 1.0 reduce the effect globally (useful if 1.0
      over-erases and hurts image quality).
    - num_fisher_samples: Samples per prompt for Fisher estimation. More is
      more accurate but slower (each sample is a full UNet forward+backward).
    """

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"

    # Concept erasure settings
    erase_concept: str = "nudity"

    # SSD hyperparameters
    alpha: float = 200.0
    dampening_coeff: float = 1.0
    num_fisher_samples: int = 4

    # Save modified UNet weights (optional, to avoid re-running SSD)
    save_path: Optional[str] = None

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
