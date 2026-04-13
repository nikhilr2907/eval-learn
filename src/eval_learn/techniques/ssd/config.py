from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class SSDConfig(BaseConfig):
    """Configuration for Selective Synaptic Dampening (SSD)."""

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"
    use_fp16: bool = True

    # Concept erasure settings
    erase_concept: str = "nudity"

    # SSD hyperparameters
    alpha: float = 200.0
    dampening_coeff: float = 1.0
    num_fisher_samples: int = 4

    # Save modified UNet weights (optional, to avoid re-running SSD)
    save_path: Optional[str] = None

    def __post_init__(self):
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.dampening_coeff <= 0:
            raise ValueError(f"dampening_coeff must be > 0, got {self.dampening_coeff}")
        if self.num_fisher_samples <= 0:
            raise ValueError(f"num_fisher_samples must be > 0, got {self.num_fisher_samples}")

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
