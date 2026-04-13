from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class SSDConfig(BaseConfig):
    """Configuration for Selective Synaptic Dampening (SSD)."""

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
