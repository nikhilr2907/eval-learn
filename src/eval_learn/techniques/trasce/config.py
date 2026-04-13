from dataclasses import dataclass, field
from typing import Optional, Any, Dict

from ...configs.base import BaseConfig


@dataclass(frozen=True)
class TraSCEConfig(BaseConfig):
    # parameters as given and defined by the original TraSCE paper (https://arxiv.org/abs/2412.07658)
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: Optional[str] = None
    discriminator_guidance_scale: float = 5.0
    guidance_loss_scale: float = 15.0
    sigma: float = 1.0
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    erase_concept: str = "nudity"
