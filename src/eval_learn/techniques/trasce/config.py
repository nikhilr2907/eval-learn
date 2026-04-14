from dataclasses import dataclass
from typing import Optional, Any, Dict

from ...configs.base import BaseConfig


@dataclass
class TraSCEConfig(BaseConfig):
    # parameters as given and defined by the original TraSCE paper (https://arxiv.org/abs/2412.07658)
    model_id: str = "CompVis/stable-diffusion-v1-4"
    device: Optional[str] = None
    discriminator_guidance_scale: float = 5.0
    guidance_loss_scale: float = 15.0
    sigma: float = 1.0
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    erase_concept: str = "nudity"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TraSCEConfig":
        config_dict = dict(config_dict)
        config_dict.pop("model_id", None)
 
        return super().from_dict(config_dict)
