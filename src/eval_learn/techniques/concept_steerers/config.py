import os
from dataclasses import dataclass
from typing import Optional, Any, Dict

from ...configs.base import BaseConfig


@dataclass
class ConceptSteerersConfig(BaseConfig):
    model_id: str = "CompVis/stable-diffusion-v1-4"
    device: str = "cuda"
    sae_path: Optional[str] = None
    concept: str = "nudity"
    multiplier: float = 1.0

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConceptSteerersConfig':
        config_dict = dict(config_dict)
        config_dict.pop("model_id", None)

        if 'sae_path' not in config_dict or not config_dict['sae_path']:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_dict['sae_path'] = os.path.join(base_dir, "checkpoints", "i2p_sd14_l9")

        return super().from_dict(config_dict)
