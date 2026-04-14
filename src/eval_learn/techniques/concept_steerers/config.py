import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

from ...configs.base import BaseConfig

_VALID_ERASE_CONCEPTS = {"nudity"}


@dataclass(frozen=True)
class ConceptSteerersConfig(BaseConfig):
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: Optional[str] = None
    use_fp16: bool = True
    sae_path: Optional[str] = None
    erase_concept: str = "nudity"
    multiplier: float = 1.0
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConceptSteerersConfig":
        config_dict = dict(config_dict)

        erase_concept = config_dict.get("erase_concept", "nudity")
        if erase_concept.lower() not in _VALID_ERASE_CONCEPTS:
            raise ValueError(
                f"ConceptSteerers only supports nudity concept erasure. "
                f"Got erase_concept='{erase_concept}'. "
                f"Available: {sorted(_VALID_ERASE_CONCEPTS)}"
            )

        if not config_dict.get("device"):
            if torch.cuda.is_available():
                config_dict["device"] = "cuda"
            elif torch.backends.mps.is_available():
                config_dict["device"] = "mps"
            else:
                config_dict["device"] = "cpu"

        if "sae_path" not in config_dict or not config_dict["sae_path"]:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_dict["sae_path"] = os.path.join(
                base_dir, "checkpoints", "i2p_sd14_l9"
            )

        return super().from_dict(config_dict)
