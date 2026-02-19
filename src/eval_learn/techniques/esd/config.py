from dataclasses import dataclass
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig

# Available training methods for ESD
# - xattn: Fine-tunes cross-attention layers (ESD-x) - best for styles/artists
# - noxattn: Fine-tunes all layers except cross-attention
# - selfattn: Fine-tunes only self-attention layers
# - full: Fine-tunes entire UNet (most aggressive)
TRAIN_METHODS = ["xattn", "noxattn", "selfattn", "full"]


@dataclass
class ESDConfig(BaseConfig):
    """
    Configuration for Erased Stable Diffusion (ESD).

    Two main variants:
    - ESD-x (train_method='xattn'): Fine-tunes cross-attention layers
      Best for specific concepts like artist styles, objects
    - ESD-u (train_method='full'/'noxattn'): Fine-tunes broader layers
      Best for general concepts like nudity, violence
    """
    # Model settings
    model_id: str = "CompVis/stable-diffusion-v1-4"
    device: Optional[str] = None

    # Concept erasure settings
    erase_concept: str = "nudity"
    erase_from: Optional[str] = None  # Target concept to erase from (defaults to erase_concept)
    train_method: str = "xattn"
    negative_guidance: float = 2.0

    # Training settings
    train_steps: int = 200
    learning_rate: float = 5e-5
    use_fp16: bool = True

    # Save trained weights (optional)
    save_path: Optional[str] = None

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ESDConfig':
        """Create config from a dict, validating train_method."""
        data = dict(data)
        data.pop("model_id", None)

        train_method = data.get("train_method", "xattn")
        if train_method not in TRAIN_METHODS:
            raise ValueError(
                f"Unknown train_method '{train_method}'. "
                f"Available: {TRAIN_METHODS}"
            )
        return super().from_dict(data)
