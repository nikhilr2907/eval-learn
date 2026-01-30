from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class ERRConfig(BaseConfig):
    """
    Configuration for the Erasing-Retention-Robustness (ERR) metric.

    Attributes:
        clip_model_name: HuggingFace CLIP model identifier.
        device: Torch device string (default: None, auto-detect).
    """
    clip_model_name: str = "openai/clip-vit-large-patch14"
    device: Optional[str] = None
