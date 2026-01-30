from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class CLIPScoreConfig(BaseConfig):
    """
    Configuration for the CLIP Score metric.

    Attributes:
        clip_model_name: HuggingFace CLIP model identifier used by torchmetrics.
        device: Torch device string (default: None, auto-detect).
    """
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None
