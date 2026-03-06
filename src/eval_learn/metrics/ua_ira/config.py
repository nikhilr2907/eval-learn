from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class UAIRAConfig(BaseConfig):
    """
    Configuration for Unlearning Accuracy and In-domain Retain Accuracy (IRA) Metric.

    UA is calculated by the ratio of images NOT classified as the target concept
    to the total number of target images.
    IRA is calculated by the ratio of images correctly classified as the retain concept
    to the total number of retain images.

    Uses HuggingFace streaming to load target and retain prompts from the ERR dataset.
    """
    clip_model: str = "openai/clip-vit-large-patch14"
    device: Optional[str] = None
    target_concept: str = "Mickey Mouse"
    retain_concept: str = "Minnie Mouse"
    target_prompt_count: Optional[int] = 5
    retain_prompt_count: Optional[int] = 5
    

