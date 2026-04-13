from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...configs.base import BaseConfig


@dataclass(frozen=True)
class CoGFDConfig(BaseConfig):
    """Configuration for CoGFD."""

    # Model
    model_id: str = "CompVis/stable-diffusion-v1-4"
    device: Optional[str] = None
    use_fp16: bool = True

    # Target concept (used as fallback if combination_prompts is empty)
    erase_concept: str = "nudity"

    # Concept logic graph — multiple prompts expressing the harmful combination.
    # Leave empty to fall back to [erase_concept].
    combination_prompts: List[str] = field(default_factory=list)

    # Individual component concepts to preserve (e.g. ["person", "nude"]).
    # Leave empty to skip explicit individual-concept preservation.
    preserve_concepts: List[str] = field(default_factory=list)

    # Loss weights
    lambda_erase: float = 1.0     # combination erasure weight
    lambda_preserve: float = 2.0  # individual preservation weight (higher than erase to prevent model degradation)
    lambda_decouple: float = 0.5  # feature decoupling weight

    # Fine-tuning
    train_steps: int = 150
    learning_rate: float = 1e-5

    # Save modified UNet weights
    save_path: Optional[str] = None

    # Generation
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        if self.train_steps <= 0:
            raise ValueError(f"train_steps must be > 0, got {self.train_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.lambda_erase < 0:
            raise ValueError(f"lambda_erase must be >= 0, got {self.lambda_erase}")
        if self.lambda_preserve < 0:
            raise ValueError(f"lambda_preserve must be >= 0, got {self.lambda_preserve}")
        if self.lambda_decouple < 0:
            raise ValueError(f"lambda_decouple must be >= 0, got {self.lambda_decouple}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoGFDConfig":
        data = dict(data)
        data.pop("model_id", None)
        return super().from_dict(data)
