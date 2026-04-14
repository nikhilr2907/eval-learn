from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig

@dataclass(frozen=True)
class CAConfig(BaseConfig):
    """
    Configuration for Concept Ablation (CA).

    Concept Ablation fine-tunes the cross-attention layers to force the model's
    distribution for a 'target_concept' to match an 'anchor_concept'.
    """

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"

    # Concept ablation settings (Defaulted to the nudity task)
    target_concept: str = "nudity"
    anchor_concept: str = "a person wearing clothes" 

    # Training settings
    train_steps: int = 400
    learning_rate: float = 1e-5
    use_fp16: bool = True

    # Save trained weights (Highly recommended for evaluation efficiency)
    save_path: Optional[str] = None

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CAConfig":
        """Create config from a dict, with compatibility for ESD runner kwargs."""
        # Ensure we don't modify the original dict unexpectedly
        config_data = data.copy()
        
        # Compatibility with caller that might use ESD's 'erase_concept'
        if "target_concept" not in config_data and "erase_concept" in config_data:
            config_data["target_concept"] = config_data.pop("erase_concept")
            
        if "target_concept" in config_data and "anchor_concept" not in config_data:
            raise ValueError(
                "Concept Ablation requires both 'target_concept' and 'anchor_concept'. "
                f"Received only target: {config_data['target_concept']}"
            )
            
        return super().from_dict(config_data)