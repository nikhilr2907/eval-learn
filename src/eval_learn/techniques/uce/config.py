from dataclasses import dataclass
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig

# Valid preset names — weight resolution is handled by the UCE package itself
_VALID_PRESETS = {"nudity", "violence", "dog"}


@dataclass
class UCEConfig(BaseConfig):
    model_id: str = "CompVis/stable-diffusion-v1-4"
    uce_weights_path: Optional[str] = None
    device: Optional[str] = None
    preset: Optional[str] = None  # bundled preset name ("nudity", "violence", "dog")
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UCEConfig':
        preset = data.get("preset")
        if preset is not None and preset.lower() not in _VALID_PRESETS:
            raise ValueError(
                f"Unknown UCE preset '{preset}'. "
                f"Available: {sorted(_VALID_PRESETS)}"
            )
        return super().from_dict(data)

    def __post_init__(self):
        if self.preset is None and self.uce_weights_path is None:
            raise ValueError(
                "UCE requires either a 'preset' (e.g. 'nudity') "
                "or an explicit 'uce_weights_path'."
            )

    @property
    def erase_concept(self) -> Optional[str]:
        """Return the erase_concept corresponding to the preset for runner validation."""
        return self.preset.lower() if self.preset else None