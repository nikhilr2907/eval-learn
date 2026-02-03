from dataclasses import dataclass
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig

# Configure UCE parameters
# name of weight files is used to access the respective filepath
# presets are the preset maps, that maps name to respective uce weight file path
_PRESETS: Dict[str, Dict[str, Any]] = { "nudity_erasure": {
        "uce_weights_path": "models/uce_nudity_v1.4.safetensors",
    },
    "violence_erasure": {
        "uce_weights_path": "models/uce_violence_v1.4.safetensors",
    },
}

@dataclass
class UCEConfig(BaseConfig):
    # Define customisable parameters for UCE
    model_id: str = "CompVis/stable-diffusion-v1-4"
    uce_weights_path: Optional[str] = None
    device: Optional[str] = None
    preset: Optional[str] = None # what is being unlearnt
    num_inference_steps: int = 50 # higher the steps, better the quality
    guidance_scale: float = 7.5 # higher value, the more the image corresponds to the text prompt

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UCEConfig':
        """Create config from a dict, resolving preset values first.

        If ``preset`` is provided, its parameter values are used as defaults.
        Any UCE parameters explicitly present in *data* take precedence over
        the preset.
        """
        preset = data.get("preset")
        if preset is not None:
            key = preset.lower()
            if key not in _PRESETS:
                # need to extend the presets to handle other cases as well
                raise ValueError(
                    f"Unknown UCE preset '{preset}'. "
                    f"Available: {[p.upper() for p in _PRESETS]}"
                )
            merged = dict(data)
            for k, v in _PRESETS[key].items():
                if k not in data:
                    merged[k] = v
            return super().from_dict(merged)
        return super().from_dict(data)
    
    # ensures that uce_weights_path is assigned to a value, either from preset or by user
    def __post_init__(self):
        if self.uce_weights_path is None:
            raise ValueError("uce_weights_path not found, use a preset or create weights")