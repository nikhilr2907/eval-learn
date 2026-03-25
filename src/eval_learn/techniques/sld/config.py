from dataclasses import dataclass
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig

# SLD parameter keys that presets control
_SLD_PARAM_KEYS = {"sld_guidance_scale", "sld_warmup_steps", "sld_threshold", "sld_momentum_scale", "sld_mom_beta"}

_VALID_ERASE_CONCEPTS = {"nudity"}

# Preset values matching diffusers SafetyConfig
_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {
        "sld_guidance_scale": 0,
        "sld_warmup_steps": 0,
        "sld_threshold": 0.0,
        "sld_momentum_scale": 0.0,
        "sld_mom_beta": 0.0,
    },
    "weak": {
        "sld_guidance_scale": 200,
        "sld_warmup_steps": 15,
        "sld_threshold": 0.0,
        "sld_momentum_scale": 0.0,
        "sld_mom_beta": 0.0,
    },
    "medium": {
        "sld_guidance_scale": 1000,
        "sld_warmup_steps": 10,
        "sld_threshold": 0.01,
        "sld_momentum_scale": 0.3,
        "sld_mom_beta": 0.4,
    },
    "strong": {
        "sld_guidance_scale": 2000,
        "sld_warmup_steps": 7,
        "sld_threshold": 0.025,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    },
    "max": {
        "sld_guidance_scale": 5000,
        "sld_warmup_steps": 0,
        "sld_threshold": 1.0,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    },
}

@dataclass
class SLDConfig(BaseConfig):
    """
    Configuration for Safe Latent Diffusion (SLD).

    Use ``preset`` to select a named safety level (NONE, WEAK, MEDIUM, STRONG, MAX)
    instead of setting individual SLD parameters. Individual parameters can still
    be passed alongside a preset to override specific values.

    SLD only supports nudity concept erasure.
    """
    model_id: str = "AIML-TUDA/stable-diffusion-safe"
    device: Optional[str] = None
    erase_concept: str = "nudity"
    preset: Optional[str] = None
    sld_guidance_scale: float = 5000
    sld_warmup_steps: int = 0
    sld_threshold: float = 1.0
    sld_momentum_scale: float = 0.5
    sld_mom_beta: float = 0.7

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SLDConfig':
        """Create config from a dict, resolving preset values first.

        If ``preset`` is provided, its parameter values are used as defaults.
        Any SLD parameters explicitly present in *data* take precedence over
        the preset.
        """
        data = dict(data)
        data.pop("model_id", None)

        erase_concept = data.get("erase_concept", "nudity")
        if erase_concept.lower() not in _VALID_ERASE_CONCEPTS:
            raise ValueError(
                f"SLD only supports nudity concept erasure. "
                f"Got erase_concept='{erase_concept}'. "
                f"Available: {sorted(_VALID_ERASE_CONCEPTS)}"
            )

        preset = data.get("preset")
        if preset is not None:
            key = preset.lower()
            if key not in _PRESETS:
                raise ValueError(
                    f"Unknown SLD preset '{preset}'. "
                    f"Available: {[p.upper() for p in _PRESETS]}"
                )
            merged = dict(data)
            for k, v in _PRESETS[key].items():
                if k not in data:
                    merged[k] = v
            return super().from_dict(merged)
        return super().from_dict(data)
