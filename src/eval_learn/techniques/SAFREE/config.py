from dataclasses import dataclass
from typing import Any, Dict, Tuple
from ...configs.base import BaseConfig

_VALID_ERASE_CONCEPTS = {"nudity"}


@dataclass
class SAFREEConfig(BaseConfig):
    # Model settings
    model_id: str = "CompVis/stable-diffusion-v1-4"
    device: str = "cuda"

    # Stage 1: Text Projection
    alpha: float = 0.01

    # Stage 2: SVF (Self-Validation Filter)
    enable_svf: bool = True
    upperbound_timestep: int = 10

    # Stage 3: LRA (Latent Re-Attention)
    enable_lra: bool = True
    lra_filter_type: str = "high"  # "high", "low", or "all"
    freeu_b1: float = 1.0
    freeu_b2: float = 1.0
    freeu_s1: float = 0.9
    freeu_s2: float = 0.2

    # Concept specification (SAFREE only supports nudity)
    erase_concept: str = "nudity"

    # Alternative to SVF (if SVF disabled)
    re_attn_timestep_range: Tuple[int, int] = (-1, 1001)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SAFREEConfig':
        erase_concept = data.get("erase_concept", "nudity")
        if erase_concept.lower() not in _VALID_ERASE_CONCEPTS:
            raise ValueError(
                f"SAFREE only supports nudity concept erasure. "
                f"Got erase_concept='{erase_concept}'. "
                f"Available: {sorted(_VALID_ERASE_CONCEPTS)}"
            )
        return super().from_dict(data)