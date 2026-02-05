from ...configs.base import BaseConfig
from dataclasses import dataclass

@dataclass
class SAFREEConfig(BaseConfig):
    # Stage 1: Text Projection
    alpha: float = 0.01                    # Threshold for trigger token detection
    
    # Stage 2: SVF (Self-Validation Filter)
    enable_svf: bool = True                # Use adaptive timestep scheduling
    upperbound_timestep: int = 10          # Max timesteps to apply projection (up_t)
    
    # Stage 3: LRA (Latent Re-Attention)
    enable_lra: bool = True                # Enable Fourier filtering
    freeu_b1: float = 1.0                  # FreeU backbone scale factor 1
    freeu_b2: float = 1.0                  # FreeU backbone scale factor 2
    freeu_s1: float = 0.9                  # FreeU skip scale factor 1
    freeu_s2: float = 0.2                  # FreeU skip scale factor 2
    
    # Concept specification
    unsafe_concepts: list[str] = None      # e.g., ["Nudity", "Pornography", ...]
    concept_category: str = "nudity"       # "nudity" or "artists-VanGogh", etc.
    
    # Alternative to SVF (if SVF disabled)
    re_attn_timestep_range: tuple[int, int] = (-1, 1001)  # Which steps to apply projection


    