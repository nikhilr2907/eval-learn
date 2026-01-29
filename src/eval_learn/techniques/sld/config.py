from dataclasses import dataclass, field
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class SLDConfig(BaseConfig):
    """
    Configuration for Safe Latent Diffusion (SLD).
    
    Attributes:
        model_id: HuggingFace model ID (default: "AIML-TUDA/stable-diffusion-safe")
        device: Device to run on ("cuda", "cpu", "mps")
        safety_concept: Concept to suppress (default: "nudity")
        sld_guidance_scale: Scale for safety guidance (default: 1000 for MAX)
        sld_warmup_steps: Warmup steps (default: 7)
        sld_threshold: Threshold (default: 0.01)
        sld_momentum_scale: Momentum scale (default: 0.3)
        sld_mom_beta: Momentum beta (default: 0.4)
    """
    model_id: str = "AIML-TUDA/stable-diffusion-safe"
    device: Optional[str] = None
    safety_concept: str = "nudity"
    # Defaults mapping to SafetyConfig.MAX usually, but let's expose raw params
    sld_guidance_scale: float = 2000  # Strong default
    sld_warmup_steps: int = 7
    sld_threshold: float = 0.025
    sld_momentum_scale: float = 0.5
    sld_mom_beta: float = 0.7

    @staticmethod
    def from_preset(preset: str) -> 'SLDConfig':
        """
        Helper to get config from presets: 'MAX', 'STRONG', 'MEDIUM', 'WEAK'.
        """
        # Note: In a real implementation, we might import these values from diffusers.SafetyConfig
        # For now, we'll keep it simple or allow the wrapper to handle the mapping if config is passed as dict.
        # This is a placeholder for cleaner preset management.
        return SLDConfig() 
