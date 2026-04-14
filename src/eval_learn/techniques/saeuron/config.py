import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List

import saeuron
from ...configs.base import BaseConfig

_VALID_ERASE_CONCEPTS = {"nudity"}


@dataclass(frozen=True)
class SAeUronConfig(BaseConfig):
    """Configuration for SAeUron (Sparse Autoencoder Unlearning)."""

    # --- Base Model Parameters ---
    # Fixed: SAE was trained on SD 1.4 and is not compatible with other models.
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    # Target computation device
    device: str = "cuda"
    use_fp16: bool = True

    # --- Paths ---
    # Path to the directory containing the SAE weights (cfg.json & sae.safetensors)
    sae_path: Optional[str] = None

    # Path to the .pkl file containing cached concept activations for feature selection.
    acts_path: Optional[str] = None

    # --- Unlearning Parameters ---
    # The string path to the UNet module to hook.
    # Hardcoded default for Object Unlearning based on the official repository.
    position: str = "unet.up_blocks.1.attentions.1"

    # The specific concept name to unlearn. 
    erase_concept: str = "nudity"

    # Multiplier applied to the target latents.
    # Negative values indicate ablation/unlearning.
    multiplier: float = -20.0

    # The percentile threshold used to select which SAE features represent the target concept.
    # Only features with activation scores above this percentile will be modified.
    percentile: float = 99.99

    # Explicit list of latent indices to target.
    # If left empty, the wrapper will calculate them dynamically using `acts_path` and `percentile`.
    target_latents: List[int] = field(default_factory=list)

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        if not self.erase_concept:
            raise ValueError("erase_concept must not be empty.")
        if self.multiplier == 0:
            raise ValueError("multiplier must not be 0 — use a negative value to ablate.")
        if not (0 < self.percentile <= 100):
            raise ValueError(f"percentile must be in (0, 100], got {self.percentile}.")
        if self.guidance_scale <= 1.0:
            raise ValueError("guidance_scale must be > 1.0 — SAeUron requires CFG to be active.")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SAeUronConfig":
        config_dict = dict(config_dict)

        erase_concept = config_dict.get("erase_concept", "nudity")
        using_bundled_defaults = (
            not config_dict.get("acts_path") and not config_dict.get("target_latents")
        )

        # Restrict to nudity only when using bundled activation cache.
        # Custom acts_path or explicit target_latents can support any concept.
        if using_bundled_defaults and erase_concept.lower() not in _VALID_ERASE_CONCEPTS:
            raise ValueError(
                f"The bundled activation cache only supports: {sorted(_VALID_ERASE_CONCEPTS)}. "
                f"Got erase_concept='{erase_concept}'. "
                f"To erase a different concept, supply a custom acts_path or explicit target_latents."
            )

        # Get the absolute path of the saeuron package
        saeuron_pkg_path = os.path.dirname(saeuron.__file__)

        # Resolve SAE checkpoint path to bundled default if not provided
        if not config_dict.get("sae_path"):
            config_dict["sae_path"] = os.path.join(saeuron_pkg_path, "checkpoints")

        # Resolve activation cache path to bundled default if not provided
        if not config_dict.get("acts_path"):
            config_dict["acts_path"] = os.path.join(
                saeuron_pkg_path, "core", "cls_latents_dict_mini.pkl"
            )

        return super().from_dict(config_dict)
