import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List

import saeuron
from ...configs.base import BaseConfig

_VALID_ERASE_CONCEPTS = {"nudity"}


@dataclass
class SAeUronConfig(BaseConfig):
    """
    Configuration for the SAeUron (Sparse Autoencoder Unlearning) technique.

    SAeUron only supports nudity concept erasure. The cached activations pickle
    file (cls_latents_dict_mini.pkl) contains pre-computed latent importance scores
    only for the nudity concept.

    This configuration defines the paths to the SAE checkpoints and cached
    feature activations, as well as the hyper-parameters required to ablate
    or steer specific concepts during the diffusion generation process.
    """

    # --- Base Model Parameters ---
    # Identifier for the base diffusion model (must match the one used for SAE training)
    model_id: str = "CompVis/stable-diffusion-v1-4"
    # Target computation device
    device: str = "cuda"

    # --- Paths ---
    # Path to the directory containing the SAE weights (cfg.json & sae.safetensors)
    sae_path: Optional[str] = None

    # Path to the .pkl file containing cached concept activations for feature selection.
    acts_path: Optional[str] = None

    # --- Unlearning Parameters ---
    # The string path to the UNet module to hook.
    # Hardcoded default for Object Unlearning based on the official repository.
    position: str = "unet.up_blocks.1.attentions.1"

    # The specific concept name to unlearn. SAeUron only supports nudity.
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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SAeUronConfig':
        """
        Creates a SAeUronConfig instance from a dictionary.

        This method automatically resolves relative paths to absolute paths
        based on the location of this config.py file, ensuring the wrapper
        can always find the checkpoints and .pkl files regardless of where
        the evaluation script is executed from.
        """
        config_dict = dict(config_dict)

        # Pop model_id to let the global evaluation pipeline manage it if needed,
        # or fall back to the dataclass default.
        config_dict.pop("model_id", None)

        # Validate erase_concept
        erase_concept = config_dict.get("erase_concept", "nudity")
        if erase_concept.lower() not in _VALID_ERASE_CONCEPTS:
            raise ValueError(
                f"SAeUron only supports nudity concept erasure. "
                f"Got erase_concept='{erase_concept}'. "
                f"Available: {sorted(_VALID_ERASE_CONCEPTS)}"
            )

        # Get the absolute path of the saeuron package
        saeuron_pkg_path = os.path.dirname(saeuron.__file__)

        # 1. Resolve SAE Checkpoint Path automatically
        if 'sae_path' not in config_dict or not config_dict['sae_path']:
            # Points to the 'checkpoints' folder in the saeuron package
            config_dict['sae_path'] = os.path.join(saeuron_pkg_path, "checkpoints")

        # 2. Resolve Cached Activations Path automatically
        if 'acts_path' not in config_dict or not config_dict['acts_path']:
            config_dict['acts_path'] = os.path.join(
                saeuron_pkg_path,
                "core",
                "cls_latents_dict_mini.pkl"
            )

        return super().from_dict(config_dict)