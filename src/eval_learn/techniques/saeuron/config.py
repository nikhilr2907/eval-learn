import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List

from ...configs.base import BaseConfig

@dataclass
class SAeUronConfig(BaseConfig):
    """
    Configuration for the SAeUron (Sparse Autoencoder Unlearning) technique.
    
    This configuration defines the paths to the SAE checkpoints and cached 
    feature activations, as well as the hyper-parameters required to ablate 
    or steer specific concepts (e.g., objects or styles) during the diffusion 
    generation process.
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
    # We default this to the 6GB object latents file you just generated.
    acts_path: Optional[str] = None
    
    # --- Unlearning Parameters ---
    # The string path to the UNet module to hook. 
    # Hardcoded default for Object Unlearning based on the official repository.
    position: str = "unet.up_blocks.1.attentions.1"
    
    # The specific concept name to unlearn (must match a key in your acts_path .pkl file).
    concept: str = "church"
    
    # Multiplier applied to the target latents. 
    # Negative values indicate ablation/unlearning. For object unlearning, 
    # values between -10.0 and -30.0 are generally recommended.
    multiplier: float = -10.0
    
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

        # Get the absolute path of the directory containing this config.py file
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. Resolve SAE Checkpoint Path automatically
        if 'sae_path' not in config_dict or not config_dict['sae_path']:
            # Points to the 'checkpoints' folder where cfg.json and sae.safetensors are stored
            config_dict['sae_path'] = os.path.join(base_dir, "checkpoints")

        # 2. Resolve Cached Activations Path automatically
        if 'acts_path' not in config_dict or not config_dict['acts_path']:
            # Points to the generated .pkl file inside the 'core' directory
            config_dict['acts_path'] = os.path.join(
                base_dir, 
                "core", 
                "cls_latents_dict_unet.up_blocks.1.attentions.1.pkl"
            )

        return super().from_dict(config_dict)