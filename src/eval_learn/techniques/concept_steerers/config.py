import os
try:
    from dataclasses import dataclass
except ImportError:
    raise RuntimeError(
        "SDSAERunnerConfig requires 'dataclasses'. Please ensure you are using Python 3.7+ or install it via: pip install dataclasses"
    )
from typing import Optional, Any, Dict

@dataclass
class ConceptSteerersConfig:
    model_id: str = "CompVis/stable-diffusion-v1-4"
    device: str = "cuda"
    sae_path: Optional[str] = None
    concept: str = "nudity"
    multiplier: float = 1.0

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        
        if 'sae_path' not in config_dict or not config_dict['sae_path']:
            # techniques/concept_steerers/
            base_dir = os.path.dirname(os.path.abspath(__file__))
            #  checkpoint
            config_dict['sae_path'] = os.path.join(base_dir, "checkpoints", "i2p_sd14_l9")
            
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__dataclass_fields__
        })