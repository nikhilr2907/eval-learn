from dataclasses import dataclass, field
from typing import Optional
from ...configs.base import BaseConfig

_VALID_EMB_COMPUTING = {"close_regzero", "close_standardreg", "close_surrogatereg"}
_VALID_CONCEPT_TYPES = {"object", "style", "attribute"}


@dataclass(frozen=True)
class RECEConfig(BaseConfig):
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"
    use_fp16: bool = True

    # Load pre-built weights directly (skips training)
    load_path: Optional[str] = None

    # Custom concept creation — used when load_path is not given
    erase_concept: Optional[str] = None
    concept_type: str = "object"
    emb_computing: str = "close_regzero"
    save_path: Optional[str] = None

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        if self.load_path is None and self.erase_concept is None:
            raise ValueError(
                "RECE requires either 'load_path' (pre-built weights) "
                "or 'erase_concept' + 'save_path' (create weights inline)."
            )
        if self.load_path is not None and self.erase_concept is not None:
            raise ValueError(
                "RECE: 'load_path' and 'erase_concept' are mutually exclusive. "
                "Provide one or the other, not both."
            )
        if self.erase_concept is not None and not self.erase_concept.strip():
            raise ValueError("'erase_concept' must be a non-empty string.")
        if self.erase_concept is not None and not self.save_path:
            raise ValueError(
                "Creating RECE weights inline requires 'save_path' — "
                "weight creation takes 5–30 minutes and the result must be persisted."
            )
        if self.emb_computing not in _VALID_EMB_COMPUTING:
            raise ValueError(
                f"emb_computing must be one of {sorted(_VALID_EMB_COMPUTING)}, "
                f"got '{self.emb_computing}'."
            )
        if self.concept_type not in _VALID_CONCEPT_TYPES:
            raise ValueError(
                f"concept_type must be one of {sorted(_VALID_CONCEPT_TYPES)}, "
                f"got '{self.concept_type}'."
            )
