from dataclasses import dataclass, field
from typing import Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class SalUnConfig(BaseConfig):
    """
    Configuration for SalUn (Saliency Unlearning) for Stable Diffusion.

    SalUn is a two-phase fine-tuning approach:
      Phase 1 — builds a saliency mask by computing gradient magnitudes over
        the forget concept, then thresholding to keep only the top-k% of
        UNet weights (those most responsible for the forget concept).
      Phase 2 — fine-tunes the UNet with mask-gated gradient updates:
        forget loss pulls UNet(erase_concept) toward UNet(anchor_concept);
        retain loss is standard diffusion MSE on the anchor concept.

    forget_data_path: Directory of images representing the concept to forget.
        Images are used in Phase 1 (saliency) and Phase 2 (forget loss).
        Recommended: 100-500 representative images of the forget concept.

    retain_data_path: Directory of images representing general content to preserve.
        Used in Phase 2 retain loss. Should be diverse benign content
        unrelated to the forget concept.

    anchor_concept: Text prompt the model should produce instead of the erased
        concept. Must be semantically adjacent enough to be a coherent substitute
        (e.g. erase "nudity" -> anchor "a person fully clothed").

    threshold: Top fraction of UNet parameters to include in the saliency mask.
        Lower values restrict updates to fewer, more targeted weights.
        Recommended range: 0.1-0.9. Default: 0.5 (top 50%).

    train_method: "full" updates all UNet parameters (masked); "xattn"
        restricts updates to cross-attention layers only.
    """

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"
    use_fp16: bool = False

    # Concept erasure settings
    erase_concept: str = "nudity"
    anchor_concept: str = "a person fully clothed"

    # Dataset paths — both required for training
    forget_data_path: str = ""
    retain_data_path: str = ""

    # Save/load checkpoint (optional, to skip re-training)
    save_path: Optional[str] = None
    load_path: Optional[str] = None

    # Image settings
    image_size: int = 512

    # SalUn hyperparameters
    alpha: float = 0.5
    lr: float = 1e-5
    epochs: int = 5
    batch_size: int = 4
    c_guidance: float = 7.5
    threshold: float = 0.5
    train_method: str = "full"

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        if not self.erase_concept:
            raise ValueError("erase_concept must not be empty.")
        if not self.anchor_concept:
            raise ValueError("anchor_concept must not be empty.")
        if not self.forget_data_path:
            raise ValueError("forget_data_path must be provided.")
        if not self.retain_data_path:
            raise ValueError("retain_data_path must be provided.")
        if not (0.0 < self.threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.train_method not in ("full", "xattn"):
            raise ValueError(
                f"train_method must be 'full' or 'xattn', got {self.train_method!r}"
            )
