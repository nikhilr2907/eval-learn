from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig

TRAIN_METHODS = [
    "text_encoder_full",
    "xattn",
    "noxattn",
    "selfattn",
    "full",
]

ATTACK_METHODS = ["pgd", "fast_at"]

ATTACK_TYPES = ["prefix_k", "suffix_k", "replace_k", "add", "mid_k", "insert_k", "per_k_words"]

RETAIN_DATASETS = ["coco_object", "imagenet243", "coco_object_no_filter", "imagenet243_no_filter"]


@dataclass(frozen=True)
class AdvUnlearnConfig(BaseConfig):
    """Configuration for AdvUnlearn (Adversarially Robust Concept Unlearning)."""

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"

    # Concept erasure
    erase_concept: str = "nudity"

    # Training method
    train_method: str = "text_encoder_full"

    # Retention settings
    dataset_retain: str = "coco_object"
    retain_train: str = "iter"
    retain_batch: int = 5
    retain_step: int = 1
    retain_loss_w: float = 1.0

    # Unlearning loss settings
    start_guidance: float = 3.0
    negative_guidance: float = 1.0
    iterations: int = 1
    lr: float = 1e-5

    # Adversarial attack settings
    attack_method: str = "pgd"
    attack_step: int = 5
    attack_lr: float = 1e-3
    attack_type: str = "prefix_k"
    attack_init: str = "latest"
    attack_embd_type: str = "word_embd"
    adv_prompt_num: int = 1
    adv_prompt_update_step: int = 1
    warmup_iter: int = 0

    # Model component selection
    component: str = "all"
    norm_layer: bool = False

    # Training resolution / DDIM settings
    ddim_steps: int = 10
    save_interval: int = 200

    # Misc
    save_path: Optional[str] = None
    cache_dir: str = ".cache"

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvUnlearnConfig":
        data = dict(data)

        train_method = data.get("train_method", "text_encoder_full")
        if train_method not in TRAIN_METHODS:
            raise ValueError(
                f"Unknown train_method '{train_method}'. Available: {TRAIN_METHODS}"
            )

        attack_method = data.get("attack_method", "pgd")
        if attack_method not in ATTACK_METHODS:
            raise ValueError(
                f"Unknown attack_method '{attack_method}'. Available: {ATTACK_METHODS}"
            )

        attack_type = data.get("attack_type", "prefix_k")
        if attack_type not in ATTACK_TYPES:
            raise ValueError(
                f"Unknown attack_type '{attack_type}'. Available: {ATTACK_TYPES}"
            )

        return super().from_dict(data)
