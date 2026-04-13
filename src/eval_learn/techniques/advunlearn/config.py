import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig

# Fixed train_method strings
TRAIN_METHODS = [
    "text_encoder_full",
    "noxattn",
    "selfattn",
    "xattn",
    "full",
    "notime",
    "xlayer",
    "selflayer",
]
# Also accepts "text_encoder_layer<digits>" e.g. "text_encoder_layer012_910"
_TRAIN_METHOD_LAYER_RE = re.compile(r"^text_encoder_layer[\d_]+$")

RETAIN_TRAIN_METHODS = ["iter", "reg"]

ATTACK_METHODS = ["pgd", "fast_at"]

ATTACK_TYPES = ["prefix_k", "suffix_k", "replace_k", "add", "mid_k", "insert_k", "per_k_words"]

ATTACK_EMBD_TYPES = ["word_embd"]

COMPONENTS = ["all", "ffn", "attn"]

RETAIN_DATASETS = ["coco_object", "imagenet243", "coco_object_no_filter", "imagenet243_no_filter"]


@dataclass(frozen=True)
class AdvUnlearnConfig(BaseConfig):
    """Configuration for AdvUnlearn (Adversarially Robust Concept Unlearning)."""

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"
    use_fp16: bool = True

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
    train_steps: int = 5
    learning_rate: float = 1e-5

    # Adversarial attack settings
    attack_method: str = "pgd"
    attack_step: int = 30
    attack_lr: float = 1e-3
    attack_type: str = "prefix_k"
    attack_init: str = "latest"
    attack_embd_type: str = "word_embd"
    adv_prompt_num: int = 1
    adv_prompt_update_step: int = 1
    warmup_iter: int = 1

    # Model component selection
    component: str = "all"
    norm_layer: bool = False

    # Training resolution / DDIM settings
    ddim_steps: int = 50
    save_interval: int = 1

    # Misc
    save_dir: Optional[str] = None
    checkpoint_path: Optional[str] = None
    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvUnlearnConfig":
        data = dict(data)

        train_method = data.get("train_method", "text_encoder_full")
        if train_method not in TRAIN_METHODS and not _TRAIN_METHOD_LAYER_RE.match(train_method):
            raise ValueError(
                f"Unknown train_method '{train_method}'. "
                f"Available: {TRAIN_METHODS} or 'text_encoder_layer<digits>' e.g. 'text_encoder_layer012_910'"
            )

        retain_train = data.get("retain_train", "iter")
        if retain_train not in RETAIN_TRAIN_METHODS:
            raise ValueError(
                f"Unknown retain_train '{retain_train}'. Available: {RETAIN_TRAIN_METHODS}"
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

        attack_embd_type = data.get("attack_embd_type", "word_embd")
        if attack_embd_type not in ATTACK_EMBD_TYPES:
            raise ValueError(
                f"Unknown attack_embd_type '{attack_embd_type}'. Available: {ATTACK_EMBD_TYPES}"
            )

        component = data.get("component", "all")
        if component not in COMPONENTS:
            raise ValueError(
                f"Unknown component '{component}'. Available: {COMPONENTS}"
            )

        return super().from_dict(data)

    def __post_init__(self):
        if self.train_steps <= 0:
            raise ValueError(f"train_steps must be > 0, got {self.train_steps}")
        if self.attack_step <= 0:
            raise ValueError(f"attack_step must be > 0, got {self.attack_step}")
        if self.retain_batch <= 0:
            raise ValueError(f"retain_batch must be > 0, got {self.retain_batch}")
        if self.retain_step <= 0:
            raise ValueError(f"retain_step must be > 0, got {self.retain_step}")
        if self.adv_prompt_num <= 0:
            raise ValueError(f"adv_prompt_num must be > 0, got {self.adv_prompt_num}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.attack_lr <= 0:
            raise ValueError(f"attack_lr must be > 0, got {self.attack_lr}")
        if self.warmup_iter >= self.train_steps:
            raise ValueError(
                f"warmup_iter must be < train_steps, got warmup_iter={self.warmup_iter}, train_steps={self.train_steps}"
            )
