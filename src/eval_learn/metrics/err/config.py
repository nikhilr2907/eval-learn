from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class ERRConfig(BaseConfig):
    """
    Configuration for the Erasing-Retention-Robustness (ERR) metric.

    Attributes:
        clip_model_name: HuggingFace CLIP model identifier.
        device: Torch device string (default: None, auto-detect).
        i2p_path: Path to I2P benchmark CSV (target prompts).
        challenge_path: Path to ERR challenge CSV (retain prompts).
        rab_path: Path to Ring-A-Bell CSV (adversarial prompts).
        target_limit: Max target prompts to load.
        retain_limit: Max retain prompts to load.
        adversarial_limit: Max adversarial prompts to load.
    """
    clip_model_name: str = "openai/clip-vit-large-patch14"
    device: Optional[str] = None
    i2p_path: str = "data/i2p/i2p_benchmark_sample.csv"
    challenge_path: str = "data/ERR/raw_csv_data/challenge_dataset.csv"
    rab_path: str = "data/ring_a_bell/ring_a_bell_dataset.csv"
    target_limit: Optional[int] = 100
    retain_limit: Optional[int] = 100
    adversarial_limit: Optional[int] = 100
