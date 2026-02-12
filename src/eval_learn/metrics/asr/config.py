from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class ASRConfig(BaseConfig):
    """
    Configuration for Attack Success Rate (ASR) metric.

    Attributes:
        use_nudenet: Whether to use NudeNet detector (default: True).
        use_q16: Whether to use Q16/CLIP detector (default: False).
        device: Device for detectors (default: None, auto-detect).
        dataset_path: Path to I2P benchmark CSV.
        prompt_col: Column name containing prompts.
        limit: Max number of prompts to load.
    """
    use_nudenet: bool = True
    use_q16: bool = False
    device: str = None
    dataset_path: str = "data/i2p/i2p_benchmark.csv"
    prompt_col: str = "prompt"
    limit: Optional[int] = 500
