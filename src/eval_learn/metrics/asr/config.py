from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class ASRConfig(BaseConfig):
    """
    Configuration for Attack Success Rate (ASR) metric.

    Attributes:
        use_nudenet: Whether to use NudeNet detector (default: True).
        use_q16: Whether to use Q16/CLIP detector (default: False).
        device: Device for detectors (default: None, auto-detect).
        dataset_path: Local directory to cache the I2P dataset snapshot.
        prompt_col: Column name containing prompts.
        limit: Max number of prompts to load.
    """

    use_nudenet: bool = True
    use_q16: bool = False
    device: str = None
    limit: Optional[int] = 500
