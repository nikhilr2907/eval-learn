from dataclasses import dataclass
from ...configs.base import BaseConfig

@dataclass
class ASRConfig(BaseConfig):
    """
    Configuration for Attack Success Rate (ASR) metric.
    
    Attributes:
        use_nudenet: Whether to use NudeNet detector (default: True).
        use_q16: Whether to use Q16/CLIP detector (default: False).
        device: Device for detectors (default: None, auto-detect).
    """
    use_nudenet: bool = True
    use_q16: bool = False
    device: str = None
