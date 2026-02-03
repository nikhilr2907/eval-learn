from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class FIDConfig(BaseConfig):
    """
    Configuration for Frechet Inception Distance (FID) metric.

    Attributes:
        real_images_dir: Path to directory of real reference images.
        batch_size: Batch size for InceptionV3 feature extraction.
        device: Device string (unused by TensorFlow but kept for interface consistency).
    """
    real_images_dir: str = ""
    batch_size: int = 32
    device: Optional[str] = None
