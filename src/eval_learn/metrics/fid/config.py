from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class FIDConfig(BaseConfig):
    """
    Configuration for Frechet Inception Distance (FID) metric.

    Uses PyTorch InceptionV3 for feature extraction — no TensorFlow required.
    Real images are loaded directly from the parquet into memory
    without writing to disk.

    Attributes:
        batch_size: Batch size for InceptionV3 feature extraction.
        device: Device string (e.g. "cuda", "cpu").
        parquet_path: Path to COCO parquet file.
        caption_col: Column name containing captions.
        image_col: Column name containing image dicts.
        limit: Max number of rows to load.
    """
    batch_size: int = 32
    device: Optional[str] = None
    parquet_path: str = "data/coco/coco_3k_sample.parquet"
    caption_col: str = "caption"
    image_col: str = "image"
    limit: Optional[int] = 1000
