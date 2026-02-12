from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class FIDConfig(BaseConfig):
    """
    Configuration for Frechet Inception Distance (FID) metric.

    Attributes:
        real_images_dir: Path to directory of real reference images.
            Populated automatically by load_dataset() when using the
            default COCO parquet.
        batch_size: Batch size for InceptionV3 feature extraction.
        device: Device string (unused by TensorFlow but kept for interface consistency).
        parquet_path: Path to COCO parquet file.
        caption_col: Column name containing captions.
        image_col: Column name containing image dicts.
        limit: Max number of rows to load.
    """
    real_images_dir: str = "data/coco/real_images"
    batch_size: int = 32
    device: Optional[str] = None
    parquet_path: str = "data/coco/coco_3k_sample.parquet"
    caption_col: str = "caption"
    image_col: str = "image"
    limit: Optional[int] = 1000
