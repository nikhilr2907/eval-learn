import os
import io
from typing import Optional
from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger

logger = get_logger(__name__)


@register_dataset("coco_parquet")
def load_coco_parquet(
    path: str = "data/coco/coco_3k_sample.parquet",
    limit: Optional[int] = None,
    caption_col: str = "caption",
    image_col: str = "image",
) -> Dataset:
    """
    Load captions and reference images from a COCO parquet file.

    Each row is expected to have an ``image`` column containing a dict
    with a ``bytes`` key (raw JPEG/PNG bytes) and a ``caption`` column
    with the text prompt.

    PIL images are returned in ``metadata["images"]`` for downstream use
    (e.g. feature extraction by FIDMetric).

    Args:
        path: Path to the parquet file.
        limit: Max number of rows to load.
        caption_col: Column name containing captions.
        image_col: Column name containing image dicts.
    """
    import pandas as pd
    from PIL import Image

    if not os.path.exists(path):
        raise FileNotFoundError(f"COCO parquet not found at: {path}")

    logger.info(f"Loading COCO parquet from {path}...")
    df = pd.read_parquet(path)

    if caption_col not in df.columns:
        raise ValueError(f"Column '{caption_col}' not found. Columns: {df.columns.tolist()}")
    if image_col not in df.columns:
        raise ValueError(f"Column '{image_col}' not found. Columns: {df.columns.tolist()}")

    if limit:
        df = df.head(limit)

    captions = df[caption_col].tolist()

    logger.info(f"Decoding {len(df)} reference images...")
    images = []
    for _, row in df.iterrows():
        img_data = row[image_col]
        img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
        images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    logger.info(f"Loaded {len(captions)} captions and {len(images)} images.")

    return Dataset(
        prompts=captions,
        metadata={
            "source": "coco_parquet",
            "path": path,
            "total_loaded": len(captions),
            "images": images,
        },
    )
