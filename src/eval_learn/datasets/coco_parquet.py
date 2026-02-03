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
    real_images_dir: Optional[str] = None,
) -> Dataset:
    """
    Load captions and reference images from a COCO parquet file.

    Each row is expected to have an ``image`` column containing a dict
    with a ``bytes`` key (raw JPEG/PNG bytes) and a ``caption`` column
    with the text prompt.

    Reference images are extracted to *real_images_dir* (defaults to
    ``data/coco/real_images/``) so that the FID metric can read them.

    Args:
        path: Path to the parquet file.
        limit: Max number of rows to load.
        caption_col: Column name containing captions.
        image_col: Column name containing image dicts.
        real_images_dir: Directory to extract reference images into.
            If ``None``, defaults to ``data/coco/real_images/``.
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

    # --- extract reference images to disk ---
    if real_images_dir is None:
        real_images_dir = os.path.join(os.path.dirname(path), "real_images")

    os.makedirs(real_images_dir, exist_ok=True)

    # Only extract if the directory is empty or has fewer images than we need
    existing = [f for f in os.listdir(real_images_dir) if f.endswith((".jpg", ".png"))]
    if len(existing) >= len(df):
        logger.info(f"Real images already extracted ({len(existing)} files in {real_images_dir}), skipping.")
    else:
        logger.info(f"Extracting {len(df)} reference images to {real_images_dir}...")
        for idx, row in df.iterrows():
            img_data = row[image_col]
            img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img.save(os.path.join(real_images_dir, f"{idx:05d}.jpg"), "JPEG")
        logger.info("Extraction complete.")

    logger.info(f"Loaded {len(captions)} captions.")

    return Dataset(
        prompts=captions,
        metadata={
            "source": "coco_parquet",
            "path": path,
            "total_loaded": len(captions),
            "real_images_dir": real_images_dir,
        },
    )
