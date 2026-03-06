import io
from typing import Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset as hf_load_dataset

from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger
from .hf_stream import load_hf_config

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


@register_dataset("coco_parquet")
def load_coco_parquet(
    limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    token: Optional[str] = None,
) -> DataLoader:
    """
    Stream COCO images and captions directly from HuggingFace.

    Returns a DataLoader that yields Dataset batches. Each batch has:
      - prompts: list of caption strings
      - metadata["images"]: (B, C, H, W) float tensor, preprocessed for InceptionV3

    Args:
        limit:      Max number of rows to stream.
        batch_size: Number of images per batch.
        token:      HF token (falls back to HF_TOKEN env var).
    """
    from torchvision import transforms
    from PIL import Image

    cfg = load_hf_config("coco")

    _inception_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    logger.info(
        "Setting up HF streaming for COCO (%s, split=%s)...",
        cfg["repo_id"], cfg["split"],
    )

    hf_ds = hf_load_dataset(cfg["repo_id"], split=cfg["split"], streaming=True, token=token)
    if limit is not None:
        hf_ds = hf_ds.take(limit)

    image_col = cfg["image_col"]
    caption_col = cfg["caption_col"]

    def collate_fn(batch):
        images = []
        captions = []

        for row in batch:
            # Decode image from HF's various formats
            img = row[image_col]
            if isinstance(img, dict) and "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            elif not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert("RGB")
            else:
                img = img.convert("RGB")
            images.append(_inception_transform(img))

            # COCO captions may be a list — take the first
            caption = row[caption_col]
            if isinstance(caption, list):
                caption = caption[0]
            captions.append(caption)

        return Dataset(
            prompts=captions,
            metadata={
                "source": "coco_hf",
                "repo_id": cfg["repo_id"],
                "images": torch.stack(images),  # (B, C, H, W)
            },
        )

    return DataLoader(hf_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
