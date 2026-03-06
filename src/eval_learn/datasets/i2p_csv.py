from typing import Optional

from torch.utils.data import DataLoader
from datasets import load_dataset as hf_load_dataset

from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger
from .hf_stream import load_hf_config

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


@register_dataset("i2p_csv")
def load_i2p_csv(
    limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    token: Optional[str] = None,
) -> DataLoader:
    """
    Stream prompts from the I2P dataset directly from HuggingFace.

    Returns a DataLoader that yields Dataset batches. Each batch has:
      - prompts: list of prompt strings

    Args:
        limit:      Max number of rows to stream.
        batch_size: Number of prompts per batch.
        token:      HF token (falls back to HF_TOKEN env var).
    """
    cfg = load_hf_config("i2p")
    caption_col = cfg["caption_col"]

    logger.info(
        "Setting up HF streaming for I2P (%s, split=%s)...",
        cfg["repo_id"], cfg["split"],
    )

    hf_ds = hf_load_dataset(cfg["repo_id"], split=cfg["split"], streaming=True, token=token)
    if limit is not None:
        hf_ds = hf_ds.take(limit)

    def collate_fn(batch):
        return Dataset(
            prompts=[row[caption_col] for row in batch],
            metadata={
                "source": "i2p_hf",
                "repo_id": cfg["repo_id"],
            },
        )

    return DataLoader(hf_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
