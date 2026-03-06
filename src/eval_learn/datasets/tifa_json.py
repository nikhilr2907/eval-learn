from typing import Optional

from torch.utils.data import DataLoader
from datasets import load_dataset as hf_load_dataset

from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger
from .hf_stream import load_hf_config

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


@register_dataset("tifa_json")
def load_tifa_json(
    limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    token: Optional[str] = None,
) -> DataLoader:
    """
    Stream captions and QA pairs from the TIFA dataset directly from HuggingFace.

    Returns a DataLoader that yields Dataset batches. Each batch has:
      - prompts: list of caption strings
      - metadata["qa_pairs"]: list of QA pair lists, parallel to prompts.
        Each element is a list of {"question": str, "answer": str} dicts.

    Args:
        limit:      Max number of rows to stream.
        batch_size: Number of prompts per batch.
        token:      HF token (falls back to HF_TOKEN env var).
    """
    cfg = load_hf_config("tifa")
    caption_col = cfg["caption_col"]
    qa_col = cfg["qa_col"]

    logger.info(
        "Setting up HF streaming for TIFA (%s, split=%s)...",
        cfg["repo_id"], cfg["split"],
    )

    hf_ds = hf_load_dataset(cfg["repo_id"], split=cfg["split"], streaming=True, token=token)
    if limit is not None:
        hf_ds = hf_ds.take(limit)

    def collate_fn(batch):
        return Dataset(
            prompts=[row[caption_col] for row in batch],
            metadata={
                "source": "tifa_hf",
                "repo_id": cfg["repo_id"],
                "qa_pairs": [row[qa_col] for row in batch],
            },
        )

    return DataLoader(hf_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
