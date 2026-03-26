from typing import Optional

import torch.utils.data
from torch.utils.data import DataLoader
from datasets import load_dataset as hf_load_dataset

from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger
from .hf_stream import load_hf_config

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


class _RowDataset(torch.utils.data.Dataset):
    """Simple map-style wrapper over a flat list of (prompt, concept, category) tuples."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


@register_dataset("err_composite")
def load_err_composite(
    target_limit: Optional[int] = None,
    retain_limit: Optional[int] = None,
    adversarial_limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    token: Optional[str] = None,
) -> DataLoader:
    """
    Stream and combine ERR composite data from three HuggingFace sources:
      - I2P (target prompts)
      - ERR challenge (retain prompts)
      - Ring-A-Bell (adversarial prompts)

    Each source is streamed separately (text is lightweight) then combined
    into a single DataLoader. Each batch yields a Dataset with:
      - prompts: list of prompt strings
      - metadata["concepts"]: list of concept strings, parallel to prompts
      - metadata["categories"]: list of "target" | "retain" | "adversarial"

    Args:
        target_limit:      Max rows from I2P.
        retain_limit:      Max rows from ERR challenge.
        adversarial_limit: Max rows from Ring-A-Bell.
        batch_size:        Number of rows per batch.
        token:             HF token (falls back to HF_TOKEN env var).
    """
    i2p_cfg = load_hf_config("i2p")
    challenge_cfg = load_hf_config("err_challenge")
    rab_cfg = load_hf_config("ring_a_bell")

    rows = []

    # --- I2P (target) ---
    logger.info("Streaming I2P (target) from %s...", i2p_cfg["repo_id"])
    i2p_ds = hf_load_dataset(
        i2p_cfg["repo_id"], split=i2p_cfg["split"], streaming=True, token=token
    )
    if target_limit is not None:
        i2p_ds = i2p_ds.take(target_limit)
    for row in i2p_ds:
        rows.append(
            (row[i2p_cfg["caption_col"]], row[i2p_cfg["concept_col"]], "target")
        )

    # --- ERR challenge (retain) ---
    logger.info("Streaming ERR challenge (retain) from %s...", challenge_cfg["repo_id"])
    ch_ds = hf_load_dataset(
        challenge_cfg["repo_id"],
        split=challenge_cfg["split"],
        streaming=True,
        token=token,
    )
    if retain_limit is not None:
        ch_ds = ch_ds.take(retain_limit)
    for row in ch_ds:
        rows.append(
            (
                row[challenge_cfg["caption_col"]],
                row[challenge_cfg["concept_col"]],
                "retain",
            )
        )

    # --- Ring-A-Bell (adversarial) ---
    logger.info("Streaming Ring-A-Bell (adversarial) from %s...", rab_cfg["repo_id"])
    rab_ds = hf_load_dataset(
        rab_cfg["repo_id"], split=rab_cfg["split"], streaming=True, token=token
    )
    if adversarial_limit is not None:
        rab_ds = rab_ds.take(adversarial_limit)
    for row in rab_ds:
        rows.append(
            (row[rab_cfg["caption_col"]], row[rab_cfg["concept_col"]], "adversarial")
        )

    logger.info("ERR composite: %d total rows collected.", len(rows))

    def collate_fn(batch):
        prompts, concepts, categories = zip(*batch)
        return Dataset(
            prompts=list(prompts),
            metadata={
                "source": "err_composite_hf",
                "concepts": list(concepts),
                "categories": list(categories),
            },
        )

    return DataLoader(_RowDataset(rows), batch_size=batch_size, collate_fn=collate_fn)
