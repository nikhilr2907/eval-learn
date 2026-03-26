from typing import Optional

from torch.utils.data import DataLoader
from datasets import load_dataset as hf_load_dataset

from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger
from .hf_stream import load_hf_config

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


@register_dataset("ua_ira_hf")
def load_ua_ira_hf(
    target_concept: str = "Mickey Mouse",
    retain_concept: str = "Minnie Mouse",
    target_limit: Optional[int] = None,
    retain_limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    token: Optional[str] = None,
) -> DataLoader:
    """
    Stream target and retain prompts from HuggingFace ERR dataset.

    Returns a DataLoader that yields Dataset batches. Each batch has:
      - prompts: list of prompt strings (target and retain mixed)
      - metadata["concepts"]: list of concept names parallel to prompts
      - metadata["categories"]: list of "target" | "retain" parallel to prompts
      - metadata["target_prompt_end_index"]: index where target prompts end

    Args:
        target_concept: Name of concept to unlearn (e.g., "Mickey Mouse").
        retain_concept: Name of concept to retain (e.g., "Minnie Mouse").
        target_limit: Max target prompts to load.
        retain_limit: Max retain prompts to load.
        batch_size: Number of prompts per batch.
        token: HF token (falls back to HF_TOKEN env var).
    """
    cfg = load_hf_config("err_challenge")
    caption_col = cfg["caption_col"]
    concept_col = cfg["concept_col"]

    logger.info(
        "Setting up HF streaming for UA_IRA (%s, split=%s)...",
        cfg["repo_id"],
        cfg["split"],
    )

    hf_ds = hf_load_dataset(
        cfg["repo_id"], split=cfg["split"], streaming=True, token=token
    )

    # Collect target and retain prompts
    target_prompts = []
    retain_prompts = []

    for row in hf_ds:
        concept_name = row.get(concept_col, "").lower()
        caption = row.get(caption_col, "")

        if concept_name == target_concept.lower():
            if target_limit is None or len(target_prompts) < target_limit:
                target_prompts.append((caption, target_concept, "target"))
        elif concept_name == retain_concept.lower():
            if retain_limit is None or len(retain_prompts) < retain_limit:
                retain_prompts.append((caption, retain_concept, "retain"))

        # Early exit if both limits reached
        if (
            target_limit
            and len(target_prompts) >= target_limit
            and retain_limit
            and len(retain_prompts) >= retain_limit
        ):
            break

    logger.info(
        f"Loaded {len(target_prompts)} target and {len(retain_prompts)} retain prompts."
    )

    # Combine: target prompts first, then retain
    all_prompts = target_prompts + retain_prompts
    target_prompt_end_index = len(target_prompts)

    def collate_fn(batch):
        return Dataset(
            prompts=[item[0] for item in batch],
            metadata={
                "source": "ua_ira_hf",
                "repo_id": cfg["repo_id"],
                "concepts": [item[1] for item in batch],
                "categories": [item[2] for item in batch],
                "target_prompt_end_index": target_prompt_end_index,
            },
        )

    # Create a simple list-based dataset for batching
    class _PromptDataset:
        def __init__(self, prompts):
            self._prompts = prompts

        def __len__(self):
            return len(self._prompts)

        def __getitem__(self, idx):
            return self._prompts[idx]

    return DataLoader(
        _PromptDataset(all_prompts),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )
