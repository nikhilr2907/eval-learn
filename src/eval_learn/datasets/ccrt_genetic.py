from typing import List, Optional

import torch.utils.data
from torch.utils.data import DataLoader

from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


class _PromptSeedDataset(torch.utils.data.Dataset):
    """Map-style wrapper over parallel prompt and seed lists."""

    def __init__(self, prompts: List[str], seeds: List[int]):
        self._prompts = prompts
        self._seeds = seeds

    def __len__(self):
        return len(self._prompts)

    def __getitem__(self, idx):
        return self._prompts[idx], self._seeds[idx]


@register_dataset("ccrt_genetic")
def load_ccrt_genetic(
    prompts: List[str],
    seeds: List[int],
    concept_name: str,
    concept_desc: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DataLoader:
    """
    Wrap CCRT-generated prompts and seeds into a batched DataLoader.

    This is called after the genetic search and LLM prompt generation are
    complete. The heavy computation (run_genetic_search, generate_prompts,
    baseline image generation) belongs in the metric's load_dataset(); this
    function owns only the collation into a DataLoader.

    Each batch yields a Dataset with:
      - prompts: list of natural language prompt strings
      - metadata["seeds"]:        list of int seeds, parallel to prompts
      - metadata["concept_name"]: the target concept being tested
      - metadata["concept_desc"]: optional description of the concept
      - metadata["source"]:       "ccrt_genetic"

    Args:
        prompts:      Natural language prompts from generate_prompts().
        seeds:        Per-prompt random seeds from generate_prompts().
        concept_name: Name of the concept under evaluation.
        concept_desc: Optional human-readable description of the concept.
        batch_size:   Number of prompts per batch.
    """
    if len(prompts) != len(seeds):
        raise ValueError(
            f"prompts and seeds must have equal length, "
            f"got {len(prompts)} prompts and {len(seeds)} seeds."
        )

    logger.info(
        "Building CCRT DataLoader: %d prompts, batch_size=%d, concept='%s'.",
        len(prompts), batch_size, concept_name,
    )

    _concept_name = concept_name
    _concept_desc = concept_desc

    def collate_fn(batch):
        batch_prompts, batch_seeds = zip(*batch)
        return Dataset(
            prompts=list(batch_prompts),
            metadata={
                "source": "ccrt_genetic",
                "concept_name": _concept_name,
                "concept_desc": _concept_desc,
                "seeds": list(batch_seeds),
            },
        )

    return DataLoader(
        _PromptSeedDataset(prompts, seeds),
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
