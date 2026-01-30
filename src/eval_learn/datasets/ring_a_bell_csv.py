import pandas as pd
import os
from typing import Optional
from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger

logger = get_logger(__name__)

@register_dataset("ring_a_bell_csv")
def load_ring_a_bell_csv(
    path: str = "data/ring_a_bell/ring_a_bell_dataset.csv",
    limit: Optional[int] = None,
) -> Dataset:
    """
    Loads prompts from the Ring-A-Bell CSV file.

    Args:
        path: Path to the CSV file.
        limit: Max number of prompts to load.

    Returns:
        Dataset with prompts and concept metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ring-A-Bell dataset not found at: {path}")

    logger.info(f"Loading Ring-A-Bell dataset from {path}...")
    df = pd.read_csv(path)

    for col in ("prompt", "concept"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. Columns: {df.columns.tolist()}")

    if limit:
        df = df.head(limit)

    prompts = df["prompt"].tolist()
    concepts = df["concept"].tolist()

    logger.info(f"Loaded {len(prompts)} prompts.")

    return Dataset(
        prompts=prompts,
        metadata={
            "source": "ring_a_bell_csv",
            "path": path,
            "total_loaded": len(prompts),
            "concepts": concepts,
        },
    )
