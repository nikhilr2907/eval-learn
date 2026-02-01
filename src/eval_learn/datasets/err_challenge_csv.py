import pandas as pd
import os
from typing import Optional
from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger

logger = get_logger(__name__)

@register_dataset("err_challenge_csv")
def load_err_challenge_csv(
    path: str = "data/ERR/raw_csv_data/challenge_dataset.csv",
    limit: Optional[int] = None,
    prompt_col: str = "direct_prompt",
    concept_col: str = "concept_name",
) -> Dataset:
    """
    Loads prompts from the ERR challenge CSV file.

    Args:
        path: Path to the CSV file.
        limit: Max number of prompts to load.
        prompt_col: Column name containing the prompts.
        concept_col: Column name containing the concept names.

    Returns:
        Dataset with prompts and concept metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERR challenge dataset not found at: {path}")

    logger.info(f"Loading ERR challenge dataset from {path}...")
    df = pd.read_csv(path)

    for col in (prompt_col, concept_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. Columns: {df.columns.tolist()}")

    if limit:
        df = df.head(limit)

    prompts = df[prompt_col].tolist()
    concepts = df[concept_col].tolist()

    logger.info(f"Loaded {len(prompts)} prompts.")

    return Dataset(
        prompts=prompts,
        metadata={
            "source": "err_challenge_csv",
            "path": path,
            "total_loaded": len(prompts),
            "concepts": concepts,
        },
    )
