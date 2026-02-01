import pandas as pd
import os
from typing import Optional
from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger

logger = get_logger(__name__)

@register_dataset("err_composite")
def load_err_composite(
    i2p_path: str = "data/i2p/i2p_benchmark_sample.csv",
    challenge_path: str = "data/ERR/raw_csv_data/challenge_dataset.csv",
    rab_path: str = "data/ring_a_bell/ring_a_bell_dataset.csv",
    target_limit: Optional[int] = None,
    retain_limit: Optional[int] = None,
    adversarial_limit: Optional[int] = None,
) -> Dataset:
    """
    Composite loader that assembles the full ERR dataset from three sources:
      - I2P (target prompts)
      - ERR challenge CSV (retain prompts)
      - Ring-A-Bell (adversarial prompts)

    Returns a Dataset whose metadata contains parallel ``concepts`` and
    ``categories`` lists as required by the ERR metric.

    Args:
        i2p_path: Path to the I2P benchmark sample CSV.
        challenge_path: Path to the ERR challenge CSV.
        rab_path: Path to the Ring-A-Bell CSV.
        target_limit: Max prompts to load from I2P.
        retain_limit: Max prompts to load from ERR challenge.
        adversarial_limit: Max prompts to load from Ring-A-Bell.
    """
    all_prompts = []
    all_concepts = []
    all_categories = []

    # --- I2P (target) ---
    if not os.path.exists(i2p_path):
        raise FileNotFoundError(f"I2P dataset not found at: {i2p_path}")
    logger.info(f"Loading I2P (target) from {i2p_path}...")
    i2p_df = pd.read_csv(i2p_path)
    if target_limit:
        i2p_df = i2p_df.head(target_limit)
    all_prompts.extend(i2p_df["prompt"].tolist())
    # I2P has a 'categories' column; use it as both concept and category
    all_concepts.extend(i2p_df["categories"].tolist())
    all_categories.extend(["target"] * len(i2p_df))
    target_count = len(i2p_df)

    # --- ERR challenge (retain) ---
    if not os.path.exists(challenge_path):
        raise FileNotFoundError(f"ERR challenge dataset not found at: {challenge_path}")
    logger.info(f"Loading ERR challenge (retain) from {challenge_path}...")
    ch_df = pd.read_csv(challenge_path)
    if retain_limit:
        ch_df = ch_df.head(retain_limit)
    all_prompts.extend(ch_df["direct_prompt"].tolist())
    all_concepts.extend(ch_df["concept_name"].tolist())
    all_categories.extend(["retain"] * len(ch_df))
    retain_count = len(ch_df)

    # --- Ring-A-Bell (adversarial) ---
    if not os.path.exists(rab_path):
        raise FileNotFoundError(f"Ring-A-Bell dataset not found at: {rab_path}")
    logger.info(f"Loading Ring-A-Bell (adversarial) from {rab_path}...")
    rab_df = pd.read_csv(rab_path)
    if adversarial_limit:
        rab_df = rab_df.head(adversarial_limit)
    all_prompts.extend(rab_df["prompt"].tolist())
    all_concepts.extend(rab_df["concept"].tolist())
    all_categories.extend(["adversarial"] * len(rab_df))
    adversarial_count = len(rab_df)

    total = len(all_prompts)
    logger.info(
        f"Loaded ERR composite dataset: {total} prompts "
        f"(target={target_count}, retain={retain_count}, adversarial={adversarial_count})"
    )

    return Dataset(
        prompts=all_prompts,
        metadata={
            "source": "err_composite",
            "concepts": all_concepts,
            "categories": all_categories,
            "total_loaded": total,
            "counts": {
                "target": target_count,
                "retain": retain_count,
                "adversarial": adversarial_count,
            },
        },
    )
