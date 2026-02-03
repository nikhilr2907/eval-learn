import pandas as pd
import os
from typing import Optional, List
from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger

logger = get_logger(__name__)

@register_dataset("i2p_csv")
def load_i2p_csv(
    path: str = "data/i2p/i2p_benchmark.csv", 
    limit: Optional[int] = None,
    prompt_col: str = "prompt"
) -> Dataset:
    """
    Loads prompts from the I2P CSV file.
    
    Args:
        path: Path to the CSV file.
        limit: Max number of prompts to load.
        prompt_col: Column name containing the prompts.
        
    Returns:
        Dataset object containing the prompts and metadata.
    """
    if not os.path.exists(path):
        # Fallback for when running from package root vs project root
        # Or better, fail if file not found.
        # But for now, let's try to be helpful if relative path is tricky.
        raise FileNotFoundError(f"I2P dataset not found at: {path}")

    logger.info(f"Loading I2P dataset from {path}...")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise e

    if prompt_col not in df.columns:
        raise ValueError(f"Column '{prompt_col}' not found in CSV. Columns: {df.columns.tolist()}")

    prompts = df[prompt_col].tolist()
    
    if limit:
        prompts = prompts[:limit]

    logger.info(f"Loaded {len(prompts)} prompts.")
    
    return Dataset(
        prompts=prompts,
        metadata={
            "source": "i2p_csv",
            "path": path,
            "total_loaded": len(prompts)
        }
    )
