from datasets import load_dataset
from typing import Optional
from ..types import Dataset
from ..registry import register_dataset
from ..registry.hf_sync import HFSync
from ..logging_utils import get_logger

logger = get_logger(__name__)


@register_dataset("i2p_csv")
def load_i2p_csv(
    repo_id: str = "AIML-TUDA/i2p",
    split: str = "train",
    revision: Optional[str] = "6594f223b0544fcbf63a6889417a6b9d2e40a77c",
    limit: Optional[int] = None,
    prompt_col: str = "prompt",
    local_dir: str = "data/i2p",
    token: Optional[str] = None,
) -> Dataset:
    """
    Pulls the I2P dataset from Hugging Face to a local directory using HFSync,
    then loads prompts from the local copy.

    Args:
        repo_id:    HF dataset repo to pull from.
        split:      Dataset split to load.
        revision:   Commit hash / branch / tag to pin the version.
        limit:      Max number of prompts to load.
        prompt_col: Column name containing the prompts.
        local_dir:  Local directory to cache the dataset snapshot.
        token:      HF token (falls back to HF_TOKEN env var).
    """
    logger.info("Pulling I2P dataset (%s) to local dir: %s ...", repo_id, local_dir)

    # results_repo / images_repo are unused here but required by HFSync.__init__
    syncer = HFSync(
        datasets_repo=repo_id,
        results_repo=repo_id,
        images_repo=repo_id,
        token=token,
    )
    local_path = syncer.pull_datasets(local_dir=local_dir)

    logger.info("Loading from local path: %s", local_path)
    try:
        hf_dataset = load_dataset(local_path, split=split)
        df = hf_dataset.to_pandas()
    except Exception as e:
        logger.error("Failed to load dataset from local path: %s", e)
        raise

    if prompt_col not in df.columns:
        raise ValueError(
            f"Column '{prompt_col}' not found in dataset. Columns: {df.columns.tolist()}"
        )

    prompts = df[prompt_col].tolist()

    if limit:
        prompts = prompts[:limit]

    logger.info("Loaded %d prompts from local copy.", len(prompts))

    return Dataset(
        prompts=prompts,
        metadata={
            "source": "i2p_csv",
            "repo_id": repo_id,
            "split": split,
            "revision": revision,
            "local_dir": local_dir,
            "total_loaded": len(prompts),
        },
    )
