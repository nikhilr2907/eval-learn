import json
import os
from typing import Optional
from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger

logger = get_logger(__name__)

@register_dataset("tifa_json")
def load_tifa_json(
    text_path: str = "data/tifa/sensitive_text_inputs.json",
    qa_path: str = "data/tifa/sensitive_question_answers.json",
    limit: Optional[int] = None,
) -> Dataset:
    """
    Loads the TIFA dataset from two JSON files: captions and QA pairs.

    The returned Dataset has ``qa_pairs`` in its metadata — a list parallel
    to ``prompts`` where each element is a list of ``{"question", "answer"}``
    dicts, as expected by the TIFA metric.

    Args:
        text_path: Path to the captions JSON (list of {"id", "caption"}).
        qa_path: Path to the QA JSON (list of {"id", "qas": [{"question", "answer"}, ...]}).
        limit: Max number of prompts to load.
    """
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"TIFA text inputs not found at: {text_path}")
    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"TIFA question-answers not found at: {qa_path}")

    logger.info(f"Loading TIFA captions from {text_path}...")
    with open(text_path, "r", encoding="utf-8") as f:
        text_data = json.load(f)

    logger.info(f"Loading TIFA QA pairs from {qa_path}...")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Build a lookup from id -> qas list
    qa_lookup = {item["id"]: item["qas"] for item in qa_data}

    if limit:
        text_data = text_data[:limit]

    prompts = []
    qa_pairs = []
    for item in text_data:
        prompts.append(item["caption"])
        qa_pairs.append(qa_lookup.get(item["id"], []))

    logger.info(f"Loaded {len(prompts)} prompts with QA pairs.")

    return Dataset(
        prompts=prompts,
        metadata={
            "source": "tifa_json",
            "total_loaded": len(prompts),
            "qa_pairs": qa_pairs,
        },
    )
