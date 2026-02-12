from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class CLIPScoreConfig(BaseConfig):
    """
    Configuration for the CLIP Score metric.

    Attributes:
        clip_model_name: HuggingFace CLIP model identifier used by torchmetrics.
        device: Torch device string (default: None, auto-detect).
        text_path: Path to TIFA captions JSON.
        qa_path: Path to TIFA QA pairs JSON.
        limit: Max number of prompts to load.
    """
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None
    text_path: str = "data/tifa/sensitive_text_inputs.json"
    qa_path: str = "data/tifa/sensitive_question_answers.json"
    limit: Optional[int] = 300
