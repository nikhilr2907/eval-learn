from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig

@dataclass
class TIFAConfig(BaseConfig):
    """
    Configuration for the TIFA (Text-to-Image Faithfulness) metric.

    Attributes:
        vqa_model_name: HuggingFace model identifier for the BLIP-2 VQA model.
        device: Torch device string (default: None, auto-detect).
    """
    vqa_model_name: str = "Salesforce/blip2-flan-t5-xl"
    device: Optional[str] = None
    text_path: str = "data/tifa/sensitive_text_inputs.json"
    qa_path: str = "data/tifa/sensitive_question_answers.json"
    limit: Optional[int] = 200
