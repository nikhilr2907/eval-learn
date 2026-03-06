from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CCRTConfig:
    """
    Configuration for the CCRT metric.

    Attributes:
        original_model_id:  HF model ID of the unmodified baseline model.
                            Used in genetic search (MSE scoring) and fitness metric.
        erased_model_id:    HF model ID of the erased model being evaluated.
                            Should match the free_run technique's model_id.
        concept_name:       Human-readable name of the concept, e.g. "Van Gogh".
        concept_desc:       Short description of the concept's visual style,
                            e.g. "emotional colours, swirling brushwork".
        reference_imgs:     Path to directory containing reference images of the
                            concept (3+ images), used by the LLM eval step.
        vocab_dir:          Path to WordNet vocab directory (wnid.txt, words.txt,
                            gloss.txt, is_a.txt). None = ccrt package default.
        llm_api_key:        OpenAI API key for GPT-3.5 prompt generation and
                            GPT-4V style evaluation.
        output_dir:         Directory for intermediate artifacts (entities.pkl,
                            prompts.csv). Defaults to "results/ccrt".
        genetic_iterations: Max iterations for the genetic search loop.
        genetic_top_k:      Number of survivors kept after each selection step.
        limit:              Max number of prompts passed to the runner.
        clip_model_name:    HuggingFace CLIP model identifier.
        device:             Torch device string (None = auto-detect).
    """
    original_model_id: str = ""
    erased_model_id: str = ""
    concept_name: str = ""
    concept_desc: str = ""
    reference_imgs: str = ""
    vocab_dir: Optional[str] = None
    llm_api_key: str = ""
    output_dir: str = "results/ccrt"
    genetic_iterations: int = 6
    genetic_top_k: int = 10
    limit: Optional[int] = 100
    clip_model_name: str = "openai/clip-vit-large-patch14"
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_dict(cls, data: dict) -> "CCRTConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        instance = cls(**filtered)
        missing = [
            f for f in ("original_model_id", "erased_model_id", "concept_name",
                        "concept_desc", "reference_imgs", "llm_api_key")
            if not getattr(instance, f)
        ]
        if missing:
            raise ValueError(f"CCRTConfig missing required fields: {missing}")
        return instance

    def to_dict(self) -> dict:
        return asdict(self)
