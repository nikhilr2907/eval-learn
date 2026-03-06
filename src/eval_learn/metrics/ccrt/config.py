import os
from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig


@dataclass
class CCRTConfig(BaseConfig):
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
        batch_size:         Number of prompts per DataLoader batch.
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
    batch_size: int = 32
    clip_model_name: str = "openai/clip-vit-large-patch14"
    device: Optional[str] = None

    @classmethod
    def from_dict(cls, data):
        instance = super().from_dict(data)
        missing = [
            f for f in ("original_model_id", "erased_model_id", "concept_name",
                        "concept_desc", "reference_imgs", "llm_api_key")
            if not getattr(instance, f)
        ]
        if missing:
            raise ValueError(f"CCRTConfig missing required fields: {missing}")

        # Validate that reference_imgs directory exists
        if not os.path.isdir(instance.reference_imgs):
            raise ValueError(
                f"reference_imgs directory not found: {instance.reference_imgs}. "
                f"Must be a valid directory path with 3+ reference images."
            )

        # Validate that vocab_dir exists if specified
        if instance.vocab_dir is not None and not os.path.isdir(instance.vocab_dir):
            raise ValueError(
                f"vocab_dir directory not found: {instance.vocab_dir}. "
                f"Must be a valid WordNet vocabulary directory with wnid.txt, words.txt, gloss.txt, is_a.txt."
            )

        return instance
