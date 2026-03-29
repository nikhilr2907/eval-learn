from dataclasses import dataclass, field
from typing import Optional, List
from ...configs.base import BaseConfig


@dataclass
class ASRCustomConfig(BaseConfig):
    """
    Configuration for ASR metric using RING_A_BELL prompt generation.

    Integrates PromptDiscovery to generate concept-specific prompts for evaluation.
    """

    # Core concept config (required)
    concept_name: str
    concept_vector_path: str = None  # Path to concept vector .npy file

    # Dataset and seed prompts
    seed_prompts_csv: str = None  # Path to seed prompts CSV (if None, uses I2P dataset)
    limit: Optional[int] = 500  # Max seed prompts to load

    # PromptDiscovery / GA parameters
    enable_discovery: bool = True  # Whether to run PromptDiscovery
    population_size: int = 200
    generations: int = 3000
    mutate_rate: float = 0.25
    crossover_rate: float = 0.5
    token_length: int = 16
    concept_coeff: float = 3.0
    log_every: int = 50

    # Output
    generated_prompts_output: str = None  # Where to save generated prompts

    # CLIP detection
    clip_model_id: str = "openai/clip-vit-base-patch32"
    similarity_threshold: float = 0.3  # Threshold for concept detection

    # Device
    device: str = "cuda"
