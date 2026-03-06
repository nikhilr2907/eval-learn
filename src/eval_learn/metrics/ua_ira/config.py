from dataclasses import dataclass
from typing import List, Optional
from ...configs.base import BaseConfig

@dataclass
class UAIRAConfig(BaseConfig):
    """
    Configuration for Unlearning accuracy and In-domain Retain Accuracy (IRA)
    Original paper in https://arxiv.org/html/2402.11846v4, uses LLM to generate prompts
    Current config uses prompts from ERR data set
    """
    clip_model: str = "openai/clip-vit-large-patch14" # commonly used clip model by openAI
    device: str = None
    target_concept: str = "Mickey Mouse"
    retain_concept: str = "Minnie Mouse"
    dataset_path: str = "data/ERR/raw_csv_data/challenge_dataset.csv"
    target_prompt_count: Optional[int] = 5 # number of prompts containing target concept taken from dataset
    retain_prompt_count: Optional[int] = 5
    

