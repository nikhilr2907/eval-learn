from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

class UnlearningTechnique(ABC):
    """
    Abstract Base Class for all Unlearning Techniques.
    Ensures a consistent interface for the orchestrator.
    """
    
    @abstractmethod
    def __init__(self, model_id: str, device: str):
        pass

    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[Any]:
        """
        Generates a list of images based on the provided prompts.
        
        Args:
            prompts: List of text prompts.
            **kwargs: Additional generation parameters (e.g., guidance_scale, safety_config).
            
        Returns:
            List of generated images (PIL.Image or similar).
        """
        pass
