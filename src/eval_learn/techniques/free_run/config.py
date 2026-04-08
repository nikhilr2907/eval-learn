from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...configs.base import BaseConfig


@dataclass(frozen=True)
class FreeRunConfig(BaseConfig):
    """Configuration for the free run technique — any HF text-to-image model."""

    model_id: str = ""
    device: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FreeRunConfig":
        if not data.get("model_id"):
            raise ValueError("FreeRunConfig requires 'model_id' to be specified.")
        return super().from_dict(data)
