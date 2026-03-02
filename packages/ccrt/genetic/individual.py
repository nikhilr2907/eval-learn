from dataclasses import dataclass, field
from typing import List


@dataclass
class Individual:
    id: List[str]
    concepts: List[str] = field(default_factory=list)
    score: float = 0.0
