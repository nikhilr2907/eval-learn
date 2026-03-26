import time
from abc import ABC, abstractmethod
from typing import Any, Dict
from ...artifacts import ArtifactWriter
from ...logging_utils import get_logger

logger = get_logger(__name__)


class BaseRunner(ABC):
    """
    Abstract base class for all benchmark runners.

    Provides common infrastructure:
    - Artifact management (output directory, ArtifactWriter)
    - Report building with standard fields
    - Logging utilities

    Subclasses implement run() with their specific benchmark logic.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the base runner.

        Args:
            output_dir: Directory where artifacts will be saved.
        """
        self.output_dir = output_dir
        self.writer = ArtifactWriter(base_dir=output_dir)

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Execute the benchmark and return a report.

        Each runner type implements its own benchmark logic here.

        Returns:
            Report dictionary containing run results.
        """
        pass

    def _build_base_report(
        self, run_id: str, timestamp: float, **kwargs
    ) -> Dict[str, Any]:
        """
        Build a report dict with common base fields.

        Args:
            run_id: Unique identifier for this run.
            timestamp: Unix timestamp when the run started.
            **kwargs: Additional fields to include in the report.

        Returns:
            Report dict with run_id, timestamp, and any additional fields.
        """
        return {"run_id": run_id, "timestamp": timestamp, **kwargs}

    def _log_phase(self, phase_name: str):
        """
        Log the start of a benchmark phase.

        Args:
            phase_name: Name of the phase (e.g., "Loading dataset", "Generating images").
        """
        logger.info(f"{phase_name}...")
