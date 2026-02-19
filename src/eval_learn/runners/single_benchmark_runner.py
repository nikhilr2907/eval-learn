import hashlib
import json
import time
from typing import Any, Dict, Optional

from ..types import Dataset, MetricResult
from ..logging_utils import get_logger
from ..registry import get_technique, get_metric
from ..registry.entrypoints import load_entrypoints
from .core.base_runner import BaseRunner

logger = get_logger(__name__)


def generate_run_id(
    technique_name: str,
    technique_config: Dict[str, Any],
    metric_name: str,
    metric_config: Dict[str, Any],
    dataset_name: str,
    timestamp: float,
) -> str:
    """Generate a short hash from run configuration and timestamp."""
    payload = json.dumps({
        "technique": technique_name,
        "technique_config": technique_config,
        "metric": metric_name,
        "metric_config": metric_config,
        "dataset": dataset_name,
        "timestamp": timestamp,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


class SingleBenchmarkRunner(BaseRunner):
    """
    Single technique x single metric benchmark runner.

    Resolves technique and metric from the registry by name,
    validates the configuration, then orchestrates image generation
    and metric evaluation.
    """

    def __init__(
        self,
        technique_name: str,
        metric_name: str,
        technique_config: Optional[Dict[str, Any]] = None,
        metric_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "results",
    ):
        super().__init__(output_dir)

        self.technique_config = technique_config or {}
        self.metric_config = metric_config or {}
        self.technique_name = technique_name
        self.metric_name = metric_name

        # Ensure registry is populated, then validate
        load_entrypoints()
        self._validate()

    def _validate(self):
        """Resolve factories from registry. Raises ValueError on failure."""
        self.technique_factory = get_technique(self.technique_name)
        self.metric_factory = get_metric(self.metric_name)

    def run(self) -> Dict[str, Any]:
        """Execute single technique x single metric benchmark."""
        logger.info("Starting Benchmark Run...")
        timestamp = time.time()

        # 1. Initialize Metric
        self._log_phase("Initializing metric")
        metric = self.metric_factory(**self.metric_config)

        # 2. Load Dataset (owned by the metric)
        self._log_phase("Loading dataset")
        dataset: Dataset = metric.load_dataset()
        dataset_name = dataset.metadata.get("source", "unknown")
        logger.info(f"Loaded {len(dataset.prompts)} prompts from '{dataset_name}'.")

        # Generate run_id now that we have dataset_name
        run_id = generate_run_id(
            technique_name=self.technique_name,
            technique_config=self.technique_config,
            metric_name=self.metric_name,
            metric_config=self.metric_config,
            dataset_name=dataset_name,
            timestamp=timestamp,
        )
        logger.info(f"Run ID: {run_id}")

        # 3. Initialize Technique
        self._log_phase("Initializing technique")
        technique = self.technique_factory(**self.technique_config)

        # 4. Generate Images
        self._log_phase("Generating images")
        images = technique.generate(prompts=dataset.prompts)
        logger.info(f"Generated {len(images)} images.")

        # 5. Compute Metrics
        self._log_phase("Computing metrics")
        result: MetricResult = metric.compute(
            images=images, prompts=dataset.prompts, metadata=dataset.metadata
        )
        logger.info(f"Metric Result ({result.name}): {result.value}")

        # 6. Prepare Report
        report = self._build_base_report(
            run_id=run_id,
            timestamp=timestamp,
            technique_name=self.technique_name,
            metric_name=self.metric_name,
            dataset_name=dataset_name,
            dataset_metadata=dataset.metadata,
            technique_config=self.technique_config,
            metric_config=self.metric_config,
            metric_result={
                "name": result.name,
                "value": result.value,
                "details": result.details,
            },
        )

        # 7. Save Artifacts
        self.writer.save_run(
            run_id=run_id,
            technique_name=self.technique_name,
            metric_name=self.metric_name,
            images=images,
            report=report,
            metadata=dataset.metadata,
        )

        logger.info("Benchmark run completed.")
        return report
