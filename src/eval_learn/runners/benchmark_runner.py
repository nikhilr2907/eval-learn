import hashlib
import json
import time
from typing import Any, Dict
from ..types import Dataset, MetricResult
from ..logging_utils import get_logger
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


class BenchmarkRunner(BaseRunner):
    """
    Single technique × single metric benchmark runner.

    Orchestrates the execution of a benchmark run where one technique
    generates images for one metric to evaluate.

    Each metric owns its dataset via ``metric.load_dataset()``.
    """

    def __init__(
        self,
        technique_factory: Any,
        metric_factory: Any,
        technique_name: str,
        metric_name: str,
        technique_config: Dict[str, Any],
        metric_config: Dict[str, Any],
        output_dir: str = "results",
    ):
        """
        Initialize the single-benchmark runner.

        Args:
            technique_factory: Technique class (not instantiated).
            metric_factory: Metric class (not instantiated).
            technique_name: Name of the technique (e.g., "sld").
            metric_name: Name of the metric (e.g., "asr").
            technique_config: Config dict to pass to technique.__init__().
            metric_config: Config dict to pass to metric.__init__().
            output_dir: Directory where artifacts will be saved.
        """
        super().__init__(output_dir)
        self.technique_factory = technique_factory
        self.metric_factory = metric_factory
        self.technique_name = technique_name
        self.metric_name = metric_name
        self.technique_config = technique_config
        self.metric_config = metric_config

    def run(self) -> Dict[str, Any]:
        """Execute single technique × single metric benchmark."""
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
