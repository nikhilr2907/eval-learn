import hashlib
import json
import time
from typing import Any, Dict, List, Optional
from ..types import Dataset, MetricResult
from ..logging_utils import get_logger
from ..configs.base import BaseConfig
from ..artifacts import ArtifactWriter

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

class BenchmarkRunner:
    """
    Orchestrates the execution of a benchmark run.
    """

    def __init__(
        self,
        dataset_loader: Any,
        technique_factory: Any,
        metric_factory: Any,
        technique_name: str,
        metric_name: str,
        dataset_name: str,
        technique_config: Dict[str, Any],
        metric_config: Dict[str, Any],
        dataset_config: Dict[str, Any] = {},
        output_dir: str = "results",
    ):
        self.dataset_loader = dataset_loader
        self.technique_factory = technique_factory
        self.metric_factory = metric_factory
        self.technique_name = technique_name
        self.metric_name = metric_name
        self.dataset_name = dataset_name
        self.technique_config = technique_config
        self.metric_config = metric_config
        self.dataset_config = dataset_config
        self.writer = ArtifactWriter(base_dir=output_dir)

    def run(self) -> Dict[str, Any]:
        logger.info("Starting Benchmark Run...")
        ts = time.time()

        run_id = generate_run_id(
            technique_name=self.technique_name,
            technique_config=self.technique_config,
            metric_name=self.metric_name,
            metric_config=self.metric_config,
            dataset_name=self.dataset_name,
            timestamp=ts,
        )
        logger.info(f"Run ID: {run_id}")

        # 1. Load Dataset
        logger.info("Loading dataset...")
        dataset: Dataset = self.dataset_loader(**self.dataset_config)
        logger.info(f"Loaded {len(dataset.prompts)} prompts.")

        # 2. Initialize Technique
        logger.info("Initializing technique...")
        technique = self.technique_factory(**self.technique_config)

        # 3. Generate Images
        logger.info("Generating images...")
        images = technique.generate(prompts=dataset.prompts)
        logger.info(f"Generated {len(images)} images.")

        # 4. Initialize Metric
        logger.info("Initializing metric...")
        metric = self.metric_factory(**self.metric_config)

        # 5. Compute Metrics
        logger.info("Computing metrics...")
        result: MetricResult = metric.compute(
            images=images, prompts=dataset.prompts, metadata=dataset.metadata
        )
        logger.info(f"Metric Result ({result.name}): {result.value}")

        # 6. Prepare Report
        report = {
            "run_id": run_id,
            "timestamp": ts,
            "technique_name": self.technique_name,
            "metric_name": self.metric_name,
            "dataset_name": self.dataset_name,
            "dataset_metadata": dataset.metadata,
            "technique_config": self.technique_config,
            "metric_config": self.metric_config,
            "metric_result": {
                "name": result.name,
                "value": result.value,
                "details": result.details,
            },
        }

        # 7. Save Artifacts
        self.writer.save_run(
            run_id=run_id,
            technique_name=self.technique_name,
            metric_name=self.metric_name,
            images=images,
            report=report,
            metadata=dataset.metadata,
        )

        return report
