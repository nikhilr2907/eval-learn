import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from ..types import MetricResult
from ..logging_utils import get_logger
from ..registry import get_technique, get_metric
from ..registry.entrypoints import load_entrypoints
from .core.base_runner import BaseRunner

logger = get_logger(__name__)


def generate_multi_run_id(
    technique_name: str,
    technique_config: Dict[str, Any],
    metric_names: List[str],
    metric_configs: Dict[str, Dict[str, Any]],
    dataset_name: str,
    timestamp: float,
) -> str:
    """Generate a short hash from multi-benchmark run configuration."""
    payload = json.dumps({
        "technique": technique_name,
        "technique_config": technique_config,
        "metric_names": sorted(metric_names),
        "metric_configs": metric_configs,
        "dataset": dataset_name,
        "timestamp": timestamp,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


class MultiBenchmarkRunner(BaseRunner):
    """
    Single technique x multiple metrics benchmark runner.

    All metrics must implement the DataLoader interface:
      - load_dataset() -> DataLoader
      - update(images, prompts, metadata)
      - compute() -> MetricResult

    The first metric's DataLoader drives image generation. All other metrics
    have load_dataset() called for side effects only (e.g. FID pre-extracts
    real-image features). Every metric's update() is called once per batch
    with the same generated images.
    """

    def __init__(
        self,
        technique_name: str,
        metric_names: List[str],
        technique_config: Optional[Dict[str, Any]] = None,
        metric_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        output_dir: str = "results",
    ):
        super().__init__(output_dir)

        self.technique_name = technique_name
        self.metric_names = metric_names
        self.technique_config = technique_config or {}
        self.metric_configs = metric_configs or {}

        load_entrypoints()
        self._validate()

    def _validate(self):
        """Resolve factories from registry. Raises ValueError on failure."""
        if not self.metric_names:
            raise ValueError("metric_names must not be empty.")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric_names contains duplicates.")

        self.technique_factory = get_technique(self.technique_name)
        self.metric_factories = {}
        for name in self.metric_names:
            self.metric_factories[name] = get_metric(name)

    def run(self) -> Dict[str, Any]:
        """Execute single technique x multiple metrics benchmark."""
        logger.info("Starting Multi-Benchmark Run...")
        timestamp = time.time()

        # 1. Initialize all metrics
        self._log_phase("Initializing metrics")
        metrics: Dict[str, Any] = {}
        for name in self.metric_names:
            config = self.metric_configs.get(name, {})
            metrics[name] = self.metric_factories[name](**config)

        # 2. Load datasets — first metric drives generation; others for side effects
        self._log_phase("Loading datasets")
        first_metric_name = self.metric_names[0]
        loader = metrics[first_metric_name].load_dataset()

        for name in self.metric_names[1:]:
            metrics[name].load_dataset()

        # 3. Initialize technique
        self._log_phase("Initializing technique")
        technique = self.technique_factory(**self.technique_config)

        # 4+5. Generate and evaluate batch by batch
        self._log_phase("Generating images and computing metrics")
        all_images: List[Any] = []
        dataset_name = "unknown"
        total_generated = 0
        accumulated_metadata: Dict[str, Any] = {}

        for batch in loader:
            dataset_name = batch.metadata.get("source", dataset_name)

            batch_images = technique.generate(prompts=batch.prompts)
            all_images.extend(batch_images)
            total_generated += len(batch_images)

            for key, val in batch.metadata.items():
                if isinstance(val, list):
                    accumulated_metadata.setdefault(key, []).extend(val)
                else:
                    accumulated_metadata[key] = val

            for metric in metrics.values():
                metric.update(batch_images, batch.prompts, batch.metadata)

        logger.info(f"Generated and evaluated {total_generated} images from '{dataset_name}'.")

        # 6. Finalise all metrics
        metric_results: Dict[str, Any] = {}
        for name, metric in metrics.items():
            self._log_phase(f"Finalising metric '{name}'")
            result: MetricResult = metric.compute()
            metric_results[name] = {
                "name": result.name,
                "value": result.value,
                "details": result.details,
            }
            logger.info(f"Metric Result ({result.name}): {result.value}")

        run_id = generate_multi_run_id(
            technique_name=self.technique_name,
            technique_config=self.technique_config,
            metric_names=self.metric_names,
            metric_configs=self.metric_configs,
            dataset_name=dataset_name,
            timestamp=timestamp,
        )
        logger.info(f"Run ID: {run_id}")

        # 7. Build report
        report = self._build_base_report(
            run_id=run_id,
            timestamp=timestamp,
            technique_name=self.technique_name,
            metric_names=self.metric_names,
            dataset_name=dataset_name,
            dataset_metadata={**accumulated_metadata, "total_loaded": total_generated},
            technique_config=self.technique_config,
            metric_configs=self.metric_configs,
            metric_results=metric_results,
        )

        # 8. Save artifacts
        self.writer.save_run(
            run_id=run_id,
            technique_name=self.technique_name,
            metric_name="multi",
            images=all_images,
            report=report,
            metadata=accumulated_metadata,
        )

        logger.info("Multi-benchmark run completed.")
        return report
