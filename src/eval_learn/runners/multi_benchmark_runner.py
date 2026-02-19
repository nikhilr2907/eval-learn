import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from ..types import Dataset, MetricResult
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

    Loads one technique, generates images once, then evaluates
    against each metric in sequence. The first metric's dataset
    drives image generation; other metrics still have load_dataset()
    called for side effects.
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

        # 1. Initialize first metric and load dataset
        first_metric_name = self.metric_names[0]
        first_metric_config = self.metric_configs.get(first_metric_name, {})
        self._log_phase("Initializing first metric")
        first_metric = self.metric_factories[first_metric_name](**first_metric_config)

        self._log_phase("Loading dataset")
        dataset: Dataset = first_metric.load_dataset()
        dataset_name = dataset.metadata.get("source", "unknown")
        logger.info(f"Loaded {len(dataset.prompts)} prompts from '{dataset_name}'.")

        # 2. Generate run_id
        run_id = generate_multi_run_id(
            technique_name=self.technique_name,
            technique_config=self.technique_config,
            metric_names=self.metric_names,
            metric_configs=self.metric_configs,
            dataset_name=dataset_name,
            timestamp=timestamp,
        )
        logger.info(f"Run ID: {run_id}")

        # 3. Initialize technique and generate images ONCE
        self._log_phase("Initializing technique")
        technique = self.technique_factory(**self.technique_config)

        self._log_phase("Generating images")
        images = technique.generate(prompts=dataset.prompts)
        logger.info(f"Generated {len(images)} images.")

        # 4. Compute each metric
        metric_results = {}
        for metric_name in self.metric_names:
            self._log_phase(f"Computing metric '{metric_name}'")
            metric_config = self.metric_configs.get(metric_name, {})

            if metric_name == first_metric_name:
                metric = first_metric
            else:
                metric = self.metric_factories[metric_name](**metric_config)
                metric.load_dataset()  # called for side effects

            result: MetricResult = metric.compute(
                images=images, prompts=dataset.prompts, metadata=dataset.metadata
            )
            metric_results[metric_name] = {
                "name": result.name,
                "value": result.value,
                "details": result.details,
            }
            logger.info(f"Metric Result ({result.name}): {result.value}")

        # 5. Build report
        report = self._build_base_report(
            run_id=run_id,
            timestamp=timestamp,
            technique_name=self.technique_name,
            metric_names=self.metric_names,
            dataset_name=dataset_name,
            dataset_metadata=dataset.metadata,
            technique_config=self.technique_config,
            metric_configs=self.metric_configs,
            metric_results=metric_results,
        )

        # 6. Save artifacts — images saved once, report has all metric results
        self.writer.save_run(
            run_id=run_id,
            technique_name=self.technique_name,
            metric_name="multi",
            images=images,
            report=report,
            metadata=dataset.metadata,
        )

        logger.info("Multi-benchmark run completed.")
        return report
