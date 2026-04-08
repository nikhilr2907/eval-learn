import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from ..types import MetricResult
from ..logging_utils import get_logger
from ..registry import get_technique, get_metric
from ..registry.entrypoints import load_entrypoints
from .core.base_runner import BaseRunner
from .validation import validate_technique_metric_pair, ValidationError

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
    payload = json.dumps(
        {
            "technique": technique_name,
            "technique_config": technique_config,
            "metric": metric_name,
            "metric_config": metric_config,
            "dataset": dataset_name,
            "timestamp": timestamp,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


class SingleBenchmarkRunner(BaseRunner):
    """
    Single technique x single metric benchmark runner.

    Metrics must implement the DataLoader interface:
      - load_dataset() -> DataLoader  (batches of Dataset with prompts + metadata)
      - update(images, prompts, metadata)  (called once per batch)
      - compute() -> MetricResult  (called once to finalise the score)

    The runner iterates the DataLoader, calls technique.generate() per batch,
    feeds results to metric.update(), then finalises with metric.compute().
    Images are accumulated across batches for artifact saving.
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

        load_entrypoints()
        self._validate()

    def _validate(self):
        """Resolve factories from registry and validate technique-metric compatibility."""
        # 1. Check factories exist
        self.technique_factory = get_technique(self.technique_name)
        self.metric_factory = get_metric(self.metric_name)

        # 2. Validate technique-metric pair
        try:
            validate_technique_metric_pair(
                technique_name=self.technique_name,
                technique_config=self.technique_config,
                metric_name=self.metric_name,
                metric_config=self.metric_config,
            )
        except ValidationError as e:
            logger.error(f"Invalid technique-metric pair: {e}")
            raise ValueError(str(e))

    def run(self) -> Dict[str, Any]:
        """Execute single technique x single metric benchmark."""
        logger.info("Starting Benchmark Run...")
        timestamp = time.time()

        # 1. Initialize Metric
        self._log_phase("Initializing metric")
        resolved = self._resolve_mma_clip_model(
            {self.metric_name: self.metric_config},
            self.technique_name,
            self.technique_config,
        )
        metric_config = resolved.get(self.metric_name, self.metric_config)
        metric = self.metric_factory(**metric_config)

        # 2. Load Dataset
        self._log_phase("Loading dataset")
        loader = metric.load_dataset()

        # 3. Initialize Technique
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
            metric.update(batch_images, batch.prompts, batch.metadata)

            all_images.extend(batch_images)
            total_generated += len(batch_images)

            for key, val in batch.metadata.items():
                if isinstance(val, list):
                    accumulated_metadata.setdefault(key, []).extend(val)
                else:
                    accumulated_metadata[key] = val

        logger.info(
            f"Generated and evaluated {total_generated} images from '{dataset_name}'."
        )

        # 6. Finalise metric
        result: MetricResult = metric.compute()
        logger.info(f"Metric Result ({result.name}): {result.value}")

        run_id = generate_run_id(
            technique_name=self.technique_name,
            technique_config=self.technique_config,
            metric_name=self.metric_name,
            metric_config=self.metric_config,
            dataset_name=dataset_name,
            timestamp=timestamp,
        )
        logger.info(f"Run ID: {run_id}")

        # 7. Prepare Report
        report = self._build_base_report(
            run_id=run_id,
            timestamp=timestamp,
            technique_name=self.technique_name,
            metric_name=self.metric_name,
            dataset_name=dataset_name,
            dataset_metadata={**accumulated_metadata, "total_loaded": total_generated},
            technique_config=self.technique_config,
            metric_config=self.metric_config,
            metric_result={
                "name": result.name,
                "value": result.value,
                "details": result.details,
            },
        )

        # 8. Save Artifacts
        self.writer.save_run(
            run_id=run_id,
            technique_name=self.technique_name,
            metric_name=self.metric_name,
            images=all_images,
            report=report,
            metadata=accumulated_metadata,
        )

        logger.info("Benchmark run completed.")
        return report
