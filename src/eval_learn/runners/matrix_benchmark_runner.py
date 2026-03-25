import gc
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

from ..logging_utils import get_logger
from ..registry import get_technique, get_metric
from ..registry.entrypoints import load_entrypoints
from .core.base_runner import BaseRunner
from .multi_benchmark_runner import MultiBenchmarkRunner

logger = get_logger(__name__)


def generate_matrix_run_id(
    technique_names: List[str],
    technique_configs: Dict[str, Dict[str, Any]],
    metric_names: List[str],
    metric_configs: Dict[str, Dict[str, Any]],
    timestamp: float,
) -> str:
    """Generate a short hash from matrix benchmark run configuration."""
    payload = json.dumps({
        "technique_names": sorted(technique_names),
        "technique_configs": technique_configs,
        "metric_names": sorted(metric_names),
        "metric_configs": metric_configs,
        "timestamp": timestamp,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


class MatrixBenchmarkRunner(BaseRunner):
    """
    Multiple techniques x multiple metrics benchmark runner.

    Composes MultiBenchmarkRunner instances — one per technique.
    Each technique is loaded, evaluated against all metrics, then
    unloaded before the next technique. VRAM is cleared between
    technique runs via gc.collect() + torch.cuda.empty_cache().
    """

    def __init__(
        self,
        technique_names: List[str],
        metric_names: List[str],
        technique_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        metric_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        output_dir: str = "results",
    ):
        super().__init__(output_dir)

        self.technique_names = technique_names
        self.metric_names = metric_names
        self.technique_configs = technique_configs or {}
        self.metric_configs = metric_configs or {}

        load_entrypoints()
        self._validate()

    def _validate(self):
        """Fast-fail: verify all names exist in the registry."""
        if not self.technique_names:
            raise ValueError("technique_names must not be empty.")
        if not self.metric_names:
            raise ValueError("metric_names must not be empty.")
        if len(set(self.technique_names)) != len(self.technique_names):
            raise ValueError("technique_names contains duplicates.")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric_names contains duplicates.")

        # CCRT can only be used in matrix benchmark if all techniques are free_run
        if "ccrt" in self.metric_names:
            non_free_run = [t for t in self.technique_names if t != "free_run"]
            if non_free_run:
                raise ValueError(
                    f"CCRT metric can only be used with 'free_run' technique. "
                    f"Got techniques: {self.technique_names}. "
                    "CCRT requires specifying original_model_id and erased_model_id, "
                    "which only free_run supports. "
                    "For other techniques, use UA_IRA or other concept-agnostic metrics."
                )

        # ASR and ERR metrics are nudity-specific
        nudity_metrics = {"asr", "err"}
        used_nudity_metrics = set(self.metric_names) & nudity_metrics
        if used_nudity_metrics:
            for technique_name in self.technique_names:
                tech_config = self.technique_configs.get(technique_name, {})
                erase_concept = tech_config.get("erase_concept", "").lower()
                if erase_concept and erase_concept != "nudity":
                    raise ValueError(
                        f"Metrics {used_nudity_metrics} are designed for nudity evaluation only. "
                        f"Got technique '{technique_name}' with erase_concept='{erase_concept}'. "
                        "ASR and ERR use hardcoded nudity datasets (I2P) and detectors (NudeNet). "
                        "For other concepts, use CCRT or UA_IRA instead."
                    )

        for name in self.technique_names:
            get_technique(name)
        for name in self.metric_names:
            get_metric(name)

    def run(self) -> Dict[str, Any]:
        """Execute N techniques x M metrics benchmark matrix."""
        logger.info("Starting Matrix Benchmark Run...")
        timestamp = time.time()

        run_id = generate_matrix_run_id(
            technique_names=self.technique_names,
            technique_configs=self.technique_configs,
            metric_names=self.metric_names,
            metric_configs=self.metric_configs,
            timestamp=timestamp,
        )
        logger.info(f"Matrix Run ID: {run_id}")

        technique_reports = {}

        for technique_name in self.technique_names:
            self._log_phase(f"Running technique '{technique_name}'")

            technique_config = self.technique_configs.get(technique_name, {})

            multi_runner = MultiBenchmarkRunner(
                technique_name=technique_name,
                metric_names=self.metric_names,
                technique_config=technique_config,
                metric_configs=self.metric_configs,
                output_dir=self.output_dir,
            )

            sub_report = multi_runner.run()
            technique_reports[technique_name] = sub_report

            # VRAM cleanup between techniques
            del multi_runner
            self._cleanup_vram()

        # Build matrix report
        report = self._build_base_report(
            run_id=run_id,
            timestamp=timestamp,
            technique_names=self.technique_names,
            metric_names=self.metric_names,
            technique_configs=self.technique_configs,
            metric_configs=self.metric_configs,
            technique_reports=technique_reports,
            comparison=self._build_comparison(technique_reports),
        )

        # Save matrix-level report
        self._save_matrix_report(run_id, report)

        logger.info("Matrix benchmark run completed.")
        return report

    def _build_comparison(self, technique_reports: Dict[str, Dict]) -> Dict[str, Any]:
        """Build a comparison table: metric_name -> {technique_name -> value}."""
        comparison = {}
        for metric_name in self.metric_names:
            comparison[metric_name] = {}
            for technique_name, report in technique_reports.items():
                metric_results = report.get("metric_results", {})
                if metric_name in metric_results:
                    comparison[metric_name][technique_name] = metric_results[metric_name]["value"]
                else:
                    comparison[metric_name][technique_name] = None
        return comparison

    def _save_matrix_report(self, run_id: str, report: Dict[str, Any]):
        """Save the matrix-level comparison report."""
        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, f"matrix_{run_id}_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Matrix report saved to {report_path}")

    @staticmethod
    def _cleanup_vram():
        """Free GPU memory between technique runs."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("VRAM cleared.")
        except ImportError:
            pass
