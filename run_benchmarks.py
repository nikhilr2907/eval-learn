"""
run_benchmarks.py — Run benchmarks from a config file.

Supports all three runner modes, auto-detected from config structure:
  - Single:  technique  + metric   → SingleBenchmarkRunner
  - Multi:   technique  + metrics  → MultiBenchmarkRunner
  - Matrix:  techniques + metrics  → MatrixBenchmarkRunner

Usage:
    python run_benchmarks.py --config examples/smoke_config.json
    python run_benchmarks.py --config examples/multi_config.json
    python run_benchmarks.py --config examples/matrix_config.json

Requires GPU for real runs. Set HF_TOKEN in .env for model downloads.
"""

import argparse
import json
import sys
import os

from dotenv import load_dotenv
load_dotenv(override=True)

from eval_learn.runners import SingleBenchmarkRunner, MultiBenchmarkRunner, MatrixBenchmarkRunner
from eval_learn.logging_utils import get_logger

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        if path.endswith(('.yaml', '.yml')):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run eval-learn benchmarks")
    parser.add_argument("--config", "-c", required=True, help="Path to config file (JSON/YAML)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get("output_dir", "results")

    has_techniques = "techniques" in config
    has_technique = "technique" in config
    has_metrics = "metrics" in config
    has_metric = "metric" in config

    try:
        if has_techniques and has_metrics:
            runner = _build_matrix(config, output_dir)
        elif has_technique and has_metrics:
            runner = _build_multi(config, output_dir)
        elif has_technique and has_metric:
            runner = _build_single(config, output_dir)
        else:
            print("Invalid config: must have technique+metric, technique+metrics, or techniques+metrics")
            sys.exit(1)
    except ValueError as e:
        print(f"Validation error: {e}")
        sys.exit(1)

    report = runner.run()

    print(f"\nRun completed. Output: {output_dir}")
    print(f"Run ID: {report['run_id']}")

    # Print summary
    if "metric_result" in report:
        # Single
        r = report["metric_result"]
        print(f"  {r['name']}: {r['value']}")
    elif "metric_results" in report and "comparison" not in report:
        # Multi
        for name, r in report["metric_results"].items():
            print(f"  {r['name']}: {r['value']}")
    elif "comparison" in report:
        # Matrix
        print("\nComparison table:")
        comparison = report["comparison"]
        techniques = report["technique_names"]
        header = f"{'metric':<20}" + "".join(f"{t:<15}" for t in techniques)
        print(header)
        print("-" * len(header))
        for metric_name, scores in comparison.items():
            row = f"{metric_name:<20}"
            for t in techniques:
                val = scores.get(t)
                row += f"{val:<15.4f}" if val is not None else f"{'N/A':<15}"
            print(row)


def _build_single(config, output_dir):
    tech = config["technique"]
    met = config["metric"]
    logger.info(f"Mode: single | {tech['name']} x {met['name']}")
    return SingleBenchmarkRunner(
        technique_name=tech["name"],
        metric_name=met["name"],
        technique_config=tech.get("config", {}),
        metric_config=met.get("config", {}),
        output_dir=output_dir,
    )


def _build_multi(config, output_dir):
    tech = config["technique"]
    metric_names = []
    metric_configs = {}
    for m in config["metrics"]:
        metric_names.append(m["name"])
        cfg = m.get("config", {})
        if cfg:
            metric_configs[m["name"]] = cfg
    logger.info(f"Mode: multi | {tech['name']} x {metric_names}")
    return MultiBenchmarkRunner(
        technique_name=tech["name"],
        metric_names=metric_names,
        technique_config=tech.get("config", {}),
        metric_configs=metric_configs,
        output_dir=output_dir,
    )


def _build_matrix(config, output_dir):
    technique_names = []
    technique_configs = {}
    for t in config["techniques"]:
        technique_names.append(t["name"])
        cfg = t.get("config", {})
        if cfg:
            technique_configs[t["name"]] = cfg
    metric_names = []
    metric_configs = {}
    for m in config["metrics"]:
        metric_names.append(m["name"])
        cfg = m.get("config", {})
        if cfg:
            metric_configs[m["name"]] = cfg
    logger.info(f"Mode: matrix | {technique_names} x {metric_names}")
    return MatrixBenchmarkRunner(
        technique_names=technique_names,
        metric_names=metric_names,
        technique_configs=technique_configs,
        metric_configs=metric_configs,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
