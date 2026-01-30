import argparse
import sys
import os
import json
from typing import Dict, Any

# Ensure registry is populated
import eval_learn.techniques.sld.wrapper
import eval_learn.metrics.asr.metric
import eval_learn.metrics.fid.metric
import eval_learn.metrics.err.metric
import eval_learn.metrics.tifa.metric
import eval_learn.metrics.clip_score.metric
import eval_learn.datasets.i2p_csv

from eval_learn.registry import get_technique, get_metric, get_dataset
from eval_learn.registry.entrypoints import load_entrypoints
from eval_learn.runners import BenchmarkRunner
from eval_learn.logging_utils import get_logger

logger = get_logger("cli")

def load_config(path: str) -> Dict[str, Any]:
    """Load config from JSON or YAML file."""
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        sys.exit(1)
        
    with open(path, 'r') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                logger.error("PyYAML not installed. Install it to use .yaml files, or use .json")
                sys.exit(1)
        else:
            return json.load(f)

def run_benchmark(args):
    """Execute the benchmark run."""
    config = load_config(args.config)
    
    # Extract sections
    run_name = config.get("run_name", "Benchmark_Run")
    output_dir = config.get("output_dir", "results/benchmarks")
    
    # Dataset
    ds_conf = config.get("dataset", {})
    ds_name = ds_conf.get("name")
    if not ds_name:
        logger.error("Config must specify 'dataset.name'")
        sys.exit(1)
        
    # Technique
    tech_conf = config.get("technique", {})
    tech_name = tech_conf.get("name")
    if not tech_name:
        logger.error("Config must specify 'technique.name'")
        sys.exit(1)
        
    # Metric
    met_conf = config.get("metric", {})
    met_name = met_conf.get("name")
    if not met_name:
        logger.error("Config must specify 'metric.name'")
        sys.exit(1)

    logger.info(f"Preparing run '{run_name}'...")
    logger.info(f"Dataset: {ds_name} | Technique: {tech_name} | Metric: {met_name}")

    try:
        dataset_loader = get_dataset(ds_name)
        technique_factory = get_technique(tech_name)
        metric_factory = get_metric(met_name)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    runner = BenchmarkRunner(
        dataset_loader=dataset_loader,
        technique_factory=technique_factory,
        metric_factory=metric_factory,
        technique_config=tech_conf.get("config", {}),
        metric_config=met_conf.get("config", {}),
        dataset_config=ds_conf.get("config", {}),
        output_dir=output_dir,
        run_name=run_name
    )
    
    try:
        report = runner.run()
        logger.info("Run completed successfully.")
    except Exception as e:
        logger.exception("Run failed.")
        sys.exit(1)

def main():
    # Load plugins from entry points
    load_entrypoints()

    parser = argparse.ArgumentParser(description="Eval-Learn CLI")
    parser.add_argument("--version", action="store_true", help="Show version")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run Command
    run_parser = subparsers.add_parser("run", help="Execute a benchmark run")
    run_parser.add_argument("--config", "-c", required=True, help="Path to config file (JSON/YAML)")
    
    args = parser.parse_args()

    if args.version:
        from eval_learn import __version__
        print(f"Eval-Learn v{__version__}")
        sys.exit(0)

    if args.command == "run":
        run_benchmark(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()