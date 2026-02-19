import argparse
import sys
import os
import json
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv(override=True)

from eval_learn.registry.hf_sync import HFSync
from eval_learn.runners import SingleBenchmarkRunner, MultiBenchmarkRunner, MatrixBenchmarkRunner
from eval_learn.logging_utils import get_logger

logger = get_logger("cli")


def _build_hf_sync(args) -> HFSync:
    """Construct HFSync from CLI args, falling back to env vars."""
    datasets_repo = args.datasets_repo or os.environ.get("HF_DATASETS_REPO")
    results_repo = args.results_repo or os.environ.get("HF_RESULTS_REPO")
    images_repo = args.images_repo or os.environ.get("HF_IMAGES_REPO")

    missing = []
    if not datasets_repo:
        missing.append("--datasets-repo or HF_DATASETS_REPO")
    if not results_repo:
        missing.append("--results-repo or HF_RESULTS_REPO")
    if not images_repo:
        missing.append("--images-repo or HF_IMAGES_REPO")

    if missing:
        logger.error("Missing required HF repo IDs: %s", ", ".join(missing))
        sys.exit(1)

    return HFSync(
        datasets_repo=datasets_repo,
        results_repo=results_repo,
        images_repo=images_repo,
        create_pr=getattr(args, 'create_pr', False),
    )

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


def _parse_metrics_list(metrics_list: List[Dict[str, Any]]) -> tuple:
    """Extract metric names and configs from a list of metric dicts."""
    metric_names = []
    metric_configs = {}
    for m in metrics_list:
        name = m.get("name")
        if not name:
            logger.error("Each entry in 'metrics' must have a 'name'")
            sys.exit(1)
        metric_names.append(name)
        cfg = m.get("config", {})
        if cfg:
            metric_configs[name] = cfg
    return metric_names, metric_configs


def _build_single_runner(config: Dict[str, Any], output_dir: str) -> SingleBenchmarkRunner:
    """Build a SingleBenchmarkRunner from config."""
    tech_conf = config.get("technique", {})
    tech_name = tech_conf.get("name")
    if not tech_name:
        logger.error("Config must specify 'technique.name'")
        sys.exit(1)

    met_conf = config.get("metric", {})
    met_name = met_conf.get("name")
    if not met_name:
        logger.error("Config must specify 'metric.name'")
        sys.exit(1)

    logger.info(f"Mode: single | Technique: {tech_name} | Metric: {met_name}")

    return SingleBenchmarkRunner(
        technique_name=tech_name,
        metric_name=met_name,
        technique_config=tech_conf.get("config", {}),
        metric_config=met_conf.get("config", {}),
        output_dir=output_dir,
    )


def _build_multi_runner(config: Dict[str, Any], output_dir: str) -> MultiBenchmarkRunner:
    """Build a MultiBenchmarkRunner from config."""
    tech_conf = config.get("technique", {})
    tech_name = tech_conf.get("name")
    if not tech_name:
        logger.error("Config must specify 'technique.name'")
        sys.exit(1)

    metrics_list = config.get("metrics", [])
    if not metrics_list:
        logger.error("Config must specify 'metrics' as a non-empty list")
        sys.exit(1)

    metric_names, metric_configs = _parse_metrics_list(metrics_list)

    logger.info(f"Mode: multi | Technique: {tech_name} | Metrics: {metric_names}")

    return MultiBenchmarkRunner(
        technique_name=tech_name,
        metric_names=metric_names,
        technique_config=tech_conf.get("config", {}),
        metric_configs=metric_configs,
        output_dir=output_dir,
    )


def _build_matrix_runner(config: Dict[str, Any], output_dir: str) -> MatrixBenchmarkRunner:
    """Build a MatrixBenchmarkRunner from config."""
    techniques_list = config.get("techniques", [])
    if not techniques_list:
        logger.error("Config must specify 'techniques' as a non-empty list")
        sys.exit(1)

    metrics_list = config.get("metrics", [])
    if not metrics_list:
        logger.error("Config must specify 'metrics' as a non-empty list")
        sys.exit(1)

    technique_names = []
    technique_configs = {}
    for t in techniques_list:
        name = t.get("name")
        if not name:
            logger.error("Each entry in 'techniques' must have a 'name'")
            sys.exit(1)
        technique_names.append(name)
        cfg = t.get("config", {})
        if cfg:
            technique_configs[name] = cfg

    metric_names, metric_configs = _parse_metrics_list(metrics_list)

    logger.info(f"Mode: matrix | Techniques: {technique_names} | Metrics: {metric_names}")

    return MatrixBenchmarkRunner(
        technique_names=technique_names,
        metric_names=metric_names,
        technique_configs=technique_configs,
        metric_configs=metric_configs,
        output_dir=output_dir,
    )


def run_benchmark(args):
    """Execute the benchmark run, auto-detecting mode from config structure."""
    config = load_config(args.config)
    output_dir = config.get("output_dir", "results")

    has_techniques = "techniques" in config
    has_technique = "technique" in config
    has_metrics = "metrics" in config
    has_metric = "metric" in config

    try:
        if has_techniques and has_metrics:
            runner = _build_matrix_runner(config, output_dir)
        elif has_technique and has_metrics:
            runner = _build_multi_runner(config, output_dir)
        elif has_technique and has_metric:
            runner = _build_single_runner(config, output_dir)
        else:
            logger.error(
                "Invalid config: must specify technique+metric, "
                "technique+metrics, or techniques+metrics"
            )
            sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        report = runner.run()
        logger.info("Run completed successfully.")
    except Exception as e:
        logger.exception("Run failed.")
        sys.exit(1)

def hf_push(args):
    """Push run artifacts to HF Hub."""
    sync = _build_hf_sync(args)
    run_dir = args.run_dir
    run_id = args.run_id

    if getattr(args, "matrix", False):
        # Matrix mode: push all sub-runs + matrix report
        result = sync.push_matrix_run(run_dir, run_id)
        logger.info("Matrix report pushed: %s", result["matrix_report_url"])
        for folder, urls in result["sub_runs"].items():
            logger.info("  %s — report: %s, images: %s",
                        folder, urls["report_url"], urls["images_url"])
        return

    if args.target == "report":
        url = sync.push_report(run_dir, run_id)
        logger.info("Report pushed: %s", url)
    elif args.target == "images":
        url = sync.push_images(run_dir, run_id)
        logger.info("Images pushed: %s", url)
    elif args.target == "all":
        urls = sync.push_run(run_dir, run_id)
        logger.info("Report pushed: %s", urls["report_url"])
        logger.info("Images pushed: %s", urls["images_url"])


def hf_pull(args):
    """Pull artifacts from HF Hub."""
    sync = _build_hf_sync(args)

    if args.target == "datasets":
        path = sync.pull_datasets(args.local_dir)
        logger.info("Datasets downloaded to %s", path)
    elif args.target == "results":
        path = sync.pull_results(args.local_dir)
        logger.info("Results downloaded to %s", path)
    elif args.target == "images":
        if not args.run_folder:
            logger.error("--run-folder is required when pulling images")
            sys.exit(1)
        path = sync.pull_run_images(args.run_folder, args.local_dir)
        logger.info("Images downloaded to %s", path)


def main():
    parser = argparse.ArgumentParser(description="Eval-Learn CLI")
    parser.add_argument("--version", action="store_true", help="Show version")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run Command
    run_parser = subparsers.add_parser("run", help="Execute a benchmark run")
    run_parser.add_argument("--config", "-c", required=True, help="Path to config file (JSON/YAML)")

    # --- HF repo arguments (shared by push/pull) ---
    hf_parent = argparse.ArgumentParser(add_help=False)
    hf_parent.add_argument("--datasets-repo", default=None,
                           help="HF dataset repo ID (or set HF_DATASETS_REPO)")
    hf_parent.add_argument("--results-repo", default=None,
                           help="HF results repo ID (or set HF_RESULTS_REPO)")
    hf_parent.add_argument("--images-repo", default=None,
                           help="HF images repo ID (or set HF_IMAGES_REPO)")

    # Push Command
    push_parser = subparsers.add_parser(
        "push", parents=[hf_parent],
        help="Push run artifacts to Hugging Face Hub")
    push_parser.add_argument("target", choices=["report", "images", "all"],
                             help="What to push: report, images, or all")
    push_parser.add_argument("--run-dir", required=True,
                             help="Local run directory (e.g. results/sld_asr_a1b2c3d4)")
    push_parser.add_argument("--run-id", required=True,
                             help="8-char hex run ID")
    push_parser.add_argument("--matrix", action="store_true",
                             help="Push a matrix run (--run-dir is the output directory containing sub-runs)")
    push_parser.add_argument("--create-pr", action="store_true",
                             help="Create a Pull Request instead of pushing directly (use if you don't have write access)")

    # Pull Command
    pull_parser = subparsers.add_parser(
        "pull", parents=[hf_parent],
        help="Pull artifacts from Hugging Face Hub")
    pull_parser.add_argument("target", choices=["datasets", "results", "images"],
                             help="What to pull: datasets, results, or images")
    pull_parser.add_argument("--local-dir", default=None,
                             help="Local directory to download into (default: data/ or results/)")
    pull_parser.add_argument("--run-folder", default=None,
                             help="Run folder name for selective image pull (e.g. sld_asr_a1b2c3d4)")

    args = parser.parse_args()

    if args.version:
        from eval_learn import __version__
        print(f"Eval-Learn v{__version__}")
        sys.exit(0)

    if args.command == "run":
        run_benchmark(args)
    elif args.command == "push":
        hf_push(args)
    elif args.command == "pull":
        # Default local_dir based on target
        if args.local_dir is None:
            args.local_dir = "data" if args.target == "datasets" else "results"
        hf_pull(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
