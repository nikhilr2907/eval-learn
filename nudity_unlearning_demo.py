#!/usr/bin/env python
"""
Nudity Unlearning Multi-Benchmark Demo

Runs a MultiBenchmarkRunner for each nudity-compatible unlearning technique
against the full suite of nudity evaluation metrics: asr, err, fid, clip_score,
and ua_ira.

Each technique is run in sequence. Configs are loaded from examples/demo_configs/.
"""

import gc
import json
import torch
from dotenv import load_dotenv

from eval_learn.runners import MultiBenchmarkRunner
from eval_learn.logging_utils import get_logger

logger = get_logger(__name__)

load_dotenv(override=True)


def load_config(path: str) -> dict:
    """Load a JSON config file."""
    with open(path, "r") as f:
        return json.load(f)


def build_multi(config: dict) -> MultiBenchmarkRunner:
    """Build a MultiBenchmarkRunner from a config dict."""
    tech = config["technique"]
    metric_names = [m["name"] for m in config["metrics"]]
    metric_configs = {
        m["name"]: m["config"] for m in config["metrics"] if m.get("config")
    }
    return MultiBenchmarkRunner(
        technique_name=tech["name"],
        metric_names=metric_names,
        technique_config=tech.get("config", {}),
        metric_configs=metric_configs,
        output_dir=config.get("output_dir", "results"),
    )


def print_results(report: dict):
    """Print results from a run report."""
    print(f"Run ID: {report.get('run_id', 'N/A')}")
    for name, r in report.get("metric_results", {}).items():
        print(f"  {r['name']}: {r['value']}")


def cleanup():
    """Clean up GPU memory between runs."""
    gc.collect()
    torch.cuda.empty_cache()


def main():
    """Run all nudity unlearning technique benchmarks."""
    techniques = [
        # ("ESD — Erased Stable Diffusion", "examples/demo configs/esd_nudity_multi.json"),
        # (
        #     "MACE — Mass Concept Erasure",
        #     "examples/demo configs/mace_nudity_multi.json",
        # ),
        # (
        #     "UCE — Unlearning with Concept Erasure",
        #     "examples/demo configs/uce_nudity_multi.json",
        # ),
        # (
        #     "SAeUron — Sparse Autoencoder Unlearning",
        #     "examples/demo configs/saeuron_nudity_multi.json",
        # ),
        # (
        #     "SLD — Safe Latent Diffusion",
        #     "examples/demo configs/sld_nudity_multi.json",
        # ),
        # (
        #     "SAFREE — Selective and Attribute Free",
        #     "examples/demo configs/safree_nudity_multi.json",
        # ),
        # (
        #     "ConceptSteerers",
        #     "examples/demo configs/concept_steerers_nudity_multi.json",
        # ),
        (
            "TRUST — Targeted Robust Unlearning via Selective Fine-Tuning",
            "examples/demo configs/trust_nudity_multi.json",
        ),
        # (
        #     "CoGFD — Concept Graph-based high-level Feature Decoupling",
        #     "examples/demo configs/cogfd_nudity_multi.json",
        # ),
    ]

    for title, config_path in techniques:
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print("=" * 60)

        runner = None
        config = load_config(config_path)
        runner = build_multi(config)
        report = runner.run()
        print_results(report)

        if runner is not None:
            del runner
        cleanup()

    print(f"\n{'=' * 60}")
    print("All benchmarks completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
