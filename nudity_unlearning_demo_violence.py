#!/usr/bin/env python
"""
Violence Unlearning Multi-Benchmark Demo

Runs a MultiBenchmarkRunner for each violence-compatible unlearning technique
against the full suite of violence evaluation metrics: asr_p4d, asr_mma_diffusion,
err, fid, clip_score, ua_ira, and tifa.

Each technique is run in sequence. Configs are loaded from examples/demo_configs/.

NOTE: SAeUron and ConceptSteerers require external pre-built violence checkpoint
files before they can be run. Update the placeholder paths in their configs:
  - saeuron_violence_multi.json        → acts_path
  - concept_steerers_violence_multi.json → sae_path
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
    """Run all violence unlearning technique benchmarks."""
    techniques = [
        (
            "CoGFD — Concept Graph-based high-level Feature Decoupling",
            "examples/demo_configs/cogfd_violence_multi.json",
        ),
        (
            "SSD — Selective Synaptic Dampening",
            "examples/demo_configs/ssd_violence_multi.json",
        ),
        (
            "TraSCE — Training-free Stable Concept Editing",
            "examples/demo_configs/trasce_violence_multi.json",
        ),
        (
            "ESD — Erased Stable Diffusion",
            "examples/demo_configs/esd_violence_multi.json",
        ),
        (
            "MACE — Mass Concept Erasure",
            "examples/demo_configs/mace_violence_multi.json",
        ),
        (
            "UCE — Unlearning with Concept Erasure",
            "examples/demo_configs/uce_violence_multi.json",
        ),
        (
            "SLD — Safe Latent Diffusion",
            "examples/demo_configs/sld_violence_multi.json",
        ),
        (
            "SAFREE — Selective and Attribute Free (SVF disabled for violence)",
            "examples/demo_configs/safree_violence_multi.json",
        ),
        (
            "AdvUnlearn — Adversarial Unlearning",
            "examples/demo_configs/advunlearn_violence_multi.json",
        ),
        (
            "CA — Concept Ablation",
            "examples/demo_configs/ca_violence_multi.json",
        ),
    ]

    # Techniques that need external pre-built files — commented out by default.
    # Uncomment after supplying the required checkpoint paths in their configs.
    #
    # ("SAeUron — Sparse Autoencoder Unlearning (needs acts_path)",
    #  "examples/demo_configs/saeuron_violence_multi.json"),
    # ("ConceptSteerers (needs sae_path)",
    #  "examples/demo_configs/concept_steerers_violence_multi.json"),

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
