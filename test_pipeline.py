"""
test_pipeline.py — Integration test for the BenchmarkRunner saving structure.

Runs SLD against each available metric with minimal prompts (1-2) to verify
that the folder layout is correct:

    results/<technique>_<metric>_<run_id>/
        images/
            <category_subdirs if applicable>/
                0.png, ...
        <run_id>_report.json

Usage:
    python test_pipeline.py

Requires GPU and HF_TOKEN in .env for model downloads.
"""

import os
import json
import torch

from dotenv import load_dotenv
load_dotenv(override=True)

from eval_learn.registry import get_dataset, get_technique, get_metric
from eval_learn.runners import BenchmarkRunner
from eval_learn.logging_utils import get_logger

# Trigger registration
import eval_learn.techniques.sld.wrapper
import eval_learn.techniques.SAFREE.wrapper
import eval_learn.metrics.asr.metric
import eval_learn.metrics.fid.metric
import eval_learn.metrics.err.metric
import eval_learn.metrics.tifa.metric
import eval_learn.metrics.clip_score.metric
import eval_learn.datasets.i2p_csv
import eval_learn.datasets.err_composite
import eval_learn.datasets.tifa_json
import eval_learn.datasets.coco_parquet

logger = get_logger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results/pipeline_test"
SLD_MODEL_ID = "AIML-TUDA/stable-diffusion-safe"
PROMPT_LIMIT = 2


def run_test(metric_name, dataset_name, dataset_config, metric_config):
    """Run a single BenchmarkRunner test and verify output structure."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing: sld + {metric_name} on {dataset_name}")
    logger.info(f"{'='*50}")

    runner = BenchmarkRunner(
        dataset_loader=get_dataset(dataset_name),
        technique_factory=get_technique("safree"),
        metric_factory=get_metric(metric_name),
        technique_name="safree",
        metric_name=metric_name,
        dataset_name=dataset_name,
        technique_config={"model_id": SLD_MODEL_ID, "device": DEVICE},
        metric_config=metric_config,
        dataset_config=dataset_config,
        output_dir=OUTPUT_DIR,
    )

    report = runner.run()
    run_id = report["run_id"]

    # Verify folder structure
    run_dir = os.path.join(OUTPUT_DIR, f"sld_{metric_name}_{run_id}")
    images_dir = os.path.join(run_dir, "images")
    report_path = os.path.join(run_dir, f"{run_id}_report.json")

    assert os.path.isdir(run_dir), f"Run directory missing: {run_dir}"
    assert os.path.isdir(images_dir), f"Images directory missing: {images_dir}"
    assert os.path.isfile(report_path), f"Report file missing: {report_path}"

    # Verify report content
    with open(report_path) as f:
        saved_report = json.load(f)
    assert saved_report["run_id"] == run_id
    assert saved_report["technique_name"] == "sld"
    assert saved_report["metric_name"] == metric_name
    assert "metric_result" in saved_report

    logger.info(f"PASS: sld_{metric_name}_{run_id}")
    logger.info(f"  Score: {report['metric_result']['value']}")
    logger.info(f"  Dir: {run_dir}")

    return report


def test_sld_asr():
    """SLD + ASR on I2P (flat images, no categories)."""
    return run_test(
        metric_name="asr",
        dataset_name="i2p_csv",
        dataset_config={"limit": PROMPT_LIMIT},
        metric_config={"use_nudenet": True, "device": DEVICE},
    )


def test_sld_clip_score():
    """SLD + CLIP Score on I2P (flat images)."""
    return run_test(
        metric_name="clip_score",
        dataset_name="i2p_csv",
        dataset_config={"limit": PROMPT_LIMIT},
        metric_config={"device": DEVICE},
    )


def test_sld_err():
    """SLD + ERR on err_composite (category subdirectories)."""
    report = run_test(
        metric_name="err",
        dataset_name="err_composite",
        dataset_config={
            "target_limit": PROMPT_LIMIT,
            "retain_limit": PROMPT_LIMIT,
            "adversarial_limit": PROMPT_LIMIT,
        },
        metric_config={"device": DEVICE},
    )

    # Verify category subdirectories exist
    run_id = report["run_id"]
    images_dir = os.path.join(OUTPUT_DIR, f"sld_err_{run_id}", "images")
    for cat in ("target", "retain", "adversarial"):
        cat_dir = os.path.join(images_dir, cat)
        assert os.path.isdir(cat_dir), f"Category dir missing: {cat_dir}"
        pngs = [f for f in os.listdir(cat_dir) if f.endswith(".png")]
        assert len(pngs) > 0, f"No images in {cat_dir}"
        logger.info(f"  {cat}/: {len(pngs)} images")

    return report


def test_sld_tifa():
    """SLD + TIFA on tifa_json (flat images)."""
    return run_test(
        metric_name="tifa",
        dataset_name="tifa_json",
        dataset_config={"limit": PROMPT_LIMIT},
        metric_config={"device": DEVICE},
    )


def test_sld_fid():
    """SLD + FID on COCO (flat images)."""
    return run_test(
        metric_name="fid",
        dataset_name="coco_parquet",
        dataset_config={"limit": PROMPT_LIMIT},
        metric_config={
            "real_images_dir": "data/coco/real_images",
        },
    )


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}\n")

    results = {}
    tests = [
        ("asr", test_sld_asr),
        ("clip_score", test_sld_clip_score),
        ("err", test_sld_err),
        ("tifa", test_sld_tifa),
        ("fid", test_sld_fid),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
            print(f"  [{name}] PASS")
        except Exception as e:
            print(f"  [{name}] FAIL: {e}")
            results[name] = None

    # Summary
    print(f"\n{'='*50}")
    print("Pipeline Test Summary")
    print(f"{'='*50}")
    passed = sum(1 for v in results.values() if v is not None)
    print(f"  {passed}/{len(tests)} tests passed")
    for name, report in results.items():
        if report:
            score = report["metric_result"]["value"]
            run_id = report["run_id"]
            print(f"  {name}: {score:.4f} (run_id: {run_id})")
        else:
            print(f"  {name}: FAILED")
