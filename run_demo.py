"""
run_demo.py — Demonstrates eval-learn library workflows.

Runs 6 scenarios showing SLD (Safe Latent Diffusion) evaluated
with different metric combinations on appropriate datasets.

  Scenario 1: SLD MAX  + ASR, CLIPScore  (I2P)
  Scenario 2: SLD WEAK + ASR, CLIPScore  (I2P)
  Scenario 3: Full Suite — ERR/ASR/CLIPScore + TIFA (multi-pass)
  Scenario 4: SLD MAX  + ASR only        (I2P)
  Scenario 5: SLD MAX  + TIFA only       (TIFA captions)
  Scenario 6: SLD MAX  + FID             (COCO)
  Scenario 7: UCE + ASR (I2P)            (requires UCE weights)

All results saved to results/demo_runs/.

Usage:
    python run_demo.py
"""

import time
import json
import os
from typing import Dict, List, Any, Optional

import torch

from eval_learn.registry import get_dataset, get_technique, get_metric
from eval_learn.artifacts import ArtifactWriter
from eval_learn.types import Dataset, MetricResult
from eval_learn.logging_utils import get_logger

# Trigger registration of all components
import eval_learn.datasets.i2p_csv
import eval_learn.datasets.tifa_json
import eval_learn.datasets.coco_parquet
import eval_learn.datasets.err_composite
import eval_learn.metrics.asr.metric
import eval_learn.metrics.tifa.metric
import eval_learn.metrics.fid.metric
import eval_learn.metrics.err.metric
import eval_learn.metrics.clip_score.metric
import eval_learn.techniques.sld.wrapper

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results/demo_runs"
PROMPT_LIMIT = 5
NUM_INFERENCE_STEPS = 25
SLD_MODEL_ID = "AIML-TUDA/stable-diffusion-safe"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def run_scenario(
    scenario_name: str,
    dataset: Dataset,
    dataset_name: str,
    metric_configs: List[Dict[str, Any]],
    technique=None,
    images: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Run a single evaluation scenario.

    Either *technique* or pre-computed *images* must be supplied.
    If *technique* is given, images are generated from ``dataset.prompts``.
    Each metric in *metric_configs* is attempted independently; failures
    (e.g. missing optional deps) are caught and the metric is recorded as
    skipped.

    Returns the report dict that was persisted by ArtifactWriter.
    """
    # ------------------------------------------------------------------
    # 1. Generate images (once per scenario)
    # ------------------------------------------------------------------
    if images is None:
        if technique is None:
            raise ValueError("Either technique or images must be provided")
        logger.info(f"[{scenario_name}] Generating images for {len(dataset.prompts)} prompts...")
        images = technique.generate(
            prompts=dataset.prompts,
            num_inference_steps=NUM_INFERENCE_STEPS,
        )

    # ------------------------------------------------------------------
    # 2. Evaluate with each metric
    # ------------------------------------------------------------------
    metric_results: Dict[str, Any] = {}
    skipped_metrics: List[str] = []

    for mcfg in metric_configs:
        name = mcfg["name"]
        kwargs = mcfg.get("kwargs", {})
        try:
            metric_cls = get_metric(name)
            metric = metric_cls(**kwargs)
            result: MetricResult = metric.compute(
                images=images,
                prompts=dataset.prompts,
                metadata=dataset.metadata,
            )
            metric_results[name] = {
                "name": result.name,
                "value": result.value,
                "details": result.details,
            }
        except RuntimeError as exc:
            logger.warning(f"[{scenario_name}] Skipping metric '{name}': {exc}")
            skipped_metrics.append(name)
        except Exception as exc:
            logger.warning(f"[{scenario_name}] Skipping metric '{name}' (unexpected): {exc}")
            skipped_metrics.append(name)

    # ------------------------------------------------------------------
    # 3. Build report and persist
    # ------------------------------------------------------------------
    report = {
        "scenario": scenario_name,
        "dataset": {
            "name": dataset_name,
            "total_prompts": len(dataset.prompts),
        },
        "technique": {
            "name": "sld",
            "config": {
                "model_id": SLD_MODEL_ID,
                "device": DEVICE,
                "num_inference_steps": NUM_INFERENCE_STEPS,
            },
        },
        "metric_results": metric_results,
        "skipped_metrics": skipped_metrics,
    }

    writer = ArtifactWriter(base_dir=OUTPUT_DIR)
    writer.save_run(run_name=scenario_name, images=images, report=report)

    # ------------------------------------------------------------------
    # 4. Print summary table
    # ------------------------------------------------------------------
    _print_summary(scenario_name, dataset_name, len(dataset.prompts), metric_results, skipped_metrics)

    return report


def _print_summary(
    scenario_name: str,
    dataset_name: str,
    n_prompts: int,
    metric_results: Dict[str, Any],
    skipped_metrics: List[str],
) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {scenario_name}")
    print(f"{'=' * 55}")
    print(f"  Dataset:   {dataset_name} ({n_prompts} prompts)")
    print(f"  Technique: SLD ({SLD_MODEL_ID})")
    print()
    print(f"  {'Metric':<16}{'Score':>10}    Status")
    print(f"  {'------':<16}{'-----':>10}    ------")

    for name, result in metric_results.items():
        value = result["value"]
        print(f"  {result['name']:<16}{value:>10.3f}    ok")

    for name in skipped_metrics:
        print(f"  {name:<16}{'--':>10}    SKIPPED")

    print(f"\n  Results saved to: {OUTPUT_DIR}/{scenario_name}/\n")


# ---------------------------------------------------------------------------
# Scenario 1: Safety + Faithfulness
# ---------------------------------------------------------------------------
def scenario_1_sld_max_with_asr_clip():
    """SLD x i2p_csv dataset x [ASR, TIFA, CLIPScore]."""
    scenario_name = "scenario_1_sld_max_with_asr_tifa_clip"
    dataset_name = "i2p_csv"

    loader = get_dataset(dataset_name)
    dataset = loader(limit=PROMPT_LIMIT)

    technique_cls = get_technique("sld")
    technique = technique_cls(model_id=SLD_MODEL_ID, device=DEVICE, preset="max")

    metric_configs = [
        {"name": "asr", "kwargs": {"use_nudenet": True, "device": DEVICE}},
        {"name": "clip_score", "kwargs": {"device": DEVICE}},
    ]

    run_scenario(
        scenario_name=scenario_name,
        dataset=dataset,
        dataset_name=dataset_name,
        metric_configs=metric_configs,
        technique=technique,
    )

# Scenario 2: SLD Weak + ASR, TIFA, CLIPScore
def scenario_2_sld_weak_with_asr_clip():
    """SLD x TIFA dataset x [ASR, TIFA, CLIPScore]."""
    scenario_name = "scenario_2_sld_weak_with_asr_tifa_clip"
    dataset_name = "i2p_csv"

    loader = get_dataset(dataset_name)
    dataset = loader(limit=PROMPT_LIMIT)

    technique_cls = get_technique("sld")
    technique = technique_cls(model_id=SLD_MODEL_ID, device=DEVICE, preset="weak")

    metric_configs = [
        {"name": "asr", "kwargs": {"use_nudenet": True, "device": DEVICE}},
        {"name": "clip_score", "kwargs": {"device": DEVICE}},
    ]

    run_scenario(
        scenario_name=scenario_name,
        dataset=dataset,
        dataset_name=dataset_name,
        metric_configs=metric_configs,
        technique=technique,
    )



# ---------------------------------------------------------------------------
# Scenario 2: Alignment on Adversarial Prompts
# ---------------------------------------------------------------------------
def scenario_2_alignment_adversarial():
    """SLD x I2P dataset x [ASR, CLIPScore]."""
    scenario_name = "Scenario2_Alignment_Adversarial"
    dataset_name = "i2p_csv"

    loader = get_dataset(dataset_name)
    dataset = loader(limit=PROMPT_LIMIT)

    technique_cls = get_technique("sld")
    technique = technique_cls(model_id=SLD_MODEL_ID, device=DEVICE)

    metric_configs = [
        {"name": "asr", "kwargs": {"use_nudenet": True, "device": DEVICE}},
        {"name": "clip_score", "kwargs": {"device": DEVICE}},
    ]

    run_scenario(
        scenario_name=scenario_name,
        dataset=dataset,
        dataset_name=dataset_name,
        metric_configs=metric_configs,
        technique=technique,
    )


# ---------------------------------------------------------------------------
# Scenario 3: Full Evaluation Suite
# ---------------------------------------------------------------------------
def scenario_3_full_suite():
    """All metrics across their appropriate datasets (two generation passes).

    Pass A: err_composite -> ERR, ASR, CLIPScore
    Pass B: tifa_json     -> TIFA
    Combined report saved under one scenario name.
    """
    scenario_name = "Scenario3_Full_Suite"

    technique_cls = get_technique("sld")
    technique = technique_cls(model_id=SLD_MODEL_ID, device=DEVICE)

    # --- Pass A: ERR composite dataset ---
    err_loader = get_dataset("err_composite")
    err_dataset = err_loader(
        target_limit=PROMPT_LIMIT,
        retain_limit=PROMPT_LIMIT,
        adversarial_limit=PROMPT_LIMIT,
    )

    logger.info(f"[{scenario_name}] Pass A: generating images for ERR composite ({len(err_dataset.prompts)} prompts)...")
    err_images = technique.generate(
        prompts=err_dataset.prompts,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    pass_a_metrics: Dict[str, Any] = {}
    pass_a_skipped: List[str] = []
    for mcfg in [
        {"name": "err", "kwargs": {"device": DEVICE}},
        {"name": "asr", "kwargs": {"use_nudenet": True, "device": DEVICE}},
        {"name": "clip_score", "kwargs": {"device": DEVICE}},
    ]:
        name = mcfg["name"]
        kwargs = mcfg.get("kwargs", {})
        try:
            metric_cls = get_metric(name)
            metric = metric_cls(**kwargs)
            result = metric.compute(
                images=err_images,
                prompts=err_dataset.prompts,
                metadata=err_dataset.metadata,
            )
            pass_a_metrics[name] = {
                "name": result.name,
                "value": result.value,
                "details": result.details,
            }
        except RuntimeError as exc:
            logger.warning(f"[{scenario_name}] Skipping metric '{name}': {exc}")
            pass_a_skipped.append(name)
        except Exception as exc:
            logger.warning(f"[{scenario_name}] Skipping metric '{name}' (unexpected): {exc}")
            pass_a_skipped.append(name)

    # --- Pass B: TIFA dataset ---
    tifa_loader = get_dataset("tifa_json")
    tifa_dataset = tifa_loader(limit=PROMPT_LIMIT)

    logger.info(f"[{scenario_name}] Pass B: generating images for TIFA ({len(tifa_dataset.prompts)} prompts)...")
    tifa_images = technique.generate(
        prompts=tifa_dataset.prompts,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    pass_b_metrics: Dict[str, Any] = {}
    pass_b_skipped: List[str] = []
    try:
        tifa_metric_cls = get_metric("tifa")
        tifa_metric = tifa_metric_cls(device=DEVICE)
        result = tifa_metric.compute(
            images=tifa_images,
            prompts=tifa_dataset.prompts,
            metadata=tifa_dataset.metadata,
        )
        pass_b_metrics["tifa"] = {
            "name": result.name,
            "value": result.value,
            "details": result.details,
        }
    except RuntimeError as exc:
        logger.warning(f"[{scenario_name}] Skipping metric 'tifa': {exc}")
        pass_b_skipped.append("tifa")
    except Exception as exc:
        logger.warning(f"[{scenario_name}] Skipping metric 'tifa' (unexpected): {exc}")
        pass_b_skipped.append("tifa")

    # --- Combined report ---
    ts = time.time()
    all_metric_results = {**pass_a_metrics, **pass_b_metrics}
    all_skipped = pass_a_skipped + pass_b_skipped

    report = {
        "scenario": scenario_name,
        "passes": {
            "pass_a": {
                "dataset": "err_composite",
                "total_prompts": len(err_dataset.prompts),
                "metrics": list(pass_a_metrics.keys()),
                "skipped": pass_a_skipped,
            },
            "pass_b": {
                "dataset": "tifa_json",
                "total_prompts": len(tifa_dataset.prompts),
                "metrics": list(pass_b_metrics.keys()),
                "skipped": pass_b_skipped,
            },
        },
        "technique": {
            "name": "sld",
            "config": {
                "model_id": SLD_MODEL_ID,
                "device": DEVICE,
                "num_inference_steps": NUM_INFERENCE_STEPS,
            },
        },
        "metric_results": all_metric_results,
        "skipped_metrics": all_skipped,
    }

    # Save ERR images first, then TIFA images in a separate pass
    writer = ArtifactWriter(base_dir=OUTPUT_DIR)
    writer.save_run(run_name=scenario_name, images=err_images, report=report, timestamp=ts)
    writer.save_run(run_name=scenario_name, images=tifa_images, report=report, timestamp=ts + 1)

    # --- Print summary ---
    total_prompts = len(err_dataset.prompts) + len(tifa_dataset.prompts)
    print(f"\n{'=' * 55}")
    print(f"  {scenario_name}")
    print(f"{'=' * 55}")
    print(f"  Pass A: err_composite ({len(err_dataset.prompts)} prompts)")
    print(f"  Pass B: tifa_json ({len(tifa_dataset.prompts)} prompts)")
    print(f"  Technique: SLD ({SLD_MODEL_ID})")
    print()
    print(f"  {'Metric':<16}{'Score':>10}    Status")
    print(f"  {'------':<16}{'-----':>10}    ------")
    for name, result in all_metric_results.items():
        print(f"  {result['name']:<16}{result['value']:>10.3f}    ok")
    for name in all_skipped:
        print(f"  {name:<16}{'--':>10}    SKIPPED")
    print(f"\n  Results saved to: {OUTPUT_DIR}/{scenario_name}/\n")


# ---------------------------------------------------------------------------
# Scenario 4: SLD MAX + ASR only
# ---------------------------------------------------------------------------
def scenario_4_sld_max_asr():
    """SLD-MAX x I2P dataset x [ASR]."""
    scenario_name = "Scenario4_SLD_Max_ASR"
    dataset_name = "i2p_csv"

    loader = get_dataset(dataset_name)
    dataset = loader(limit=PROMPT_LIMIT)

    technique_cls = get_technique("sld")
    technique = technique_cls(model_id=SLD_MODEL_ID, device=DEVICE, preset="max")

    metric_configs = [
        {"name": "asr", "kwargs": {"use_nudenet": True, "device": DEVICE}},
    ]

    run_scenario(
        scenario_name=scenario_name,
        dataset=dataset,
        dataset_name=dataset_name,
        metric_configs=metric_configs,
        technique=technique,
    )


# ---------------------------------------------------------------------------
# Scenario 5: SLD MAX + TIFA
# ---------------------------------------------------------------------------
def scenario_5_sld_max_tifa():
    """SLD-MAX x TIFA dataset x [TIFA]."""
    scenario_name = "Scenario5_SLD_Max_TIFA"
    dataset_name = "tifa_json"

    loader = get_dataset(dataset_name)
    dataset = loader(limit=PROMPT_LIMIT)

    technique_cls = get_technique("sld")
    technique = technique_cls(model_id=SLD_MODEL_ID, device=DEVICE, preset="max")

    metric_configs = [
        {"name": "tifa", "kwargs": {"device": DEVICE}},
    ]

    run_scenario(
        scenario_name=scenario_name,
        dataset=dataset,
        dataset_name=dataset_name,
        metric_configs=metric_configs,
        technique=technique,
    )


# ---------------------------------------------------------------------------
# Scenario 6: SLD MAX + FID (image quality on COCO)
# ---------------------------------------------------------------------------
def scenario_6_sld_max_fid():
    """SLD-MAX x COCO dataset x [FID].

    Loads captions from the COCO parquet file, generates images with
    SLD-MAX, then computes FID against the real COCO reference images
    that the dataset loader extracts to disk.
    """
    scenario_name = "Scenario6_SLD_Max_FID"
    dataset_name = "coco_parquet"

    loader = get_dataset(dataset_name)
    dataset = loader(limit=PROMPT_LIMIT)

    technique_cls = get_technique("sld")
    technique = technique_cls(model_id=SLD_MODEL_ID, device=DEVICE, preset="max")

    # The coco_parquet loader stores the extracted reference image directory
    # in metadata so we can pass it straight to the FID metric.
    real_images_dir = dataset.metadata["real_images_dir"]

    metric_configs = [
        {"name": "fid", "kwargs": {"real_images_dir": real_images_dir}},
    ]

    run_scenario(
        scenario_name=scenario_name,
        dataset=dataset,
        dataset_name=dataset_name,
        metric_configs=metric_configs,
        technique=technique,
    )


# ---------------------------------------------------------------------------
# Scenario 7: UCE + ASR only (I2P)
# ---------------------------------------------------------------------------
def scenario_7_uce_nudity_asr():
    """UCE x I2P dataset x [ASR].
    Uses pre-generated UCE weights from:
    python src/eval_learn/external/UCE/weights/

    *Had to install nudenet to work*

    For this example using nudity unlearning weights.
    """
    scenario_name = "Scenario7_UCE_Nudity_ASR"
    dataset_name = "i2p_csv"

    loader = get_dataset(dataset_name)
    dataset = loader(limit=PROMPT_LIMIT)

    technique_cls = get_technique("uce")
    technique = technique_cls(
        model_id="CompVis/stable-diffusion-v1-4",
        uce_weights_path="src/eval_learn/external/UCE/weights/uce_nudity.safetensors",
        device=DEVICE
    )

    metric_configs = [
        {"name": "asr", "kwargs": {"use_nudenet": True, "device": DEVICE}},
    ]

    run_scenario(
        scenario_name=scenario_name,
        dataset=dataset,
        dataset_name=dataset_name,
        metric_configs=metric_configs,
        technique=technique,
    )
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("eval-learn Demo: Evaluating SLD Technique")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    scenario_1_sld_max_with_asr_clip()
    scenario_2_sld_weak_with_asr_clip()
    # scenario_3_full_suite()
    # scenario_4_sld_max_asr()
    scenario_5_sld_max_tifa()
    scenario_6_sld_max_fid()
    
    scenario_7_uce_nudity_asr()

    print("\nAll scenarios complete. Results in:", OUTPUT_DIR)
