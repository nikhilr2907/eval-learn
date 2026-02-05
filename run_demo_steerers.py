"""
run_demo_steerers.py — Demonstrates SAE-based Concept Steering workflows.

Runs 6 scenarios showing Concept Steerers (SAE) evaluated with 
different metric combinations on appropriate datasets.

  Scenario 1: SAE MAX (m=4.0) + ASR, CLIPScore (I2P)
  Scenario 2: SAE WEAK (m=1.0) + ASR, CLIPScore (I2P)
  Scenario 3: Full Suite — ERR/ASR/CLIPScore + TIFA (multi-pass)
  Scenario 4: SAE MAX + ASR only (I2P)
  Scenario 5: SAE MAX + TIFA only (TIFA captions)
  Scenario 6: SAE MAX + FID (COCO)
"""
import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import time
import torch
from typing import Dict, List, Any, Optional

from eval_learn.registry import get_dataset, get_technique, get_metric
from eval_learn.artifacts import ArtifactWriter
from eval_learn.types import Dataset, MetricResult
from eval_learn.logging_utils import get_logger

# Import modules to trigger registration
import eval_learn.datasets.i2p_csv
import eval_learn.datasets.tifa_json
import eval_learn.datasets.coco_parquet
import eval_learn.datasets.err_composite
import eval_learn.metrics.asr.metric
import eval_learn.metrics.tifa.metric
import eval_learn.metrics.fid.metric
import eval_learn.metrics.err.metric
import eval_learn.metrics.clip_score.metric
import eval_learn.techniques.concept_steerers.wrapper

logger = get_logger(__name__)

# --- Config constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results/steerers_demo_full"
PROMPT_LIMIT = 5
NUM_INFERENCE_STEPS = 50
MODEL_ID = "CompVis/stable-diffusion-v1-4"

def run_scenario(
    scenario_name: str,
    dataset: Dataset,
    dataset_name: str,
    metric_configs: List[Dict[str, Any]],
    technique=None,
    images: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Run a single evaluation scenario."""
    
    # 1. Generate images
    if images is None:
        if technique is None:
            raise ValueError("Either technique or images must be provided")
        logger.info(f"[{scenario_name}] Generating images for {len(dataset.prompts)} prompts...")
        images = technique.generate(
            prompts=dataset.prompts,
            num_inference_steps=NUM_INFERENCE_STEPS,
        )

    # 2. Evaluate with each metric
    metric_results: Dict[str, Any] = {}
    skipped_metrics: List[str] = []
    for mcfg in metric_configs:
        name = mcfg["name"]
        kwargs = mcfg.get("kwargs", {})
        try:
            logger.info(f"[{scenario_name}] Running metric: {name}...")
            metric_cls = get_metric(name)
            metric = metric_cls(**kwargs)
            
            # Add explicit memory clearing before metric computation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        except Exception as exc:
            # Enhanced error logging
            logger.error(f"[{scenario_name}] CRITICAL: Metric '{name}' failed!")
            logger.error(f"Error Type: {type(exc).__name__}")
            logger.error(f"Error Message: {exc}")
            skipped_metrics.append(name)

    # 3. Build report and persist
    report = {
        "scenario": scenario_name,
        "dataset": {"name": dataset_name, "total_prompts": len(dataset.prompts)},
        "technique": {
            "name": "concept_steerers",
            "config": {"concept": getattr(technique.config, 'concept', 'N/A'), "multiplier": getattr(technique.config, 'multiplier', 0.0)}
        },
        "metric_results": metric_results,
        "skipped_metrics": skipped_metrics,
    }

    writer = ArtifactWriter(base_dir=OUTPUT_DIR)
    writer.save_run(run_name=scenario_name, images=images, report=report)
    _print_summary(scenario_name, dataset_name, len(dataset.prompts), metric_results, skipped_metrics)
    return report

def _print_summary(scenario_name, dataset_name, n_prompts, metric_results, skipped_metrics):
    print(f"\n{'=' * 55}\n  {scenario_name}\n{'=' * 55}")
    print(f"  Dataset: {dataset_name} ({n_prompts} prompts)\n")
    for name, result in metric_results.items():
        print(f"  {result['name']:<16}{result['value']:>10.3f}    ok")
    for name in skipped_metrics:
        print(f"  {name:<16}{'--':>10}    SKIPPED")

# --- Scenarios ---

def scenario_1_sae_max():
    """Scenario 1: SAE MAX (m=4.0) + ASR, CLIPScore (I2P)."""
    loader = get_dataset("i2p_csv")
    dataset = loader(limit=PROMPT_LIMIT)
    tech = get_technique("concept_steerers")(model_id=MODEL_ID, device=DEVICE, multiplier=4.0, concept="nudity, blood")
    metrics = [{"name": "asr", "kwargs": {"use_nudenet": True, "device": DEVICE}}, {"name": "clip_score", "kwargs": {"device": DEVICE}}]
    run_scenario("Scenario1_SAE_Max", dataset, "i2p_csv", metrics, technique=tech)

def scenario_2_sae_weak():
    """Scenario 2: SAE WEAK (m=1.0) + ASR, CLIPScore (I2P)."""
    loader = get_dataset("i2p_csv")
    dataset = loader(limit=PROMPT_LIMIT)
    tech = get_technique("concept_steerers")(model_id=MODEL_ID, device=DEVICE, multiplier=1.0, concept="nudity, blood")
    metrics = [{"name": "asr", "kwargs": {"use_nudenet": True, "device": DEVICE}}, {"name": "clip_score", "kwargs": {"device": DEVICE}}]
    run_scenario("Scenario2_SAE_Weak", dataset, "i2p_csv", metrics, technique=tech)

def scenario_3_full_suite():
    """Scenario 3: Full Evaluation Suite (Multi-dataset pass)."""
    scenario_name = "Scenario3_Full_Suite"
    tech = get_technique("concept_steerers")(model_id=MODEL_ID, device=DEVICE, multiplier=2.5)
    
    # Pass A: ERR Composite
    loader_a = get_dataset("err_composite")
    ds_a = loader_a(target_limit=PROMPT_LIMIT, retain_limit=PROMPT_LIMIT)
    run_scenario(f"{scenario_name}_PassA", ds_a, "err_composite", [{"name": "err", "kwargs": {"device": DEVICE}}], technique=tech)

    # Pass B: TIFA
    loader_b = get_dataset("tifa_json")
    ds_b = loader_b(limit=PROMPT_LIMIT)
    run_scenario(f"{scenario_name}_PassB", ds_b, "tifa_json", [{"name": "tifa", "kwargs": {"device": DEVICE}}], technique=tech)

def scenario_4_sae_max_asr():
    """Scenario 4: SAE MAX + ASR only (I2P)."""
    dataset = get_dataset("i2p_csv")(limit=PROMPT_LIMIT)
    tech = get_technique("concept_steerers")(model_id=MODEL_ID, device=DEVICE, multiplier=4.0)
    run_scenario("Scenario4_SAE_Max_ASR", dataset, "i2p_csv", [{"name": "asr", "kwargs": {"device": DEVICE}}], technique=tech)

def scenario_5_sae_max_tifa():
    """Scenario 5: SAE MAX + TIFA only (TIFA captions)."""
    dataset = get_dataset("tifa_json")(limit=PROMPT_LIMIT)
    tech = get_technique("concept_steerers")(model_id=MODEL_ID, device=DEVICE, multiplier=4.0)
    run_scenario("Scenario5_SAE_Max_TIFA", dataset, "tifa_json", [{"name": "tifa", "kwargs": {"device": DEVICE}}], technique=tech)

def scenario_6_sae_max_fid():
    """Scenario 6: SAE MAX + FID (COCO)."""
    dataset = get_dataset("coco_parquet")(limit=PROMPT_LIMIT)
    tech = get_technique("concept_steerers")(model_id=MODEL_ID, device=DEVICE, multiplier=4.0)
    metrics = [{"name": "fid", "kwargs": {"real_images_dir": dataset.metadata["real_images_dir"]}}]
    run_scenario("Scenario6_SAE_Max_FID", dataset, "coco_parquet", metrics, technique=tech)

if __name__ == "__main__":
    scenario_1_sae_max()
    scenario_2_sae_weak()
    #scenario_3_full_suite()
    #scenario_4_sae_max_asr()
    scenario_5_sae_max_tifa()
    scenario_6_sae_max_fid()