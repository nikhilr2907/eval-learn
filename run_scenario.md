 Plan: Demo Script — Simulated User Workflows

 Overview

 Create a single standalone Python script run_demo.py in the project root that demonstrates how a user interacts with the eval-learn library. It uses the        
 lower-level API (load dataset → generate images once → score with multiple metrics → save artifacts), skips FID, and gracefully skips any metric whose optional 
  dependencies are missing.

 Results are saved to results/demo_runs/.

 File to Create

 run_demo.py    (project root, next to pyproject.toml)

 No other files modified.

 Script Structure

 """
 run_demo.py — Demonstrates eval-learn library workflows.

 Runs 3 scenarios showing SLD (Safe Latent Diffusion) evaluated
 with different metric combinations on appropriate datasets.
 All results saved to results/demo_runs/.

 Usage:
     python run_demo.py
 """

 Top-Level Config Constants

 DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 OUTPUT_DIR = "results/demo_runs"
 PROMPT_LIMIT = 5                    # top 5 rows per dataset
 NUM_INFERENCE_STEPS = 25            # lower = faster, less quality
 SLD_MODEL_ID = "AIML-TUDA/stable-diffusion-safe"

 Shared Helper: run_scenario()

 A reusable function that encapsulates the full pipeline:

 1. Look up dataset loader via get_dataset(name), call with config to get Dataset
 2. Look up technique via get_technique("sld"), instantiate with config
 3. Call technique.generate(prompts=dataset.prompts) — images generated once
 4. Loop over requested metrics:
   - try: instantiate metric, call metric.compute(images, prompts, metadata)
   - except RuntimeError: print skip warning (missing dep), continue
 5. Build combined report dict with all metric results
 6. Use ArtifactWriter to save images + JSON report
 7. Print summary table to stdout

 Three Scenarios

 Scenario 1: Safety + Faithfulness — SLD × TIFA dataset × [ASR, TIFA, CLIPScore]

 - Dataset: tifa_json (limit=5) — provides qa_pairs for TIFA; ASR and CLIPScore don't need special metadata
 - Why this grouping: TIFA dataset has the richest metadata, supporting all 3 metrics
 - Metrics:
   - asr — checks if SLD prevents unsafe image generation (config: use_nudenet=True; gracefully skipped if nudenet missing)
   - tifa — VQA-based faithfulness check against QA pairs
   - clip_score — text-image alignment score
 - Run name: "Scenario1_Safety_Faithfulness"

 Scenario 2: Alignment on Adversarial Prompts — SLD × I2P dataset × [ASR, CLIPScore]

 - Dataset: i2p_csv (limit=5) — adversarial prompts designed to elicit unsafe content
 - Why this grouping: Tests SLD's safety mechanism on its target use case
 - Metrics:
   - asr — the primary safety metric for this dataset
   - clip_score — how well SLD images match the (adversarial) prompts
 - Run name: "Scenario2_Alignment_Adversarial"

 Scenario 3: Full Evaluation Suite — All 4 metrics across their appropriate datasets

 Since ERR needs concepts+categories and TIFA needs qa_pairs, this scenario runs two generation passes:

 - Pass A: err_composite dataset (5 per source = 15 prompts) → ERR, ASR, CLIPScore
 - Pass B: tifa_json dataset (limit=5) → TIFA

 Combined report saved under "Scenario3_Full_Suite" with sub-sections for each pass.

 Output Format

 Each scenario produces:
 results/demo_runs/
 ├── Scenario1_Safety_Faithfulness/
 │   ├── images/run_{ts}/0.png ... 4.png
 │   └── report_{ts}.json
 ├── Scenario2_Alignment_Adversarial/
 │   ├── images/run_{ts}/0.png ... 4.png
 │   └── report_{ts}.json
 └── Scenario3_Full_Suite/
     ├── images/run_{ts}/           (ERR composite images)
     ├── images/run_{ts+1}/         (TIFA images)
     └── report_{ts}.json           (combined report)

 Report JSON structure (per scenario):
 {
     "scenario": "Scenario1_Safety_Faithfulness",
     "dataset": {"name": "tifa_json", "total_prompts": 5, "metadata": {...}},
     "technique": {"name": "sld", "config": {...}},
     "metric_results": {
         "asr": {"name": "ASR", "value": 0.0, "details": {...}},
         "tifa": {"name": "TIFA", "value": 0.85, "details": {...}},
         "clip_score": {"name": "CLIPScore", "value": 24.5, "details": {...}}
     },
     "skipped_metrics": ["asr"],
     "image_paths": [...],
     "timestamp": 1234567890.0
 }

 Stdout Output

 The script prints a readable summary after each scenario:

 === Scenario 1: Safety + Faithfulness ===
 Dataset: tifa_json (5 prompts)
 Technique: SLD (AIML-TUDA/stable-diffusion-safe)

   Metric        Score    Status
   ──────        ─────    ──────
   ASR           0.000    ✓
   TIFA          0.850    ✓
   CLIPScore     24.50    ✓

 Results saved to: results/demo_runs/Scenario1_Safety_Faithfulness/

 === Scenario 2: Alignment on Adversarial Prompts ===
 ...

 Dependency Handling

 At the top of the script, before any scenario runs:
 # Hard requirements (script won't run without these)
 import torch
 from eval_learn.registry import get_dataset, get_technique, get_metric
 from eval_learn.artifacts import ArtifactWriter
 from eval_learn.types import Dataset, MetricResult

 # Trigger registration of all components
 import eval_learn.datasets.i2p_csv
 import eval_learn.datasets.tifa_json
 import eval_learn.datasets.err_composite
 import eval_learn.metrics.asr.metric
 import eval_learn.metrics.tifa.metric
 import eval_learn.metrics.err.metric
 import eval_learn.metrics.clip_score.metric
 import eval_learn.techniques.sld.wrapper

 Individual metric failures (e.g., nudenet not installed) are caught per-metric in the try/except inside run_scenario().

 Dataset Paths

 Uses the actual files in data/:
 - data/tifa/sensitive_text_inputs.json + data/tifa/sensitive_question_answers.json
 - data/i2p/i2p_benchmark_sample.csv
 - data/ERR/raw_csv_data/challenge_dataset.csv
 - data/ring_a_bell/ring_a_bell_dataset.csv

 Guard and Entry Point

 if __name__ == "__main__":
     print("eval-learn Demo: Evaluating SLD Technique")
     print(f"Device: {DEVICE}")
     print(f"Output: {OUTPUT_DIR}\n")

     scenario_1_safety_faithfulness()
     scenario_2_alignment_adversarial()
     scenario_3_full_suite()

     print("\nAll scenarios complete. Results in:", OUTPUT_DIR)

 Verification

 # Run the full demo (requires GPU for reasonable speed, but works on CPU)
 python run_demo.py

 # Check results were created
 ls results/demo_runs/

 # Inspect a report
 python -c "import json; print(json.dumps(json.load(open('results/demo_runs/Scenario1_Safety_Faithfulness/report_*.json')), indent=2))"

 Key Design Decisions

 1. Lower-level API — generates images once per scenario, evaluates with multiple metrics. More efficient than running BenchmarkRunner N times.
 2. FID skipped — requires tensorflow + reference image directory. Not worth the complexity.
 3. Graceful skip — each metric instantiation is wrapped in try/except; missing deps produce a warning, not a crash.
 4. Scenario 3 uses two passes — because ERR and TIFA need different datasets. This is realistic: a real user would do the same.
 5. Registration imports at top — mirrors what cli.py does; ensures all components are available in the registry.