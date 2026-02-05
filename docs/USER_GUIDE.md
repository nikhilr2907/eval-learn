# Eval-Learn User Guide

Welcome to the **Eval-Learn** library. This guide focuses on using the library programmatically (via Python code) to benchmark unlearning techniques in text-to-image diffusion models.

## 1. Installation

Install the library with the specific dependencies you need (or `all` for everything).

```bash
# Install core + dependencies for all metrics/techniques
pip install -e .[all]

# OR install specific extras
# pip install -e .[diffusers,asr,fid]
```

---

## 2. Core Concepts

The library is built around a **Runner** architecture. You don't write loop logic; you configure components and let the runner execute them.

1.  **Dataset:** Provides a list of prompts (and optionally metadata like questions or target concepts).
2.  **Technique:** An "Unlearning" method (e.g., Safe Latent Diffusion) that takes prompts and generates images.
3.  **Metric:** Evaluates the generated images (e.g., detects nudity, checks prompt adherence).
4.  **Runner:** Ties them together: `Dataset` -> `Technique` -> `Images` -> `Metric` -> `Report`.

---

## 3. Step-by-Step Guide

Here is the standard workflow to create a benchmark run.

### Step 1: Import Registry & Runner

We use a registry system to load components. This ensures modularity.

```python
from eval_learn.registry import get_dataset, get_technique, get_metric
from eval_learn.runners import BenchmarkRunner
```

### Step 2: Load a Dataset

Currently, we support the **I2P (Inappropriate Image Prompts)** dataset.

```python
# Get the loader function
DatasetLoader = get_dataset("i2p_csv")

# Configuration for the dataset
dataset_config = {
    "path": "data/i2p/i2p_benchmark.csv", # Path to your CSV
    "limit": 10,                          # Number of prompts to use (Optional)
    "prompt_col": "prompt"                # Column name in CSV
}
```

### Step 3: Configure a Technique

Select and configure the unlearning technique. Example: **SLD (Safe Latent Diffusion)**.

```python
# Get the technique factory
TechniqueFactory = get_technique("sld")

# Configuration for SLD
technique_config = {
    "model_id": "CompVis/stable-diffusion-v1-4", # Base model
    "device": "cuda",                            # "cuda", "cpu", or "mps"
    
    # SLD Specific Hyperparameters (SafetyConfig.MAX equivalent)
    "safety_concept": "nudity",
    "sld_guidance_scale": 2000,
    "sld_warmup_steps": 7,
    "sld_threshold": 0.025,
    "sld_momentum_scale": 0.5,
    "sld_mom_beta": 0.7
}
```

### Step 4: Configure a Metric

Select the metric to evaluate the performance. Example: **ASR (Attack Success Rate)** for NSFW detection.

```python
# Get the metric factory
MetricFactory = get_metric("asr")

# Configuration for ASR
metric_config = {
    "use_nudenet": True,  # Enable NudeNet classifier
    "use_q16": False      # Enable Q16 classifier (Optional)
}
```

### Step 5: Run the Benchmark

Initialize the runner and execute.

```python
runner = BenchmarkRunner(
    dataset_loader=DatasetLoader,
    technique_factory=TechniqueFactory,
    metric_factory=MetricFactory,
    dataset_config=dataset_config,
    technique_config=technique_config,
    metric_config=metric_config,
    output_dir="results/my_experiment",
    run_name="SLD_Max_I2P"
)

# Returns a dictionary containing the results
report = runner.run()

print(f"Final Score: {report['metric_result']['value']}")
```

---

## 4. Available Components & Configurations

### A. Techniques

#### `sld` (Safe Latent Diffusion)
Wraps a Stable Diffusion pipeline with safety guidance.

| Config Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_id` | `str` | `"AIML-TUDA/..."` | The HuggingFace model ID to load. |
| `device` | `str` | `None` | Execution device (`cuda`, `cpu`, `mps`). |
| `safety_concept` | `str` | `"nudity"` | The concept to suppress. |
| `preset` | `str` | `None` | Named preset: `"none"`, `"weak"`, `"medium"`, `"strong"`, `"max"`. Sets all SLD parameters below automatically. |
| `sld_guidance_scale` | `float` | `5000` | Strength of the safety guidance. |
| `sld_warmup_steps` | `int` | `0` | Steps to apply safety at the start. |
| `sld_threshold` | `float` | `1.0` | Activation threshold for safety. |
| `sld_momentum_scale` | `float` | `0.5` | Momentum for safety guidance. |
| `sld_mom_beta` | `float` | `0.7` | Momentum beta. |

**SLD Preset Reference:**

Instead of setting individual parameters, you can use the `preset` field. Any individual SLD parameter you also pass will override the preset value for that parameter.

| Preset | `sld_guidance_scale` | `sld_warmup_steps` | `sld_threshold` | `sld_momentum_scale` | `sld_mom_beta` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `none` | 0 | 0 | 0.0 | 0.0 | 0.0 |
| `weak` | 200 | 15 | 0.0 | 0.0 | 0.0 |
| `medium` | 1000 | 10 | 0.01 | 0.3 | 0.4 |
| `strong` | 2000 | 7 | 0.025 | 0.5 | 0.7 |
| `max` | 5000 | 0 | 1.0 | 0.5 | 0.7 |

---

### B. Metrics

#### 1. `asr` (Attack Success Rate)
Measures the percentage of images that are detected as unsafe (NSFW). **Lower is better.**

| Config Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `use_nudenet` | `bool` | `True` | Uses NudeNet library for detection. |
| `use_q16` | `bool` | `False` | Uses Q16/CLIP-based classifier. |
| `device` | `str` | `None` | Device for the classifier models. |

#### 2. `fid` (Frechet Inception Distance)
Measures the distance between the distribution of generated images and a directory of real images. **Lower is better.**

| Config Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `real_images_dir` | `str` | **Required** | Path to folder containing reference images (jpg/png). |
| `batch_size` | `int` | `32` | Batch size for InceptionV3 inference. |
| `device` | `str` | `None` | (Not used by TF implementation, but available). |

#### 3. `err` (Erasing-Retention-Robustness)
Uses CLIP to measure if specific concepts are erased, retained, or robust against adversarial prompts. **Higher is better.**
*Requires dataset metadata (`concepts`, `categories`) to function.*

| Config Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `clip_model_name` | `str` | `"openai/clip-vit-large-patch14"` | CLIP model for scoring. |
| `device` | `str` | `None` | Device for CLIP. |

#### 4. `tifa` (Text-to-Image Faithfulness)
Uses BLIP-2 VQA to answer questions about the image to verify prompt adherence. **Higher is better.**
*Requires dataset metadata (`qa_pairs`) to function.*

| Config Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `vqa_model_name` | `str` | `"Salesforce/blip2-flan-t5-xl"` | VQA model to use. |
| `device` | `str` | `None` | Device for BLIP-2. |

---

### C. Datasets

#### `i2p_csv`
Loads prompts from a CSV file. Designed for the I2P benchmark but works with any CSV containing a prompt column.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` | `"data/i2p/..."` | Path to the CSV file. |
| `limit` | `int` | `None` | Limit the number of prompts loaded (useful for testing). |
| `prompt_col` | `str` | `"prompt"` | The column name containing the text prompts. |

---

## 5. Advanced: Running Multiple Benchmarks

You can easily loop over configurations to compare hyperparameters.

```python
scales = [0, 500, 1000, 2000]

for scale in scales:
    tech_config = {
        "model_id": "CompVis/stable-diffusion-v1-4",
        "sld_guidance_scale": scale,
        # ... other params ...
    }

    runner = BenchmarkRunner(
        # ... same factories ...
        technique_config=tech_config,
        # ... same metric config ...
        run_name=f"SLD_Scale_{scale}"
    )

    runner.run()
```

---

## 6. Recipes: Common Evaluation Configurations

Below are complete, copy-pasteable examples for common evaluation setups.

### Recipe 1 — SLD MAX with ASR Only

Evaluate the strongest SLD safety preset and measure how many generated images are still detected as unsafe (NSFW). This uses a single metric with the `BenchmarkRunner`.

```python
from eval_learn.registry import get_dataset, get_technique, get_metric
from eval_learn.runners import BenchmarkRunner

# Components
DatasetLoader = get_dataset("i2p_csv")
TechniqueFactory = get_technique("sld")
MetricFactory = get_metric("asr")

# Configs
dataset_config = {
    "path": "data/i2p/i2p_benchmark.csv",
    "limit": 50,
}

technique_config = {
    "preset": "max",       # Uses SLD-MAX safety parameters
    "device": "cuda",
}

metric_config = {
    "use_nudenet": True,
}

# Run
runner = BenchmarkRunner(
    dataset_loader=DatasetLoader,
    technique_factory=TechniqueFactory,
    metric_factory=MetricFactory,
    dataset_config=dataset_config,
    technique_config=technique_config,
    metric_config=metric_config,
    output_dir="results/sld_max_asr",
    run_name="SLD_Max_ASR",
)

report = runner.run()
print(f"ASR Score: {report['metric_result']['value']}")
```

**CLI equivalent** — save this as `config_sld_max_asr.json` and run with `eval-learn run --config config_sld_max_asr.json`:

```json
{
    "run_name": "SLD_Max_ASR",
    "output_dir": "results/sld_max_asr",
    "dataset": {
        "name": "i2p_csv",
        "config": { "path": "data/i2p/i2p_benchmark.csv", "limit": 50 }
    },
    "technique": {
        "name": "sld",
        "config": { "preset": "max", "device": "cuda" }
    },
    "metric": {
        "name": "asr",
        "config": { "use_nudenet": true }
    }
}
```

---

### Recipe 2 — SLD WEAK with ASR + CLIPScore

Evaluate the lightest SLD safety preset on two metrics: safety (ASR) and image-text alignment (CLIPScore). Since `BenchmarkRunner` accepts one metric at a time, we generate images once and then evaluate each metric separately using the lower-level API.

```python
from eval_learn.registry import get_dataset, get_technique, get_metric

# 1. Load dataset
DatasetLoader = get_dataset("i2p_csv")
dataset = DatasetLoader(path="data/i2p/i2p_benchmark.csv", limit=50)

# 2. Generate images once with SLD-WEAK
TechniqueFactory = get_technique("sld")
technique = TechniqueFactory(preset="weak", device="cuda")
images = technique.generate(prompts=dataset.prompts)

# 3. Evaluate ASR
ASRMetric = get_metric("asr")
asr = ASRMetric(use_nudenet=True)
asr_result = asr.compute(images=images, prompts=dataset.prompts, metadata=dataset.metadata)
print(f"ASR Score: {asr_result.value}")

# 4. Evaluate CLIPScore
CLIPScoreMetric = get_metric("clip_score")
clip = CLIPScoreMetric(device="cuda")
clip_result = clip.compute(images=images, prompts=dataset.prompts, metadata=dataset.metadata)
print(f"CLIPScore: {clip_result.value}")
```

This pattern avoids generating images twice. You can add any number of additional metrics the same way — just instantiate and call `.compute()` on the same `images` list.
