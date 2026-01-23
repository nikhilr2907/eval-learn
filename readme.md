# Eval-Learn: Unlearning Benchmark for Text-to-Image Models

**Eval-Learn** is a comprehensive evaluation library designed to benchmark "unlearning" techniques in Text-to-Image (T2I) diffusion models. It provides a standardized framework to measure the effectiveness of safety filters and unlearning algorithms against harmful content generation.

## 📂 Project Structure

The project is organized into modular components to support easy addition of new techniques and metrics.

```text
eval-learn/
├── core/                   # Abstract base classes for techniques and benchmarks
├── evaluation_metrics/     # Metric implementations (ASR, etc.)
│   └── asr_metric/         # Attack Success Rate (NudeNet, Q16)
├── unlearning_techniques/  # Unlearning method wrappers
│   └── sld_pipeline/       # Safe Latent Diffusion (SLD)
├── data/                   # Benchmark datasets (e.g., I2P)
├── results/                # Output images and JSON reports
└── benchmark.py            # Main execution script
```

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.8 - 3.12 for CUDA Compatibility
- CUDA-capable GPU (Recommended) or Apple Silicon (MPS)
- Hugging Face Account (for model access)

### 2. Install Dependencies
Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

*Note: If you run into issues installing `nudenet`, please refer to their [official documentation](https://github.com/notAI-tech/NudeNet).*

### 3. Environment Configuration
Create a `.env` file in the root directory to store your Hugging Face token. This is required to download models like `stable-diffusion-safe`.

**File: `.env`**
```ini
HF_TOKEN=your_huggingface_write_token_here
```

## 📊 Running the Benchmark

The main entry point is `benchmark.py`. This script runs the evaluation pipeline, comparing different configurations of unlearning techniques against specified metrics.

To run the standard ASR (Attack Success Rate) benchmark with SLD:

```bash
python benchmark.py
```

### What happens when you run it?
1.  **Loads Data**: Reads prompts from `data/i2p_benchmark.csv`.
2.  **Loads Models**: Initializes the Stable Diffusion Safety pipeline (SLD).
3.  **Generates Images**: Runs generation for specified configurations (e.g., SLD Max vs. SLD Disabled).
4.  **Calculates Metrics**: Uses NudeNet (and optionally Q16) to detect unsafe content.
5.  **Saves Results**:
    - JSON Reports: `results/benchmarks/ASR_Benchmark/report_{timestamp}.json`
    - Images: `results/benchmarks/ASR_Benchmark/images/`

## 🛠️ Supported Features

### Unlearning Techniques
- **SLD (Safe Latent Diffusion)**: A white-box inference-time intervention method.
    - Supported Configs: `MAX`, `STRONG`, `MEDIUM`, `WEAK`, and `Disabled` (Standard SD).

### Evaluation Metrics
- **ASR (Attack Success Rate)**: Measures the percentage of generated images containing unsafe content (Nudity/Inappropriate).
    - **NudeNet**: Detects exposed body parts.
    - **Q16 (CLIP)**: (Coming Soon) Detects general inappropriate content.
