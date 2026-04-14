# CLI Reference

Eval-Learn provides a command-line interface for running benchmarks, managing results on
Hugging Face Hub, and inspecting installed plugins.

```
eval-learn <command> [options]
eval-learn --version
```

---

## Commands

### `run` — Execute a benchmark

Run a benchmark defined in a config file.

```bash
eval-learn run --config config.yaml
eval-learn run --config config.json
```

The runner type is determined automatically from the config:

- `technique` + `metric` → [SingleBenchmarkRunner](running-experiments/single-runner.md)
- `technique` + `metrics` → [MultiBenchmarkRunner](running-experiments/multi-runner.md)

Results are written to `output_dir` as defined in the config.

**Options:**

| Flag | Description |
|------|-------------|
| `--config`, `-c` | Path to config file (JSON or YAML). Required. |
| `--hf-repo REPO_ID` | Push results to a HF Hub dataset repo after the run (e.g. `org/my-results`). |
| `--hf-path PATH` | Remote path inside the repo. Defaults to the basename of `output_dir`. |
| `--create-pr` | Open a pull request on HF Hub instead of committing directly. |

**Examples:**

```bash
# Basic run
eval-learn run --config examples/demo_configs/esd_nudity_multi.json

# Run and push results to HF Hub
eval-learn run --config config.yaml --hf-repo my-org/eval-results

# Run and open a PR instead of committing directly
eval-learn run --config config.yaml --hf-repo my-org/eval-results --create-pr
```

---

### `push` — Upload results to HF Hub

Push a local results directory to a Hugging Face Hub dataset repo.

```bash
eval-learn push --repo REPO_ID --local-dir PATH
```

**Options:**

| Flag | Description |
|------|-------------|
| `--repo REPO_ID` | HF Hub dataset repo ID (e.g. `my-org/eval-results`). Required. |
| `--local-dir PATH` | Local directory to upload. Required. |
| `--remote-path PATH` | Destination path in the repo. Defaults to the basename of `--local-dir`. |
| `--create-pr` | Open a pull request instead of committing directly. |

**Example:**

```bash
eval-learn push --repo my-org/eval-results --local-dir results/esd_nudity_multi
```

---

### `pull` — Download results from HF Hub

Pull artifacts from a Hugging Face Hub dataset repo.

```bash
eval-learn pull --repo REPO_ID
eval-learn pull --repo REPO_ID --remote-path esd_nudity_multi
```

**Options:**

| Flag | Description |
|------|-------------|
| `--repo REPO_ID` | HF Hub dataset repo ID. Required. |
| `--remote-path PATH` | Specific path within the repo to download. Omit to pull the entire repo. |
| `--local-dir PATH` | Local directory to download into. Defaults to `results/`. |

**Examples:**

```bash
# Pull a specific run
eval-learn pull --repo my-org/eval-results --remote-path esd_nudity_multi

# Pull everything
eval-learn pull --repo my-org/eval-results --local-dir my-results/
```

---

### `plugins` — List registered plugins

Print all installed techniques, metrics, and datasets.

```bash
eval-learn plugins
```

Output example:

```
Techniques:
  advunlearn
  ca
  cogfd
  concept_steerers
  esd
  free_run
  mace
  safree
  saeuron
  sld
  ssd
  trasce
  uce

Metrics:
  asr_i2p
  asr_mma_diffusion
  asr_p4d
  asr_ring_a_bell
  clip_score
  err
  fid
  tifa
  ua_ira

Datasets:
  coco_parquet
  err_composite
  i2p_csv
  tifa_csv
  ua_ira_csv
```

Use this to confirm that optional technique or metric packages are correctly installed and
discovered via entry points.

---

### `models` — Show base models

Print the base Stable Diffusion model and evaluation models used by each technique and metric.

```bash
eval-learn models
```

Output example:

```
Techniques:
  name                 base model                                    configurable
  -------------------- --------------------------------------------- ------------
  esd                  CompVis/stable-diffusion-v1-4                 no
  free_run             (user-specified via model_id)                 required
  ...

Metrics:
  name                 model                                         configurable
  -------------------- --------------------------------------------- ------------
  asr_i2p              NudeNet / openai/clip-vit-large-patch14       yes  (config: clip_model_id: ...)
  clip_score           openai/clip-vit-base-patch32                  yes  (config: clip_model_name: ...)
  ...
```

Useful for checking compatibility and understanding which models will be downloaded before
running a benchmark.

---

## Config format

Both JSON and YAML are supported and fully equivalent. The runner is selected based on
whether `metric` (singular) or `metrics` (list) is present.

=== "YAML"

    ```yaml
    technique:
      name: esd
      config:
        erase_concept: nudity
        train_method: noxattn

    metrics:
      - name: asr
        config:
          device: cuda
          limit: 500
      - name: fid
        config:
          device: cuda

    output_dir: results/esd_nudity
    ```

=== "JSON"

    ```json
    {
      "technique": {
        "name": "esd",
        "config": {
          "erase_concept": "nudity",
          "train_method": "noxattn"
        }
      },
      "metrics": [
        { "name": "asr",  "config": { "device": "cuda", "limit": 500 } },
        { "name": "fid",  "config": { "device": "cuda" } }
      ],
      "output_dir": "results/esd_nudity"
    }
    ```

---

## HF authentication

Most techniques download model weights from HF Hub. Create a `.env` file at the project
root with your token:

```bash
HF_TOKEN=your_token_here
```

Eval-Learn loads this automatically on startup via `python-dotenv`. Alternatively, export
it in your shell:

```bash
export HF_TOKEN=your_token_here
```
