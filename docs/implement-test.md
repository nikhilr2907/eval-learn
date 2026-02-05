<!-- ````md -->

# Plan: Comprehensive Test Suite for eval-learn

## Overview

Create 11 test files + 1 shared conftest covering all components: types, configs, registry, 5 dataset loaders, 5 metrics (ASR, FID, ERR, TIFA, CLIPScore), ArtifactWriter, and BenchmarkRunner. All heavy models mocked — no GPU, no downloads, no optional deps required.

## Files to Create

```text
tests/
  conftest.py                     # Shared fixtures (7 fixtures)
  test_types.py                   # Dataset, MetricResult dataclasses (6 tests)
  test_configs.py                 # BaseConfig + all 6 config subclasses (18 tests)
  test_registry.py                # Registry decorators & lookup (13 tests)
  test_datasets.py                # All 5 dataset loaders (27 tests)
  test_metric_asr.py              # ASR metric, mocked NudeNet (10 tests)
  test_metric_fid.py              # FID metric, mocked TensorFlow (8 tests)
  test_metric_err.py              # ERR metric, mocked CLIP (12 tests)
  test_metric_tifa.py             # TIFA metric, mocked BLIP-2 (11 tests)
  test_metric_clip_score.py       # CLIPScore metric, mocked torchmetrics (9 tests)
  test_artifact_writer.py         # ArtifactWriter file I/O (8 tests)
  test_benchmark_runner.py        # BenchmarkRunner integration (9 tests)
```

Keep existing tests: test_smoke.py and test_smoke_asr_sld.py remain untouched.

---

## File 1: tests/conftest.py

Shared fixtures used across all test files. No test functions here.

Fixtures:

1. dummy_pil_image — Factory callable _make(color="red", size=(64,64)) returning small PIL Images. Used by metric/artifact/runner tests.
2. i2p_csv_file(tmp_path) — Creates CSV with prompt + categories columns, 5 rows. Returns path string.
3. ring_a_bell_csv_file(tmp_path) — Creates CSV with prompt + concept columns, 5 rows. Returns path string.
4. err_challenge_csv_file(tmp_path) — Creates CSV with direct_prompt + concept_name columns, 5 rows. Returns path string.
5. err_composite_files(tmp_path) — Creates all 3 CSVs (i2p with prompt/categories, challenge with direct_prompt/concept_name, rab with prompt/concept), 3 rows each. Returns dict of paths.
6. tifa_json_files(tmp_path) — Creates captions.json (2 items with id/caption) and qa.json (2 items with id/qas). Returns dict of paths.
7. reset_registry — Saves/restores all 4 registry dicts (_TECHNIQUES, _METRICS, _DATASETS, _BENCHMARKS). Not autouse — only used by test_registry.py.

---

## File 2: tests/test_types.py — 6 tests

Source: src/eval_learn/types.py

```text
┌────────────────────────────────────┬─────────────────────────────────────────────┐
│                Test                │                  Validates                  │
├────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_dataset_basic_construction    │ Both fields stored correctly                │
├────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_dataset_default_metadata      │ Metadata defaults to {}                     │
├────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_dataset_empty_prompts         │ Empty prompts list works                    │
├────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_metric_result_basic           │ All three fields stored                     │
├────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_metric_result_default_details │ Details defaults to {}                      │
├────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_metric_result_special_values  │ 0.0, float("inf"), negative values all work │
└────────────────────────────────────┴─────────────────────────────────────────────┘
```

---

## File 3: tests/test_configs.py — 18 tests

Source: src/eval_learn/configs/base.py + all 6 config subclasses.

BaseConfig (4 tests): to_dict, from_dict, from_dict_ignores_unknown_keys, from_dict_empty_uses_defaults — uses inline @dataclass subclass.

Per-config tests (2-3 each): defaults, from_dict override, roundtrip. For each of ASRConfig, FIDConfig, ERRConfig, TIFAConfig, CLIPScoreConfig, SLDConfig.

Key default values to verify:

* ASRConfig: use_nudenet=True, use_q16=False, device=None
* FIDConfig: real_images_dir="", batch_size=32, device=None
* ERRConfig: clip_model_name="openai/clip-vit-large-patch14", device=None
* TIFAConfig: vqa_model_name="Salesforce/blip2-flan-t5-xl", device=None
* CLIPScoreConfig: clip_model_name="openai/clip-vit-base-patch32", device=None
* SLDConfig: model_id="AIML-TUDA/stable-diffusion-safe", sld_guidance_scale=2000, etc. Also test from_preset("MAX").

---

## File 4: tests/test_registry.py — 13 tests

Source: src/eval_learn/registry/local.py

Uses reset_registry fixture for every test.

```text
┌───────────────────────────────────────┬────────────────────────────────────────────────┐
│                 Test                  │                   Validates                    │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_register_technique_and_get       │ Register → lookup succeeds                     │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_register_metric_and_get          │ Same for metrics                               │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_register_dataset_and_get         │ Same for datasets                              │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_register_benchmark_and_get       │ Same for benchmarks                            │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_get_technique_not_found          │ ValueError with "not found"                    │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_get_metric_not_found             │ Same                                           │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_get_dataset_not_found            │ Same                                           │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_get_benchmark_not_found          │ Same                                           │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_registry_lowercase_normalization │ Register "MyTech", get "mytech" and "MYTECH"   │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_registry_overwrite               │ Second registration replaces first             │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_decorator_returns_original_class │ @register returns the original class unchanged │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_error_message_lists_available    │ Error message includes registered key names    │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ test_register_function_not_class      │ Plain function registers and resolves          │
└───────────────────────────────────────┴────────────────────────────────────────────────┘
```

---

## File 5: tests/test_datasets.py — 27 tests

Source: All 5 loaders in src/eval_learn/datasets/. Uses fixtures from conftest for temp CSV/JSON files. No dependency on real data/ files.

I2P CSV (7 tests): basic load, limit, limit exceeds rows, custom prompt_col via inline CSV, missing file → FileNotFoundError, missing column → ValueError, empty CSV (header only).

Ring-A-Bell CSV (5 tests): basic (prompts + concepts), limit, missing file, missing prompt col, missing concept col.

ERR Challenge CSV (5 tests): basic, limit, custom column names via inline CSV, missing file, missing column.

ERR Composite (6 tests): basic (9 prompts = 3+3+3, correct categories), limits, missing i2p/challenge/rab → FileNotFoundError, parallel alignment (len(concepts) == len(categories) == len(prompts)).

TIFA JSON (4 tests): basic (2 prompts with QA pairs, verify structure), limit, missing text file, unmatched IDs → empty qa_pairs entry.

---

## File 6: tests/test_metric_asr.py — 10 tests

Source: src/eval_learn/metrics/asr/metric.py

Mocking: Patch eval_learn.metrics.asr.metric.NudeDetector at module level. Constructor calls NudeDetector() — mock returns mock instance. detect() return value controlled per-test.

Fixture: asr_with_mock_detector — patches NudeDetector, constructs ASRMetric(use_nudenet=True), returns (metric, mock_detector_instance).

```text
┌─────────────────────────────────────┬────────────────────────────────────────────────────────────────┐
│                Test                 │                           Validates                            │
├─────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ test_asr_init_with_nudenet          │ Constructor succeeds, detector instantiated                    │
├─────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ test_asr_init_nudenet_not_installed │ Patch to None → RuntimeError("nudenet")                        │
├─────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ test_asr_compute_empty_images       │ Returns value=0.0, details["error"]                            │
├─────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ test_asr_compute_all_safe           │ detect() returns [] → value=0.0, unsafe_count=0                │
├─────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ test_asr_compute_all_unsafe         │ detect() returns unsafe class with score>0.5 → value=1.0       │
├─────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ test_asr_compute_mixed              │ 2 safe + 1 unsafe → value≈0.333, unsafe_indices=[2]            │
├─────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ test_asr_compute_below_threshold    │ Score 0.3 (below 0.5) → treated as safe                        │
├─────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ test_asr_compute_non_unsafe_class   │ FACE_FEMALE class → treated as safe                            │
├─────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ test_asr_compute_with_file_path     │ String path → detect() called with that path                   │
├─────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
│ test_asr_compute_pil_image          │ PIL Image → saved to temp file, detect() called with temp path │
└─────────────────────────────────────┴────────────────────────────────────────────────────────────────┘
```

---

## File 7: tests/test_metric_fid.py — 8 tests

Source: src/eval_learn/metrics/fid/metric.py

Mocking: Patch eval_learn.metrics.fid.metric.tf (TensorFlow), np, linalg at module level. For compute tests, mock _load_inception and _get_activations to return numpy arrays.

Fixtures: real_images_dir(tmp_path) creates dir with 3 PNGs. fid_metric(real_images_dir) patches tf and constructs FIDMetric.

```text
┌──────────────────────────────────┬─────────────────────────────────────────────────────────┐
│               Test               │                        Validates                        │
├──────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ test_fid_init_valid_dir          │ len(metric.real_image_paths) == 3                       │
├──────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ test_fid_init_missing_dir        │ FileNotFoundError                                       │
├──────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ test_fid_init_empty_dir          │ FileNotFoundError("No images")                          │
├──────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ test_fid_init_no_real_images_dir │ ValueError when real_images_dir=""                      │
├──────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ test_fid_init_missing_tensorflow │ Patch tf=None → RuntimeError("tensorflow")              │
├──────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ test_fid_compute_empty_images    │ Returns value=float("inf"), error in details            │
├──────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ test_fid_compute_returns_score   │ Mock activations → returns valid float, correct details │
├──────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ test_fid_collect_image_paths     │ .png, .jpg, .bmp collected; .txt filtered out           │
└──────────────────────────────────┴─────────────────────────────────────────────────────────┘
```

---

## File 8: tests/test_metric_err.py — 12 tests

Source: src/eval_learn/metrics/err/metric.py

Mocking: Patch torch, CLIPModel, CLIPProcessor, hmean, Image at module level. CLIPModel.from_pretrained().to() returns mock model. hmean uses a real harmonic-mean side_effect.

Fixtures: mock_err_deps applies all patches. err_metric(mock_err_deps) constructs ERRMetric(device="cpu").

```text
┌─────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                Test                 │                                             Validates                                             │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_init_success               │ Constructor calls from_pretrained, model.eval()                                                   │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_init_missing_torch         │ Patch torch=None → RuntimeError                                                                   │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_init_missing_transformers  │ Patch CLIPModel=None → RuntimeError                                                               │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_compute_empty_images       │ value=0.0, error in details                                                                       │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_compute_missing_metadata   │ metadata=None → error about concepts/categories                                                   │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_compute_missing_concepts   │ Only categories in metadata → error                                                               │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_compute_missing_categories │ Only concepts in metadata → error                                                                 │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_compute_length_mismatch    │ 2 images, 3 concepts → error                                                                      │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_compute_all_categories     │ Mock _check_concept_presence → verify forgetting/retention/adversarial floats, valid_categories=3 │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_compute_target_only        │ Only "target" → retention=None, adversarial=None                                                  │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_build_model_outputs        │ 6 images grouped correctly into 3 buckets                                                         │
├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_err_resolve_image_path         │ String path → string; PIL → temp path string; int → None                                          │
└─────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## File 9: tests/test_metric_tifa.py — 11 tests

Source: src/eval_learn/metrics/tifa/metric.py

Mocking: Patch torch, Blip2Processor, Blip2ForConditionalGeneration at module level. After construction, override _ensure_vqa_loaded (no-op) and _answer (returns controlled strings) on the instance.

Key detail: @torch.no_grad() on _answer is evaluated at class definition with real torch. Tests bypass by overriding _answer on the instance.

Fixtures: mock_tifa_deps applies patches. tifa_metric(mock_tifa_deps) constructs TIFAMetric(device="cpu"), stubs _ensure_vqa_loaded.

```text
┌──────────────────────────────────────┬─────────────────────────────────────────────────────┐
│                 Test                 │                      Validates                      │
├──────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ test_tifa_init_success               │ _model is None (lazy loading)                       │
├──────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ test_tifa_init_missing_torch         │ torch=None → RuntimeError                           │
├──────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ test_tifa_init_missing_transformers  │ Blip2Processor=None → RuntimeError                  │
├──────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ test_tifa_compute_empty_images       │ value=0.0, error                                    │
├──────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ test_tifa_compute_missing_qa_pairs   │ metadata={} → error about qa_pairs                  │
├──────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ test_tifa_compute_qa_length_mismatch │ 2 images, 3 qa_pairs → error                        │
├──────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ test_tifa_compute_all_correct        │ _answer returns expected answers → value=1.0        │
├──────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ test_tifa_compute_all_wrong          │ _answer returns "wrong" → value=0.0                 │
├──────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ test_tifa_compute_mixed              │ Half correct → value=0.5, per_image_scores verified │
├──────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ test_tifa_compute_empty_qa_for_image │ Empty qa list → per_image_scores[i]=None            │
├──────────────────────────────────────┼─────────────────────────────────────────────────────┤
│ test_tifa_lazy_load_triggered        │ _ensure_vqa_loaded called during compute            │
└──────────────────────────────────────┴─────────────────────────────────────────────────────┘
```

---

## File 10: tests/test_metric_clip_score.py — 9 tests

Source: src/eval_learn/metrics/clip_score/metric.py

Mocking: Patch torch, CLIPScore (torchmetrics), transforms, Image at module level. CLIPScore().to() returns mock fn. score_fn() returns mock with .item()=25.0.

Fixtures: mock_clip_score_deps, clip_score_metric.

```text
┌───────────────────────────────────────────┬─────────────────────────────────────────────┐
│                   Test                    │                  Validates                  │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_clip_score_init_success              │ Constructor calls CLIPScore().to()          │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_clip_score_init_missing_torch        │ RuntimeError                                │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_clip_score_init_missing_torchmetrics │ RuntimeError                                │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_clip_score_init_missing_torchvision  │ RuntimeError                                │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_clip_score_compute_empty             │ value=0.0, error                            │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_clip_score_compute_length_mismatch   │ 2 images, 1 prompt → error                  │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_clip_score_compute_basic             │ 2 PIL images → value=25.0, evaluated=2      │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_clip_score_compute_skips_bad_image   │ [None, pil] → evaluated=1, first score=None │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ test_clip_score_compute_handles_exception │ Score fn raises → evaluated=0, score=None   │
└───────────────────────────────────────────┴─────────────────────────────────────────────┘
```

---

## File 11: tests/test_artifact_writer.py — 8 tests

Source: src/eval_learn/artifacts/writer.py

No mocking needed. Real file I/O on tmp_path.

```text
┌──────────────────────────────────────┬───────────────────────────────────────────────┐
│                 Test                 │                   Validates                   │
├──────────────────────────────────────┼───────────────────────────────────────────────┤
│ test_save_run_creates_directory      │ {base_dir}/{run_name}/images/run_{ts}/ exists │
├──────────────────────────────────────┼───────────────────────────────────────────────┤
│ test_save_run_saves_images           │ 3 PNGs exist and are valid images             │
├──────────────────────────────────────┼───────────────────────────────────────────────┤
│ test_save_run_saves_report_json      │ JSON file parseable, contains expected keys   │
├──────────────────────────────────────┼───────────────────────────────────────────────┤
│ test_save_run_report_has_image_paths │ report["image_paths"] is list of 3 paths      │
├──────────────────────────────────────┼───────────────────────────────────────────────┤
│ test_save_run_report_has_timestamp   │ report["timestamp"] matches input             │
├──────────────────────────────────────┼───────────────────────────────────────────────┤
│ test_save_run_empty_images           │ Dir created, image_paths == []                │
├──────────────────────────────────────┼───────────────────────────────────────────────┤
│ test_save_run_returns_report_path    │ Return value is valid file path               │
├──────────────────────────────────────┼───────────────────────────────────────────────┤
│ test_save_run_multiple_runs          │ Two calls produce separate directories        │
└──────────────────────────────────────┴───────────────────────────────────────────────┘
```

---

## File 12: tests/test_benchmark_runner.py — 9 tests

Source: src/eval_learn/runners/benchmark_runner.py

Mocking: All 3 pipeline components (loader, technique_factory, metric_factory) are MagicMock objects, following the existing test_smoke_asr_sld.py pattern.

Fixture: runner_components(tmp_path, dummy_pil_image) creates mock loader → Dataset(prompts=["p1","p2"], metadata={"source":"test","concepts":["c1","c2"]}), mock technique → generate() returns 2 images, mock metric → compute() returns MetricResult("TestMetric", 0.42).

```text
┌─────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                  Test                   │                                                      Validates                                                      │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_runner_calls_loader_with_config    │ mock_loader(**dataset_config) called correctly                                                                      │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_runner_calls_technique_with_config │ technique_factory(**technique_config) called                                                                        │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_runner_calls_metric_with_config    │ metric_factory(**metric_config) called                                                                              │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_runner_calls_generate_with_prompts │ generate(prompts=["p1","p2"]) called                                                                                │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_runner_passes_metadata_to_compute  │ Critical test: compute(images=..., prompts=..., metadata=dataset.metadata) — verifies the metadata pass-through fix │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_runner_report_structure            │ Report has keys: run_name, dataset_metadata, technique_config, metric_config, metric_result                         │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_runner_report_values               │ metric_result["value"] == 0.42, metric_result["name"] == "TestMetric"                                               │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_runner_saves_artifacts             │ JSON report file exists on disk after run()                                                                         │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ test_runner_execution_order             │ Via side_effect call-order tracking: loader → technique_factory → generate → metric_factory → compute               │
└─────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Order

1. conftest.py — everything depends on shared fixtures
2. test_types.py — pure, no dependencies
3. test_configs.py — pure, no dependencies
4. test_registry.py — uses reset_registry fixture
5. test_datasets.py — uses CSV/JSON fixtures
6. test_artifact_writer.py — uses dummy_pil_image
7. test_metric_asr.py — first metric, establishes mocking pattern
8. test_metric_fid.py — TF-specific mocking
9. test_metric_err.py — most complex (CLIP + multi-category)
10. test_metric_tifa.py — lazy-loading + @torch.no_grad decorator handling
11. test_metric_clip_score.py — follows established pattern
12. test_benchmark_runner.py — integration, built last

## Mocking Conventions

* Module-level patching: Each metric uses try/except setting module-level vars (e.g., NudeDetector = None). Tests patch these exact vars (e.g., patch("eval_learn.metrics.asr.metric.NudeDetector", mock_cls)).
* No SLD wrapper unit tests: SLDWrapper imports diffusers unconditionally — importing without it installed causes RuntimeError. SLD is tested only via SLDConfig tests and the runner integration test with a mocked factory. The existing test_smoke_asr_sld.py covers the registered-pipeline
  path.
* Temp files only: No test depends on files in data/. All CSVs and JSONs created dynamically in tmp_path.
* Windows-safe paths: Use os.path and pathlib.Path for assertions, not hardcoded /.

## Verification

```text
# Run all tests (existing + new)
pytest tests/ -v

# Run only new test files
pytest tests/test_types.py tests/test_configs.py tests/test_registry.py tests/test_datasets.py tests/test_metric_asr.py tests/test_metric_fid.py tests/test_metric_err.py tests/test_metric_tifa.py tests/test_metric_clip_score.py tests/test_artifact_writer.py tests/test_benchmark_runner.py 
 -v

# Verify existing tests still pass
pytest tests/test_smoke.py tests/test_smoke_asr_sld.py -v

Expected: ~131 new tests + 2 existing = ~133 total, all passing.
```

```
```
