import pytest
import os
import json
import csv
from PIL import Image


@pytest.fixture
def dummy_pil_image():
    """Factory fixture that creates small PIL Images for testing."""
    def _make(color="red", size=(64, 64)):
        return Image.new("RGB", size, color=color)
    return _make


@pytest.fixture
def i2p_csv_file(tmp_path):
    """Creates a temporary I2P-style CSV with a 'prompt' column and 5 rows."""
    path = tmp_path / "i2p_test.csv"
    rows = [{"prompt": f"test prompt {i}"} for i in range(5)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt"])
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


@pytest.fixture
def ring_a_bell_csv_file(tmp_path):
    """Creates a temporary Ring-A-Bell-style CSV with 'prompt' and 'concept' columns."""
    path = tmp_path / "rab_test.csv"
    rows = [{"prompt": f"adversarial prompt {i}", "concept": f"concept_{i}"} for i in range(5)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "concept"])
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


@pytest.fixture
def err_challenge_csv_file(tmp_path):
    """Creates a temporary ERR challenge CSV with 'direct_prompt' and 'concept_name' columns."""
    path = tmp_path / "challenge_test.csv"
    rows = [{"direct_prompt": f"retain prompt {i}", "concept_name": f"concept_{i}"} for i in range(5)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["direct_prompt", "concept_name"])
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


@pytest.fixture
def err_composite_files(tmp_path):
    """Creates all 3 CSV files needed by err_composite loader."""
    # I2P file (needs "prompt" and "categories" columns)
    i2p_path = tmp_path / "i2p_composite.csv"
    i2p_rows = [{"prompt": f"target prompt {i}", "categories": f"cat_{i}"} for i in range(3)]
    with open(i2p_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "categories"])
        w.writeheader()
        w.writerows(i2p_rows)

    # Challenge file
    ch_path = tmp_path / "challenge_composite.csv"
    ch_rows = [{"direct_prompt": f"retain prompt {i}", "concept_name": f"concept_{i}"} for i in range(3)]
    with open(ch_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["direct_prompt", "concept_name"])
        w.writeheader()
        w.writerows(ch_rows)

    # RAB file
    rab_path = tmp_path / "rab_composite.csv"
    rab_rows = [{"prompt": f"adversarial prompt {i}", "concept": f"concept_{i}"} for i in range(3)]
    with open(rab_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "concept"])
        w.writeheader()
        w.writerows(rab_rows)

    return {"i2p_path": str(i2p_path), "challenge_path": str(ch_path), "rab_path": str(rab_path)}


@pytest.fixture
def tifa_json_files(tmp_path):
    """Creates TIFA-style JSON files for captions and QA pairs."""
    text_path = tmp_path / "captions.json"
    qa_path = tmp_path / "qa.json"

    text_data = [
        {"id": 1, "caption": "a red dog running"},
        {"id": 2, "caption": "a blue cat sitting"},
    ]
    qa_data = [
        {"id": 1, "qas": [
            {"question": "Is there a dog?", "answer": "yes"},
            {"question": "What color is the dog?", "answer": "red"},
        ]},
        {"id": 2, "qas": [
            {"question": "Is there a cat?", "answer": "yes"},
        ]},
    ]

    with open(text_path, "w") as f:
        json.dump(text_data, f)
    with open(qa_path, "w") as f:
        json.dump(qa_data, f)

    return {"text_path": str(text_path), "qa_path": str(qa_path)}


@pytest.fixture
def reset_registry():
    """Saves and restores all 4 registry dicts to avoid cross-test pollution."""
    from eval_learn.registry.local import _TECHNIQUES, _METRICS, _DATASETS, _BENCHMARKS
    saved = (dict(_TECHNIQUES), dict(_METRICS), dict(_DATASETS), dict(_BENCHMARKS))
    yield
    _TECHNIQUES.clear()
    _TECHNIQUES.update(saved[0])
    _METRICS.clear()
    _METRICS.update(saved[1])
    _DATASETS.clear()
    _DATASETS.update(saved[2])
    _BENCHMARKS.clear()
    _BENCHMARKS.update(saved[3])
