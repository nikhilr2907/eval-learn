import csv
import json
import pytest
from eval_learn.datasets.i2p_csv import load_i2p_csv
from eval_learn.datasets.ring_a_bell_csv import load_ring_a_bell_csv
from eval_learn.datasets.err_challenge_csv import load_err_challenge_csv
from eval_learn.datasets.err_composite import load_err_composite
from eval_learn.datasets.tifa_json import load_tifa_json


# ==================== I2P CSV ====================

class TestI2PCSV:
    def test_basic_load(self, i2p_csv_file):
        ds = load_i2p_csv(path=i2p_csv_file)
        assert len(ds.prompts) == 5
        assert ds.metadata["source"] == "i2p_csv"
        assert ds.metadata["path"] == i2p_csv_file
        assert ds.metadata["total_loaded"] == 5

    def test_with_limit(self, i2p_csv_file):
        ds = load_i2p_csv(path=i2p_csv_file, limit=2)
        assert len(ds.prompts) == 2

    def test_limit_exceeds_rows(self, i2p_csv_file):
        ds = load_i2p_csv(path=i2p_csv_file, limit=100)
        assert len(ds.prompts) == 5

    def test_custom_prompt_col(self, tmp_path):
        path = tmp_path / "custom.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["text"])
            w.writeheader()
            w.writerow({"text": "hello"})
        ds = load_i2p_csv(path=str(path), prompt_col="text")
        assert ds.prompts == ["hello"]

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_i2p_csv(path="/nonexistent/path.csv")

    def test_missing_column(self, tmp_path):
        path = tmp_path / "bad.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["text"])
            w.writeheader()
            w.writerow({"text": "x"})
        with pytest.raises(ValueError, match="prompt"):
            load_i2p_csv(path=str(path))

    def test_empty_csv(self, tmp_path):
        path = tmp_path / "empty.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["prompt"])
            w.writeheader()
        ds = load_i2p_csv(path=str(path))
        assert len(ds.prompts) == 0


# ==================== Ring-A-Bell CSV ====================

class TestRingABellCSV:
    def test_basic_load(self, ring_a_bell_csv_file):
        ds = load_ring_a_bell_csv(path=ring_a_bell_csv_file)
        assert len(ds.prompts) == 5
        assert len(ds.metadata["concepts"]) == 5
        assert ds.metadata["source"] == "ring_a_bell_csv"

    def test_with_limit(self, ring_a_bell_csv_file):
        ds = load_ring_a_bell_csv(path=ring_a_bell_csv_file, limit=2)
        assert len(ds.prompts) == 2
        assert len(ds.metadata["concepts"]) == 2

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_ring_a_bell_csv(path="/nonexistent.csv")

    def test_missing_prompt_col(self, tmp_path):
        path = tmp_path / "bad.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["concept"])
            w.writeheader()
            w.writerow({"concept": "x"})
        with pytest.raises(ValueError, match="prompt"):
            load_ring_a_bell_csv(path=str(path))

    def test_missing_concept_col(self, tmp_path):
        path = tmp_path / "bad.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["prompt"])
            w.writeheader()
            w.writerow({"prompt": "x"})
        with pytest.raises(ValueError, match="concept"):
            load_ring_a_bell_csv(path=str(path))


# ==================== ERR Challenge CSV ====================

class TestERRChallengeCSV:
    def test_basic_load(self, err_challenge_csv_file):
        ds = load_err_challenge_csv(path=err_challenge_csv_file)
        assert len(ds.prompts) == 5
        assert ds.metadata["source"] == "err_challenge_csv"
        assert len(ds.metadata["concepts"]) == 5

    def test_with_limit(self, err_challenge_csv_file):
        ds = load_err_challenge_csv(path=err_challenge_csv_file, limit=3)
        assert len(ds.prompts) == 3

    def test_custom_cols(self, tmp_path):
        path = tmp_path / "custom.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["my_prompt", "my_concept"])
            w.writeheader()
            w.writerow({"my_prompt": "p1", "my_concept": "c1"})
        ds = load_err_challenge_csv(
            path=str(path), prompt_col="my_prompt", concept_col="my_concept"
        )
        assert ds.prompts == ["p1"]
        assert ds.metadata["concepts"] == ["c1"]

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_err_challenge_csv(path="/nonexistent.csv")

    def test_missing_column(self, tmp_path):
        path = tmp_path / "bad.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["wrong_col"])
            w.writeheader()
            w.writerow({"wrong_col": "x"})
        with pytest.raises(ValueError, match="direct_prompt"):
            load_err_challenge_csv(path=str(path))


# ==================== ERR Composite ====================

class TestERRComposite:
    def test_basic_load(self, err_composite_files):
        ds = load_err_composite(**err_composite_files)
        assert len(ds.prompts) == 9  # 3+3+3
        assert len(ds.metadata["categories"]) == 9
        assert len(ds.metadata["concepts"]) == 9
        assert ds.metadata["counts"]["target"] == 3
        assert ds.metadata["counts"]["retain"] == 3
        assert ds.metadata["counts"]["adversarial"] == 3
        assert "target" in ds.metadata["categories"]
        assert "retain" in ds.metadata["categories"]
        assert "adversarial" in ds.metadata["categories"]

    def test_with_limits(self, err_composite_files):
        ds = load_err_composite(
            **err_composite_files,
            target_limit=1,
            retain_limit=2,
            adversarial_limit=1,
        )
        assert len(ds.prompts) == 4
        assert ds.metadata["counts"]["target"] == 1
        assert ds.metadata["counts"]["retain"] == 2
        assert ds.metadata["counts"]["adversarial"] == 1

    def test_missing_i2p(self, err_composite_files):
        err_composite_files["i2p_path"] = "/nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_err_composite(**err_composite_files)

    def test_missing_challenge(self, err_composite_files):
        err_composite_files["challenge_path"] = "/nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_err_composite(**err_composite_files)

    def test_missing_rab(self, err_composite_files):
        err_composite_files["rab_path"] = "/nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_err_composite(**err_composite_files)

    def test_categories_alignment(self, err_composite_files):
        ds = load_err_composite(**err_composite_files)
        assert len(ds.metadata["categories"]) == len(ds.prompts)
        assert len(ds.metadata["concepts"]) == len(ds.prompts)


# ==================== TIFA JSON ====================

class TestTIFAJSON:
    def test_basic_load(self, tifa_json_files):
        ds = load_tifa_json(**tifa_json_files)
        assert len(ds.prompts) == 2
        assert ds.prompts[0] == "a red dog running"
        assert ds.prompts[1] == "a blue cat sitting"
        assert len(ds.metadata["qa_pairs"]) == 2
        assert len(ds.metadata["qa_pairs"][0]) == 2  # 2 QA pairs for first item
        assert len(ds.metadata["qa_pairs"][1]) == 1  # 1 QA pair for second item
        assert ds.metadata["qa_pairs"][0][0]["question"] == "Is there a dog?"

    def test_with_limit(self, tifa_json_files):
        ds = load_tifa_json(**tifa_json_files, limit=1)
        assert len(ds.prompts) == 1

    def test_missing_text_file(self, tifa_json_files):
        with pytest.raises(FileNotFoundError):
            load_tifa_json(text_path="/nonexistent.json", qa_path=tifa_json_files["qa_path"])

    def test_missing_qa_file(self, tifa_json_files):
        with pytest.raises(FileNotFoundError):
            load_tifa_json(text_path=tifa_json_files["text_path"], qa_path="/nonexistent.json")

    def test_unmatched_ids(self, tmp_path):
        text_path = tmp_path / "text.json"
        qa_path = tmp_path / "qa.json"
        with open(text_path, "w") as f:
            json.dump([{"id": 999, "caption": "orphan"}], f)
        with open(qa_path, "w") as f:
            json.dump([{"id": 1, "qas": [{"question": "q?", "answer": "a"}]}], f)
        ds = load_tifa_json(text_path=str(text_path), qa_path=str(qa_path))
        assert len(ds.prompts) == 1
        assert ds.metadata["qa_pairs"][0] == []  # unmatched ID -> empty list
