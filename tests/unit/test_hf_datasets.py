"""Verify HuggingFace datasets load with correct schemas."""

import pytest
from dotenv import load_dotenv
import os


def get_first_row(ds):
    """Extract first row from dataset, handling DatasetDict."""
    # If it's a DatasetDict (has splits), get the first available split
    if hasattr(ds, "keys"):
        split_name = list(ds.keys())[0]
        ds = ds[split_name]
    return next(iter(ds))


class TestHFDatasets:
    """Verify HuggingFace datasets load with correct schemas and columns."""

    @pytest.fixture(scope="session", autouse=True)
    def setup_env(self):
        """Load environment variables."""
        load_dotenv(override=True)

    def test_i2p_loads(self):
        """Verify i2p dataset loads from HF with correct schema."""
        from datasets import load_dataset, Features, Value

        token = os.getenv("HF_TOKEN")
        features = Features({
            'prompt': Value('string'),
            'categories': Value('string'),
        })
        ds = load_dataset(
            "AIML-TUDA/i2p",
            data_files="i2p_benchmark.csv",
            features=features,
            streaming=True,
            token=token,
        )
        print(f"[DEBUG] i2p dataset type: {type(ds)}")

        # Validate it's an IterableDataset, not DatasetDict
        from datasets import IterableDataset, DatasetDict
        assert not isinstance(ds, DatasetDict), f"Expected IterableDataset, got DatasetDict"
        assert isinstance(ds, IterableDataset), f"Expected IterableDataset, got {type(ds)}"

        first_row = get_first_row(ds)
        assert "prompt" in first_row
        assert "categories" in first_row

    def test_i2p_with_split(self):
        """Verify i2p dataset loads with explicit split (no data_files)."""
        from datasets import load_dataset, IterableDataset, DatasetDict

        token = os.getenv("HF_TOKEN")
        ds = load_dataset(
            "AIML-TUDA/i2p",
            split="train",  # ← Explicit split, no data_files
            streaming=True,
            token=token,
        )
        print(f"[DEBUG] i2p with split type: {type(ds)}")

        # Must be IterableDataset, not DatasetDict
        assert not isinstance(ds, DatasetDict), f"Expected IterableDataset, got DatasetDict"
        assert isinstance(ds, IterableDataset), f"Expected IterableDataset, got {type(ds)}"

        first_row = get_first_row(ds)
        assert "prompt" in first_row
        assert "categories" in first_row

    def test_coco_loads(self):
        """Verify COCO dataset loads with expected columns."""
        from datasets import load_dataset

        token = os.getenv("HF_TOKEN")
        ds = load_dataset(
            "phiyodr/coco2017",
            split="train",
            streaming=True,
            token=token,
        )
        first_row = get_first_row(ds)
        assert "captions" in first_row

    def test_err_challenge_loads(self):
        """Verify ERR challenge dataset loads from HF with correct schema."""
        from datasets import load_dataset, Features, Value, IterableDataset, DatasetDict

        token = os.getenv("HF_TOKEN")
        features = Features({
            'concept_type': Value('string'),
            'concept_name': Value('string'),
            'direct_prompt': Value('string'),
            'indirect_prompt': Value('string'),
            'adversarial_prompt': Value('string'),
        })
        ds = load_dataset(
            "Unlearningltd/datasets",
            data_files="ERR/raw_csv_data/challenge_dataset.csv",
            features=features,
            streaming=True,
            token=token,
        )
        print(f"[DEBUG] challenge dataset type: {type(ds)}")
        assert not isinstance(ds, DatasetDict), f"Expected IterableDataset, got DatasetDict"
        assert isinstance(ds, IterableDataset), f"Expected IterableDataset, got {type(ds)}"

        first_row = get_first_row(ds)
        assert "direct_prompt" in first_row
        assert "concept_name" in first_row

    def test_ring_a_bell_loads(self):
        """Verify Ring-A-Bell dataset loads from HF with correct schema."""
        from datasets import load_dataset, Features, Value, IterableDataset, DatasetDict

        token = os.getenv("HF_TOKEN")
        features = Features({
            'prompt': Value('string'),
            'concept': Value('string'),
        })
        ds = load_dataset(
            "Unlearningltd/datasets",
            data_files="ring_a_bell/ring_a_bell_dataset.csv",
            features=features,
            streaming=True,
            token=token,
        )
        print(f"[DEBUG] ring_a_bell dataset type: {type(ds)}")
        assert not isinstance(ds, DatasetDict), f"Expected IterableDataset, got DatasetDict"
        assert isinstance(ds, IterableDataset), f"Expected IterableDataset, got {type(ds)}"

        first_row = get_first_row(ds)
        assert "prompt" in first_row
        assert "concept" in first_row

    def test_tifa_text_loads(self):
        """Verify TIFA text file loads with correct schema."""
        from datasets import load_dataset, DatasetDict
        from eval_learn.datasets.hf_stream import load_hf_config

        token = os.getenv("HF_TOKEN")
        cfg = load_hf_config("tifa")
        text_ds = load_dataset(
            cfg["repo_id"],
            data_files=cfg.get("text_file"),
            token=token,
        )
        print(f"[DEBUG] tifa text dataset type: {type(text_ds)}")
        if isinstance(text_ds, DatasetDict):
            print(f"[WARNING] TIFA text is DatasetDict with splits: {list(text_ds.keys())}")
        text_row = get_first_row(text_ds)
        assert "caption" in text_row

    def test_tifa_qa_loads(self):
        """Verify TIFA QA file loads with correct schema."""
        from datasets import load_dataset, DatasetDict
        from eval_learn.datasets.hf_stream import load_hf_config

        token = os.getenv("HF_TOKEN")
        cfg = load_hf_config("tifa")
        qa_ds = load_dataset(
            cfg["repo_id"],
            data_files=cfg.get("qa_file"),
            token=token,
        )
        print(f"[DEBUG] tifa qa dataset type: {type(qa_ds)}")
        if isinstance(qa_ds, DatasetDict):
            print(f"[WARNING] TIFA QA is DatasetDict with splits: {list(qa_ds.keys())}")
        qa_row = get_first_row(qa_ds)
        assert "qas" in qa_row
        # Verify qas is a list of dicts with question/answer
        assert isinstance(qa_row["qas"], list)
        assert len(qa_row["qas"]) > 0
        assert "question" in qa_row["qas"][0]
        assert "answer" in qa_row["qas"][0]
