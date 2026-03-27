"""Verify HuggingFace datasets load with correct columns."""

import pytest
from dotenv import load_dotenv
import os


class TestHFDatasets:
    """Verify HuggingFace datasets load with correct columns."""

    @pytest.fixture(scope="session", autouse=True)
    def setup_env(self):
        """Load environment variables."""
        load_dotenv(override=True)

    def test_i2p_loads(self):
        """Verify i2p dataset loads from HF with expected columns."""
        from datasets import load_dataset

        token = os.getenv("HF_TOKEN")
        ds = load_dataset(
            "AIML-TUDA/i2p",
            data_files="i2p_benchmark.csv",
            streaming=True,
            token=token,
        )
        first_row = next(iter(ds))
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
        first_row = next(iter(ds))
        assert "image" in first_row
        assert "captions" in first_row

    def test_err_challenge_loads(self):
        """Verify ERR challenge dataset loads from HF."""
        from datasets import load_dataset

        token = os.getenv("HF_TOKEN")
        ds = load_dataset(
            "Unlearningltd/datasets",
            data_files="ERR/raw_csv_data/challenge_dataset.csv",
            streaming=True,
            token=token,
        )
        first_row = next(iter(ds))
        assert "direct_prompt" in first_row
        assert "concept_name" in first_row

    def test_ring_a_bell_loads(self):
        """Verify Ring-A-Bell dataset loads from HF."""
        from datasets import load_dataset

        token = os.getenv("HF_TOKEN")
        ds = load_dataset(
            "Unlearningltd/datasets",
            data_files="ring_a_bell/ring_a_bell_dataset.csv",
            streaming=True,
            token=token,
        )
        first_row = next(iter(ds))
        assert "prompt" in first_row
        assert "concept" in first_row

    def test_tifa_text_loads(self):
        """Verify TIFA text file loads."""
        from datasets import load_dataset

        token = os.getenv("HF_TOKEN")
        text_ds = load_dataset(
            "Unlearningltd/datasets",
            data_files="tifa/tifa_text_inputs.json",
            streaming=True,
            token=token,
        )
        text_row = next(iter(text_ds))
        assert "caption" in text_row

    def test_tifa_qa_loads(self):
        """Verify TIFA QA file loads."""
        from datasets import load_dataset

        token = os.getenv("HF_TOKEN")
        qa_ds = load_dataset(
            "Unlearningltd/datasets",
            data_files="tifa/tifa_question_answers.json",
            streaming=True,
            token=token,
        )
        qa_row = next(iter(qa_ds))
        assert "qas" in qa_row
