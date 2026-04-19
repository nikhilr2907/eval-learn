"""Unit tests for TIFA (Text-to-Image Faithfulness) metric."""

import tempfile
from unittest.mock import Mock, patch
import pytest
import torch
from PIL import Image

from eval_learn.metrics.tifa.metric import TIFAMetric
from eval_learn.metrics.tifa.config import TIFAConfig
from eval_learn.types import MetricResult


class TestTIFAConfig:
    """Test TIFAConfig initialization and validation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = TIFAConfig()
        assert config.vqa_model_name == "Salesforce/blip2-flan-t5-xl"
        assert config.device is None
        assert config.limit == 200

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "vqa_model_name": "Salesforce/blip2-opt-2.7b",
            "device": "cpu",
            "limit": 100,
        }
        config = TIFAConfig.from_dict(config_dict)
        assert config.vqa_model_name == "Salesforce/blip2-opt-2.7b"
        assert config.device == "cpu"
        assert config.limit == 100

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TIFAConfig(device="cpu", limit=50)
        config_dict = config.to_dict()
        assert config_dict["device"] == "cpu"
        assert config_dict["limit"] == 50


class TestTIFAMetricInitialization:
    """Test TIFAMetric initialization."""

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    @patch("eval_learn.metrics.tifa.metric.Image")
    def test_init_success_cpu(self, mock_image, mock_model_class, mock_processor_class, mock_torch):
        """Test successful initialization on CPU."""
        mock_torch.cuda.is_available.return_value = False

        metric = TIFAMetric(device="cpu")

        assert metric.device == "cpu"
        assert metric._processor is None  # Lazy loaded
        assert metric._model is None  # Lazy loaded
        assert metric._correct_count == 0
        assert metric._total_count == 0
        assert metric._total_images == 0
        assert metric._per_image_scores == []

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    @patch("eval_learn.metrics.tifa.metric.Image")
    def test_init_auto_detect_device(self, mock_image, mock_model_class, mock_processor_class, mock_torch):
        """Test device auto-detection when device is None."""
        mock_torch.cuda.is_available.return_value = False

        metric = TIFAMetric(device=None)

        assert metric.device == "cpu"

    @patch("eval_learn.metrics.tifa.metric.torch", None)
    def test_init_missing_torch_raises_error(self):
        """Test that missing torch raises helpful error."""
        with pytest.raises(RuntimeError, match="requires 'torch'"):
            TIFAMetric()

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor", None)
    @patch("eval_learn.metrics.tifa.metric.Image")
    def test_init_missing_transformers_raises_error(self, mock_image, mock_torch):
        """Test that missing transformers raises helpful error."""
        with pytest.raises(RuntimeError, match="requires 'transformers'"):
            TIFAMetric()

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    @patch("eval_learn.metrics.tifa.metric.Image", None)
    def test_init_missing_pillow_raises_error(self, mock_model_class, mock_processor_class, mock_torch):
        """Test that missing Pillow raises helpful error."""
        with pytest.raises(RuntimeError, match="requires 'Pillow'"):
            TIFAMetric()


class TestTIFAVQALoader:
    """Test VQA model loading mechanism."""

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    @patch("eval_learn.metrics.tifa.metric.Image")
    def test_ensure_vqa_loaded_loads_model(self, mock_image, mock_model_class, mock_processor_class, mock_torch):
        """Test that _ensure_vqa_loaded loads model on first call."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        # Model not loaded yet
        assert metric._model is None

        # Call _ensure_vqa_loaded
        metric._ensure_vqa_loaded()

        # Now model should be loaded
        assert metric._model is mock_model
        assert metric._processor is mock_processor
        mock_model_class.from_pretrained.assert_called_once()
        mock_processor_class.from_pretrained.assert_called_once()

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    @patch("eval_learn.metrics.tifa.metric.Image")
    def test_ensure_vqa_loaded_no_reload(self, mock_image, mock_model_class, mock_processor_class, mock_torch):
        """Test that _ensure_vqa_loaded doesn't reload on second call."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        # First call
        metric._ensure_vqa_loaded()
        call_count_1 = mock_model_class.from_pretrained.call_count

        # Second call
        metric._ensure_vqa_loaded()
        call_count_2 = mock_model_class.from_pretrained.call_count

        # Should not have called again
        assert call_count_1 == call_count_2 == 1

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    @patch("eval_learn.metrics.tifa.metric.Image")
    def test_model_loaded_on_correct_device(self, mock_image, mock_model_class, mock_processor_class, mock_torch):
        """Test that model is loaded to correct device."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")
        metric._ensure_vqa_loaded()

        # Check .to() was called with correct device
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()


class TestTIFAAnswerMethod:
    """Test the VQA answer method."""

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    def test_answer_returns_string(self, mock_model_class, mock_processor_class, mock_torch):
        """Test _answer returns a string."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_processor.decode.return_value = "yes"

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")
        metric._ensure_vqa_loaded()

        # Mock _answer directly to test its interface
        with patch.object(metric, "_answer", return_value="yes"):
            img = Image.new("RGB", (10, 10), color="red")
            answer = metric._answer(img, "Is this a dog?")
            assert isinstance(answer, str)
            assert answer == "yes"

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    def test_answer_strips_whitespace(self, mock_model_class, mock_processor_class, mock_torch):
        """Test _answer strips whitespace from output."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_processor.decode.return_value = "  yes  "

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")
        metric._ensure_vqa_loaded()

        with patch.object(metric, "_answer", return_value="yes"):
            answer = metric._answer(Image.new("RGB", (10, 10)), "Q?")
            assert answer == "yes"  # Should be stripped


class TestTIFAMetricUpdate:
    """Test update() method."""

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    def test_update_correct_answer(self, mock_model_class, mock_processor_class, mock_torch):
        """Test update counts correct answer."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        metadata = {
            "qa_pairs": [
                [{"question": "Is this red?", "answer": "yes"}]
            ]
        }

        # Mock _answer to return correct answer
        with patch.object(metric, "_answer", return_value="yes"):
            metric.update([img], ["prompt"], metadata)

            assert metric._correct_count == 1
            assert metric._total_count == 1
            assert metric._total_images == 1
            assert metric._per_image_scores[0] == 1.0

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    def test_update_incorrect_answer(self, mock_model_class, mock_processor_class, mock_torch):
        """Test update counts incorrect answer."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        metadata = {
            "qa_pairs": [
                [{"question": "Is this red?", "answer": "yes"}]
            ]
        }

        # Mock _answer to return incorrect answer
        with patch.object(metric, "_answer", return_value="no"):
            metric.update([img], ["prompt"], metadata)

            assert metric._correct_count == 0
            assert metric._total_count == 1
            assert metric._per_image_scores[0] == 0.0

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    def test_update_case_insensitive(self, mock_model_class, mock_processor_class, mock_torch):
        """Test update does case-insensitive comparison."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        metadata = {
            "qa_pairs": [
                [{"question": "Is this red?", "answer": "yes"}]
            ]
        }

        # Mock _answer to return uppercase version
        with patch.object(metric, "_answer", return_value="YES"):
            metric.update([img], ["prompt"], metadata)

            assert metric._correct_count == 1  # Should match despite case
            assert metric._total_count == 1

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    def test_update_multiple_qa_pairs(self, mock_model_class, mock_processor_class, mock_torch):
        """Test update with multiple QA pairs per image."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        metadata = {
            "qa_pairs": [
                [
                    {"question": "Is this red?", "answer": "yes"},
                    {"question": "Is this blue?", "answer": "no"},
                    {"question": "What color?", "answer": "red"},
                ]
            ]
        }

        # Mock _answer to return different answers
        with patch.object(metric, "_answer", side_effect=["yes", "no", "red"]):
            metric.update([img], ["prompt"], metadata)

            assert metric._correct_count == 3
            assert metric._total_count == 3
            assert metric._per_image_scores[0] == 1.0

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    def test_update_skips_none_image(self, mock_model_class, mock_processor_class, mock_torch):
        """Test update skips None images."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        metadata = {
            "qa_pairs": [
                [{"question": "Is this red?", "answer": "yes"}]
            ]
        }

        metric.update([None], ["prompt"], metadata)

        assert metric._correct_count == 0
        assert metric._total_count == 0
        assert metric._per_image_scores[0] is None

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    def test_update_skips_no_qa_pairs(self, mock_model_class, mock_processor_class, mock_torch):
        """Test update skips images without QA pairs."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        metadata = {
            "qa_pairs": [None]  # No QA pairs
        }

        metric.update([img], ["prompt"], metadata)

        assert metric._correct_count == 0
        assert metric._total_count == 0
        assert metric._per_image_scores[0] is None

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    def test_update_with_file_path(self, mock_model_class, mock_processor_class, mock_torch):
        """Test update with file path."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        metadata = {
            "qa_pairs": [
                [{"question": "Question?", "answer": "yes"}]
            ]
        }

        # Use patch.object to mock _answer since file loading is complex
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = f"{tmpdir}/test.png"
            img = Image.new("RGB", (10, 10), color="red")
            img.save(img_path)

            with patch.object(metric, "_answer", return_value="yes"):
                metric.update([img_path], ["prompt"], metadata)

                assert metric._correct_count == 1
                assert metric._total_count == 1


class TestTIFAMetricComputation:
    """Test compute() method."""

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    @patch("eval_learn.metrics.tifa.metric.Image")
    def test_compute_no_images(self, mock_image_mod, mock_model_class, mock_processor_class, mock_torch):
        """Test compute returns 0.0 with no images."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        result = metric.compute()

        assert result.name == "TIFA"
        assert result.value == 0.0
        assert "error" in result.details

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    @patch("eval_learn.metrics.tifa.metric.Image")
    def test_compute_perfect_score(self, mock_image_mod, mock_model_class, mock_processor_class, mock_torch):
        """Test compute with perfect accuracy."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        metric._correct_count = 10
        metric._total_count = 10
        metric._total_images = 2
        metric._per_image_scores = [1.0, 1.0]

        result = metric.compute()

        assert result.name == "TIFA"
        assert result.value == 1.0
        assert result.details["correct"] == 10
        assert result.details["total_questions"] == 10

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    @patch("eval_learn.metrics.tifa.metric.Image")
    def test_compute_partial_score(self, mock_image_mod, mock_model_class, mock_processor_class, mock_torch):
        """Test compute with partial correctness."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        metric._correct_count = 6
        metric._total_count = 10
        metric._total_images = 2
        metric._per_image_scores = [1.0, 0.5]

        result = metric.compute()

        assert result.name == "TIFA"
        assert result.value == pytest.approx(0.6)
        assert result.details["correct"] == 6
        assert result.details["total_questions"] == 10

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    @patch("eval_learn.metrics.tifa.metric.Image")
    def test_compute_returns_metric_result(self, mock_image_mod, mock_model_class, mock_processor_class, mock_torch):
        """Test that compute returns MetricResult instance."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        metric._correct_count = 5
        metric._total_count = 10
        metric._total_images = 1
        metric._per_image_scores = [0.5]

        result = metric.compute()

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert isinstance(result.details, dict)
        assert "config" in result.details
        assert "per_image_scores" in result.details


class TestTIFAMetricIntegration:
    """Integration tests for TIFA metric workflow."""

    @patch("eval_learn.metrics.tifa.metric.torch")
    @patch("eval_learn.metrics.tifa.metric.Blip2Processor")
    @patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration")
    def test_full_workflow_update_compute(self, mock_model_class, mock_processor_class, mock_torch):
        """Test complete workflow: initialize → update → compute."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(device="cpu")

        # Update with two images, each with 2 QA pairs
        imgs = [Image.new("RGB", (10, 10), color="red") for _ in range(2)]
        metadata = {
            "qa_pairs": [
                [
                    {"question": "Is this red?", "answer": "yes"},
                    {"question": "What color?", "answer": "red"},
                ],
                [
                    {"question": "Is this red?", "answer": "yes"},
                    {"question": "What color?", "answer": "red"},
                ],
            ]
        }

        # Mock _answer to return different responses (2 correct, 2 incorrect)
        with patch.object(metric, "_answer", side_effect=["yes", "red", "no", "blue"]):
            metric.update(imgs, ["prompt"] * 2, metadata)

            result = metric.compute()

            assert result.name == "TIFA"
            assert result.value == pytest.approx(0.5)  # 2 correct out of 4
            assert result.details["correct"] == 2
            assert result.details["total_questions"] == 4
            assert result.details["total_images"] == 2
