"""Unit tests for ERR (Erasing-Retention-Robustness) metric."""

from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
import numpy as np
from PIL import Image

from eval_learn.metrics.err.metric import ERRMetric
from eval_learn.metrics.err.config import ERRConfig
from eval_learn.types import MetricResult


class TestERRConfig:
    """Test ERRConfig initialization and validation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ERRConfig()
        assert config.clip_model_name == "openai/clip-vit-large-patch14"
        assert config.device is None
        assert config.target_limit == 100
        assert config.retain_limit == 100
        assert config.adversarial_limit == 100

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "clip_model_name": "openai/clip-vit-base-patch32",
            "device": "cpu",
            "target_limit": 50,
            "retain_limit": 75,
            "adversarial_limit": 60,
        }
        config = ERRConfig.from_dict(config_dict)
        assert config.clip_model_name == "openai/clip-vit-base-patch32"
        assert config.device == "cpu"
        assert config.target_limit == 50
        assert config.retain_limit == 75
        assert config.adversarial_limit == 60

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ERRConfig(device="cpu", target_limit=25)
        config_dict = config.to_dict()
        assert config_dict["device"] == "cpu"
        assert config_dict["target_limit"] == 25


class TestERRMetricInitialization:
    """Test ERRMetric initialization."""

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_init_success_cpu(self, mock_image, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test successful initialization on CPU."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        assert metric.device == torch.device("cpu")
        assert metric.model is mock_model
        assert metric.processor is mock_processor
        assert metric._counts == {"target": {"success": 0, "evaluated": 0},
                                   "retain": {"success": 0, "evaluated": 0},
                                   "adversarial": {"success": 0, "evaluated": 0}}

    @patch("eval_learn.metrics.err.metric.torch", None)
    def test_init_missing_torch_raises_error(self):
        """Test that missing torch raises helpful error."""
        with pytest.raises(RuntimeError, match="requires 'torch'"):
            ERRMetric()

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel", None)
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_init_missing_transformers_raises_error(self, mock_image, mock_hmean, mock_processor, mock_torch):
        """Test that missing transformers raises helpful error."""
        with pytest.raises(RuntimeError, match="requires 'transformers'"):
            ERRMetric()

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean", None)
    @patch("eval_learn.metrics.err.metric.Image")
    def test_init_missing_scipy_raises_error(self, mock_image, mock_processor, mock_model, mock_torch):
        """Test that missing scipy raises helpful error."""
        with pytest.raises(RuntimeError, match="requires 'scipy'"):
            ERRMetric()

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image", None)
    def test_init_missing_pillow_raises_error(self, mock_hmean, mock_processor, mock_model, mock_torch):
        """Test that missing Pillow raises helpful error."""
        with pytest.raises(RuntimeError, match="requires 'Pillow'"):
            ERRMetric()


class TestERRMetricConceptPresence:
    """Test concept presence detection via CLIP."""

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_concept_check_handles_none_image(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test that None image returns None."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        result = metric._check_concept_presence(None, "dog")

        assert result is None

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_concept_check_handles_empty_concept(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test that empty concept returns None."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        result = metric._check_concept_presence(img, "")

        assert result is None

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_concept_check_handles_clip_error(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test that CLIP errors return None gracefully."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        # Make processor raise an exception
        mock_processor.side_effect = Exception("CLIP error")

        img = Image.new("RGB", (10, 10), color="red")
        result = metric._check_concept_presence(img, "dog")

        assert result is None


class TestERRMetricUpdate:
    """Test update() method."""

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_update_target_success(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test update counts success for target category (concept should be absent)."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        # Mock _check_concept_presence to return False (absent, which is success for target)
        with patch.object(metric, "_check_concept_presence", return_value=False):
            img = Image.new("RGB", (10, 10), color="red")
            metadata = {
                "concepts": ["dog"],
                "categories": ["target"],
            }

            metric.update([img], ["prompt"], metadata)

            assert metric._counts["target"]["evaluated"] == 1
            assert metric._counts["target"]["success"] == 1

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_update_target_failure(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test update counts failure for target category (concept present when should be absent)."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        # Concept is present (True), but target expects absent (False) - failure
        with patch.object(metric, "_check_concept_presence", return_value=True):
            img = Image.new("RGB", (10, 10), color="red")
            metadata = {
                "concepts": ["dog"],
                "categories": ["target"],
            }

            metric.update([img], ["prompt"], metadata)

            assert metric._counts["target"]["evaluated"] == 1
            assert metric._counts["target"]["success"] == 0

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_update_retain_success(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test update counts success for retain category (concept should be present)."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        # Concept is present (True), retain expects present (True) - success
        with patch.object(metric, "_check_concept_presence", return_value=True):
            img = Image.new("RGB", (10, 10), color="red")
            metadata = {
                "concepts": ["cat"],
                "categories": ["retain"],
            }

            metric.update([img], ["prompt"], metadata)

            assert metric._counts["retain"]["evaluated"] == 1
            assert metric._counts["retain"]["success"] == 1

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_update_adversarial_success(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test update counts success for adversarial category."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        # Concept is absent (False), adversarial expects absent (False) - success
        with patch.object(metric, "_check_concept_presence", return_value=False):
            img = Image.new("RGB", (10, 10), color="red")
            metadata = {
                "concepts": ["dog"],
                "categories": ["adversarial"],
            }

            metric.update([img], ["prompt"], metadata)

            assert metric._counts["adversarial"]["evaluated"] == 1
            assert metric._counts["adversarial"]["success"] == 1

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_update_skips_none_image(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test that None images are skipped."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        metadata = {
            "concepts": ["dog"],
            "categories": ["target"],
        }

        metric.update([None], ["prompt"], metadata)

        assert metric._counts["target"]["evaluated"] == 0

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_update_skips_unknown_category(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test that unknown categories are skipped."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        metadata = {
            "concepts": ["dog"],
            "categories": ["unknown_category"],
        }

        metric.update([img], ["prompt"], metadata)

        # Should not have incremented any category
        for cat in metric._counts.values():
            assert cat["evaluated"] == 0

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_update_multiple_categories(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test update with multiple images and categories."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        # Mock different returns for each call
        with patch.object(metric, "_check_concept_presence", side_effect=[False, True, False]):
            imgs = [Image.new("RGB", (10, 10), color="red") for _ in range(3)]
            metadata = {
                "concepts": ["dog", "cat", "dog"],
                "categories": ["target", "retain", "adversarial"],
            }

            metric.update(imgs, ["prompt"] * 3, metadata)

            assert metric._counts["target"]["evaluated"] == 1
            assert metric._counts["target"]["success"] == 1
            assert metric._counts["retain"]["evaluated"] == 1
            assert metric._counts["retain"]["success"] == 1
            assert metric._counts["adversarial"]["evaluated"] == 1
            assert metric._counts["adversarial"]["success"] == 1

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_update_skips_failed_concept_check(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test that failed concept checks (None) are skipped."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        # _check_concept_presence returns None (failure)
        with patch.object(metric, "_check_concept_presence", return_value=None):
            img = Image.new("RGB", (10, 10), color="red")
            metadata = {
                "concepts": ["dog"],
                "categories": ["target"],
            }

            metric.update([img], ["prompt"], metadata)

            assert metric._counts["target"]["evaluated"] == 0


class TestERRMetricComputation:
    """Test compute() method."""

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_compute_no_images(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test compute returns 0.0 with no evaluated images."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        metric = ERRMetric(device="cpu")

        result = metric.compute()

        assert result.name == "ERR"
        assert result.value == 0.0
        assert "error" in result.details

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_compute_all_categories_perfect(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test compute with perfect accuracy in all categories."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        # Mock hmean to return average
        mock_hmean.return_value = 1.0

        metric = ERRMetric(device="cpu")

        # Set perfect counts: all success
        metric._counts["target"]["success"] = 10
        metric._counts["target"]["evaluated"] = 10
        metric._counts["retain"]["success"] = 10
        metric._counts["retain"]["evaluated"] = 10
        metric._counts["adversarial"]["success"] = 10
        metric._counts["adversarial"]["evaluated"] = 10

        result = metric.compute()

        assert result.name == "ERR"
        assert result.value == 1.0
        assert result.details["forgetting"] == 1.0
        assert result.details["retention"] == 1.0
        assert result.details["adversarial"] == 1.0

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_compute_partial_success(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test compute with partial success."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        # Mock hmean to compute harmonic mean
        def harmonic_mean(values):
            return len(values) / sum(1.0/v for v in values)

        mock_hmean.side_effect = harmonic_mean

        metric = ERRMetric(device="cpu")

        # Set partial counts
        metric._counts["target"]["success"] = 5
        metric._counts["target"]["evaluated"] = 10  # 0.5
        metric._counts["retain"]["success"] = 8
        metric._counts["retain"]["evaluated"] = 10  # 0.8
        metric._counts["adversarial"]["success"] = 6
        metric._counts["adversarial"]["evaluated"] = 10  # 0.6

        result = metric.compute()

        assert result.name == "ERR"
        assert isinstance(result.value, float)
        assert 0 <= result.value <= 1
        assert result.details["forgetting"] == pytest.approx(0.5)
        assert result.details["retention"] == pytest.approx(0.8)
        assert result.details["adversarial"] == pytest.approx(0.6)

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_compute_missing_category(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test compute with missing category (0 evaluated)."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        def harmonic_mean(values):
            return len(values) / sum(1.0/v for v in values)

        mock_hmean.side_effect = harmonic_mean

        metric = ERRMetric(device="cpu")

        # Only target and retain have data, adversarial is 0
        metric._counts["target"]["success"] = 10
        metric._counts["target"]["evaluated"] = 10
        metric._counts["retain"]["success"] = 10
        metric._counts["retain"]["evaluated"] = 10
        metric._counts["adversarial"]["success"] = 0
        metric._counts["adversarial"]["evaluated"] = 0

        result = metric.compute()

        assert result.name == "ERR"
        assert result.details["forgetting"] == 1.0
        assert result.details["retention"] == 1.0
        assert result.details["adversarial"] is None  # No data
        assert result.details["valid_categories"] == 2

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_compute_returns_metric_result(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test that compute returns MetricResult instance."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_hmean.return_value = 0.75

        metric = ERRMetric(device="cpu")

        metric._counts["target"]["success"] = 8
        metric._counts["target"]["evaluated"] = 10
        metric._counts["retain"]["success"] = 8
        metric._counts["retain"]["evaluated"] = 10
        metric._counts["adversarial"]["success"] = 8
        metric._counts["adversarial"]["evaluated"] = 10

        result = metric.compute()

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert isinstance(result.details, dict)
        assert "config" in result.details
        assert "counts" in result.details

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_compute_includes_config(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test that compute includes config in details."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_hmean.return_value = 1.0

        metric = ERRMetric(device="cpu", target_limit=50)

        metric._counts["target"]["success"] = 1
        metric._counts["target"]["evaluated"] = 1
        metric._counts["retain"]["success"] = 1
        metric._counts["retain"]["evaluated"] = 1
        metric._counts["adversarial"]["success"] = 1
        metric._counts["adversarial"]["evaluated"] = 1

        result = metric.compute()

        assert "config" in result.details
        assert result.details["config"]["target_limit"] == 50


class TestERRMetricIntegration:
    """Integration tests for ERR metric workflow."""

    @patch("eval_learn.metrics.err.metric.torch")
    @patch("eval_learn.metrics.err.metric.CLIPModel")
    @patch("eval_learn.metrics.err.metric.CLIPProcessor")
    @patch("eval_learn.metrics.err.metric.hmean")
    @patch("eval_learn.metrics.err.metric.Image")
    def test_full_workflow_update_compute(self, mock_image_mod, mock_hmean, mock_processor_class, mock_model_class, mock_torch):
        """Test complete workflow: initialize → update → compute."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = torch.device

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        def harmonic_mean(values):
            return len(values) / sum(1.0/v for v in values)

        mock_hmean.side_effect = harmonic_mean

        metric = ERRMetric(device="cpu")

        # Simulate multiple updates
        concept_results = [False, True, False, True, False, True, False, True, False]
        with patch.object(metric, "_check_concept_presence", side_effect=concept_results):
            imgs = [Image.new("RGB", (10, 10), color="red") for _ in range(9)]
            metadata = {
                "concepts": ["dog", "cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"],
                "categories": ["target", "retain", "adversarial", "target", "retain",
                             "adversarial", "target", "retain", "adversarial"],
            }

            metric.update(imgs, ["prompt"] * 9, metadata)

            result = metric.compute()

            assert result.name == "ERR"
            assert isinstance(result.value, float)
            assert 0 <= result.value <= 1
            assert result.details["forgetting"] is not None
            assert result.details["retention"] is not None
            assert result.details["adversarial"] is not None
