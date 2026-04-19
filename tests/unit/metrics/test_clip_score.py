"""Unit tests for CLIP Score metric."""

from unittest.mock import Mock, patch
import pytest

from eval_learn.metrics.clip_score.metric import CLIPScoreMetric
from eval_learn.metrics.clip_score.config import CLIPScoreConfig
from eval_learn.types import MetricResult


class TestCLIPScoreConfig:
    """Test CLIPScoreConfig initialization and validation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = CLIPScoreConfig()
        assert config.clip_model_name == "openai/clip-vit-large-patch14"
        assert config.device is None
        assert config.limit == 300

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "clip_model_name": "openai/clip-vit-large-patch14",
            "device": "cuda",
            "limit": 100,
        }
        config = CLIPScoreConfig.from_dict(config_dict)
        assert config.clip_model_name == "openai/clip-vit-large-patch14"
        assert config.device == "cuda"
        assert config.limit == 100

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = CLIPScoreConfig(device="cpu", limit=50)
        config_dict = config.to_dict()
        assert config_dict["device"] == "cpu"
        assert config_dict["limit"] == 50


class TestCLIPScoreMetricInitialization:
    """Test CLIPScoreMetric initialization."""

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_init_success_cpu(self, mock_image, mock_transforms, mock_clip_score_class, mock_torch):
        """Test successful initialization on CPU."""
        mock_torch.cuda.is_available.return_value = False
        mock_clip_score_instance = Mock()
        mock_clip_score_class.return_value = mock_clip_score_instance
        mock_clip_score_instance.to.return_value = mock_clip_score_instance

        metric = CLIPScoreMetric(device="cpu")

        assert metric.device == "cpu"
        assert metric._total_score == 0.0
        assert metric._evaluated == 0
        assert metric._total_images == 0
        assert metric._per_image_scores == []

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_init_auto_detect_device_cpu(self, mock_image, mock_transforms, mock_clip_score_class, mock_torch):
        """Test device auto-detection defaults to CPU when CUDA unavailable."""
        mock_torch.cuda.is_available.return_value = False
        mock_clip_score_instance = Mock()
        mock_clip_score_class.return_value = mock_clip_score_instance
        mock_clip_score_instance.to.return_value = mock_clip_score_instance

        metric = CLIPScoreMetric(device=None)
        assert metric.device == "cpu"

    @patch("eval_learn.metrics.clip_score.metric.torch", None)
    def test_init_missing_torch_raises_error(self):
        """Test that missing torch raises helpful error."""
        with pytest.raises(RuntimeError, match="requires 'torch'"):
            CLIPScoreMetric()

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore", None)
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_init_missing_torchmetrics_raises_error(self, mock_image, mock_transforms, mock_torch):
        """Test that missing torchmetrics raises helpful error."""
        with pytest.raises(RuntimeError, match="requires 'torchmetrics'"):
            CLIPScoreMetric()

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms", None)
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_init_missing_torchvision_raises_error(self, mock_image, mock_clip_score_class, mock_torch):
        """Test that missing torchvision raises helpful error."""
        with pytest.raises(RuntimeError, match="requires 'torchvision'"):
            CLIPScoreMetric()

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image", None)
    def test_init_missing_pillow_raises_error(self, mock_transforms, mock_clip_score_class, mock_torch):
        """Test that missing Pillow raises helpful error."""
        with pytest.raises(RuntimeError, match="requires 'Pillow'"):
            CLIPScoreMetric()


class TestCLIPScoreMetricUpdate:
    """Test update method for scoring images."""

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_update_with_valid_images(self, mock_image_class, mock_transforms, mock_clip_class, mock_torch):
        """Test update with valid image-prompt pairs."""
        mock_torch.uint8 = Mock()
        mock_clip_instance = Mock()
        mock_clip_class.return_value = mock_clip_instance
        mock_clip_instance.to.return_value = mock_clip_instance

        # Setup tensor mock
        mock_tensor = Mock()
        mock_score_tensor = Mock()
        mock_score_tensor.item.return_value = 0.75
        mock_clip_instance.return_value = mock_score_tensor

        metric = CLIPScoreMetric(device="cpu")

        # Mock _to_uint8_tensor to return a mock tensor
        with patch.object(metric, "_to_uint8_tensor", return_value=mock_tensor):
            metric.update([Mock(), Mock()], ["prompt1", "prompt2"])

            assert metric._total_images == 2
            assert metric._evaluated == 2
            assert metric._total_score == pytest.approx(1.5)  # 0.75 + 0.75
            assert len(metric._per_image_scores) == 2

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_update_with_failed_image_loading(self, mock_image_class, mock_transforms, mock_clip_class, mock_torch):
        """Test update when image loading fails."""
        mock_torch.uint8 = Mock()
        mock_clip_instance = Mock()
        mock_clip_class.return_value = mock_clip_instance
        mock_clip_instance.to.return_value = mock_clip_instance

        metric = CLIPScoreMetric(device="cpu")

        # Mock _to_uint8_tensor to fail
        with patch.object(metric, "_to_uint8_tensor", return_value=None):
            metric.update([Mock()], ["prompt"])

            assert metric._total_images == 1
            assert metric._evaluated == 0  # Failed to load
            assert metric._per_image_scores[0] is None

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_update_multiple_batches_accumulate(self, mock_image_class, mock_transforms, mock_clip_class, mock_torch):
        """Test that update accumulates across multiple calls."""
        mock_torch.uint8 = Mock()
        mock_clip_instance = Mock()
        mock_clip_class.return_value = mock_clip_instance
        mock_clip_instance.to.return_value = mock_clip_instance

        metric = CLIPScoreMetric(device="cpu")

        # Create mock tensors for two updates
        mock_tensor = Mock()
        mock_score1 = Mock(item=Mock(return_value=0.8))
        mock_score2 = Mock(item=Mock(return_value=0.9))
        mock_clip_instance.side_effect = [mock_score1, mock_score2]

        with patch.object(metric, "_to_uint8_tensor", return_value=mock_tensor):
            metric.update([Mock()], ["prompt1"])
            assert metric._total_images == 1

            metric.update([Mock()], ["prompt2"])
            assert metric._total_images == 2
            assert metric._evaluated == 2


class TestCLIPScoreMetricComputation:
    """Test compute method."""

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_compute_with_no_images(self, mock_image_class, mock_transforms, mock_clip_class, mock_torch):
        """Test compute with no images evaluated."""
        mock_torch.uint8 = Mock()
        mock_clip_instance = Mock()
        mock_clip_class.return_value = mock_clip_instance
        mock_clip_instance.to.return_value = mock_clip_instance

        metric = CLIPScoreMetric(device="cpu")
        result = metric.compute()

        assert result.name == "CLIPScore"
        assert result.value == 0.0
        assert "error" in result.details

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_compute_with_all_successful(self, mock_image_class, mock_transforms, mock_clip_class, mock_torch):
        """Test compute with all images scored successfully."""
        mock_torch.uint8 = Mock()
        mock_clip_instance = Mock()
        mock_clip_class.return_value = mock_clip_instance
        mock_clip_instance.to.return_value = mock_clip_instance

        metric = CLIPScoreMetric(device="cpu")
        metric._total_images = 4
        metric._evaluated = 4
        metric._total_score = 3.2  # 0.8 * 4
        metric._per_image_scores = [0.8, 0.8, 0.8, 0.8]

        result = metric.compute()

        assert result.name == "CLIPScore"
        assert result.value == pytest.approx(0.8)
        assert result.details["evaluated"] == 4
        assert result.details["total_images"] == 4

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_compute_with_partial_success(self, mock_image_class, mock_transforms, mock_clip_class, mock_torch):
        """Test compute with some images failing to load."""
        mock_torch.uint8 = Mock()
        mock_clip_instance = Mock()
        mock_clip_class.return_value = mock_clip_instance
        mock_clip_instance.to.return_value = mock_clip_instance

        metric = CLIPScoreMetric(device="cpu")
        metric._total_images = 5
        metric._evaluated = 3
        metric._total_score = 2.1  # 0.7 * 3
        metric._per_image_scores = [0.7, None, 0.7, 0.7, None]

        result = metric.compute()

        assert result.name == "CLIPScore"
        assert result.value == pytest.approx(0.7)
        assert result.details["evaluated"] == 3
        assert result.details["total_images"] == 5

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_compute_returns_metric_result(self, mock_image_class, mock_transforms, mock_clip_class, mock_torch):
        """Test that compute returns MetricResult instance."""
        mock_torch.uint8 = Mock()
        mock_clip_instance = Mock()
        mock_clip_class.return_value = mock_clip_instance
        mock_clip_instance.to.return_value = mock_clip_instance

        metric = CLIPScoreMetric(device="cpu")
        metric._total_images = 1
        metric._evaluated = 1
        metric._total_score = 0.85
        metric._per_image_scores = [0.85]

        result = metric.compute()

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert isinstance(result.details, dict)
        assert "config" in result.details

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_compute_includes_per_image_scores(self, mock_image_class, mock_transforms, mock_clip_class, mock_torch):
        """Test that compute includes per-image scores in details."""
        mock_torch.uint8 = Mock()
        mock_clip_instance = Mock()
        mock_clip_class.return_value = mock_clip_instance
        mock_clip_instance.to.return_value = mock_clip_instance

        metric = CLIPScoreMetric(device="cpu")
        metric._total_images = 3
        metric._evaluated = 3
        metric._total_score = 2.25
        metric._per_image_scores = [0.7, 0.75, 0.8]

        result = metric.compute()

        assert result.details["per_image_scores"] == [0.7, 0.75, 0.8]

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_compute_zero_evaluated_with_images(self, mock_image_class, mock_transforms, mock_clip_class, mock_torch):
        """Test compute when images exist but none were evaluated successfully."""
        mock_torch.uint8 = Mock()
        mock_clip_instance = Mock()
        mock_clip_class.return_value = mock_clip_instance
        mock_clip_instance.to.return_value = mock_clip_instance

        metric = CLIPScoreMetric(device="cpu")
        metric._total_images = 3
        metric._evaluated = 0
        metric._total_score = 0.0
        metric._per_image_scores = [None, None, None]

        result = metric.compute()

        assert result.name == "CLIPScore"
        assert result.value == 0.0


class TestCLIPScoreMetricIntegration:
    """Integration tests for CLIP Score metric workflow."""

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_full_workflow(self, mock_image_class, mock_transforms, mock_clip_class, mock_torch):
        """Test complete workflow from init to compute."""
        mock_torch.uint8 = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_clip_instance = Mock()
        mock_clip_class.return_value = mock_clip_instance
        mock_clip_instance.to.return_value = mock_clip_instance

        # Setup scoring for three images
        mock_tensor = Mock()
        mock_scores = [Mock(item=Mock(return_value=s)) for s in [0.75, 0.8, 0.85]]
        mock_clip_instance.side_effect = mock_scores

        metric = CLIPScoreMetric(device="cpu")

        with patch.object(metric, "_to_uint8_tensor", return_value=mock_tensor):
            metric.update([Mock(), Mock(), Mock()], ["prompt1", "prompt2", "prompt3"])

            result = metric.compute()

            assert result.name == "CLIPScore"
            assert result.value == pytest.approx(0.8)  # (0.75 + 0.8 + 0.85) / 3
            assert result.details["evaluated"] == 3
            assert result.details["total_images"] == 3

    @patch("eval_learn.metrics.clip_score.metric.torch")
    @patch("eval_learn.metrics.clip_score.metric.CLIPScore")
    @patch("eval_learn.metrics.clip_score.metric.transforms")
    @patch("eval_learn.metrics.clip_score.metric.Image")
    def test_config_included_in_result(self, mock_image_class, mock_transforms, mock_clip_class, mock_torch):
        """Test that config is included in result details."""
        mock_torch.uint8 = Mock()
        mock_clip_instance = Mock()
        mock_clip_class.return_value = mock_clip_instance
        mock_clip_instance.to.return_value = mock_clip_instance

        metric = CLIPScoreMetric(device="cpu", clip_model_name="custom-model")
        metric._total_images = 1
        metric._evaluated = 1
        metric._total_score = 0.75
        metric._per_image_scores = [0.75]

        result = metric.compute()

        assert "config" in result.details
        assert result.details["config"]["clip_model_name"] == "custom-model"
