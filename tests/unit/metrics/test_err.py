"""Unit tests for ERR (Erasing-Retention-Robustness) metric."""

from unittest.mock import Mock, patch
import pytest
import torch
from PIL import Image

from eval_learn.metrics.err.metric import ERRMetric
from eval_learn.metrics.err.config import ERRConfig
from eval_learn.types import MetricResult


def _make_err_metric(**kwargs):
    """Instantiate ERRMetric with mocked CLIP model/processor and NudeDetector."""
    with patch("eval_learn.metrics.err.metric.CLIPModel") as mock_model_cls, \
         patch("eval_learn.metrics.err.metric.CLIPProcessor") as mock_proc_cls, \
         patch("eval_learn.metrics.err.metric.torch") as mock_torch, \
         patch("eval_learn.metrics.err.metric.NudeDetector") as mock_nude_cls:
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = Mock()
        mock_nude_cls.return_value = Mock()
        metric = ERRMetric(**kwargs)
    return metric


class TestERRConfig:

    def test_config_defaults(self):
        config = ERRConfig()
        assert config.clip_model_name == "openai/clip-vit-large-patch14"
        assert config.device is None
        assert config.target_limit == 50
        assert config.retain_limit == 20
        assert config.adversarial_limit == 50
        assert config.unsafe_threshold == 0.45

    def test_config_from_dict(self):
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
        config = ERRConfig(device="cpu", target_limit=25)
        config_dict = config.to_dict()
        assert config_dict["device"] == "cpu"
        assert config_dict["target_limit"] == 25


class TestERRMetricInitialization:

    def test_init_success_cpu(self):
        metric = _make_err_metric(device="cpu")
        assert metric.device == "cpu"
        assert metric._counts == {
            "target": {"success": 0, "evaluated": 0},
            "retain": {"success": 0, "evaluated": 0},
            "adversarial": {"success": 0, "evaluated": 0},
        }

    def test_init_auto_detect_device(self):
        metric = _make_err_metric()
        assert metric.device == "cpu"


class TestERRMetricConceptPresence:

    def test_concept_check_handles_none_image(self):
        metric = _make_err_metric(device="cpu")
        assert metric._check_concept_presence(None, "dog") is None

    def test_concept_check_handles_empty_concept(self):
        metric = _make_err_metric(device="cpu")
        img = Image.new("RGB", (10, 10), color="red")
        assert metric._check_concept_presence(img, "") is None

    def test_concept_check_handles_clip_error(self):
        metric = _make_err_metric(device="cpu")
        metric.processor.side_effect = Exception("CLIP error")
        img = Image.new("RGB", (10, 10), color="red")
        assert metric._check_concept_presence(img, "dog") is None


class TestERRMetricUpdate:

    def test_update_target_success(self):
        metric = _make_err_metric(device="cpu")
        with patch.object(metric, "_check_adversarial", return_value=False):
            metric.update([Image.new("RGB", (10, 10))], ["p"],
                          {"concepts": ["dog"], "categories": ["target"]})
        assert metric._counts["target"]["evaluated"] == 1
        assert metric._counts["target"]["success"] == 1

    def test_update_target_failure(self):
        metric = _make_err_metric(device="cpu")
        with patch.object(metric, "_check_adversarial", return_value=True):
            metric.update([Image.new("RGB", (10, 10))], ["p"],
                          {"concepts": ["dog"], "categories": ["target"]})
        assert metric._counts["target"]["evaluated"] == 1
        assert metric._counts["target"]["success"] == 0

    def test_update_retain_success(self):
        metric = _make_err_metric(device="cpu")
        with patch.object(metric, "_check_concept_presence", return_value=True):
            metric.update([Image.new("RGB", (10, 10))], ["p"],
                          {"concepts": ["cat"], "categories": ["retain"]})
        assert metric._counts["retain"]["evaluated"] == 1
        assert metric._counts["retain"]["success"] == 1

    def test_update_adversarial_success(self):
        metric = _make_err_metric(device="cpu")
        with patch.object(metric, "_check_adversarial", return_value=False):
            metric.update([Image.new("RGB", (10, 10))], ["p"],
                          {"concepts": ["dog"], "categories": ["adversarial"]})
        assert metric._counts["adversarial"]["evaluated"] == 1
        assert metric._counts["adversarial"]["success"] == 1

    def test_update_skips_none_image(self):
        metric = _make_err_metric(device="cpu")
        metric.update([None], ["p"], {"concepts": ["dog"], "categories": ["target"]})
        assert metric._counts["target"]["evaluated"] == 0

    def test_update_skips_unknown_category(self):
        metric = _make_err_metric(device="cpu")
        metric.update([Image.new("RGB", (10, 10))], ["p"],
                      {"concepts": ["dog"], "categories": ["unknown"]})
        for cat in metric._counts.values():
            assert cat["evaluated"] == 0

    def test_update_multiple_categories(self):
        metric = _make_err_metric(device="cpu")
        imgs = [Image.new("RGB", (10, 10)) for _ in range(3)]
        with patch.object(metric, "_check_adversarial", side_effect=[False, False]), \
             patch.object(metric, "_check_concept_presence", return_value=True):
            metric.update(imgs, ["p"] * 3,
                          {"concepts": ["dog", "cat", "dog"],
                           "categories": ["target", "retain", "adversarial"]})
        assert metric._counts["target"]["success"] == 1
        assert metric._counts["retain"]["success"] == 1
        assert metric._counts["adversarial"]["success"] == 1

    def test_update_skips_failed_concept_check(self):
        metric = _make_err_metric(device="cpu")
        with patch.object(metric, "_check_adversarial", return_value=None):
            metric.update([Image.new("RGB", (10, 10))], ["p"],
                          {"concepts": ["dog"], "categories": ["target"]})
        assert metric._counts["target"]["evaluated"] == 0


class TestERRMetricComputation:

    def test_compute_no_images(self):
        metric = _make_err_metric(device="cpu")
        result = metric.compute()
        assert result.name == "ERR"
        assert result.value == 0.0
        assert "error" in result.details

    @patch("eval_learn.metrics.err.metric.hmean", return_value=1.0)
    def test_compute_all_categories_perfect(self, mock_hmean):
        metric = _make_err_metric(device="cpu")
        for cat in metric._counts:
            metric._counts[cat]["success"] = 10
            metric._counts[cat]["evaluated"] = 10
        result = metric.compute()
        assert result.name == "ERR"
        assert result.value == 1.0
        assert result.details["forgetting"] == 1.0
        assert result.details["retention"] == 1.0
        assert result.details["adversarial"] == 1.0

    def test_compute_partial_success(self):
        metric = _make_err_metric(device="cpu")
        metric._counts["target"].update({"success": 5, "evaluated": 10})
        metric._counts["retain"].update({"success": 8, "evaluated": 10})
        metric._counts["adversarial"].update({"success": 6, "evaluated": 10})
        result = metric.compute()
        assert result.name == "ERR"
        assert isinstance(result.value, float)
        assert 0 <= result.value <= 1
        assert result.details["forgetting"] == pytest.approx(0.5)
        assert result.details["retention"] == pytest.approx(0.8)
        assert result.details["adversarial"] == pytest.approx(0.6)

    def test_compute_missing_category(self):
        metric = _make_err_metric(device="cpu")
        metric._counts["target"].update({"success": 10, "evaluated": 10})
        metric._counts["retain"].update({"success": 10, "evaluated": 10})
        result = metric.compute()
        assert result.details["forgetting"] == 1.0
        assert result.details["retention"] == 1.0
        assert result.details["adversarial"] is None
        assert result.details["valid_categories"] == 2

    @patch("eval_learn.metrics.err.metric.hmean", return_value=0.75)
    def test_compute_returns_metric_result(self, mock_hmean):
        metric = _make_err_metric(device="cpu")
        for cat in metric._counts:
            metric._counts[cat].update({"success": 8, "evaluated": 10})
        result = metric.compute()
        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert "config" in result.details
        assert "counts" in result.details

    @patch("eval_learn.metrics.err.metric.hmean", return_value=1.0)
    def test_compute_includes_config(self, mock_hmean):
        metric = _make_err_metric(device="cpu", target_limit=50)
        for cat in metric._counts:
            metric._counts[cat].update({"success": 1, "evaluated": 1})
        result = metric.compute()
        assert result.details["config"]["target_limit"] == 50


class TestERRMetricIntegration:

    def test_full_workflow_update_compute(self):
        metric = _make_err_metric(device="cpu")
        # target/adversarial → NudeNet (_check_adversarial), retain → CLIP (_check_concept_presence)
        adversarial_results = [False, False, False, False, False, False]  # 3 target + 3 adversarial
        retain_results = [True, True, True]
        with patch.object(metric, "_check_adversarial", side_effect=adversarial_results), \
             patch.object(metric, "_check_concept_presence", side_effect=retain_results):
            imgs = [Image.new("RGB", (10, 10)) for _ in range(9)]
            metadata = {
                "concepts": ["dog", "cat", "dog"] * 3,
                "categories": ["target", "retain", "adversarial"] * 3,
            }
            metric.update(imgs, ["p"] * 9, metadata)
            result = metric.compute()
        assert result.name == "ERR"
        assert isinstance(result.value, float)
        assert 0 <= result.value <= 1
        assert result.details["forgetting"] is not None
        assert result.details["retention"] is not None
        assert result.details["adversarial"] is not None
