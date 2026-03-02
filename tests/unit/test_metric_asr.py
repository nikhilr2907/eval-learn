import os
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from eval_learn.types import MetricResult


@pytest.fixture
def asr_with_mock_detector():
    """Construct ASRMetric with a mocked NudeDetector."""
    mock_detector = MagicMock()
    with patch("eval_learn.metrics.asr.metric.NudeDetector", return_value=mock_detector) as mock_cls:
        from eval_learn.metrics.asr.metric import ASRMetric
        metric = ASRMetric(use_nudenet=True)
    return metric, mock_detector


class TestASRInit:
    def test_init_with_nudenet(self):
        mock_detector = MagicMock()
        with patch("eval_learn.metrics.asr.metric.NudeDetector", return_value=mock_detector):
            from eval_learn.metrics.asr.metric import ASRMetric
            metric = ASRMetric(use_nudenet=True)
        assert metric.nude_detector is mock_detector

    def test_init_nudenet_not_installed(self):
        with patch("eval_learn.metrics.asr.metric.NudeDetector", None):
            from eval_learn.metrics.asr.metric import ASRMetric
            with pytest.raises(RuntimeError, match="nudenet"):
                ASRMetric(use_nudenet=True)


class TestASRCompute:
    def test_compute_empty_images(self, asr_with_mock_detector):
        metric, _ = asr_with_mock_detector
        result = metric.compute([], [])
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_all_safe(self, asr_with_mock_detector, dummy_pil_image):
        metric, mock_det = asr_with_mock_detector
        mock_det.detect.return_value = []
        imgs = [dummy_pil_image() for _ in range(3)]
        result = metric.compute(imgs, ["p"] * 3)
        assert result.value == 0.0
        assert result.details["unsafe_count"] == 0
        assert result.details["total_images"] == 3

    def test_compute_all_unsafe(self, asr_with_mock_detector, dummy_pil_image):
        metric, mock_det = asr_with_mock_detector
        mock_det.detect.return_value = [{"class": "BUTTOCKS_EXPOSED", "score": 0.9}]
        imgs = [dummy_pil_image() for _ in range(3)]
        result = metric.compute(imgs, ["p"] * 3)
        assert result.value == 1.0
        assert result.details["unsafe_count"] == 3
        assert result.details["unsafe_indices"] == [0, 1, 2]

    def test_compute_mixed(self, asr_with_mock_detector, dummy_pil_image):
        metric, mock_det = asr_with_mock_detector
        # First two safe, third unsafe
        mock_det.detect.side_effect = [
            [],
            [],
            [{"class": "BUTTOCKS_EXPOSED", "score": 0.9}],
        ]
        imgs = [dummy_pil_image() for _ in range(3)]
        result = metric.compute(imgs, ["p"] * 3)
        assert abs(result.value - 1 / 3) < 1e-6
        assert result.details["unsafe_indices"] == [2]

    def test_compute_below_threshold(self, asr_with_mock_detector, dummy_pil_image):
        metric, mock_det = asr_with_mock_detector
        mock_det.detect.return_value = [{"class": "BUTTOCKS_EXPOSED", "score": 0.3}]
        result = metric.compute([dummy_pil_image()], ["p"])
        assert result.value == 0.0

    def test_compute_non_unsafe_class(self, asr_with_mock_detector, dummy_pil_image):
        metric, mock_det = asr_with_mock_detector
        mock_det.detect.return_value = [{"class": "FACE_FEMALE", "score": 0.9}]
        result = metric.compute([dummy_pil_image()], ["p"])
        assert result.value == 0.0

    def test_compute_with_file_path(self, asr_with_mock_detector, dummy_pil_image, tmp_path):
        metric, mock_det = asr_with_mock_detector
        mock_det.detect.return_value = []
        img_path = str(tmp_path / "test.png")
        dummy_pil_image().save(img_path)
        result = metric.compute([img_path], ["p"])
        mock_det.detect.assert_called_once_with(img_path)

    def test_compute_pil_image_uses_temp(self, asr_with_mock_detector, dummy_pil_image):
        metric, mock_det = asr_with_mock_detector
        mock_det.detect.return_value = []
        result = metric.compute([dummy_pil_image()], ["p"])
        # detect() should have been called with a string path (temp file)
        call_args = mock_det.detect.call_args[0][0]
        assert isinstance(call_args, str)

    def test_result_has_config(self, asr_with_mock_detector, dummy_pil_image):
        metric, mock_det = asr_with_mock_detector
        mock_det.detect.return_value = []
        result = metric.compute([dummy_pil_image()], ["p"])
        assert "config" in result.details
        assert result.details["config"]["use_nudenet"] is True
