import pytest
from unittest.mock import patch, MagicMock
from PIL import Image


@pytest.fixture
def mock_clip_score_deps():
    """Patch all external deps for CLIPScore metric."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.uint8 = "uint8"

    mock_clip_score_cls = MagicMock()
    mock_clip_score_fn = MagicMock()
    mock_clip_score_cls.return_value.to.return_value = mock_clip_score_fn
    # score_fn(tensor, prompt) returns a tensor-like with .item()
    mock_score_tensor = MagicMock()
    mock_score_tensor.item.return_value = 25.0
    mock_clip_score_fn.return_value = mock_score_tensor

    mock_transforms = MagicMock()
    mock_to_tensor_fn = MagicMock()
    mock_transforms.ToTensor.return_value = mock_to_tensor_fn
    # ToTensor()(img) returns a tensor-like that supports * 255 and .to()
    mock_tensor = MagicMock()
    mock_tensor.__mul__ = MagicMock(return_value=mock_tensor)
    mock_tensor.__rmul__ = MagicMock(return_value=mock_tensor)
    mock_tensor.to = MagicMock(return_value=mock_tensor)
    mock_to_tensor_fn.return_value = mock_tensor

    mock_image = MagicMock()
    mock_image.Image = Image.Image

    patches = [
        patch("eval_learn.metrics.clip_score.metric.torch", mock_torch),
        patch("eval_learn.metrics.clip_score.metric.CLIPScore", mock_clip_score_cls),
        patch("eval_learn.metrics.clip_score.metric.transforms", mock_transforms),
        patch("eval_learn.metrics.clip_score.metric.Image", mock_image),
    ]
    for p in patches:
        p.start()
    yield {
        "torch": mock_torch,
        "clip_score_cls": mock_clip_score_cls,
        "clip_score_fn": mock_clip_score_fn,
        "transforms": mock_transforms,
        "tensor": mock_tensor,
        "score_tensor": mock_score_tensor,
    }
    for p in patches:
        p.stop()


@pytest.fixture
def clip_score_metric(mock_clip_score_deps):
    from eval_learn.metrics.clip_score.metric import CLIPScoreMetric
    metric = CLIPScoreMetric(device="cpu")
    return metric


class TestCLIPScoreInit:
    def test_init_success(self, mock_clip_score_deps):
        from eval_learn.metrics.clip_score.metric import CLIPScoreMetric
        metric = CLIPScoreMetric(device="cpu")
        mock_clip_score_deps["clip_score_cls"].assert_called_once()

    def test_init_missing_torch(self):
        with patch("eval_learn.metrics.clip_score.metric.torch", None):
            from eval_learn.metrics.clip_score.metric import CLIPScoreMetric
            with pytest.raises(RuntimeError, match="torch"):
                CLIPScoreMetric()

    def test_init_missing_torchmetrics(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch("eval_learn.metrics.clip_score.metric.torch", mock_torch), \
             patch("eval_learn.metrics.clip_score.metric.CLIPScore", None):
            from eval_learn.metrics.clip_score.metric import CLIPScoreMetric
            with pytest.raises(RuntimeError, match="torchmetrics"):
                CLIPScoreMetric()

    def test_init_missing_torchvision(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_clip = MagicMock()
        with patch("eval_learn.metrics.clip_score.metric.torch", mock_torch), \
             patch("eval_learn.metrics.clip_score.metric.CLIPScore", mock_clip), \
             patch("eval_learn.metrics.clip_score.metric.transforms", None):
            from eval_learn.metrics.clip_score.metric import CLIPScoreMetric
            with pytest.raises(RuntimeError, match="torchvision"):
                CLIPScoreMetric()


class TestCLIPScoreCompute:
    def test_compute_empty(self, clip_score_metric):
        result = clip_score_metric.compute([], [])
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_length_mismatch(self, clip_score_metric, dummy_pil_image):
        result = clip_score_metric.compute(
            [dummy_pil_image(), dummy_pil_image()], ["only one prompt"]
        )
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_basic(self, clip_score_metric, dummy_pil_image):
        imgs = [dummy_pil_image(), dummy_pil_image()]
        # _to_pil returns the Image since it's already PIL
        clip_score_metric._to_pil = MagicMock(side_effect=lambda x: x if isinstance(x, Image.Image) else None)

        result = clip_score_metric.compute(imgs, ["prompt1", "prompt2"])
        assert result.value == 25.0
        assert result.details["evaluated"] == 2
        assert result.details["per_image_scores"] == [25.0, 25.0]

    def test_compute_skips_bad_image(self, clip_score_metric, dummy_pil_image):
        clip_score_metric._to_pil = MagicMock(side_effect=[None, dummy_pil_image()])

        result = clip_score_metric.compute([None, dummy_pil_image()], ["p1", "p2"])
        assert result.details["evaluated"] == 1
        assert result.details["per_image_scores"][0] is None
        assert result.details["per_image_scores"][1] == 25.0

    def test_compute_handles_exception(self, clip_score_metric, mock_clip_score_deps, dummy_pil_image):
        clip_score_metric._to_pil = MagicMock(side_effect=lambda x: x if isinstance(x, Image.Image) else None)
        clip_score_metric._pil_to_tensor = MagicMock(side_effect=RuntimeError("bad"))

        result = clip_score_metric.compute([dummy_pil_image()], ["p"])
        assert result.details["evaluated"] == 0
        assert result.details["per_image_scores"][0] is None

    def test_result_has_config(self, clip_score_metric, dummy_pil_image):
        clip_score_metric._to_pil = MagicMock(side_effect=lambda x: x if isinstance(x, Image.Image) else None)
        result = clip_score_metric.compute([dummy_pil_image()], ["p"])
        assert "config" in result.details
