import os
import contextlib
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image


@pytest.fixture
def mock_err_deps():
    """Patch all external deps for ERR metric."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.device.return_value = MagicMock()

    mock_clip_model_cls = MagicMock()
    mock_clip_model = MagicMock()
    mock_clip_model_cls.from_pretrained.return_value.to.return_value = mock_clip_model

    mock_clip_processor_cls = MagicMock()
    mock_clip_processor = MagicMock()
    mock_clip_processor_cls.from_pretrained.return_value = mock_clip_processor

    mock_hmean = MagicMock(side_effect=lambda x: sum(x) / len(x))

    mock_image = MagicMock()
    mock_image.Image = Image.Image

    patches = {
        "eval_learn.metrics.err.metric.torch": mock_torch,
        "eval_learn.metrics.err.metric.CLIPModel": mock_clip_model_cls,
        "eval_learn.metrics.err.metric.CLIPProcessor": mock_clip_processor_cls,
        "eval_learn.metrics.err.metric.hmean": mock_hmean,
        "eval_learn.metrics.err.metric.Image": mock_image,
    }
    stack = contextlib.ExitStack()
    for target, mock_obj in patches.items():
        stack.enter_context(patch(target, mock_obj))
    yield {
        "torch": mock_torch,
        "model": mock_clip_model,
        "processor": mock_clip_processor,
        "hmean": mock_hmean,
        "model_cls": mock_clip_model_cls,
        "processor_cls": mock_clip_processor_cls,
    }
    stack.close()


@pytest.fixture
def err_metric(mock_err_deps):
    from eval_learn.metrics.err.metric import ERRMetric
    return ERRMetric(device="cpu")


class TestERRInit:
    def test_init_success(self, mock_err_deps):
        from eval_learn.metrics.err.metric import ERRMetric
        metric = ERRMetric(device="cpu")
        mock_err_deps["model_cls"].from_pretrained.assert_called_once()
        mock_err_deps["processor_cls"].from_pretrained.assert_called_once()

    def test_init_missing_torch(self):
        with patch("eval_learn.metrics.err.metric.torch", None):
            from eval_learn.metrics.err.metric import ERRMetric
            with pytest.raises(RuntimeError, match="torch"):
                ERRMetric()

    def test_init_missing_transformers(self):
        mock_torch = MagicMock()
        with patch("eval_learn.metrics.err.metric.torch", mock_torch), \
             patch("eval_learn.metrics.err.metric.CLIPModel", None):
            from eval_learn.metrics.err.metric import ERRMetric
            with pytest.raises(RuntimeError, match="transformers"):
                ERRMetric()


class TestERRCompute:
    def test_compute_empty_images(self, err_metric):
        result = err_metric.compute([], [])
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_missing_metadata(self, err_metric, dummy_pil_image):
        result = err_metric.compute([dummy_pil_image()], ["p"], metadata=None)
        assert result.value == 0.0
        assert "error" in result.details
        assert "concepts" in result.details["error"]

    def test_compute_missing_concepts(self, err_metric, dummy_pil_image):
        result = err_metric.compute(
            [dummy_pil_image()], ["p"],
            metadata={"categories": ["target"]}
        )
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_missing_categories(self, err_metric, dummy_pil_image):
        result = err_metric.compute(
            [dummy_pil_image()], ["p"],
            metadata={"concepts": ["c1"]}
        )
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_length_mismatch(self, err_metric, dummy_pil_image):
        result = err_metric.compute(
            [dummy_pil_image(), dummy_pil_image()], ["p1", "p2"],
            metadata={"concepts": ["c1", "c2", "c3"], "categories": ["target", "target", "target"]}
        )
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_all_categories(self, err_metric, tmp_path, mock_err_deps):
        # Create actual image files so _resolve_image_path works
        paths = []
        for i in range(3):
            p = str(tmp_path / f"img_{i}.png")
            Image.new("RGB", (10, 10)).save(p)
            paths.append(p)

        # Mock _check_concept_presence to return controlled values
        err_metric._check_concept_presence = MagicMock(return_value=False)

        result = err_metric.compute(
            paths, ["p1", "p2", "p3"],
            metadata={
                "concepts": ["c1", "c2", "c3"],
                "categories": ["target", "retain", "adversarial"],
            }
        )
        assert result.name == "ERR"
        assert "forgetting" in result.details
        assert "retention" in result.details
        assert "adversarial" in result.details

    def test_compute_target_only(self, err_metric, tmp_path):
        p = str(tmp_path / "img.png")
        Image.new("RGB", (10, 10)).save(p)
        err_metric._check_concept_presence = MagicMock(return_value=False)
        result = err_metric.compute(
            [p], ["p1"],
            metadata={"concepts": ["c1"], "categories": ["target"]}
        )
        assert result.details.get("retention") is None
        assert result.details.get("adversarial") is None


class TestERRHelpers:
    def test_build_model_outputs(self, err_metric, tmp_path):
        paths = []
        for i in range(6):
            p = str(tmp_path / f"img_{i}.png")
            Image.new("RGB", (10, 10)).save(p)
            paths.append(p)

        outputs = err_metric._build_model_outputs(
            paths,
            ["c0", "c1", "c2", "c3", "c4", "c5"],
            ["target", "target", "retain", "retain", "adversarial", "adversarial"],
        )
        assert len(outputs["target"]) == 2
        assert len(outputs["retain"]) == 2
        assert len(outputs["adversarial"]) == 2

    def test_resolve_image_path_string(self, tmp_path):
        from eval_learn.metrics.err.metric import ERRMetric
        p = str(tmp_path / "test.png")
        Image.new("RGB", (10, 10)).save(p)
        result = ERRMetric._resolve_image_path(p)
        assert result == p

    def test_resolve_image_path_pil(self):
        from eval_learn.metrics.err.metric import ERRMetric
        img = Image.new("RGB", (10, 10))
        result = ERRMetric._resolve_image_path(img)
        assert isinstance(result, str)
        assert os.path.isfile(result)
        # Cleanup
        os.remove(result)

    def test_resolve_image_path_unknown(self):
        from eval_learn.metrics.err.metric import ERRMetric
        result = ERRMetric._resolve_image_path(42)
        assert result is None
