import os
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np


@pytest.fixture
def real_images_dir(tmp_path):
    """Create a temporary directory with 3 PNG files."""
    for i in range(3):
        img = Image.new("RGB", (10, 10), color="blue")
        img.save(tmp_path / f"real_{i}.png")
    return str(tmp_path)


@pytest.fixture
def fid_patches():
    """Patch all optional deps at module level for FIDMetric."""
    mock_tf = MagicMock()
    mock_np = MagicMock()
    mock_linalg = MagicMock()
    mock_image = MagicMock()
    # Make isinstance checks work for PIL Image
    mock_image.Image = Image.Image

    patches = [
        patch("eval_learn.metrics.fid.metric.tf", mock_tf),
        patch("eval_learn.metrics.fid.metric.np", mock_np),
        patch("eval_learn.metrics.fid.metric.linalg", mock_linalg),
        patch("eval_learn.metrics.fid.metric.Image", mock_image),
    ]
    for p in patches:
        p.start()
    yield {"tf": mock_tf, "np": mock_np, "linalg": mock_linalg, "image": mock_image}
    for p in patches:
        p.stop()


class TestFIDInit:
    def test_init_valid_dir(self, real_images_dir, fid_patches):
        from eval_learn.metrics.fid.metric import FIDMetric
        metric = FIDMetric(real_images_dir=real_images_dir)
        assert len(metric.real_image_paths) == 3

    def test_init_missing_dir(self, fid_patches):
        from eval_learn.metrics.fid.metric import FIDMetric
        with pytest.raises(FileNotFoundError):
            FIDMetric(real_images_dir="/nonexistent/dir")

    def test_init_empty_dir(self, tmp_path, fid_patches):
        from eval_learn.metrics.fid.metric import FIDMetric
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        with pytest.raises(FileNotFoundError, match="No images"):
            FIDMetric(real_images_dir=empty_dir)

    def test_init_no_real_images_dir(self, fid_patches):
        from eval_learn.metrics.fid.metric import FIDMetric
        with pytest.raises(ValueError):
            FIDMetric(real_images_dir="")

    def test_init_missing_tensorflow(self):
        with patch("eval_learn.metrics.fid.metric.tf", None):
            from eval_learn.metrics.fid.metric import FIDMetric
            with pytest.raises(RuntimeError, match="tensorflow"):
                FIDMetric(real_images_dir="/some/dir")


class TestFIDCompute:
    def test_compute_empty_images(self, real_images_dir, fid_patches):
        from eval_learn.metrics.fid.metric import FIDMetric
        metric = FIDMetric(real_images_dir=real_images_dir)
        result = metric.compute([], [])
        assert result.value == float("inf")
        assert "error" in result.details

    def test_compute_returns_score(self, real_images_dir, fid_patches):
        from eval_learn.metrics.fid.metric import FIDMetric
        metric = FIDMetric(real_images_dir=real_images_dir)

        # Mock the internal methods to return controlled numpy arrays
        feat = np.random.randn(3, 2048)
        metric._load_inception = MagicMock(return_value=MagicMock())
        metric._get_activations = MagicMock(return_value=feat)

        # Use real numpy for the calculation
        with patch("eval_learn.metrics.fid.metric.np", np):
            from scipy import linalg as real_linalg
            with patch("eval_learn.metrics.fid.metric.linalg", real_linalg):
                result = metric.compute(["img1.png", "img2.png"], ["p1", "p2"])

        assert result.name == "FID"
        assert isinstance(result.value, float)
        assert result.details["total_generated"] == 2
        assert result.details["total_real"] == 3


class TestFIDCollectImagePaths:
    def test_collect_filters_by_extension(self, tmp_path):
        from eval_learn.metrics.fid.metric import _collect_image_paths
        # Create various files
        Image.new("RGB", (2, 2)).save(tmp_path / "a.png")
        Image.new("RGB", (2, 2)).save(tmp_path / "b.jpg")
        Image.new("RGB", (2, 2)).save(tmp_path / "c.bmp")
        (tmp_path / "d.txt").write_text("not an image")
        paths = _collect_image_paths(str(tmp_path))
        assert len(paths) == 3
        assert all(not p.endswith(".txt") for p in paths)
