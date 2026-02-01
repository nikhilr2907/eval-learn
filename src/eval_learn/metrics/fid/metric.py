import os
import tempfile
from typing import List, Any, Dict, Optional
from ...types import MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import FIDConfig

logger = get_logger(__name__)

# Optional imports
try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy import linalg
except ImportError:
    linalg = None

try:
    from PIL import Image
except ImportError:
    Image = None


def _collect_image_paths(directory: str) -> List[str]:
    """Return sorted list of image file paths from a directory."""
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    paths = []
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in supported:
            paths.append(os.path.join(directory, fname))
    return paths


@register_metric("fid")
class FIDMetric:
    """
    Frechet Inception Distance (FID) Metric.
    Measures the quality and diversity of generated images compared to real images.
    Lower FID scores indicate better image quality.
    """

    def __init__(self, **kwargs):
        self.config = FIDConfig.from_dict(kwargs)
        self.inception_model = None

        # Validate required dependencies
        for name, mod in [("tensorflow", tf), ("numpy", np), ("scipy", linalg), ("Pillow", Image)]:
            if mod is None:
                raise RuntimeError(
                    f"FID metric requires '{name}'. "
                    f"Install with: pip install {name}"
                )

        # Discover real reference images
        if not self.config.real_images_dir:
            raise ValueError("FIDConfig.real_images_dir must be set to a directory of real reference images.")
        if not os.path.isdir(self.config.real_images_dir):
            raise FileNotFoundError(f"real_images_dir does not exist: {self.config.real_images_dir}")

        self.real_image_paths = _collect_image_paths(self.config.real_images_dir)
        if not self.real_image_paths:
            raise FileNotFoundError(f"No images found in real_images_dir: {self.config.real_images_dir}")

        logger.info(f"FIDMetric initialized with {len(self.real_image_paths)} real reference images.")

    # ------------------------------------------------------------------
    # InceptionV3 helpers (ported from legacy FIDMetric)
    # ------------------------------------------------------------------

    def _load_inception(self):
        """Load InceptionV3 model for feature extraction."""
        if self.inception_model is None:
            if not tf.config.list_physical_devices("GPU"):
                logger.warning("No GPU found for TensorFlow. FID calculation will be slow.")
            self.inception_model = tf.keras.applications.InceptionV3(
                include_top=False, weights="imagenet", pooling="avg", input_shape=(299, 299, 3),
            )
        return self.inception_model

    def _preprocess_image(self, path_or_pil):
        """Load and preprocess an image for InceptionV3."""
        if isinstance(path_or_pil, str):
            image = Image.open(path_or_pil).convert("RGB").resize((299, 299), Image.Resampling.BICUBIC)
        elif isinstance(path_or_pil, Image.Image):
            image = path_or_pil.convert("RGB").resize((299, 299), Image.Resampling.BICUBIC)
        else:
            raise ValueError(f"Unsupported image type: {type(path_or_pil)}")

        image = np.array(image)
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        return image

    def _get_activations(self, image_sources, model):
        """Extract InceptionV3 features from a list of image paths / PIL images."""
        n = len(image_sources)
        if n == 0:
            return np.array([])

        activation_batches = []
        for i in range(0, n, self.config.batch_size):
            batch = image_sources[i : i + self.config.batch_size]
            batch_images = np.stack([self._preprocess_image(p) for p in batch], axis=0)
            activations = model(batch_images, training=False).numpy()
            activation_batches.append(activations)

        return np.concatenate(activation_batches, axis=0)

    # ------------------------------------------------------------------
    # FID math (ported from legacy FIDMetric)
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_fid_score(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate Frechet Inception Distance between two Gaussian distributions."""
        mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
        sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.isclose(np.diagonal(covmean).imag, 0, atol=1e-3).all():
                return float("inf")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return float(fid)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(self, images: List[Any], prompts: List[str], metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        """
        Compute FID score between generated images and the real reference images.

        Args:
            images: List of generated images (file paths or PIL Images).
            prompts: List of prompts (unused by FID but required by metric interface).
            metadata: Optional metadata dict.

        Returns:
            MetricResult with the FID score (lower is better).
        """
        if not images:
            return MetricResult(name="FID", value=float("inf"), details={"error": "No images provided"})

        logger.info(f"Computing FID: {len(self.real_image_paths)} real vs {len(images)} generated images...")

        try:
            model = self._load_inception()

            # Extract features for real images
            real_activations = self._get_activations(self.real_image_paths, model)
            # Extract features for generated images
            gen_activations = self._get_activations(images, model)

            mu_real = np.mean(real_activations, axis=0)
            sigma_real = np.cov(real_activations, rowvar=False)
            mu_gen = np.mean(gen_activations, axis=0)
            sigma_gen = np.cov(gen_activations, rowvar=False)

            fid_score = self._calculate_fid_score(mu_real, sigma_real, mu_gen, sigma_gen)
            logger.info(f"FID Score: {fid_score:.4f}")

            return MetricResult(
                name="FID",
                value=fid_score,
                details={
                    "total_generated": len(images),
                    "total_real": len(self.real_image_paths),
                    "config": self.config.to_dict(),
                },
            )
        except Exception as e:
            logger.exception("FID computation failed.")
            return MetricResult(
                name="FID",
                value=float("inf"),
                details={"error": str(e)},
            )
