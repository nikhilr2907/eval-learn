import os
from typing import List, Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights

from ...types import Dataset, MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import FIDConfig

logger = get_logger(__name__)

# Preprocessing for InceptionV3: resize to 299x299 and normalize to [-1, 1]
_inception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),            # [0, 1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
])


def _load_inception(device: str) -> nn.Module:
    """Load InceptionV3 with the final pooling layer as output."""
    model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    # Remove the final FC layer — we want the 2048-dim pool features
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


def _pil_to_inception_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a preprocessed tensor for InceptionV3."""
    return _inception_transform(img.convert("RGB"))


@torch.no_grad()
def _get_activations(images: List[Image.Image], model: nn.Module, device: str, batch_size: int) -> np.ndarray:
    """Extract InceptionV3 pool features from a list of PIL images."""
    activations = []
    for i in range(0, len(images), batch_size):
        batch_pils = images[i : i + batch_size]
        batch = torch.stack([_pil_to_inception_tensor(img) for img in batch_pils]).to(device)
        features = model(batch)
        activations.append(features.cpu().numpy())
    return np.concatenate(activations, axis=0)


def _calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
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
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


@register_metric("fid")
class FIDMetric:
    """
    Frechet Inception Distance (FID) Metric.

    Uses PyTorch InceptionV3 for feature extraction — no TensorFlow required.
    Real images are loaded directly from the parquet into memory
    without writing to disk.
    """

    def __init__(self, **kwargs):
        self.config = FIDConfig.from_dict(kwargs)

        device = self.config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.inception_model = None
        self._real_activations = None
        self._real_count = 0

        logger.info(f"FIDMetric initialized (device={self.device}).")

    def _get_model(self) -> nn.Module:
        """Lazy-load InceptionV3."""
        if self.inception_model is None:
            logger.info("Loading InceptionV3...")
            self.inception_model = _load_inception(self.device)
        return self.inception_model

    def load_dataset(self) -> Dataset:
        """Stream COCO images from HuggingFace and extract real image features batch by batch.

        Iterates the DataLoader returned by ``load_coco_parquet``. Each batch
        yields pre-transformed tensors (inception-ready), passed directly to
        InceptionV3. No PIL images are held in memory beyond the current batch.
        Features are stored in ``self._real_activations`` for FID computation.
        """
        from ...datasets.coco_parquet import load_coco_parquet

        loader = load_coco_parquet(
            limit=self.config.limit,
            batch_size=self.config.batch_size,
        )

        model = self._get_model()
        logger.info("Extracting InceptionV3 features from real images via HF stream...")

        all_activations = []
        all_captions = []

        for dataset_batch in loader:
            # Tensors already inception-transformed by the DataLoader collate_fn
            image_batch = dataset_batch.metadata["images"].to(self.device)
            with torch.no_grad():
                features = model(image_batch)
            all_activations.append(features.cpu().numpy())
            all_captions.extend(dataset_batch.prompts)

        self._real_activations = np.concatenate(all_activations, axis=0)
        self._real_count = len(self._real_activations)
        logger.info("Extracted features for %d real images.", self._real_count)

        return Dataset(
            prompts=all_captions,
            metadata={"source": "coco_hf", "total_loaded": self._real_count},
        )

    def compute(self, images: List[Any], prompts: List[str], metadata: Optional[Dict[str, Any]] = None) -> MetricResult:
        """
        Compute FID score between generated images and the real reference images.

        Args:
            images: List of generated PIL Images (or file paths).
            prompts: List of prompts (unused by FID but required by metric interface).
            metadata: Optional metadata dict.

        Returns:
            MetricResult with the FID score (lower is better).
        """
        if not images:
            return MetricResult(name="FID", value=float("inf"), details={"error": "No images provided"})

        if self._real_activations is None:
            raise RuntimeError("Real images not loaded. Call load_dataset() before compute().")

        if len(images) < 2:
            return MetricResult(name="FID", value=float("inf"), details={"error": "At least 2 generated images are required to compute FID."})

        logger.info(f"Computing FID: {self._real_count} real vs {len(images)} generated images...")

        try:
            # Convert file paths to PIL if needed
            pil_images = []
            for img in images:
                if isinstance(img, str):
                    img = Image.open(img)
                pil_images.append(img)

            model = self._get_model()
            gen_activations = _get_activations(pil_images, model, self.device, self.config.batch_size)

            mu_real = np.mean(self._real_activations, axis=0)
            sigma_real = np.cov(self._real_activations, rowvar=False)
            mu_gen = np.mean(gen_activations, axis=0)
            sigma_gen = np.cov(gen_activations, rowvar=False)

            fid_score = _calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
            logger.info(f"FID Score: {fid_score:.4f}")

            return MetricResult(
                name="FID",
                value=fid_score,
                details={
                    "total_generated": len(images),
                    "total_real": self._real_count,
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
