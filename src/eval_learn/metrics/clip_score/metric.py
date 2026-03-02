from typing import List, Any, Dict, Optional
from torch.utils.data import DataLoader
from ...types import MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import CLIPScoreConfig

logger = get_logger(__name__)

try:
    import torch
except ImportError:
    torch = None

try:
    from torchmetrics.multimodal.clip_score import CLIPScore
except ImportError:
    CLIPScore = None

try:
    from torchvision import transforms
except ImportError:
    transforms = None

try:
    from PIL import Image
except ImportError:
    Image = None


@register_metric("clip_score")
class CLIPScoreMetric:
    """
    CLIP Score Metric.

    Measures text-to-image alignment via cosine similarity between CLIP
    embeddings of each generated image and its prompt.

    update() runs the CLIP forward pass immediately and accumulates a running
    score total + count. compute() returns the average — no images are retained.
    """

    def __init__(self, **kwargs):
        self.config = CLIPScoreConfig.from_dict(kwargs)

        for name, mod in [("torch", torch), ("torchmetrics", CLIPScore),
                          ("torchvision", transforms), ("Pillow", Image)]:
            if mod is None:
                raise RuntimeError(
                    f"CLIP Score metric requires '{name}'. "
                    f"Install with: pip install {name}"
                )

        device_str = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device_str

        logger.info("Loading CLIPScore model '%s' on %s...", self.config.clip_model_name, self.device)
        self._clip_score_fn = CLIPScore(model_name_or_path=self.config.clip_model_name).to(self.device)

        self._total_score = 0.0
        self._evaluated = 0
        self._total_images = 0
        self._per_image_scores: List[Optional[float]] = []
        logger.info("CLIPScoreMetric ready.")

    def load_dataset(self) -> DataLoader:
        """Return a DataLoader over the TIFA dataset."""
        from ...datasets.tifa_json import load_tifa_json

        self._total_score = 0.0
        self._evaluated = 0
        self._total_images = 0
        self._per_image_scores = []

        return load_tifa_json(limit=self.config.limit)

    def _to_uint8_tensor(self, img) -> Optional["torch.Tensor"]:
        """Convert a PIL Image or file path to a uint8 tensor on device."""
        pil_img = None
        if Image and isinstance(img, Image.Image):
            pil_img = img
        elif isinstance(img, str):
            try:
                pil_img = Image.open(img).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                logger.warning("Could not load image %s: %s", img, e)
                return None
        if pil_img is None:
            return None
        tensor = transforms.ToTensor()(pil_img)
        return (tensor * 255).to(torch.uint8).to(self.device)

    def update(self, images: List[Any], prompts: List[str], _metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Run CLIP on each image-prompt pair and accumulate the running score total.

        Args:
            images:    Generated PIL Images or file paths.
            prompts:   Text prompts parallel to images.
            _metadata: Unused.
        """
        for img, prompt in zip(images, prompts):
            tensor = self._to_uint8_tensor(img)
            if tensor is None:
                logger.warning("Skipping image at index %d: could not load.", self._total_images)
                self._per_image_scores.append(None)
                self._total_images += 1
                continue

            try:
                score_val = self._clip_score_fn(tensor, prompt).item()
                self._per_image_scores.append(score_val)
                self._total_score += score_val
                self._evaluated += 1
            except Exception as e:
                logger.error("Error scoring image %d: %s", self._total_images, e)
                self._per_image_scores.append(None)

            self._total_images += 1

    def compute(self) -> MetricResult:
        """
        Return average CLIP score across all evaluated image-prompt pairs.
        All CLIP inference was done in update() — this is division only.
        """
        if self._total_images == 0:
            return MetricResult(name="CLIPScore", value=0.0, details={"error": "No images evaluated"})

        avg_score = self._total_score / self._evaluated if self._evaluated > 0 else 0.0
        logger.info(
            "CLIP Score: %.4f (evaluated %d/%d)", avg_score, self._evaluated, self._total_images
        )

        return MetricResult(
            name="CLIPScore",
            value=avg_score,
            details={
                "per_image_scores": self._per_image_scores,
                "evaluated": self._evaluated,
                "total_images": self._total_images,
                "config": self.config.to_dict(),
            },
        )
