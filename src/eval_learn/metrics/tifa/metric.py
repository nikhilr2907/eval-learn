from typing import List, Any, Dict, Optional
from ...types import MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import TIFAConfig

logger = get_logger(__name__)

# Optional imports
try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except ImportError:
    Blip2Processor = None
    Blip2ForConditionalGeneration = None

try:
    from PIL import Image
except ImportError:
    Image = None


@register_metric("tifa")
class TIFAMetric:
    """
    TIFA (Text-to-Image Faithfulness) Metric.

    Uses a BLIP-2 VQA model to verify whether generated images contain
    elements described in the original prompt. Each image is evaluated
    against a set of question-answer pairs; the score is the fraction
    of correctly answered questions.

    The ``compute()`` method expects ``metadata`` to contain:
    - ``qa_pairs``: list (parallel to ``images``) where each element is a
      list of dicts with ``"question"`` and ``"answer"`` keys.
    """

    def __init__(self, **kwargs):
        self.config = TIFAConfig.from_dict(kwargs)

        # Validate required dependencies
        for name, mod in [("torch", torch), ("transformers", Blip2Processor),
                          ("Pillow", Image)]:
            if mod is None:
                raise RuntimeError(
                    f"TIFA metric requires '{name}'. "
                    f"Install with: pip install {name}"
                )

        device_str = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device_str

        # Lazy-load the VQA model on first use to save VRAM when only
        # constructing the metric object (e.g. during registration checks).
        self._processor = None
        self._model = None

    # ------------------------------------------------------------------
    # VQA engine (ported from legacy VQAModel)
    # ------------------------------------------------------------------

    def _ensure_vqa_loaded(self):
        """Lazy-load the BLIP-2 VQA model."""
        if self._model is not None:
            return

        logger.info("Loading BLIP-2 VQA model '%s' on %s...",
                     self.config.vqa_model_name, self.device)
        self._processor = Blip2Processor.from_pretrained(self.config.vqa_model_name)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.config.vqa_model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self._model.eval()
        logger.info("BLIP-2 VQA model ready.")

    @torch.no_grad()
    def _answer(self, pil_image, question: str, max_new_tokens: int = 10) -> str:
        """Run VQA on a single PIL image and question."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        inputs = self._processor(
            images=pil_image,
            text=question,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Image loading helper
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pil(img) -> Optional["Image.Image"]: # type: ignore
        """Return a PIL Image from a path string or PIL Image."""
        if Image and isinstance(img, Image.Image):
            return img
        if isinstance(img, str):
            try:
                return Image.open(img).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                logger.warning("Could not load image %s: %s", img, e)
                return None
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        images: List[Any],
        prompts: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricResult:
        """
        Compute the TIFA score.

        Args:
            images: List of generated images (file paths or PIL Images).
            prompts: List of prompts used to generate the images.
            metadata: Must contain:
                - ``qa_pairs``: list parallel to *images*, where each
                  element is a list of ``{"question": str, "answer": str}``
                  dicts.

        Returns:
            MetricResult with the TIFA accuracy score (higher is better).
        """
        if not images:
            return MetricResult(name="TIFA", value=0.0, details={"error": "No images provided"})

        metadata = metadata or {}
        qa_pairs = metadata.get("qa_pairs")

        if not qa_pairs:
            return MetricResult(
                name="TIFA",
                value=0.0,
                details={"error": "metadata must contain 'qa_pairs' list"},
            )

        if len(qa_pairs) != len(images):
            return MetricResult(
                name="TIFA",
                value=0.0,
                details={"error": f"qa_pairs length ({len(qa_pairs)}) must match images length ({len(images)})"},
            )

        self._ensure_vqa_loaded()

        correct_count = 0
        total_count = 0
        per_image_scores: List[Optional[float]] = []

        logger.info("Computing TIFA for %d images...", len(images))

        for idx, (img, questions) in enumerate(zip(images, qa_pairs)):
            pil_img = self._load_pil(img)
            if pil_img is None:
                logger.warning("Skipping image %d: could not load.", idx)
                per_image_scores.append(None)
                continue

            if not questions:
                per_image_scores.append(None)
                continue

            img_correct = 0
            for qa in questions:
                question = qa.get("question", "")
                expected = qa.get("answer", "")
                if not question or not expected:
                    continue

                prediction = self._answer(pil_img, question)
                total_count += 1

                if prediction.lower().strip() == expected.lower().strip():
                    correct_count += 1
                    img_correct += 1

            img_total = len(questions)
            per_image_scores.append(img_correct / img_total if img_total > 0 else None)

        tifa_score = correct_count / total_count if total_count > 0 else 0.0
        logger.info("TIFA Score: %.4f (%d/%d correct)", tifa_score, correct_count, total_count)

        return MetricResult(
            name="TIFA",
            value=tifa_score,
            details={
                "correct": correct_count,
                "total_questions": total_count,
                "total_images": len(images),
                "per_image_scores": per_image_scores,
                "config": self.config.to_dict(),
            },
        )
