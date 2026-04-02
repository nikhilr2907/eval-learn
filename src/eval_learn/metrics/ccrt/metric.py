import gc
import os
from typing import List, Any, Dict, Optional

import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader

from ...types import MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import CCRTConfig

logger = get_logger(__name__)

try:
    from ccrt.genetic.search import run_genetic_search
    from ccrt.genetic.llm_prompts import generate_prompts
    from ccrt.scoring.llm_eval import evaluate_style
except ImportError as e:
    raise RuntimeError(
        "CCRT metric requires the 'ccrt' package. "
        "Install it before running this metric."
    ) from e

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None


@register_metric("ccrt")
class CCRTMetric:
    """
    Cross-Category Retention Test (CCRT) Metric.

    load_dataset() runs the genetic concept search and LLM prompt generation,
    generates baseline images from the original model, then returns a DataLoader
    of the resulting prompt batches for generation by the erased technique.

    update() accumulates generated images alongside their prompts and seeds.
    compute() calls _calculate_ccrt() — currently a stub pending full implementation.
    """

    def __init__(self, **kwargs):
        self.config = CCRTConfig.from_dict(kwargs)

        for name, mod in [("transformers", CLIPModel), ("Pillow", PILImage)]:
            if mod is None:
                raise RuntimeError(
                    f"CCRT metric requires '{name}'. "
                    f"Install with: pip install {name}"
                )

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(
            f"Initializing CLIP model '{self.config.clip_model_name}' on {self.device}..."
        )
        self.model = CLIPModel.from_pretrained(self.config.clip_model_name).to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
        self.model.eval()

        self._baseline_images: List[Any] = []
        self._reference_images: List[Any] = []
        self._pending_images: List[Any] = []
        self._pending_prompts: List[str] = []
        self._pending_seeds: List[Any] = []
        logger.info("CCRTMetric ready.")

    def load_dataset(self) -> DataLoader:
        """
        Run genetic search, generate LLM prompts, generate baseline images,
        then return a DataLoader of prompt batches for the erased technique.

        Steps:
          1. Load reference images from config.reference_imgs.
          2. Run genetic concept search — finds concepts where the erased model's
             noise predictions diverge most from the original.
          3. Generate prompts from survivors via GPT-3.5.
          4. Generate baseline images from the original (unmodified) model.
          5. Unload baseline pipeline and clear VRAM.
          6. Return a DataLoader of (prompt, seed) batches.
        """
        self._pending_images = []
        self._pending_prompts = []
        self._pending_seeds = []

        os.makedirs(self.config.output_dir, exist_ok=True)

        # --- 1. Reference images ---
        if not os.path.isdir(self.config.reference_imgs):
            raise FileNotFoundError(
                f"reference_imgs directory not found: {self.config.reference_imgs}"
            )
        ref_paths = sorted(
            [
                os.path.join(self.config.reference_imgs, f)
                for f in os.listdir(self.config.reference_imgs)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        if len(ref_paths) < 3:
            raise ValueError(
                f"At least 3 reference images required in {self.config.reference_imgs}, "
                f"found {len(ref_paths)}."
            )
        self._reference_images = [PILImage.open(p).convert("RGB") for p in ref_paths]
        logger.info("Loaded %d reference images.", len(self._reference_images))

        # --- 2. Genetic search ---
        logger.info(
            f"Running genetic search: original={self.config.original_model_id}, "
            f"erased={self.config.erased_model_id}, concept={self.config.concept_name}"
        )
        entities = run_genetic_search(
            original_model_id=self.config.original_model_id,
            erased_model_id=self.config.erased_model_id,
            concept_name=self.config.concept_name,
            vocab_dir=self.config.vocab_dir,
            output_dir=self.config.output_dir,
            max_iterations=self.config.genetic_iterations,
            top_k=self.config.genetic_top_k,
        )
        logger.info("Genetic search complete: %d survivors.", len(entities))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 3. LLM prompt generation ---
        logger.info("Generating prompts via LLM...")
        prompts, seeds = generate_prompts(
            entities=entities,
            api_key=self.config.llm_api_key,
            output_dir=self.config.output_dir,
            limit=self.config.limit,
        )
        logger.info("Generated %d prompts.", len(prompts))

        # --- 4. Baseline image generation ---
        logger.info(
            f"Generating baseline images from original model {self.config.original_model_id}..."
        )
        from ...techniques.free_run.wrapper import FreeRunTechnique

        baseline_technique = FreeRunTechnique(model_id=self.config.original_model_id)
        self._baseline_images = baseline_technique.generate(prompts=prompts)
        logger.info("Generated %d baseline images.", len(self._baseline_images))

        # --- 5. Unload baseline pipeline ---
        del baseline_technique
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Baseline pipeline unloaded.")

        # --- 6. Build DataLoader via the datasets module ---
        from ...datasets.ccrt_genetic import load_ccrt_genetic

        return load_ccrt_genetic(
            prompts=prompts,
            seeds=seeds,
            concept_name=self.config.concept_name,
            concept_desc=self.config.concept_desc,
            batch_size=getattr(self.config, "batch_size", len(prompts)),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        images: List[Any],
        _prompts: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Accumulate a batch of generated images and their metadata.

        Args:
            images:   Generated PIL Images from the erased technique.
            prompts:  Prompts used to generate the images.
            metadata: Must contain ``seeds`` parallel to images.
        """
        metadata = metadata or {}
        seeds = metadata.get("seeds", [None] * len(images))
        self._pending_images.extend(images)
        self._pending_prompts.extend(_prompts)
        self._pending_seeds.extend(seeds)

    def _calculate_ccrt(
        self, images: List[Any], _prompts: List[str], _seeds: List[Any]
    ) -> Dict[str, Any]:
        """
        Score concept erasure using GPT-4V style evaluation.

        Calls evaluate_style() from the ccrt package, which asks GPT-4V whether
        each generated image still visually exhibits the target concept's style,
        using self._reference_images as few-shot exemplars.

        CCRT_Score = 1 - style_precision
          style_precision: fraction of images still showing the concept (lower = better erasure)
          CCRT_Score:      fraction of images where the concept is successfully erased (higher = better)

        Args:
            images:   Generated PIL Images from the erased technique.
            _prompts: Unused — scoring is purely visual via GPT-4V.
            _seeds:   Unused.

        Returns:
            Dict with ``"CCRT_Score"`` and ``"style_precision"`` keys.
        """

        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(PILImage.open(img).convert("RGB"))
            elif isinstance(img, PILImage.Image):
                pil_images.append(img.convert("RGB"))
            else:
                logger.warning("Skipping image of unexpected type %s", type(img))

        if not pil_images:
            return {"CCRT_Score": 0.0, "style_precision": 0.0, "evaluated": 0}

        logger.info(
            f"Running GPT-4V style evaluation on {len(pil_images)} images for concept '{self.config.concept_name}'..."
        )

        style_precision = evaluate_style(
            generated_images=pil_images,
            reference_images=self._reference_images,
            concept_name=self.config.concept_name,
            concept_desc=self.config.concept_desc,
            api_key=self.config.llm_api_key,
        )

        ccrt_score = 1.0 - style_precision

        logger.info(
            f"CCRT: style_precision={style_precision:.4f}  →  CCRT_Score={ccrt_score:.4f}"
        )

        return {
            "CCRT_Score": ccrt_score,
            "style_precision": style_precision,
            "evaluated": len(pil_images),
        }

    def compute(self) -> MetricResult:
        """
        Compute CCRT score from all accumulated update() calls.
        """
        if not self._pending_images:
            return MetricResult(
                name="CCRT", value=0.0, details={"error": "No images accumulated"}
            )

        logger.info("Computing CCRT for %d images...", len(self._pending_images))

        try:
            result = self._calculate_ccrt(
                self._pending_images, self._pending_prompts, self._pending_seeds
            )
            score = result["CCRT_Score"]
            logger.info("CCRT Score: %.4f", score)

            return MetricResult(
                name="CCRT",
                value=score,
                details={
                    **{k: v for k, v in result.items() if k != "CCRT_Score"},
                    "config": self.config.to_dict(),
                },
            )
        except Exception as e:
            logger.exception("CCRT computation failed.")
            raise
