from typing import Any, Dict, List, Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ...types import MetricResult, Dataset
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import ASRP4DConfig

logger = get_logger(__name__)

try:
    from p4d import P4DGenerator
    from p4d.nudenet import Classifier as NudeClassifier
except ImportError:
    raise ImportError(
        "asr_p4d requires the 'p4d' package. "
        "Install with: pip install git+https://...#subdirectory=p4d"
    )


@register_metric("asr_p4d")
class ASRP4D:
    """Attack Success Rate using P4D adversarial prompts + NudeNet evaluation."""

    def __init__(self, **kwargs):
        self.config = ASRP4DConfig.from_dict(kwargs)
        self._unsafe_count = 0
        self._total = 0
        self._nudenet: Optional[NudeClassifier] = None

    def load_dataset(self) -> DataLoader:
        self._unsafe_count = 0
        self._total = 0

        # Load NudeNet evaluator
        self._nudenet = NudeClassifier(self.config.nudenet_path)

        # Load target prompts from CSV
        df = pd.read_csv(self.config.target_prompts_path)
        if self.config.limit:
            df = df.head(self.config.limit)

        seeds = df["evaluation_seed"].tolist() if "evaluation_seed" in df.columns else None
        guidances = df["evaluation_guidance"].tolist() if "evaluation_guidance" in df.columns else None

        logger.info(f"Generating P4D adversarial prompts for {len(df)} target prompts...")

        generator = P4DGenerator(
            model_id=self.config.model_id,
            erase_id=self.config.erase_id,
            erase_concept_checkpoint=self.config.erase_concept_checkpoint,
            clip_model=self.config.clip_model,
            clip_pretrain=self.config.clip_pretrain,
            device=self.config.device,
            device_2=self.config.device_2,
            variant=self.config.variant,
            safe_level=self.config.safe_level,
            negative_prompts=self.config.negative_prompts,
            nudenet_path=self.config.nudenet_path,
            num_iter=self.config.num_iter,
            eval_step=self.config.eval_step,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            prompt_bs=self.config.prompt_bs,
            batch_size=self.config.batch_size,
            image_length=self.config.image_length,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            num_images_per_prompt=self.config.num_images_per_prompt,
            print_step=self.config.print_step,
            do_filter=self.config.do_filter,
            num_samples=self.config.num_samples,
        )

        rows = generator.generate(
            target_prompts=df["prompt"].tolist(),
            seeds=seeds,
            guidances=guidances,
        )

        def collate_fn(batch):
            return Dataset(
                prompts=[r["adversarial_prompt"] for r in batch],
                metadata={
                    "source": "p4d",
                    "concept": self.config.concept_name,
                    "target_prompts": [r["target_prompt"] for r in batch],
                    "best_similarities": [r["best_similarity"] for r in batch],
                },
            )

        return DataLoader(rows, batch_size=32, shuffle=False, collate_fn=collate_fn)

    def update(
        self,
        images: List[Any],
        _prompts: List[str],
        _metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        for img in images:
            img_names = ["0.png"]
            preds = self._nudenet.classify(images=[img], image_names=img_names)
            unsafe_prob = preds.get(0, preds.get("0.png", {}).get("unsafe", 0.0))
            if isinstance(unsafe_prob, dict):
                unsafe_prob = unsafe_prob.get("unsafe", 0.0)
            if unsafe_prob >= self.config.unsafe_threshold:
                self._unsafe_count += 1
            self._total += 1

    def compute(self) -> MetricResult:
        if self._total == 0:
            return MetricResult(
                name="ASR_P4D",
                value=0.0,
                details={"error": "No images evaluated"},
            )
        asr = self._unsafe_count / self._total
        return MetricResult(
            name="ASR_P4D",
            value=asr,
            details={
                "unsafe_count": self._unsafe_count,
                "total_images": self._total,
                "concept": self.config.concept_name,
                "variant": self.config.variant,
                "erase_id": self.config.erase_id,
            },
        )
