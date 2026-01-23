"""ERR Benchmark Task - Downloads datasets, generates images, and evaluates ERR metric."""

import csv
import torch
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

from core.base_benchmark import BenchmarkTask
from core.base_technique import UnlearningTechnique
from .err import ERREvaluator
from unlearning_techniques.sld_pipeline.sld_wrapper import SLDWrapper


class ERRBenchmarkTask(BenchmarkTask):
    """ERR Benchmark: loads datasets, generates images, and evaluates ERR metric."""

    RAB_DATASET_HF = "Chia15/RingABell-Nudity"
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(
        self,
        technique: UnlearningTechnique,
        technique_name: str,
        base_dir: Optional[str] = None,
        num_target_prompts: int = 100,
        num_retain_prompts: int = 100,
        num_adversarial_prompts: int = 100,
        device: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="ERR_Benchmark")

        self.technique = technique
        self.technique_name = technique_name
        self.num_target_prompts = num_target_prompts
        self.num_retain_prompts = num_retain_prompts
        self.num_adversarial_prompts = num_adversarial_prompts
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_config = generation_config or {}

        # Setup paths
        self.base_dir = (
            Path(base_dir) if base_dir else Path(__file__).parent.parent.parent
        )
        self._setup_directories()
        self._initialize_evaluator()

        # Prompt storage
        self.target_prompts: List[str] = []
        self.retain_prompts: List[str] = []
        self.adversarial_prompts: List[str] = []

        # Generated images storage (for proper _calculate_metric integration)
        self.generated_images: Dict[str, List[Any]] = {
            "target": [],
            "retain": [],
            "adversarial": [],
        }

    def _setup_directories(self) -> None:
        """Create all necessary directories."""
        # Source data directories
        self.i2p_dir = self.base_dir / "data" / "i2p"
        self.rab_dir = self.base_dir / "data" / "ring_a_bell"
        self.err_data_dir = self.base_dir / "data" / "ERR"
        self.prompt_data_dir = self.err_data_dir / "prompt_data"

        # Image output directories
        self.image_base_dir = (
            self.base_dir
            / "results"
            / "benchmarks"
            / "ERR_benchmark"
            / "images"
            / self.technique_name
        )
        self.target_dir = self.image_base_dir / "target"
        self.retain_dir = self.image_base_dir / "retain"
        self.adversarial_dir = self.image_base_dir / "adversarial"

        # Source CSV paths (I2P already exists, Ring-A-Bell will be downloaded)
        self.i2p_csv_path = self.i2p_dir / "i2p_benchmark_sample.csv"
        self.rab_csv_path = self.rab_dir / "ring_a_bell_dataset.csv"
        self.good_dataset_path = (
            self.err_data_dir / "raw_csv_data" / "challenge_dataset.csv"
        )

        # Generated prompt CSV paths
        self.target_csv_path = self.prompt_data_dir / "target_prompts.csv"
        self.retain_csv_path = self.prompt_data_dir / "retain_prompts.csv"
        self.adversarial_csv_path = self.prompt_data_dir / "adversarial_prompts.csv"

        for d in [
            self.rab_dir,
            self.prompt_data_dir,
            self.target_dir,
            self.retain_dir,
            self.adversarial_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _initialize_evaluator(self) -> None:
        """Initialize CLIP model and ERR evaluator."""
        print(f"Initializing CLIP on {self.device}...")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.metric_calculator = ERREvaluator(
            model, processor, torch.device(self.device)
        )
        print("✓ ERR Evaluator ready")

    # =========================================================================
    # Dataset Download
    # =========================================================================

    def _load_i2p_dataset(self) -> List[Dict[str, str]]:
        """Load I2P dataset from existing CSV file."""
        if not self.i2p_csv_path.exists():
            print(f"  ❌ I2P CSV not found: {self.i2p_csv_path}")
            return []

        print(f"  Loading I2P from {self.i2p_csv_path}...")
        with open(self.i2p_csv_path, "r", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        print(f"    ✓ {len(data)} items loaded")
        return data

    def _load_rab_dataset(self) -> List[Dict[str, str]]:
        """Load Ring-A-Bell adversarial prompts dataset."""
        if self.rab_csv_path.exists():
            print(f"  Loading Ring-A-Bell from {self.rab_csv_path}...")
            with open(self.rab_csv_path, "r", encoding="utf-8") as f:
                data = list(csv.DictReader(f))
            print(f"    ✓ {len(data)} items loaded")
            return data

        print(f"  Downloading Ring-A-Bell from {self.RAB_DATASET_HF}...")
        try:
            dataset = load_dataset(self.RAB_DATASET_HF, split="train")

            self.rab_dir.mkdir(parents=True, exist_ok=True)
            with open(self.rab_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "prompt", "concept"])
                for idx, item in enumerate(dataset):
                    prompt = item.get("prompt", "") or item.get("text", "")
                    concept = item.get("concept", "") or item.get("category", "") or "nudity"
                    writer.writerow([idx, prompt, concept])

            print(f"    ✓ {len(dataset)} items downloaded and saved")
            return [
                {
                    "prompt": item.get("prompt", "") or item.get("text", ""),
                    "concept": item.get("concept", "") or item.get("category", "") or "nudity",
                }
                for item in dataset
            ]

        except Exception as e:
            print(f"  ⚠️ Failed to download Ring-A-Bell: {e}")
            return []

    def _load_datasets(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Load I2P and Ring-A-Bell datasets from local CSV files."""
        print("\n[STEP 1] Loading datasets")
        return self._load_i2p_dataset(), self._load_rab_dataset()

    # =========================================================================
    # Data Generation (CSV + Images)
    # =========================================================================

    def _save_image(self, img: Any, output_dir: Path, idx: int) -> Optional[str]:
        """Save a single image to directory. Returns path if successful, None if failed."""
        try:
            path = output_dir / f"{idx:04d}.png"
            img.save(path)
            return str(path)
        except Exception as e:
            print(f"    ⚠️ Image {idx}: {e}")
            return None

    def _write_csv(self, path: Path, data: List[Tuple[int, str, str, str]]) -> None:
        """Write prompt data to CSV."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "prompt", "section", "concept"])
            for row in data:
                writer.writerow(row)

    def _generate_category_data(
        self,
        data: List[Dict[str, str]],
        num_items: int,
        csv_path: Path,
        image_dir: Path,
        section: str,
        prompt_key: str = "prompt",
        concept_key: str = "categories",
    ) -> Tuple[List[str], List[str], List[Any]]:
        """Generate data for a category. Only successfully saved images are recorded."""
        items = min(num_items, len(data))

        input_prompts = [row.get(prompt_key, "") for row in data[:items]]
        input_concepts = [
            row.get(concept_key, "") or row.get("concept", "") for row in data[:items]
        ]

        print(f"  Generating {len(input_prompts)} images...")

        saved_prompts: List[str] = []
        saved_concepts: List[str] = []
        saved_images: List[Any] = []
        csv_data: List[Tuple[int, str, str, str]] = []

        try:
            images = self.technique.generate(input_prompts, **self.generation_config)

            for img, prompt, concept in zip(images, input_prompts, input_concepts):
                saved_path = self._save_image(img, image_dir, len(saved_images))
                if saved_path is not None:
                    saved_prompts.append(prompt)
                    saved_concepts.append(concept)
                    saved_images.append(img)
                    csv_data.append((len(csv_data), prompt, section, concept))

            self._write_csv(csv_path, csv_data)
            print(f"  ✓ {section}: {len(saved_prompts)}/{len(input_prompts)} images saved")

        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
            return [], [], []

        return saved_prompts, saved_concepts, saved_images

    def _generate_retention_data(self) -> Tuple[List[str], List[str], List[Any]]:
        """Generate retention data from challenge dataset."""
        print("\n[STEP 2] Generating RETENTION data")

        if not self.good_dataset_path.exists():
            print(f"  ❌ Challenge dataset not found: {self.good_dataset_path}")
            return [], [], []

        with open(self.good_dataset_path, "r", encoding="utf-8") as f:
            data = list(csv.DictReader(f))

        data = data[: self.num_retain_prompts]

        input_prompts = [r.get("direct_prompt") or r.get("prompt", "") for r in data]
        input_concepts = [r.get("concept_name") or r.get("concept", "") for r in data]

        print(f"  Generating {len(input_prompts)} images...")

        saved_prompts: List[str] = []
        saved_concepts: List[str] = []
        saved_images: List[Any] = []
        csv_data: List[Tuple[int, str, str, str]] = []

        try:
            images = self.technique.generate(input_prompts, **self.generation_config)

            for img, prompt, concept in zip(images, input_prompts, input_concepts):
                saved_path = self._save_image(img, self.retain_dir, len(saved_images))
                if saved_path is not None:
                    saved_prompts.append(prompt)
                    saved_concepts.append(concept)
                    saved_images.append(img)
                    csv_data.append((len(csv_data), prompt, "retain", concept))

            self._write_csv(self.retain_csv_path, csv_data)
            print(f"  ✓ retain: {len(saved_prompts)}/{len(input_prompts)} images saved")

        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
            return [], [], []

        return saved_prompts, saved_concepts, saved_images

    def _generate_target_data(
        self, i2p_data: List[Dict[str, str]]
    ) -> Tuple[List[str], List[str], List[Any]]:
        """Generate target (forgetting) data from I2P dataset."""
        print("\n[STEP 3] Generating TARGET data")
        if not i2p_data:
            print("  ❌ No I2P data available")
            return [], [], []
        return self._generate_category_data(
            i2p_data,
            self.num_target_prompts,
            self.target_csv_path,
            self.target_dir,
            "target",
            concept_key="categories",
        )

    def _generate_adversarial_data(
        self, rab_data: List[Dict[str, str]]
    ) -> Tuple[List[str], List[str], List[Any]]:
        """Generate adversarial data from Ring-A-Bell dataset."""
        print("\n[STEP 4] Generating ADVERSARIAL data")
        if not rab_data:
            print("  ⚠️ Ring-A-Bell unavailable, skipping")
            return [], [], []
        return self._generate_category_data(
            rab_data,
            self.num_adversarial_prompts,
            self.adversarial_csv_path,
            self.adversarial_dir,
            "adversarial",
            concept_key="concept",
        )

    # =========================================================================
    # Evaluation
    # =========================================================================

    def _load_data(self) -> List[str]:
        """Load prompts from CSV files. Required by BenchmarkTask."""

        def load_csv(path: Path) -> List[str]:
            if not path.exists():
                return []
            with open(path, "r", encoding="utf-8") as f:
                return [
                    r.get("prompt", "") for r in csv.DictReader(f) if r.get("prompt")
                ]

        self.target_prompts = load_csv(self.target_csv_path)
        self.retain_prompts = load_csv(self.retain_csv_path)
        self.adversarial_prompts = load_csv(self.adversarial_csv_path)

        return self.target_prompts + self.retain_prompts + self.adversarial_prompts

    def _get_image_concept_pairs(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load image-concept pairs from CSV and image directories."""

        def load_pairs(csv_path: Path, image_dir: Path) -> List[Tuple[str, str]]:
            if not csv_path.exists() or not image_dir.exists():
                return []

            concepts = {}
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("index") and row.get("concept"):
                        concepts[int(row["index"])] = row["concept"]

            images = sorted(
                p
                for p in image_dir.iterdir()
                if p.suffix.lower() in self.IMAGE_EXTENSIONS
            )
            return [(str(img), concepts.get(idx, "")) for idx, img in enumerate(images)]

        return {
            "target": load_pairs(self.target_csv_path, self.target_dir),
            "retain": load_pairs(self.retain_csv_path, self.retain_dir),
            "adversarial": load_pairs(self.adversarial_csv_path, self.adversarial_dir),
        }

    def _calculate_metric(
        self, generated_images: List[Any], prompts: List[str]
    ) -> Dict[str, Any]:
        """Calculate ERR metric. Required by BenchmarkTask."""
        print("\n[STEP 5] Calculating ERR metric")

        data = self._get_image_concept_pairs()
        print(
            f"  Pairs - Target: {len(data['target'])}, Retain: {len(data['retain'])}, Adversarial: {len(data['adversarial'])}"
        )

        if not any(data.values()):
            print("  ❌ No data found!")
            return {"ERR_Score": 0.0, "Details": {}}

        result = self.metric_calculator.calculate_err(data)
        print(f"  ✓ ERR Score: {result['ERR_Score']:.4f}")
        print(
            f"    Forgetting: {result['Details']['Forgetting']:.4f}, "
            f"Retention: {result['Details']['Retention']:.4f}, "
            f"Adversarial: {result['Details']['Adversarial']:.4f}"
        )
        return result

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def run(self) -> Dict[str, Any]:
        """Run complete ERR benchmark pipeline."""
        print("=" * 60)
        print(f"ERR BENCHMARK: {self.technique_name}")
        print("=" * 60)

        # Load datasets
        i2p, rab = self._load_datasets()
        if not i2p:
            print("\n❌ Failed to load I2P dataset")
            return {"ERR_Score": 0.0, "Details": {}}

        # Generate data for each category (concepts are saved to CSV, loaded later)
        target_prompts, _, target_images = self._generate_target_data(i2p)
        retain_prompts, _, retain_images = self._generate_retention_data()
        adv_prompts, _, adv_images = self._generate_adversarial_data(rab)

        # Store for later access
        self.target_prompts = target_prompts
        self.retain_prompts = retain_prompts
        self.adversarial_prompts = adv_prompts

        self.generated_images = {
            "target": target_images,
            "retain": retain_images,
            "adversarial": adv_images,
        }

        all_prompts = target_prompts + retain_prompts + adv_prompts
        all_images = target_images + retain_images + adv_images

        # Calculate metric
        result = self._calculate_metric(all_images, all_prompts)

        # Summary
        total = len(target_prompts) + len(retain_prompts) + len(adv_prompts)
        print("\n" + "=" * 60)
        print(f"COMPLETE: {total} prompts, ERR Score: {result['ERR_Score']:.4f}")
        print("=" * 60)

        return result


if __name__ == "__main__":
    # Example usage

    sld_wrapper = SLDWrapper()
    benchmark = ERRBenchmarkTask(
        technique=sld_wrapper,
        technique_name="SLD",
        num_target_prompts=1,
        num_retain_prompts=0,
        num_adversarial_prompts=0,
        generation_config={},
    )
    result = benchmark.run()
    print(f"Final result: {result}")
