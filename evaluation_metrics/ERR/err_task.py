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
    """
    Complete ERR Benchmark: loads local datasets, generates images, and evaluates ERR metric.
    Inherits from BenchmarkTask for framework integration.
    
    - I2P: Loaded from existing local CSV (data/i2p/i2p_benchmark_sample.csv)
    - Ring-A-Bell: Downloaded from HuggingFace if not cached
    """
    
    RAB_DATASET = "AIML-TUDA/ring-a-bell"
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    
    def __init__(
        self,
        technique: UnlearningTechnique,
        technique_name: str,
        base_dir: Optional[str] = None,
        num_target_prompts: int = 100,
        num_retain_prompts: int = 100,
        num_adversarial_prompts: int = 100,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(name="ERR_Benchmark")
        
        self.technique = technique
        self.technique_name = technique_name
        self.num_target_prompts = num_target_prompts
        self.num_retain_prompts = num_retain_prompts
        self.num_adversarial_prompts = num_adversarial_prompts
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent.parent
        self._setup_directories()
        self._initialize_evaluator()
        
        # Prompt storage
        self.target_prompts: List[str] = []
        self.retain_prompts: List[str] = []
        self.adversarial_prompts: List[str] = []
    
    def _setup_directories(self) -> None:
        """Create all necessary directories."""
        # Source data directories
        self.i2p_dir = self.base_dir / "data" / "i2p"
        self.rab_dir = self.base_dir / "data" / "ring_a_bell"
        self.err_data_dir = self.base_dir / "data" / "ERR"
        self.prompt_data_dir = self.err_data_dir / "prompt_data"
        
        # Image output directories
        self.image_base_dir = self.base_dir / "results" / "benchmarks" / "ERR_benchmark" / "images" / self.technique_name
        self.target_dir = self.image_base_dir / "target"
        self.retain_dir = self.image_base_dir / "retain"
        self.adversarial_dir = self.image_base_dir / "adversarial"
        
        # Source CSV paths (I2P already exists, Ring-A-Bell will be downloaded)
        self.i2p_csv_path = self.i2p_dir / "i2p_benchmark_sample.csv"
        self.rab_csv_path = self.rab_dir / "ring_a_bell_dataset.csv"
        self.good_dataset_path = self.err_data_dir / "raw_csv_data" / "challenge_dataset.csv"
        
        # Generated prompt CSV paths
        self.target_csv_path = self.prompt_data_dir / "target_prompts.csv"
        self.retain_csv_path = self.prompt_data_dir / "retain_prompts.csv"
        self.adversarial_csv_path = self.prompt_data_dir / "adversarial_prompts.csv"
        
        for d in [self.rab_dir, self.prompt_data_dir, self.target_dir, self.retain_dir, self.adversarial_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _initialize_evaluator(self) -> None:
        """Initialize CLIP model and ERR evaluator."""
        print(f"Initializing CLIP on {self.device}...")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.metric_calculator = ERREvaluator(model, processor, torch.device(self.device))
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
        with open(self.i2p_csv_path, 'r', encoding='utf-8') as f:
            data = list(csv.DictReader(f))
        print(f"    ✓ {len(data)} items loaded")
        return data
    
    def _download_rab_dataset(self) -> List[Dict[str, str]]:
        """Download Ring-A-Bell dataset from HuggingFace and save as CSV."""
        if self.rab_csv_path.exists():
            print(f"  Ring-A-Bell already exists, loading...")
            with open(self.rab_csv_path, 'r', encoding='utf-8') as f:
                data = list(csv.DictReader(f))
            print(f"    ✓ {len(data)} items loaded")
            return data
        
        try:
            print(f"  Downloading {self.RAB_DATASET}...")
            dataset = load_dataset(self.RAB_DATASET, split="train")
            
            with open(self.rab_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["index", "prompt", "concept"])
                for idx, item in enumerate(dataset):
                    prompt = item.get('prompt', '')
                    concept = item.get('concept', '') or item.get('category', '')
                    writer.writerow([idx, prompt, concept])
            
            print(f"    ✓ {len(dataset)} items saved to {self.rab_csv_path}")
            # Return as list of dicts for consistency
            return [{'prompt': item.get('prompt', ''), 'concept': item.get('concept', '') or item.get('category', '')} 
                    for item in dataset]
        except Exception as e:
            print(f"    ⚠️ {e}")
            return []
    
    def _load_datasets(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Load I2P (from existing CSV) and download Ring-A-Bell."""
        print("\n[STEP 1] Loading datasets")
        return self._load_i2p_dataset(), self._download_rab_dataset()
    
    # =========================================================================
    # Data Generation (CSV + Images)
    # =========================================================================
    
    def _save_images(self, images: List[Any], output_dir: Path) -> int:
        """Save images to directory, return count saved."""
        saved = 0
        for idx, img in enumerate(images):
            try:
                img.save(output_dir / f"{idx:04d}.png")
                saved += 1
            except Exception as e:
                print(f"    ⚠️ Image {idx}: {e}")
        return saved
    
    def _write_csv(self, path: Path, prompts: List[str], concepts: List[str], section: str) -> None:
        """Write prompts and concepts to CSV."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "prompt", "section", "concept"])
            for idx, (p, c) in enumerate(zip(prompts, concepts)):
                writer.writerow([idx, p, section, c])
    
    def _generate_category_data(
        self, data: List[Dict[str, str]], num_items: int, csv_path: Path, image_dir: Path,
        section: str, config: Optional[Dict], prompt_key: str = "prompt",
        concept_key: str = "categories"
    ) -> List[str]:
        """Generic method to generate data for any category."""
        items = min(num_items, len(data))
        
        prompts = [row.get(prompt_key, "") for row in data[:items]]
        concepts = [row.get(concept_key, "") or row.get("concept", "") for row in data[:items]]
        
        # Write CSV
        self._write_csv(csv_path, prompts, concepts, section)
        
        # Generate and save images
        print(f"  Generating {len(prompts)} images...")
        try:
            images = self.technique.generate(prompts, **(config or {}))
            saved = self._save_images(images, image_dir)
            print(f"  ✓ {section}: {len(prompts)} prompts, {saved} images")
        except Exception as e:
            print(f"  ❌ {e}")
            return []
        
        return prompts
    
    def _generate_retention_data(self, config: Optional[Dict] = None) -> List[str]:
        """Generate retention data from challenge dataset."""
        print("\n[STEP 2] Generating RETENTION data")
        
        if not self.good_dataset_path.exists():
            print(f"  ❌ Challenge dataset not found: {self.good_dataset_path}")
            return []
        
        with open(self.good_dataset_path, 'r', encoding='utf-8') as f:
            data = list(csv.DictReader(f))
        
        data = data[:self.num_retain_prompts]
        
        prompts = [r.get("direct_prompt") or r.get("prompt", "") for r in data]
        concepts = [r.get("concept_name") or r.get("concept", "") for r in data]
        
        self._write_csv(self.retain_csv_path, prompts, concepts, "retain")
        
        print(f"  Generating {len(prompts)} images...")
        try:
            images = self.technique.generate(prompts, **(config or {}))
            saved = self._save_images(images, self.retain_dir)
            print(f"  ✓ retain: {len(prompts)} prompts, {saved} images")
        except Exception as e:
            print(f"  ❌ {e}")
            return []
        
        return prompts
    
    def _generate_target_data(self, i2p_data: List[Dict[str, str]], config: Optional[Dict] = None) -> List[str]:
        """Generate target (forgetting) data from I2P dataset."""
        print("\n[STEP 3] Generating TARGET data")
        if not i2p_data:
            print("  ❌ No I2P data available")
            return []
        return self._generate_category_data(
            i2p_data, self.num_target_prompts, self.target_csv_path,
            self.target_dir, "target", config, concept_key="categories"
        )
    
    def _generate_adversarial_data(self, rab_data: List[Dict[str, str]], config: Optional[Dict] = None) -> List[str]:
        """Generate adversarial data from Ring-A-Bell dataset."""
        print("\n[STEP 4] Generating ADVERSARIAL data")
        if not rab_data:
            print("  ⚠️ Ring-A-Bell unavailable, skipping")
            return []
        return self._generate_category_data(
            rab_data, self.num_adversarial_prompts, self.adversarial_csv_path,
            self.adversarial_dir, "adversarial", config, concept_key="concept"
        )
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    def _load_data(self) -> List[str]:
        """Load prompts from CSV files. Required by BenchmarkTask."""
        print("\n[STEP 5] Loading prompts")
        
        def load_csv(path: Path) -> List[str]:
            if not path.exists():
                return []
            with open(path, 'r', encoding='utf-8') as f:
                return [r.get('prompt', '') for r in csv.DictReader(f) if r.get('prompt')]
        
        self.target_prompts = load_csv(self.target_csv_path)
        self.retain_prompts = load_csv(self.retain_csv_path)
        self.adversarial_prompts = load_csv(self.adversarial_csv_path)
        
        print(f"  Target: {len(self.target_prompts)}, Retain: {len(self.retain_prompts)}, Adversarial: {len(self.adversarial_prompts)}")
        return self.target_prompts + self.retain_prompts + self.adversarial_prompts
    
    def _get_image_concept_pairs(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load image-concept pairs from CSV and image directories."""
        def load_pairs(csv_path: Path, image_dir: Path) -> List[Tuple[str, str]]:
            if not csv_path.exists() or not image_dir.exists():
                return []
            
            concepts = {}
            with open(csv_path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('index') and row.get('concept'):
                        concepts[int(row['index'])] = row['concept']
            
            images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in self.IMAGE_EXTENSIONS)
            return [(str(img), concepts.get(idx, '')) for idx, img in enumerate(images)]
        
        return {
            'target': load_pairs(self.target_csv_path, self.target_dir),
            'retain': load_pairs(self.retain_csv_path, self.retain_dir),
            'adversarial': load_pairs(self.adversarial_csv_path, self.adversarial_dir),
        }
    
    def _calculate_metric(self, generated_images: List[Any], prompts: List[str]) -> Dict:
        """Calculate ERR metric. Required by BenchmarkTask."""
        print("\n[STEP 6] Calculating ERR metric")
        
        data = self._get_image_concept_pairs()
        print(f"  Pairs - Target: {len(data['target'])}, Retain: {len(data['retain'])}, Adversarial: {len(data['adversarial'])}")
        
        if not any(data.values()):
            print("  ❌ No data found!")
            return {"ERR_Score": 0.0, "Details": {}}
        
        result = self.metric_calculator.calculate_err(data)
        print(f"  ✓ ERR Score: {result['ERR_Score']:.4f}")
        print(f"    Forgetting: {result['Details']['Forgetting']:.4f}, "
              f"Retention: {result['Details']['Retention']:.4f}, "
              f"Adversarial: {result['Details']['Adversarial']:.4f}")
        return result
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    def run(self, config: Optional[Dict] = None) -> None:
        """Run complete ERR benchmark pipeline."""
        print("=" * 60)
        print(f"ERR BENCHMARK: {self.technique_name}")
        print("=" * 60)
        
        # Load datasets
        i2p, rab = self._load_datasets()
        if not i2p:
            print("\n❌ Failed to load I2P dataset")
            return
        
        # Generate data
        self.target_prompts = self._generate_target_data(i2p, config)
        self.retain_prompts = self._generate_retention_data(config)
        self.adversarial_prompts = self._generate_adversarial_data(rab, config)
        
        # Evaluate
        self._load_data()
        result = self._calculate_metric([], [])
        
        # Summary
        total = len(self.target_prompts) + len(self.retain_prompts) + len(self.adversarial_prompts)
        print("\n" + "=" * 60)
        print(f"COMPLETE: {total} prompts, ERR Score: {result['ERR_Score']:.4f}")
        print("=" * 60)

if __name__ == "__main__":
    # Example usage
    
    sld_wrapper = SLDWrapper()
    benchmark = ERRBenchmarkTask(
        technique=sld_wrapper,
        technique_name="SLD_Example",
        num_target_prompts=1,
        num_retain_prompts=1,
        num_adversarial_prompts=1,
    )
    benchmark.run()