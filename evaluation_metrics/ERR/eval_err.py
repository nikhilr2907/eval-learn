# evaluation_metrics/ERR/err_task.py
import os
import csv
from typing import List, Any, Dict
from pathlib import Path

from core.base_benchmark import BenchmarkTask
from .err import ERREvaluator
from transformers import CLIPProcessor, CLIPModel
import torch


class ERRBenchmarkTask(BenchmarkTask):
    """
    Benchmark Task for ERR (Erasing-Retention-Robustness) metric.
    
    This benchmark evaluates pre-generated images across three categories:
    - Forgetting: Target concept should NOT appear (I2P prompts)
    - Retention: Benign concepts should STILL appear (Challenge dataset)
    - Adversarial: Target concept should NOT appear with adversarial prompts (Ring-A-Bell)
    
    Note: Images should be pre-generated using ERRDataGenerator before running this benchmark.
    """
    
    def __init__(
        self,
        image_base_dir: str = None,
        target_prompts_path: str = None,
        retain_prompts_path: str = None,
        adversarial_prompts_path: str = None,
        target_concept: str = "inappropriate content",
        retain_concept: str = "the requested object",
        device: str = None
    ):
        """
        Initialize ERR Benchmark Task.
        
        Args:
            image_base_dir: Base directory containing technique folders with images
            target_prompts_path: Path to CSV with target (forgetting) prompts
            retain_prompts_path: Path to CSV with retention prompts
            adversarial_prompts_path: Path to CSV with adversarial prompts
            target_concept: The concept that should be forgotten
            retain_concept: The concept that should be retained
            device: Device for CLIP model (cuda/cpu)
        """
        super().__init__(name="ERR_Benchmark")
        
        # Set default paths if not provided
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        prompt_data_dir = os.path.join(base_dir, "data", "ERR", "prompt_data")
        
        self.image_base_dir = image_base_dir or os.path.join(
            base_dir, "results", "benchmarks", "ERR_benchmark", "images"
        )
        self.target_prompts_path = target_prompts_path or os.path.join(prompt_data_dir, "target_prompts.csv")
        self.retain_prompts_path = retain_prompts_path or os.path.join(prompt_data_dir, "retain_prompts.csv")
        self.adversarial_prompts_path = adversarial_prompts_path or os.path.join(prompt_data_dir, "adversarial_prompts.csv")
        
        self.target_concept = target_concept
        self.retain_concept = retain_concept
        
        # Initialize CLIP model and ERR evaluator
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing CLIP model for ERR evaluation on {self.device}...")
        
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        self.metric_calculator = ERREvaluator(
            oracle_classifier=model,
            processor=processor,
            device=torch.device(self.device)
        )
        
        print("ERR Evaluator initialized successfully.")
        
        # Store prompts categorized by type
        self.target_prompts = []
        self.retain_prompts = []
        self.adversarial_prompts = []
    
    def _load_prompts_from_csv(self, csv_path: str) -> List[str]:
        """Load prompts from a CSV file."""
        prompts = []
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file {csv_path} does not exist")
            return prompts
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append(row.get('prompt', ''))
        
        return prompts
    
    def _load_data(self) -> List[str]:
        """
        Load all prompts from the three categories.
        Returns a combined list for consistency with base class.
        """
        print("\nLoading ERR benchmark prompts...")
        
        self.target_prompts = self._load_prompts_from_csv(self.target_prompts_path)
        self.retain_prompts = self._load_prompts_from_csv(self.retain_prompts_path)
        self.adversarial_prompts = self._load_prompts_from_csv(self.adversarial_prompts_path)
        
        print(f"  Target (forgetting) prompts:     {len(self.target_prompts)}")
        print(f"  Retain (retention) prompts:      {len(self.retain_prompts)}")
        print(f"  Adversarial (robustness) prompts: {len(self.adversarial_prompts)}")
        
        # Return combined list
        all_prompts = self.target_prompts + self.retain_prompts + self.adversarial_prompts
        
        if not all_prompts:
            print("Warning: No prompts loaded!")
            return []
        
        return all_prompts
    
    def _get_image_paths_for_technique(self, technique_name: str) -> Dict[str, List[str]]:
        """Get image paths for a specific technique, organized by category."""
        technique_dir = os.path.join(self.image_base_dir, technique_name)
        
        categories = {
            'target': os.path.join(technique_dir, 'target'),
            'retain': os.path.join(technique_dir, 'retain'),
            'adversarial': os.path.join(technique_dir, 'adversarial')
        }
        
        image_paths = {}
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        
        for category, directory in categories.items():
            paths = []
            if os.path.exists(directory):
                for file in sorted(os.listdir(directory)):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(file)[1].lower()
                        if ext in image_extensions:
                            paths.append(file_path)
            else:
                print(f"  Warning: Directory {directory} does not exist")
            
            image_paths[category] = paths
        
        return image_paths
    
    def _calculate_metric(self, generated_images: List[Any], prompts: List[str]) -> Dict:
        """
        This method is overridden to work with pre-generated images.
        The base class calls this, but we ignore the generated_images parameter
        since we load images from disk instead.
        """
        # This should not be called in ERR benchmark since we override run()
        raise NotImplementedError("ERR benchmark uses pre-generated images. Use run() method.")
    
    def run(self):
        """
        Override run method to evaluate pre-generated images instead of generating new ones.
        """
        print(f"=== Running Benchmark: {self.name} ===")
        
        # 1. Load prompts (for reference)
        prompts = self._load_data()
        print(f"Loaded {len(prompts)} total prompts for evaluation.")
        
        overall_results = {
            "benchmark": self.name,
            "timestamp": int(__import__('time').time()),
            "results": []
        }
        
        for item in self.techniques:
            run_name = item['name']
            config = item['config']
            
            print(f"\n--- Evaluating Technique: {run_name} ---")
            
            # A. Load pre-generated images
            print(f"Loading pre-generated images from disk...")
            image_paths = self._get_image_paths_for_technique(run_name)
            
            print(f"  Found images:")
            print(f"    Target:       {len(image_paths['target'])}")
            print(f"    Retain:       {len(image_paths['retain'])}")
            print(f"    Adversarial:  {len(image_paths['adversarial'])}")
            
            if not any(image_paths.values()):
                print(f"  ⚠️  No images found for {run_name}. Skipping evaluation.")
                print(f"  Please run ERRDataGenerator first to generate images.")
                continue
            
            # B. Calculate ERR metric
            print(f"Calculating ERR metric...")
            
            concepts = {
                'target_concept': self.target_concept,
                'retain_concept': self.retain_concept
            }
            
            err_results = self.metric_calculator.calculate_err(image_paths, concepts)
            
            print(f"\n  Results for {run_name}:")
            print(f"    ERR Score:     {err_results['ERR_Score']:.4f}")
            print(f"    - Forgetting:  {err_results['Details']['Forgetting']:.4f}")
            print(f"    - Retention:   {err_results['Details']['Retention']:.4f}")
            print(f"    - Adversarial: {err_results['Details']['Adversarial']:.4f}")
            
            overall_results["results"].append({
                "technique": run_name,
                "config": str(config),
                "score": err_results,
                "image_counts": {
                    "target": len(image_paths['target']),
                    "retain": len(image_paths['retain']),
                    "adversarial": len(image_paths['adversarial'])
                }
            })
        
        # Save Final Report
        report_path = os.path.join(self.output_dir, f"report_{int(__import__('time').time())}.json")
        with open(report_path, 'w') as f:
            __import__('json').dump(overall_results, f, indent=4)
        print(f"\n=== Benchmark Complete. Report saved to {report_path} ===")