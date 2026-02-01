from abc import ABC, abstractmethod
import os
import time
import json
from typing import List, Dict, Any
from core.base_technique import UnlearningTechnique

class BenchmarkTask(ABC):
    """
    Abstract Base Class for a specific Benchmark Task (e.g., ASR, TIFA).
    It manages:
    1. The Dataset specific to this benchmark.
    2. Multiple Unlearning Techniques to evaluate against this benchmark.
    3. The Metric Calculation logic.
    """

    def __init__(self, name: str, output_dir: str = "results/benchmarks"):
        self.name = name
        self.output_dir = os.path.join(output_dir, name)
        self.techniques = []
        os.makedirs(self.output_dir, exist_ok=True)

    def add_technique(self, technique: UnlearningTechnique, config: Dict[str, Any], name: str):
        """
        Add a technique execution configuration to this benchmark.
        """
        self.techniques.append({
            "technique": technique,
            "config": config,
            "name": name
        })

    @abstractmethod
    def _load_data(self) -> List[str]:
        """Load and return the list of prompts for this benchmark."""
        pass

    @abstractmethod
    def _calculate_metric(self, generated_images: List[Any], prompts: List[str]) -> Any:
        """Calculate the specific metric for this benchmark."""
        pass

    def run(self):
        """
        Runs the benchmark for all added techniques.
        """
        print(f"=== Running Benchmark: {self.name} ===")
        
        # 1. Load Data
        prompts = self._load_data()
        print(f"Loaded {len(prompts)} prompts for evaluation.")
        
        overall_results = {
            "benchmark": self.name,
            "timestamp": time.time(),
            "results": []
        }

        for item in self.techniques:
            tech = item['technique']
            config = item['config']
            run_name = item['name']
            
            print(f"\n--- Evaluating Technique: {run_name} ---")
            
            # A. Generate
            print(f"Generating images...")
            # Note: We pass the config unpacked
            generated_images = tech.generate(prompts, **config)
            
            # B. Save Images (Optional but recommended)
            self._save_images(generated_images, run_name)

            # C. Calculate Metric
            print(f"Calculating metric...")
            score = self._calculate_metric(generated_images, prompts)
            print(f"Result for {run_name}: {score}")

            overall_results["results"].append({
                "technique": run_name,
                "config": str(config),
                "score": score
            })

        # Save Final Report
        report_path = os.path.join(self.output_dir, f"report_{int(time.time())}.json")
        with open(report_path, 'w') as f:
            json.dump(overall_results, f, indent=4)
        print(f"\n=== Benchmark Complete. Report saved to {report_path} ===")

    def _save_images(self, images: List[Any], run_name: str):
        path = os.path.join(self.output_dir, "images", run_name)
        os.makedirs(path, exist_ok=True)
        for i, img in enumerate(images):
            try:
                img.save(os.path.join(path, f"{i}.png"))
            except Exception as e:
                print(f"Error saving image: {e}")
