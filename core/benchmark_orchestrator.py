import json
import os
import time
from typing import List, Dict, Any
from core.base_technique import UnlearningTechnique

class BenchmarkOrchestrator:
    """
    The Orchestrator class responsible for running the evaluation loop:
    1. Manage the Unlearning Technique (Model)
    2. Manage Datasets (Prompts)
    3. Calculate Metrics
    4. Save Results
    """
    
    def __init__(self, technique: UnlearningTechnique, output_dir: str = "results/benchmark_runs"):
        self.technique = technique
        self.metrics = []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def add_metric(self, metric: Any):
        """
        Registers a metric to be calculated after generation.
        The metric object must have a 'calculate(generated_images, prompts)' method.
        """
        self.metrics.append(metric)

    def run_benchmark(self, prompts: List[str], generation_config: Dict[str, Any] = None, run_name: str = "default_run") -> Dict[str, Any]:
        """
        Executes the benchmark: generates images and runs metrics.
        
        Args:
            prompts: List of inputs to test.
            generation_config: Configuration dict passed to the model's generate method.
            run_name: Identifier for this run (used for file naming).
            
        Returns:
            Dictionary containing metric results.
        """
        if generation_config is None:
            generation_config = {}

        print(f"--- Starting Benchmark Run: {run_name} ---")
        
        # 1. Generation Phase
        print(f"Generating images for {len(prompts)} prompts...")
        # Note: In a real scenario with thousands of prompts, you'd batch this.
        generated_images = self.technique.generate(prompts, **generation_config)
        print("Generation complete.")

        # Save generated images
        images_dir = os.path.join(self.output_dir, "images", run_name)
        os.makedirs(images_dir, exist_ok=True)
        print(f"Saving images to {images_dir}...")
        for i, img in enumerate(generated_images):
            # Sanitize filename or just use index
            img_path = os.path.join(images_dir, f"image_{i}.png") 
            try:
                img.save(img_path)
            except Exception as e:
                print(f"failed to save image {i}: {e}")

        results = {
            "run_name": run_name,
            "timestamp": time.time(),
            "config": str(generation_config),
            "num_prompts": len(prompts),
            "metric_results": {}
        }

        # 2. Evaluation Phase
        for metric in self.metrics:
            metric_name = getattr(metric, 'name', metric.__class__.__name__)
            print(f"Calculating metric: {metric_name}...")
            
            try:
                # Pass both images and prompts in case the metric needs them
                score = metric.calculate(generated_images=generated_images, prompts=prompts)
                results["metric_results"][metric_name] = score
                print(f"-> {metric_name}: {score}")
            except Exception as e:
                print(f"Error calculating {metric_name}: {e}")
                results["metric_results"][metric_name] = f"Error: {str(e)}"

        # 3. Save Results
        output_file = os.path.join(self.output_dir, f"{run_name}_{int(time.time())}.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {output_file}")
        except Exception as e:
             print(f"Failed to save results json: {e}")

        print("--- Benchmark Run Complete ---")
        return results
