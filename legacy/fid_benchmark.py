import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from unlearning_techniques.sld_pipeline.sld_wrapper import SLDWrapper
from evaluation_metrics.fid_metric.fid_task import FIDBenchmarkTask


def main():
    # Configuration
    num_samples = 5  # Number of images to test
    batch_size = 32
    seed = 42
    
    # Initialize the SLD model
    try:
        sld_wrapper = SLDWrapper(model_id="AIML-TUDA/stable-diffusion-safe")
    except Exception as e:
        print(f"Failed to load SLD Pipeline: {e}")
        return
    
    # Create FID benchmark task
    # Tests: Can SLD still generate high-quality images on OTHER concepts?
    fid_task = FIDBenchmarkTask(
        num_samples=num_samples,
        dataset_split="validation",
        batch_size=batch_size,
        seed=seed,
        exclude_categories=["dog"]  # Exclude unlearned category
    )
    
    # Configuration : SLD Disabled
    print("\nAdding configuration: SLD Disabled (Baseline)")
    fid_task.add_technique(
        technique=sld_wrapper,
        config={"config": {"sld_guidance_scale": 0}},
        name="SLD_Disabled"
    )
    
    # Configuration : SLD Max (Highest Safety)
    print("Adding configuration: SLD Max")
    fid_task.add_technique(
        technique=sld_wrapper,
        config={"config": SafetyConfig.MAX},
        name="SLD_Max"
    )
    
    fid_task.run()


if __name__ == "__main__":
    main()

