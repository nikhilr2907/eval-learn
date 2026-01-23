import sys
import os

# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from unlearning_techniques.sld_pipeline.sld_wrapper import SLDWrapper
from evaluation_metrics.asr_metric.asr_task import ASRBenchmarkTask

def main():
    print("=== Initializing Eval-Learn Benchmark ===")

    # Configuration
    csv_path = "data/i2p/i2p_benchmark_sample.csv"
    use_nudenet = True 
    use_q_16 = False

    # 1. Initialize the Unlearning Technique(s)
    print("Loading SLD Pipeline...")
    try:
        sld_wrapper = SLDWrapper(model_id="AIML-TUDA/stable-diffusion-safe")
    except Exception as e:
        print(f"Failed to load SLD Pipeline: {e}")
        return

    # 2. Initialize the Benchmark Task
    print(f"Setting up ASR Benchmark Task with dataset: {csv_path}")
    asr_task = ASRBenchmarkTask(
        dataset_path=csv_path, 
        use_nudenet=use_nudenet,
        use_q16=use_q_16
    )

    # 3. Add Configurations to Evaluate
    
    # Configuration A: SLD Max (Highest Safety)
    asr_task.add_technique(
        technique=sld_wrapper,
        config={"config": SafetyConfig.MAX},
        name="SLD_Max"
    )
    
    # Configuration B: SLD Disabled (Baseline - Standard SD)
    asr_task.add_technique(
        technique=sld_wrapper,
        config={"config": {"sld_guidance_scale": 0}},
        name="SLD_Disabled"
    )

    # 4. Execute the Benchmark
    print("Starting execution...")
    asr_task.run()
    print("Execution finished.")

if __name__ == "__main__":
    main()
