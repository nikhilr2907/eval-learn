import sys
import os

# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from unlearning_techniques.sld_pipeline.sld_wrapper import SLDWrapper
from evaluation_metrics.text_to_image_fidelity.tifa_task import TIFABenchmarkTask

def main():
    """
    Main entry point for running the TIFA (Text-to-Image Faithfulness Evaluation) benchmark.
    This script compares SLD (Safe Latent Diffusion) in Max safety mode vs. Disabled mode.
    """
    print("=== Initializing TIFA Benchmark for SLD Evaluation ===")

    # 1. Dataset Configuration
    # Ensure these paths point to the sensitive samples created earlier
    text_path = "data/tifa/sensitive_text_inputs.json"
    qa_path = "data/tifa/sensitive_question_answers.json"

    # 2. Initialize the Unlearning Technique (SLD)
    print("Loading SLD Pipeline...")
    try:
        # Initializing SLDWrapper (it will handle its own device selection and HF login)
        sld_wrapper = SLDWrapper(model_id="AIML-TUDA/stable-diffusion-safe")
    except Exception as e:
        print(f"CRITICAL: Failed to load SLD Pipeline: {e}")
        return

    # 3. Initialize the TIFA Benchmark Task
    print(f"Setting up TIFA Benchmark Task with dataset: {text_path}")
    tifa_task = TIFABenchmarkTask(
        text_path=text_path, 
        qa_path=qa_path,
        name="TIFA_Benchmark"
    )

    # 4. Add Configurations to Evaluate
    # Note: We pass the config in a way that matches SLDWrapper.generate(prompts, config=...)
    
    # Configuration A: SLD Max (Highest Safety Guidance)
    print("Adding SLD_Max configuration...")
    tifa_task.add_technique(
        technique=sld_wrapper,
        # This mapping ensures tech.generate(prompts, **config) 
        # calls sld_wrapper.generate(prompts, config=SafetyConfig.MAX)
        config={"config": SafetyConfig.MAX},
        name="SLD_Max"
    )
    
    # Configuration B: SLD Disabled (Baseline - Standard SD v1.5 behavior)
    print("Adding SLD_Disabled (Baseline) configuration...")
    tifa_task.add_technique(
        technique=sld_wrapper,
        # This shuts down the safety guidance scale in SLD
        config={"config": {"sld_guidance_scale": 0}},
        name="SLD_Disabled"
    )

    # 5. Execute the Benchmark
    # This will: 1. Load Data, 2. Generate Images for both modes, 3. Run VQA, 4. Save JSON Report
    print("\n>>> Starting Benchmark Execution...")
    tifa_task.run()
    
    print("\n" + "="*50)
    print("TIFA Benchmark finished successfully.")
    print(f"Results and generated images are saved in: results/benchmarks/TIFA_Benchmark/")
    print("="*50)

if __name__ == "__main__":
    main()
