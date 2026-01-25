

import sys
from pathlib import Path

from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from unlearning_techniques.sld_pipeline.sld_wrapper import SLDWrapper
from evaluation_metrics.ERR.err_task import ERRBenchmarkTask


def main() -> None:
    """Run the ERR benchmark."""
    print("=" * 80)
    print("ERR BENCHMARK - Erasing-Retention-Robustness Evaluation")
    print("=" * 80)
    
    # Initialize the unlearning technique
    print("\nInitializing SLD Wrapper...")
    sld_wrapper = SLDWrapper()
    
    # Create and run benchmark with SLD Max
    print("\nSetting up ERR Benchmark Task...")
    benchmark = ERRBenchmarkTask(
        technique=sld_wrapper,
        technique_name="SLD_Max",
        num_target_prompts=5,       # I2P prompts for forgetting
        num_retain_prompts=5,       # Challenge dataset prompts for retention
        num_adversarial_prompts=5,  # Ring-A-Bell prompts for adversarial
    )
    
    print("\nRunning benchmark with SLD Max configuration...")
    benchmark.run(config={"config": SafetyConfig.MAX})
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
