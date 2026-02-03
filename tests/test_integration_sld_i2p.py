import pytest
import os
from eval_learn.registry import get_dataset, get_technique, get_metric
from eval_learn.runners import BenchmarkRunner

# Check for heavy dependencies
try:
    import torch
    import diffusers
    import transformers
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

@pytest.mark.skipif(not DEPS_AVAILABLE, reason="Requires torch/diffusers/transformers")
def test_sld_on_i2p_integration(tmp_path):
    """
    Integration test running SLD technique on I2P dataset with ASR metric.
    Uses a sample dataset and limited prompts for speed.
    """
    
    # 1. Setup Paths
    # Prefer sample file if available for tests
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(repo_root, "data", "i2p", "i2p_benchmark_sample.csv")
    
    if not os.path.exists(dataset_path):
        # Fallback to original if sample not found
        dataset_path = os.path.join(repo_root, "data", "i2p", "i2p_benchmark_original.csv")
        
    if not os.path.exists(dataset_path):
        pytest.skip("I2P dataset csv not found in data/i2p/")

    # 2. Get Components via Registry
    # Explicitly import to ensure registration happens if not auto-loaded
    import eval_learn.datasets.i2p_csv
    import eval_learn.techniques.sld.wrapper
    import eval_learn.metrics.asr.metric

    try:
        print("Looking up components in registry...")
        DatasetLoader = get_dataset("i2p_csv")
        TechniqueFactory = get_technique("sld")
        MetricFactory = get_metric("asr")
    except ValueError as e:
        pytest.fail(f"Component lookup failed: {e}")

    # 3. Configure
    # Use CPU or convenient device for test to ensure it runs everywhere (though slow on CPU)
    # But usually tests run on machines that might not have GPU. 
    # For a 'test', we might want to mock the heavy model generation if this is just unit testing logic.
    # But the user asked "how would i do it in coding" implying a real run.
    # We will use a real run but extremely limited.
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_config = {
        "dataset_config": {
            "path": dataset_path,
            "limit": 1,  # Just 1 image for smoke test
            "prompt_col": "prompt"
        },
        "technique_config": {
            "model_id": "CompVis/stable-diffusion-v1-4", 
            "device": device,
            "sld_guidance_scale": 1000,
            "sld_warmup_steps": 10,
            "sld_threshold": 0.01,
            "sld_momentum_scale": 0.3,
            "sld_mom_beta": 0.4,
            # Speed up generation for test
            "num_inference_steps": 2 
        },
        "metric_config": {
            # Disable actual NudeNet model loading for speed/CI unless needed
            # Set to True if you have NudeNet installed and want to test it
            "use_nudenet": False 
        },
        "output_dir": str(tmp_path / "results"),
        "run_name": "Test_SLD_I2P"
    }

    # 4. Initialize Runner
    print("Initializing Benchmark Runner...")
    runner = BenchmarkRunner(
        dataset_loader=DatasetLoader,
        technique_factory=TechniqueFactory,
        metric_factory=MetricFactory,
        **run_config
    )

    # 5. Run
    print(f"Running integration test on {device}...")
    report = runner.run()

    # 6. Assertions
    assert report["run_name"] == "Test_SLD_I2P"
    assert "metric_result" in report
    assert report["metric_result"]["name"] == "ASR"
    
    # Check artifacts
    run_dir = tmp_path / "results" / "Test_SLD_I2P"
    assert run_dir.exists()
    assert (run_dir / "images").exists()
    
    # Ensure at least one image file was created
    # images/timestamp/0.png
    image_subdirs = list((run_dir / "images").iterdir())
    assert len(image_subdirs) > 0
    assert len(list(image_subdirs[0].glob("*.png"))) == 1

if __name__ == "__main__":
    pytest.main([__file__])
