import pytest
from unittest.mock import MagicMock
from eval_learn.registry import get_technique, get_metric, get_dataset
from eval_learn.runners import SingleBenchmarkRunner
from eval_learn.types import Dataset
# Explicitly import to trigger registration
import eval_learn.techniques.sld.wrapper
import eval_learn.metrics.asr.metric

# Mock loader for smoke test to avoid file I/O dependence
def mock_loader(limit=None, **kwargs):
    prompts = ["a photo of a cat"] * (limit or 1)
    return Dataset(prompts=prompts, metadata={"source": "mock"})

def test_smoke_asr_sld_pipeline(tmp_path):
    """
    Smoke test for the ASR + SLD pipeline.
    Uses mocks where possible to avoid heavy model loading during CI/Smoke tests,
    unless specific flags are enabled (which they aren't here).
    
    Actually, to test 'integration', we want to see if classes instantiate.
    But loading SD model takes time. 
    
    For this smoke test, we will MOCK the technique's generate method 
    but use the REAL Runner and Writer logic.
    """
    
    # 1. Setup Registry / Components
    # Use real registry lookups to ensure they are registered
    try:
        sld_cls = get_technique("sld")
        asr_cls = get_metric("asr")
    except ValueError:
        pytest.skip("SLD or ASR not registered, skipping smoke test")

    # 2. Mock the heavy lifting
    # Mock SLD instance
    mock_sld_instance = MagicMock()
    # Mock return images (simple PIL images)
    from PIL import Image
    mock_img = Image.new('RGB', (64, 64), color='red')
    mock_sld_instance.generate.return_value = [mock_img]
    
    # Create a factory that returns our mock
    mock_sld_factory = MagicMock(return_value=mock_sld_instance)
    
    # Mock ASR instance (avoid NudeNet load)
    mock_asr_instance = MagicMock()
    from eval_learn.types import MetricResult
    mock_asr_instance.compute.return_value = MetricResult("ASR", 0.0)
    mock_asr_factory = MagicMock(return_value=mock_asr_instance)

    # 3. Configure Runner
    output_dir = tmp_path / "results"
    
    runner = SingleBenchmarkRunner(
        dataset_loader=mock_loader,
        technique_factory=mock_sld_factory,
        metric_factory=mock_asr_factory,
        technique_config={"model_id": "dummy"},
        metric_config={"use_nudenet": False},
        dataset_config={"limit": 1},
        output_dir=str(output_dir),
        run_name="SmokeTest"
    )
    
    # 4. Run
    report = runner.run()
    
    # 5. Assertions
    assert report["run_name"] == "SmokeTest"
    assert report["metric_result"]["value"] == 0.0
    
    # Check artifacts
    # Navigate: output_dir / SmokeTest / images / run_... / 0.png
    run_dir = output_dir / "SmokeTest"
    assert run_dir.exists()
    
    # Check JSON
    json_files = list(run_dir.glob("*.json"))
    assert len(json_files) == 1
    
    # Check Images
    images_dir = run_dir / "images"
    assert images_dir.exists()
    # There should be one subdirectory for the timestamp
    subdirs = list(images_dir.iterdir())
    assert len(subdirs) == 1
    assert (subdirs[0] / "0.png").exists()

if __name__ == "__main__":
    pytest.main([__file__])
