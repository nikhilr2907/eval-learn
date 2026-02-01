import json
import os
import pytest
from unittest.mock import MagicMock, call
from eval_learn.runners import BenchmarkRunner
from eval_learn.types import Dataset, MetricResult


@pytest.fixture
def runner_deps(tmp_path, dummy_pil_image):
    """Build mocked components for BenchmarkRunner."""
    mock_img = dummy_pil_image()

    mock_loader = MagicMock(return_value=Dataset(
        prompts=["prompt1", "prompt2"],
        metadata={"source": "test", "concepts": ["c1", "c2"]}
    ))

    mock_technique_instance = MagicMock()
    mock_technique_instance.generate.return_value = [mock_img, mock_img]
    mock_technique_factory = MagicMock(return_value=mock_technique_instance)

    mock_metric_instance = MagicMock()
    mock_metric_instance.compute.return_value = MetricResult("TestMetric", 0.42, {"detail": "val"})
    mock_metric_factory = MagicMock(return_value=mock_metric_instance)

    return {
        "loader": mock_loader,
        "technique_factory": mock_technique_factory,
        "technique_instance": mock_technique_instance,
        "metric_factory": mock_metric_factory,
        "metric_instance": mock_metric_instance,
        "output_dir": str(tmp_path / "results"),
        "img": mock_img,
    }


def _make_runner(deps, **overrides):
    """Helper to build a BenchmarkRunner from deps dict."""
    kwargs = dict(
        dataset_loader=deps["loader"],
        technique_factory=deps["technique_factory"],
        metric_factory=deps["metric_factory"],
        technique_config={"model_id": "test-model"},
        metric_config={"use_nudenet": False},
        dataset_config={"limit": 5},
        output_dir=deps["output_dir"],
        run_name="TestRun",
    )
    kwargs.update(overrides)
    return BenchmarkRunner(**kwargs)


class TestRunnerCalls:
    def test_calls_loader_with_config(self, runner_deps):
        runner = _make_runner(runner_deps, dataset_config={"limit": 5})
        runner.run()
        runner_deps["loader"].assert_called_once_with(limit=5)

    def test_calls_technique_factory_with_config(self, runner_deps):
        runner = _make_runner(runner_deps, technique_config={"model_id": "test-model"})
        runner.run()
        runner_deps["technique_factory"].assert_called_once_with(model_id="test-model")

    def test_calls_metric_factory_with_config(self, runner_deps):
        runner = _make_runner(runner_deps, metric_config={"use_nudenet": False})
        runner.run()
        runner_deps["metric_factory"].assert_called_once_with(use_nudenet=False)

    def test_calls_generate_with_prompts(self, runner_deps):
        runner = _make_runner(runner_deps)
        runner.run()
        runner_deps["technique_instance"].generate.assert_called_once_with(
            prompts=["prompt1", "prompt2"]
        )

    def test_passes_metadata_to_compute(self, runner_deps):
        runner = _make_runner(runner_deps)
        runner.run()
        compute_call = runner_deps["metric_instance"].compute.call_args
        assert compute_call.kwargs["metadata"] == {"source": "test", "concepts": ["c1", "c2"]}


class TestRunnerReport:
    def test_report_structure(self, runner_deps):
        runner = _make_runner(runner_deps)
        report = runner.run()
        for key in ("run_name", "dataset_metadata", "technique_config",
                     "metric_config", "metric_result"):
            assert key in report
        assert "name" in report["metric_result"]
        assert "value" in report["metric_result"]
        assert "details" in report["metric_result"]

    def test_report_values(self, runner_deps):
        runner = _make_runner(runner_deps)
        report = runner.run()
        assert report["run_name"] == "TestRun"
        assert report["metric_result"]["value"] == 0.42
        assert report["metric_result"]["name"] == "TestMetric"


class TestRunnerArtifacts:
    def test_saves_artifacts(self, runner_deps):
        runner = _make_runner(runner_deps)
        runner.run()
        run_dir = os.path.join(runner_deps["output_dir"], "TestRun")
        assert os.path.isdir(run_dir)
        # Should have a JSON report
        json_files = [f for f in os.listdir(run_dir) if f.endswith(".json")]
        assert len(json_files) == 1

    def test_execution_order(self, runner_deps):
        call_order = []
        runner_deps["loader"].side_effect = lambda **kw: (
            call_order.append("loader"),
            Dataset(prompts=["p1"], metadata={"source": "test"})
        )[1]
        runner_deps["technique_factory"].side_effect = lambda **kw: (
            call_order.append("technique_factory"),
            runner_deps["technique_instance"]
        )[1]
        runner_deps["technique_instance"].generate.side_effect = lambda **kw: (
            call_order.append("generate"),
            [runner_deps["img"]]
        )[1]
        runner_deps["metric_factory"].side_effect = lambda **kw: (
            call_order.append("metric_factory"),
            runner_deps["metric_instance"]
        )[1]
        runner_deps["metric_instance"].compute.side_effect = lambda **kw: (
            call_order.append("compute"),
            MetricResult("T", 0.0)
        )[1]

        runner = _make_runner(runner_deps)
        runner.run()
        assert call_order == ["loader", "technique_factory", "generate", "metric_factory", "compute"]
