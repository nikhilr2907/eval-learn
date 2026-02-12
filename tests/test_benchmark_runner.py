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

    mock_dataset = Dataset(
        prompts=["prompt1", "prompt2"],
        metadata={"source": "test", "concepts": ["c1", "c2"]}
    )

    mock_technique_instance = MagicMock()
    mock_technique_instance.generate.return_value = [mock_img, mock_img]
    mock_technique_factory = MagicMock(return_value=mock_technique_instance)

    mock_metric_instance = MagicMock()
    mock_metric_instance.compute.return_value = MetricResult("TestMetric", 0.42, {"detail": "val"})
    mock_metric_instance.load_dataset.return_value = mock_dataset
    mock_metric_factory = MagicMock(return_value=mock_metric_instance)

    return {
        "technique_factory": mock_technique_factory,
        "technique_instance": mock_technique_instance,
        "metric_factory": mock_metric_factory,
        "metric_instance": mock_metric_instance,
        "dataset": mock_dataset,
        "output_dir": str(tmp_path / "results"),
        "img": mock_img,
    }


def _make_runner(deps, **overrides):
    """Helper to build a BenchmarkRunner from deps dict."""
    kwargs = dict(
        technique_factory=deps["technique_factory"],
        metric_factory=deps["metric_factory"],
        technique_name="sld",
        metric_name="asr",
        technique_config={"model_id": "test-model"},
        metric_config={"use_nudenet": False},
        output_dir=deps["output_dir"],
    )
    kwargs.update(overrides)
    return BenchmarkRunner(**kwargs)


class TestRunnerCalls:
    def test_calls_technique_factory_with_config(self, runner_deps):
        runner = _make_runner(runner_deps, technique_config={"model_id": "test-model"})
        runner.run()
        runner_deps["technique_factory"].assert_called_once_with(model_id="test-model")

    def test_calls_metric_factory_with_config(self, runner_deps):
        runner = _make_runner(runner_deps, metric_config={"use_nudenet": False})
        runner.run()
        runner_deps["metric_factory"].assert_called_once_with(use_nudenet=False)

    def test_metric_load_dataset_called(self, runner_deps):
        runner = _make_runner(runner_deps)
        runner.run()
        runner_deps["metric_instance"].load_dataset.assert_called_once()

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
        for key in ("run_id", "timestamp", "technique_name", "metric_name",
                     "dataset_name", "dataset_metadata", "technique_config",
                     "metric_config", "metric_result"):
            assert key in report
        assert "name" in report["metric_result"]
        assert "value" in report["metric_result"]
        assert "details" in report["metric_result"]

    def test_report_values(self, runner_deps):
        runner = _make_runner(runner_deps)
        report = runner.run()
        assert report["technique_name"] == "sld"
        assert report["metric_name"] == "asr"
        assert report["dataset_name"] == "test"
        assert report["metric_result"]["value"] == 0.42
        assert report["metric_result"]["name"] == "TestMetric"

    def test_run_id_is_8_char_hex(self, runner_deps):
        runner = _make_runner(runner_deps)
        report = runner.run()
        run_id = report["run_id"]
        assert len(run_id) == 8
        assert all(c in "0123456789abcdef" for c in run_id)


class TestRunnerArtifacts:
    def test_saves_artifacts_with_new_naming(self, runner_deps):
        runner = _make_runner(runner_deps)
        report = runner.run()
        run_id = report["run_id"]
        run_dir = os.path.join(runner_deps["output_dir"], f"sld_asr_{run_id}")
        assert os.path.isdir(run_dir)
        # Should have a JSON report named <run_id>_report.json
        json_files = [f for f in os.listdir(run_dir) if f.endswith(".json")]
        assert len(json_files) == 1
        assert json_files[0] == f"{run_id}_report.json"

    def test_saves_images_flat_without_categories(self, runner_deps):
        runner = _make_runner(runner_deps)
        report = runner.run()
        run_id = report["run_id"]
        images_dir = os.path.join(runner_deps["output_dir"], f"sld_asr_{run_id}", "images")
        assert os.path.isdir(images_dir)

    def test_saves_images_in_category_subdirs(self, runner_deps, dummy_pil_image):
        """When metadata has categories, images go into subdirectories."""
        imgs = [dummy_pil_image(), dummy_pil_image(), dummy_pil_image()]
        cat_dataset = Dataset(
            prompts=["p1", "p2", "p3"],
            metadata={
                "source": "test",
                "concepts": ["c1", "c2", "c3"],
                "categories": ["target", "retain", "adversarial"],
            }
        )
        runner_deps["metric_instance"].load_dataset.return_value = cat_dataset
        runner_deps["technique_instance"].generate.return_value = imgs

        runner = _make_runner(runner_deps, metric_name="err")
        report = runner.run()
        run_id = report["run_id"]

        images_dir = os.path.join(runner_deps["output_dir"], f"sld_err_{run_id}", "images")
        assert os.path.isdir(os.path.join(images_dir, "target"))
        assert os.path.isdir(os.path.join(images_dir, "retain"))
        assert os.path.isdir(os.path.join(images_dir, "adversarial"))

    def test_execution_order(self, runner_deps):
        call_order = []
        runner_deps["metric_factory"].side_effect = lambda **kw: (
            call_order.append("metric_factory"),
            runner_deps["metric_instance"]
        )[1]
        runner_deps["metric_instance"].load_dataset.side_effect = lambda: (
            call_order.append("load_dataset"),
            runner_deps["dataset"]
        )[1]
        runner_deps["technique_factory"].side_effect = lambda **kw: (
            call_order.append("technique_factory"),
            runner_deps["technique_instance"]
        )[1]
        runner_deps["technique_instance"].generate.side_effect = lambda **kw: (
            call_order.append("generate"),
            [runner_deps["img"]]
        )[1]
        runner_deps["metric_instance"].compute.side_effect = lambda **kw: (
            call_order.append("compute"),
            MetricResult("T", 0.0)
        )[1]

        runner = _make_runner(runner_deps)
        runner.run()
        assert call_order == ["metric_factory", "load_dataset", "technique_factory", "generate", "compute"]
