import json
import os
import pytest
from unittest.mock import MagicMock
from eval_learn.runners import SingleBenchmarkRunner
from eval_learn.registry.local import _TECHNIQUES, _METRICS
from eval_learn.types import Dataset, MetricResult


@pytest.fixture
def mock_registry(reset_registry, dummy_pil_image):
    """Register mock technique and metric factories in the registry."""
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

    # Register directly into the registry dicts
    _TECHNIQUES["mock_tech"] = mock_technique_factory
    _METRICS["mock_metric"] = mock_metric_factory

    return {
        "technique_factory": mock_technique_factory,
        "technique_instance": mock_technique_instance,
        "metric_factory": mock_metric_factory,
        "metric_instance": mock_metric_instance,
        "dataset": mock_dataset,
        "img": mock_img,
    }


def _make_runner(tmp_path, **overrides):
    """Helper to build a SingleBenchmarkRunner with defaults."""
    kwargs = dict(
        technique_name="mock_tech",
        metric_name="mock_metric",
        technique_config={"device": "cpu"},
        metric_config={"use_nudenet": False},
        output_dir=str(tmp_path / "results"),
    )
    kwargs.update(overrides)
    return SingleBenchmarkRunner(**kwargs)


class TestValidation:
    def test_raises_on_unknown_technique(self, mock_registry, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            _make_runner(tmp_path, technique_name="nonexistent")

    def test_raises_on_unknown_metric(self, mock_registry, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            _make_runner(tmp_path, metric_name="nonexistent")

    def test_accepts_valid_names(self, mock_registry, tmp_path):
        runner = _make_runner(tmp_path)
        assert runner.technique_name == "mock_tech"
        assert runner.metric_name == "mock_metric"


class TestRunnerCalls:
    def test_calls_technique_factory_with_config(self, mock_registry, tmp_path):
        runner = _make_runner(tmp_path, technique_config={"device": "cpu"})
        runner.run()
        mock_registry["technique_factory"].assert_called_once_with(device="cpu")

    def test_calls_metric_factory_with_config(self, mock_registry, tmp_path):
        runner = _make_runner(tmp_path, metric_config={"use_nudenet": False})
        runner.run()
        mock_registry["metric_factory"].assert_called_once_with(use_nudenet=False)

    def test_metric_load_dataset_called(self, mock_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        mock_registry["metric_instance"].load_dataset.assert_called_once()

    def test_calls_generate_with_prompts(self, mock_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        mock_registry["technique_instance"].generate.assert_called_once_with(
            prompts=["prompt1", "prompt2"]
        )

    def test_passes_metadata_to_compute(self, mock_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        compute_call = mock_registry["metric_instance"].compute.call_args
        assert compute_call.kwargs["metadata"] == {"source": "test", "concepts": ["c1", "c2"]}


class TestRunnerReport:
    def test_report_structure(self, mock_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        for key in ("run_id", "timestamp", "technique_name", "metric_name",
                     "dataset_name", "dataset_metadata", "technique_config",
                     "metric_config", "metric_result"):
            assert key in report
        assert "name" in report["metric_result"]
        assert "value" in report["metric_result"]
        assert "details" in report["metric_result"]

    def test_report_values(self, mock_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        assert report["technique_name"] == "mock_tech"
        assert report["metric_name"] == "mock_metric"
        assert report["dataset_name"] == "test"
        assert report["metric_result"]["value"] == 0.42
        assert report["metric_result"]["name"] == "TestMetric"

    def test_run_id_is_8_char_hex(self, mock_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        run_id = report["run_id"]
        assert len(run_id) == 8
        assert all(c in "0123456789abcdef" for c in run_id)


class TestRunnerArtifacts:
    def test_saves_artifacts_with_new_naming(self, mock_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        run_id = report["run_id"]
        run_dir = os.path.join(str(tmp_path / "results"), f"mock_tech_mock_metric_{run_id}")
        assert os.path.isdir(run_dir)
        json_files = [f for f in os.listdir(run_dir) if f.endswith(".json")]
        assert len(json_files) == 1
        assert json_files[0] == f"{run_id}_report.json"

    def test_saves_images_flat_without_categories(self, mock_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        run_id = report["run_id"]
        images_dir = os.path.join(str(tmp_path / "results"), f"mock_tech_mock_metric_{run_id}", "images")
        assert os.path.isdir(images_dir)

    def test_saves_images_in_category_subdirs(self, mock_registry, tmp_path, dummy_pil_image):
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
        mock_registry["metric_instance"].load_dataset.return_value = cat_dataset
        mock_registry["technique_instance"].generate.return_value = imgs

        runner = _make_runner(tmp_path, metric_name="mock_metric")
        report = runner.run()
        run_id = report["run_id"]

        images_dir = os.path.join(str(tmp_path / "results"), f"mock_tech_mock_metric_{run_id}", "images")
        assert os.path.isdir(os.path.join(images_dir, "target"))
        assert os.path.isdir(os.path.join(images_dir, "retain"))
        assert os.path.isdir(os.path.join(images_dir, "adversarial"))

    def test_execution_order(self, mock_registry, tmp_path):
        call_order = []
        mock_registry["metric_factory"].side_effect = lambda **kw: (
            call_order.append("metric_factory"),
            mock_registry["metric_instance"]
        )[1]
        mock_registry["metric_instance"].load_dataset.side_effect = lambda: (
            call_order.append("load_dataset"),
            mock_registry["dataset"]
        )[1]
        mock_registry["technique_factory"].side_effect = lambda **kw: (
            call_order.append("technique_factory"),
            mock_registry["technique_instance"]
        )[1]
        mock_registry["technique_instance"].generate.side_effect = lambda **kw: (
            call_order.append("generate"),
            [mock_registry["img"]]
        )[1]
        mock_registry["metric_instance"].compute.side_effect = lambda **kw: (
            call_order.append("compute"),
            MetricResult("T", 0.0)
        )[1]

        runner = _make_runner(tmp_path)
        runner.run()
        assert call_order == ["metric_factory", "load_dataset", "technique_factory", "generate", "compute"]
