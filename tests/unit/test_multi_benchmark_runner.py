import json
import os
import pytest
from unittest.mock import MagicMock
from eval_learn.runners import MultiBenchmarkRunner
from eval_learn.registry.local import _TECHNIQUES, _METRICS
from eval_learn.types import Dataset, MetricResult


@pytest.fixture
def mock_multi_registry(reset_registry, dummy_pil_image):
    """Register one mock technique and two mock metrics."""
    mock_img = dummy_pil_image()

    mock_dataset = Dataset(
        prompts=["prompt1", "prompt2"],
        metadata={"source": "test", "concepts": ["c1", "c2"]}
    )

    # Technique
    mock_technique_instance = MagicMock()
    mock_technique_instance.generate.return_value = [mock_img, mock_img]
    mock_technique_factory = MagicMock(return_value=mock_technique_instance)

    # Metric A
    mock_metric_a_instance = MagicMock()
    mock_metric_a_instance.compute.return_value = MetricResult("MetricA", 0.42, {"detail": "a"})
    mock_metric_a_instance.load_dataset.return_value = mock_dataset
    mock_metric_a_factory = MagicMock(return_value=mock_metric_a_instance)

    # Metric B
    mock_metric_b_instance = MagicMock()
    mock_metric_b_instance.compute.return_value = MetricResult("MetricB", 0.85, {"detail": "b"})
    mock_metric_b_instance.load_dataset.return_value = mock_dataset
    mock_metric_b_factory = MagicMock(return_value=mock_metric_b_instance)

    _TECHNIQUES["mock_tech"] = mock_technique_factory
    _METRICS["mock_metric_a"] = mock_metric_a_factory
    _METRICS["mock_metric_b"] = mock_metric_b_factory

    return {
        "technique_factory": mock_technique_factory,
        "technique_instance": mock_technique_instance,
        "metric_a_factory": mock_metric_a_factory,
        "metric_a_instance": mock_metric_a_instance,
        "metric_b_factory": mock_metric_b_factory,
        "metric_b_instance": mock_metric_b_instance,
        "dataset": mock_dataset,
        "img": mock_img,
    }


def _make_runner(tmp_path, **overrides):
    """Helper to build a MultiBenchmarkRunner with defaults."""
    kwargs = dict(
        technique_name="mock_tech",
        metric_names=["mock_metric_a", "mock_metric_b"],
        technique_config={"device": "cpu"},
        metric_configs={
            "mock_metric_a": {"use_nudenet": False},
            "mock_metric_b": {"device": "cpu"},
        },
        output_dir=str(tmp_path / "results"),
    )
    kwargs.update(overrides)
    return MultiBenchmarkRunner(**kwargs)


class TestValidation:
    def test_raises_on_unknown_technique(self, mock_multi_registry, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            _make_runner(tmp_path, technique_name="nonexistent")

    def test_raises_on_unknown_metric(self, mock_multi_registry, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            _make_runner(tmp_path, metric_names=["mock_metric_a", "nonexistent"])

    def test_raises_on_empty_metric_names(self, mock_multi_registry, tmp_path):
        with pytest.raises(ValueError, match="must not be empty"):
            _make_runner(tmp_path, metric_names=[])

    def test_raises_on_duplicate_metric_names(self, mock_multi_registry, tmp_path):
        with pytest.raises(ValueError, match="duplicates"):
            _make_runner(tmp_path, metric_names=["mock_metric_a", "mock_metric_a"])

    def test_accepts_valid_names(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        assert runner.technique_name == "mock_tech"
        assert runner.metric_names == ["mock_metric_a", "mock_metric_b"]


class TestRunnerCalls:
    def test_technique_factory_called_once(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        mock_multi_registry["technique_factory"].assert_called_once_with(device="cpu")

    def test_generate_called_once(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        mock_multi_registry["technique_instance"].generate.assert_called_once_with(
            prompts=["prompt1", "prompt2"]
        )

    def test_first_metric_loads_dataset(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        mock_multi_registry["metric_a_instance"].load_dataset.assert_called_once()

    def test_second_metric_loads_dataset_for_side_effects(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        mock_multi_registry["metric_b_instance"].load_dataset.assert_called_once()

    def test_all_metrics_compute_called(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        mock_multi_registry["metric_a_instance"].compute.assert_called_once()
        mock_multi_registry["metric_b_instance"].compute.assert_called_once()

    def test_all_metrics_receive_same_images(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        imgs_a = mock_multi_registry["metric_a_instance"].compute.call_args.kwargs["images"]
        imgs_b = mock_multi_registry["metric_b_instance"].compute.call_args.kwargs["images"]
        assert imgs_a is imgs_b

    def test_all_metrics_receive_same_prompts(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        prompts_a = mock_multi_registry["metric_a_instance"].compute.call_args.kwargs["prompts"]
        prompts_b = mock_multi_registry["metric_b_instance"].compute.call_args.kwargs["prompts"]
        assert prompts_a == prompts_b == ["prompt1", "prompt2"]

    def test_metric_configs_passed_correctly(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        mock_multi_registry["metric_a_factory"].assert_called_once_with(use_nudenet=False)
        mock_multi_registry["metric_b_factory"].assert_called_once_with(device="cpu")


class TestRunnerReport:
    def test_report_has_multi_structure(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        for key in ("run_id", "timestamp", "technique_name", "metric_names",
                     "dataset_name", "dataset_metadata", "technique_config",
                     "metric_configs", "metric_results"):
            assert key in report

    def test_report_metric_results_keyed_by_name(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        assert "mock_metric_a" in report["metric_results"]
        assert "mock_metric_b" in report["metric_results"]

    def test_report_values_correct(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        assert report["technique_name"] == "mock_tech"
        assert report["metric_names"] == ["mock_metric_a", "mock_metric_b"]
        assert report["dataset_name"] == "test"
        assert report["metric_results"]["mock_metric_a"]["value"] == 0.42
        assert report["metric_results"]["mock_metric_a"]["name"] == "MetricA"
        assert report["metric_results"]["mock_metric_b"]["value"] == 0.85
        assert report["metric_results"]["mock_metric_b"]["name"] == "MetricB"

    def test_run_id_is_8_char_hex(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        run_id = report["run_id"]
        assert len(run_id) == 8
        assert all(c in "0123456789abcdef" for c in run_id)


class TestRunnerArtifacts:
    def test_creates_multi_folder(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        run_id = report["run_id"]
        run_dir = os.path.join(str(tmp_path / "results"), f"mock_tech_multi_{run_id}")
        assert os.path.isdir(run_dir)

    def test_images_saved_once(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        run_id = report["run_id"]
        images_dir = os.path.join(str(tmp_path / "results"), f"mock_tech_multi_{run_id}", "images")
        assert os.path.isdir(images_dir)
        pngs = [f for f in os.listdir(images_dir) if f.endswith(".png")]
        assert len(pngs) == 2

    def test_report_json_contains_all_metrics(self, mock_multi_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        run_id = report["run_id"]
        report_path = os.path.join(
            str(tmp_path / "results"), f"mock_tech_multi_{run_id}", f"{run_id}_report.json"
        )
        assert os.path.isfile(report_path)
        with open(report_path) as f:
            saved = json.load(f)
        assert "mock_metric_a" in saved["metric_results"]
        assert "mock_metric_b" in saved["metric_results"]


class TestExecutionOrder:
    def test_execution_order(self, mock_multi_registry, tmp_path):
        call_order = []

        mock_multi_registry["metric_a_factory"].side_effect = lambda **kw: (
            call_order.append("metric_a_factory"),
            mock_multi_registry["metric_a_instance"],
        )[1]
        mock_multi_registry["metric_a_instance"].load_dataset.side_effect = lambda: (
            call_order.append("load_dataset"),
            mock_multi_registry["dataset"],
        )[1]
        mock_multi_registry["technique_factory"].side_effect = lambda **kw: (
            call_order.append("technique_factory"),
            mock_multi_registry["technique_instance"],
        )[1]
        mock_multi_registry["technique_instance"].generate.side_effect = lambda **kw: (
            call_order.append("generate"),
            [mock_multi_registry["img"], mock_multi_registry["img"]],
        )[1]
        mock_multi_registry["metric_a_instance"].compute.side_effect = lambda **kw: (
            call_order.append("compute_a"),
            MetricResult("A", 0.0),
        )[1]
        mock_multi_registry["metric_b_factory"].side_effect = lambda **kw: (
            call_order.append("metric_b_factory"),
            mock_multi_registry["metric_b_instance"],
        )[1]
        mock_multi_registry["metric_b_instance"].load_dataset.side_effect = lambda: (
            call_order.append("load_dataset_b"),
            mock_multi_registry["dataset"],
        )[1]
        mock_multi_registry["metric_b_instance"].compute.side_effect = lambda **kw: (
            call_order.append("compute_b"),
            MetricResult("B", 0.0),
        )[1]

        runner = _make_runner(tmp_path)
        runner.run()

        assert call_order == [
            "metric_a_factory",
            "load_dataset",
            "technique_factory",
            "generate",
            "compute_a",
            "metric_b_factory",
            "load_dataset_b",
            "compute_b",
        ]
