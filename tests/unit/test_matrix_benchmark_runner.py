import json
import os
import pytest
from unittest.mock import MagicMock, patch
from eval_learn.runners import MatrixBenchmarkRunner
from eval_learn.registry.local import _TECHNIQUES, _METRICS
from eval_learn.types import Dataset, MetricResult


@pytest.fixture
def mock_matrix_registry(reset_registry, dummy_pil_image):
    """Register two mock techniques and two mock metrics."""
    mock_img = dummy_pil_image()

    mock_dataset = Dataset(
        prompts=["prompt1", "prompt2"],
        metadata={"source": "test", "concepts": ["c1", "c2"]}
    )

    # Technique A
    mock_tech_a_instance = MagicMock()
    mock_tech_a_instance.generate.return_value = [mock_img, mock_img]
    mock_tech_a_factory = MagicMock(return_value=mock_tech_a_instance)

    # Technique B
    mock_tech_b_instance = MagicMock()
    mock_tech_b_instance.generate.return_value = [mock_img, mock_img]
    mock_tech_b_factory = MagicMock(return_value=mock_tech_b_instance)

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

    _TECHNIQUES["mock_tech_a"] = mock_tech_a_factory
    _TECHNIQUES["mock_tech_b"] = mock_tech_b_factory
    _METRICS["mock_metric_a"] = mock_metric_a_factory
    _METRICS["mock_metric_b"] = mock_metric_b_factory

    return {
        "tech_a_factory": mock_tech_a_factory,
        "tech_a_instance": mock_tech_a_instance,
        "tech_b_factory": mock_tech_b_factory,
        "tech_b_instance": mock_tech_b_instance,
        "metric_a_factory": mock_metric_a_factory,
        "metric_a_instance": mock_metric_a_instance,
        "metric_b_factory": mock_metric_b_factory,
        "metric_b_instance": mock_metric_b_instance,
        "dataset": mock_dataset,
        "img": mock_img,
    }


def _make_runner(tmp_path, **overrides):
    """Helper to build a MatrixBenchmarkRunner with defaults."""
    kwargs = dict(
        technique_names=["mock_tech_a", "mock_tech_b"],
        metric_names=["mock_metric_a", "mock_metric_b"],
        technique_configs={
            "mock_tech_a": {"device": "cpu"},
            "mock_tech_b": {"device": "cpu"},
        },
        metric_configs={
            "mock_metric_a": {"use_nudenet": False},
            "mock_metric_b": {"device": "cpu"},
        },
        output_dir=str(tmp_path / "results"),
    )
    kwargs.update(overrides)
    return MatrixBenchmarkRunner(**kwargs)


class TestValidation:
    def test_raises_on_unknown_technique(self, mock_matrix_registry, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            _make_runner(tmp_path, technique_names=["nonexistent"])

    def test_raises_on_unknown_metric(self, mock_matrix_registry, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            _make_runner(tmp_path, metric_names=["nonexistent"])

    def test_raises_on_empty_technique_names(self, mock_matrix_registry, tmp_path):
        with pytest.raises(ValueError, match="must not be empty"):
            _make_runner(tmp_path, technique_names=[])

    def test_raises_on_empty_metric_names(self, mock_matrix_registry, tmp_path):
        with pytest.raises(ValueError, match="must not be empty"):
            _make_runner(tmp_path, metric_names=[])

    def test_raises_on_duplicate_technique_names(self, mock_matrix_registry, tmp_path):
        with pytest.raises(ValueError, match="duplicates"):
            _make_runner(tmp_path, technique_names=["mock_tech_a", "mock_tech_a"])

    def test_raises_on_duplicate_metric_names(self, mock_matrix_registry, tmp_path):
        with pytest.raises(ValueError, match="duplicates"):
            _make_runner(tmp_path, metric_names=["mock_metric_a", "mock_metric_a"])

    def test_accepts_valid_names(self, mock_matrix_registry, tmp_path):
        runner = _make_runner(tmp_path)
        assert runner.technique_names == ["mock_tech_a", "mock_tech_b"]
        assert runner.metric_names == ["mock_metric_a", "mock_metric_b"]


class TestComposition:
    def test_each_technique_generates_images(self, mock_matrix_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        mock_matrix_registry["tech_a_instance"].generate.assert_called_once()
        mock_matrix_registry["tech_b_instance"].generate.assert_called_once()

    def test_each_technique_evaluated_by_all_metrics(self, mock_matrix_registry, tmp_path):
        runner = _make_runner(tmp_path)
        runner.run()
        # Each metric factory is called twice (once per technique)
        assert mock_matrix_registry["metric_a_factory"].call_count == 2
        assert mock_matrix_registry["metric_b_factory"].call_count == 2


class TestVRAMCleanup:
    def test_cleanup_called_between_techniques(self, mock_matrix_registry, tmp_path):
        with patch("eval_learn.runners.matrix_benchmark_runner.gc.collect") as mock_gc:
            runner = _make_runner(tmp_path)
            runner.run()
            # Called once per technique (after each, including the last)
            assert mock_gc.call_count == len(runner.technique_names)

    def test_cleanup_works_without_torch(self, mock_matrix_registry, tmp_path):
        """Cleanup should not fail if torch is not installed."""
        with patch("eval_learn.runners.matrix_benchmark_runner.gc.collect"):
            with patch.dict("sys.modules", {"torch": None}):
                runner = _make_runner(tmp_path)
                # Should not raise
                runner.run()


class TestRunnerReport:
    def test_report_has_matrix_structure(self, mock_matrix_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        for key in ("run_id", "timestamp", "technique_names", "metric_names",
                     "technique_configs", "metric_configs",
                     "technique_reports", "comparison"):
            assert key in report

    def test_technique_reports_embedded(self, mock_matrix_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        assert "mock_tech_a" in report["technique_reports"]
        assert "mock_tech_b" in report["technique_reports"]
        # Each sub-report should have metric_results
        for tech_report in report["technique_reports"].values():
            assert "metric_results" in tech_report

    def test_comparison_table_structure(self, mock_matrix_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        comparison = report["comparison"]
        assert "mock_metric_a" in comparison
        assert "mock_metric_b" in comparison
        assert "mock_tech_a" in comparison["mock_metric_a"]
        assert "mock_tech_b" in comparison["mock_metric_a"]

    def test_comparison_values_correct(self, mock_matrix_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        comparison = report["comparison"]
        # Both techniques use same mock metrics, so values should match
        assert comparison["mock_metric_a"]["mock_tech_a"] == 0.42
        assert comparison["mock_metric_a"]["mock_tech_b"] == 0.42
        assert comparison["mock_metric_b"]["mock_tech_a"] == 0.85
        assert comparison["mock_metric_b"]["mock_tech_b"] == 0.85

    def test_run_id_is_8_char_hex(self, mock_matrix_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        run_id = report["run_id"]
        assert len(run_id) == 8
        assert all(c in "0123456789abcdef" for c in run_id)


class TestRunnerArtifacts:
    def test_creates_subfolder_per_technique(self, mock_matrix_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        results_dir = str(tmp_path / "results")
        # Each technique should have a {technique}_multi_{sub_id}/ folder
        tech_a_report = report["technique_reports"]["mock_tech_a"]
        tech_b_report = report["technique_reports"]["mock_tech_b"]
        tech_a_dir = os.path.join(results_dir, f"mock_tech_a_multi_{tech_a_report['run_id']}")
        tech_b_dir = os.path.join(results_dir, f"mock_tech_b_multi_{tech_b_report['run_id']}")
        assert os.path.isdir(tech_a_dir)
        assert os.path.isdir(tech_b_dir)

    def test_matrix_report_saved(self, mock_matrix_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        run_id = report["run_id"]
        report_path = os.path.join(str(tmp_path / "results"), f"matrix_{run_id}_report.json")
        assert os.path.isfile(report_path)
        with open(report_path) as f:
            saved = json.load(f)
        assert "comparison" in saved
        assert "technique_reports" in saved

    def test_images_separate_per_technique(self, mock_matrix_registry, tmp_path):
        runner = _make_runner(tmp_path)
        report = runner.run()
        for tech_name in ["mock_tech_a", "mock_tech_b"]:
            sub_report = report["technique_reports"][tech_name]
            images_dir = os.path.join(
                str(tmp_path / "results"),
                f"{tech_name}_multi_{sub_report['run_id']}",
                "images",
            )
            assert os.path.isdir(images_dir)
            pngs = [f for f in os.listdir(images_dir) if f.endswith(".png")]
            assert len(pngs) == 2
