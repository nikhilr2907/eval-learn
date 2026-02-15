import json
import os
import pytest
from unittest.mock import patch, MagicMock, call
from eval_learn.registry.hf_sync import HFSync


FAKE_REPOS = {
    "datasets_repo": "test-org/datasets",
    "results_repo": "test-org/results",
    "images_repo": "test-org/images",
}


@pytest.fixture
def mock_hf_api():
    """Patch HfApi so no real HTTP calls are made."""
    with patch("eval_learn.registry.hf_sync.HfApi") as MockApi:
        instance = MockApi.return_value
        instance.upload_file.return_value = "https://huggingface.co/uploaded/file"
        instance.upload_folder.return_value = "https://huggingface.co/uploaded/folder"
        yield instance


@pytest.fixture
def mock_snapshot():
    """Patch snapshot_download to return a fake path."""
    with patch("eval_learn.registry.hf_sync.snapshot_download") as mock_sd:
        mock_sd.return_value = "/fake/download/path"
        yield mock_sd


@pytest.fixture
def sync(mock_hf_api):
    """Create an HFSync instance with mocked API."""
    return HFSync(**FAKE_REPOS, token="hf_fake_token")


@pytest.fixture
def run_dir(tmp_path):
    """Create a realistic local run directory with report and images."""
    run_id = "a1b2c3d4"
    folder = tmp_path / "results" / f"sld_asr_{run_id}"
    images_dir = folder / "images"
    images_dir.mkdir(parents=True)

    # Create a report JSON
    report = {"run_id": run_id, "metric_result": {"value": 0.42}}
    with open(folder / f"{run_id}_report.json", "w") as f:
        json.dump(report, f)

    # Create some fake images
    (images_dir / "0.png").write_bytes(b"fake image 0")
    (images_dir / "1.png").write_bytes(b"fake image 1")

    return {"path": str(folder), "run_id": run_id}


@pytest.fixture
def run_dir_with_categories(tmp_path):
    """Create a run directory with category subdirectories (ERR-style)."""
    run_id = "e5f6g7h8"
    folder = tmp_path / "results" / f"sld_err_{run_id}"
    images_dir = folder / "images"

    for cat in ("target", "retain", "adversarial"):
        cat_dir = images_dir / cat
        cat_dir.mkdir(parents=True)
        (cat_dir / "0.png").write_bytes(b"fake image")

    report = {"run_id": run_id, "metric_result": {"value": 0.75}}
    with open(folder / f"{run_id}_report.json", "w") as f:
        json.dump(report, f)

    return {"path": str(folder), "run_id": run_id}


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------

class TestHFSyncInit:
    def test_stores_repo_ids(self, mock_hf_api):
        sync = HFSync(**FAKE_REPOS, token="hf_test")
        assert sync.datasets_repo == "test-org/datasets"
        assert sync.results_repo == "test-org/results"
        assert sync.images_repo == "test-org/images"

    def test_token_from_param(self, mock_hf_api):
        sync = HFSync(**FAKE_REPOS, token="hf_explicit")
        assert sync.token == "hf_explicit"

    def test_token_from_env(self, mock_hf_api, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        sync = HFSync(**FAKE_REPOS)
        assert sync.token == "hf_from_env"


# ------------------------------------------------------------------
# Push
# ------------------------------------------------------------------

class TestPushReport:
    def test_uploads_report_to_results_repo(self, sync, mock_hf_api, run_dir):
        url = sync.push_report(run_dir["path"], run_dir["run_id"])

        assert url == "https://huggingface.co/uploaded/file"
        mock_hf_api.upload_file.assert_called_once()
        call_kwargs = mock_hf_api.upload_file.call_args.kwargs
        assert call_kwargs["repo_id"] == "test-org/results"
        assert call_kwargs["repo_type"] == "dataset"
        assert call_kwargs["path_in_repo"] == "sld_asr_a1b2c3d4/a1b2c3d4_report.json"

    def test_raises_if_report_missing(self, sync, tmp_path):
        fake_dir = str(tmp_path / "nonexistent_run")
        os.makedirs(fake_dir, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="Report not found"):
            sync.push_report(fake_dir, "deadbeef")


class TestPushImages:
    def test_uploads_images_folder_to_images_repo(self, sync, mock_hf_api, run_dir):
        url = sync.push_images(run_dir["path"], run_dir["run_id"])

        assert url == "https://huggingface.co/uploaded/folder"
        mock_hf_api.upload_folder.assert_called_once()
        call_kwargs = mock_hf_api.upload_folder.call_args.kwargs
        assert call_kwargs["repo_id"] == "test-org/images"
        assert call_kwargs["repo_type"] == "dataset"
        assert call_kwargs["path_in_repo"] == "sld_asr_a1b2c3d4"

    def test_raises_if_images_dir_missing(self, sync, tmp_path):
        fake_dir = str(tmp_path / "no_images")
        os.makedirs(fake_dir, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            sync.push_images(fake_dir, "deadbeef")

    def test_preserves_category_structure(self, sync, mock_hf_api, run_dir_with_categories):
        sync.push_images(run_dir_with_categories["path"], run_dir_with_categories["run_id"])

        call_kwargs = mock_hf_api.upload_folder.call_args.kwargs
        # folder_path should point to the images/ dir containing target/retain/adversarial
        images_dir = os.path.join(run_dir_with_categories["path"], "images")
        assert call_kwargs["folder_path"] == images_dir


class TestPushRun:
    def test_calls_both_push_methods(self, sync, mock_hf_api, run_dir):
        result = sync.push_run(run_dir["path"], run_dir["run_id"])

        assert "report_url" in result
        assert "images_url" in result
        mock_hf_api.upload_file.assert_called_once()
        mock_hf_api.upload_folder.assert_called_once()


class TestPushMatrixRun:
    @pytest.fixture
    def matrix_run_dir(self, tmp_path):
        """Create a matrix-style output directory with sub-runs and matrix report."""
        matrix_id = "m1a2b3c4"
        output = tmp_path / "results"
        output.mkdir()

        # Sub-run A
        sub_a_id = "aa111111"
        sub_a = output / f"sld_multi_{sub_a_id}"
        (sub_a / "images").mkdir(parents=True)
        (sub_a / "images" / "0.png").write_bytes(b"fake")
        with open(sub_a / f"{sub_a_id}_report.json", "w") as f:
            json.dump({"run_id": sub_a_id}, f)

        # Sub-run B
        sub_b_id = "bb222222"
        sub_b = output / f"esd_multi_{sub_b_id}"
        (sub_b / "images").mkdir(parents=True)
        (sub_b / "images" / "0.png").write_bytes(b"fake")
        with open(sub_b / f"{sub_b_id}_report.json", "w") as f:
            json.dump({"run_id": sub_b_id}, f)

        # Matrix report
        with open(output / f"matrix_{matrix_id}_report.json", "w") as f:
            json.dump({"run_id": matrix_id, "comparison": {}}, f)

        return {
            "path": str(output),
            "matrix_id": matrix_id,
            "sub_a_id": sub_a_id,
            "sub_b_id": sub_b_id,
        }

    def test_pushes_matrix_report(self, sync, mock_hf_api, matrix_run_dir):
        result = sync.push_matrix_run(matrix_run_dir["path"], matrix_run_dir["matrix_id"])

        assert "matrix_report_url" in result
        # First upload_file call is the matrix report
        first_call = mock_hf_api.upload_file.call_args_list[0]
        assert first_call.kwargs["path_in_repo"] == "matrix_m1a2b3c4_report.json"
        assert first_call.kwargs["repo_id"] == "test-org/results"

    def test_pushes_all_sub_runs(self, sync, mock_hf_api, matrix_run_dir):
        result = sync.push_matrix_run(matrix_run_dir["path"], matrix_run_dir["matrix_id"])

        assert "sub_runs" in result
        assert len(result["sub_runs"]) == 2
        folder_names = sorted(result["sub_runs"].keys())
        assert "esd_multi_bb222222" in folder_names
        assert "sld_multi_aa111111" in folder_names

    def test_each_sub_run_has_report_and_images_url(self, sync, mock_hf_api, matrix_run_dir):
        result = sync.push_matrix_run(matrix_run_dir["path"], matrix_run_dir["matrix_id"])

        for folder_name, urls in result["sub_runs"].items():
            assert "report_url" in urls
            assert "images_url" in urls

    def test_raises_if_matrix_report_missing(self, sync, tmp_path):
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="Matrix report not found"):
            sync.push_matrix_run(empty_dir, "deadbeef")


# ------------------------------------------------------------------
# Pull
# ------------------------------------------------------------------

class TestPullDatasets:
    def test_calls_snapshot_download_for_datasets(self, sync, mock_snapshot):
        path = sync.pull_datasets("data")

        assert path == "/fake/download/path"
        mock_snapshot.assert_called_once_with(
            repo_id="test-org/datasets",
            repo_type="dataset",
            local_dir="data",
            token="hf_fake_token",
        )

    def test_default_local_dir(self, sync, mock_snapshot):
        sync.pull_datasets()
        call_kwargs = mock_snapshot.call_args.kwargs
        assert call_kwargs["local_dir"] == "data"


class TestPullResults:
    def test_calls_snapshot_download_for_results(self, sync, mock_snapshot):
        path = sync.pull_results("my_results")

        mock_snapshot.assert_called_once_with(
            repo_id="test-org/results",
            repo_type="dataset",
            local_dir="my_results",
            token="hf_fake_token",
        )

    def test_default_local_dir(self, sync, mock_snapshot):
        sync.pull_results()
        call_kwargs = mock_snapshot.call_args.kwargs
        assert call_kwargs["local_dir"] == "results"


class TestPullRunImages:
    def test_calls_snapshot_with_allow_patterns(self, sync, mock_snapshot):
        path = sync.pull_run_images("sld_asr_a1b2c3d4", local_dir="results")

        mock_snapshot.assert_called_once_with(
            repo_id="test-org/images",
            repo_type="dataset",
            local_dir="results",
            allow_patterns="sld_asr_a1b2c3d4/**",
            token="hf_fake_token",
        )


class TestPullMatrixResults:
    def test_calls_snapshot_with_matrix_patterns(self, sync, mock_snapshot):
        path = sync.pull_matrix_results("m1a2b3c4", local_dir="results")

        assert path == "/fake/download/path"
        mock_snapshot.assert_called_once_with(
            repo_id="test-org/results",
            repo_type="dataset",
            local_dir="results",
            allow_patterns=[
                "matrix_m1a2b3c4_report.json",
                "*_multi_*/**",
            ],
            token="hf_fake_token",
        )

    def test_default_local_dir(self, sync, mock_snapshot):
        sync.pull_matrix_results("m1a2b3c4")
        call_kwargs = mock_snapshot.call_args.kwargs
        assert call_kwargs["local_dir"] == "results"
