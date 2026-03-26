"""Hugging Face Hub integration for syncing datasets, results, and images."""

import os
from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi, snapshot_download

from ..logging_utils import get_logger

logger = get_logger(__name__)


class HFSync:
    """Syncs benchmark artifacts with Hugging Face Hub dataset repositories."""

    def __init__(
        self,
        datasets_repo: str,
        results_repo: str,
        images_repo: str,
        token: Optional[str] = None,
        create_pr: bool = False,
    ):
        self.token = token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token)
        self.datasets_repo = datasets_repo
        self.results_repo = results_repo
        self.images_repo = images_repo
        self.create_pr = create_pr

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push_report(self, run_dir: str, run_id: str) -> str:
        """Upload run report JSON to results repo."""
        report_path = os.path.join(run_dir, f"{run_id}_report.json")
        if not os.path.isfile(report_path):
            raise FileNotFoundError(f"Report not found: {report_path}")

        folder_name = os.path.basename(run_dir)
        path_in_repo = f"{folder_name}/{run_id}_report.json"

        logger.info("Pushing report to %s/%s", self.results_repo, path_in_repo)
        url = self.api.upload_file(
            path_or_fileobj=report_path,
            path_in_repo=path_in_repo,
            repo_id=self.results_repo,
            repo_type="dataset",
            commit_message=f"Add report for run {run_id}",
            create_pr=self.create_pr,
        )
        logger.info("Report pushed: %s", url)
        return url

    def push_images(self, run_dir: str, run_id: str) -> str:
        """Upload run images folder to images repo."""
        images_dir = os.path.join(run_dir, "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        folder_name = os.path.basename(run_dir)

        logger.info("Pushing images to %s/%s/", self.images_repo, folder_name)
        url = self.api.upload_folder(
            folder_path=images_dir,
            path_in_repo=folder_name,
            repo_id=self.images_repo,
            repo_type="dataset",
            commit_message=f"Add images for run {run_id}",
            create_pr=self.create_pr,
        )
        logger.info("Images pushed: %s", url)
        return url

    def push_run(self, run_dir: str, run_id: str) -> Dict[str, str]:
        """Push both report and images for a completed run."""
        report_url = self.push_report(run_dir, run_id)
        images_url = self.push_images(run_dir, run_id)
        return {"report_url": report_url, "images_url": images_url}

    def push_matrix_run(self, output_dir: str, matrix_run_id: str) -> Dict[str, Any]:
        """Push all artifacts for a matrix benchmark run."""
        # Push the matrix-level report
        matrix_report_name = f"matrix_{matrix_run_id}_report.json"
        matrix_report_path = os.path.join(output_dir, matrix_report_name)
        if not os.path.isfile(matrix_report_path):
            raise FileNotFoundError(f"Matrix report not found: {matrix_report_path}")

        logger.info(
            "Pushing matrix report to %s/%s", self.results_repo, matrix_report_name
        )
        matrix_report_url = self.api.upload_file(
            path_or_fileobj=matrix_report_path,
            path_in_repo=matrix_report_name,
            repo_id=self.results_repo,
            repo_type="dataset",
            commit_message=f"Add matrix report for run {matrix_run_id}",
            create_pr=self.create_pr,
        )

        # Find and push each sub-run folder
        sub_run_dirs = sorted(
            [
                d
                for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d)) and "_multi_" in d
            ]
        )

        sub_urls = {}
        for folder_name in sub_run_dirs:
            run_dir = os.path.join(output_dir, folder_name)
            # Extract sub_run_id — last segment after final underscore
            sub_run_id = folder_name.rsplit("_", 1)[-1]
            logger.info("Pushing sub-run %s (id=%s)", folder_name, sub_run_id)
            urls = self.push_run(run_dir, sub_run_id)
            sub_urls[folder_name] = urls

        return {"matrix_report_url": matrix_report_url, "sub_runs": sub_urls}

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------

    def pull_datasets(self, local_dir: str = "data") -> str:
        """Download all datasets from HF datasets repo."""
        logger.info("Pulling datasets from %s to %s/", self.datasets_repo, local_dir)
        path = snapshot_download(
            repo_id=self.datasets_repo,
            repo_type="dataset",
            local_dir=local_dir,
            token=self.token,
        )
        logger.info("Datasets downloaded to %s", path)
        return path

    def pull_results(self, local_dir: str = "results") -> str:
        """Download all report JSONs from results repo."""
        logger.info("Pulling results from %s to %s/", self.results_repo, local_dir)
        path = snapshot_download(
            repo_id=self.results_repo,
            repo_type="dataset",
            local_dir=local_dir,
            token=self.token,
        )
        logger.info("Results downloaded to %s", path)
        return path

    def pull_run_images(self, folder_name: str, local_dir: str = "results") -> str:
        """Download images for a single run."""
        logger.info("Pulling images for %s from %s", folder_name, self.images_repo)
        path = snapshot_download(
            repo_id=self.images_repo,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=f"{folder_name}/**",
            token=self.token,
        )
        logger.info("Images downloaded to %s", path)
        return path

    def pull_matrix_results(
        self, matrix_run_id: str, local_dir: str = "results"
    ) -> str:
        """Download matrix report and all associated sub-run reports."""
        logger.info(
            "Pulling matrix results for %s from %s", matrix_run_id, self.results_repo
        )
        path = snapshot_download(
            repo_id=self.results_repo,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=[
                f"matrix_{matrix_run_id}_report.json",
                f"*_multi_*/**",
            ],
            token=self.token,
        )
        logger.info("Matrix results downloaded to %s", path)
        return path
