"""
hf_sync.py — Hugging Face Hub integration for syncing datasets, results, and images.

Uses three separate HF dataset repos:
  - datasets_repo: raw input data (CSVs, JSONs, parquets)
  - results_repo:  run report JSONs only (lightweight)
  - images_repo:   generated images organised by run (heavy)
"""

import os
from typing import Dict, Optional

from huggingface_hub import HfApi, snapshot_download

from ..logging_utils import get_logger

logger = get_logger(__name__)


class HFSync:
    """
    Syncs benchmark artifacts with Hugging Face Hub dataset repositories.

    Folder mapping::

        Local results/sld_asr_<id>/          →  results repo:  sld_asr_<id>/<id>_report.json
                                              →  images repo:   sld_asr_<id>/0.png ...
        Local data/                          ←  datasets repo: (full snapshot)
    """

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
        """
        Upload a run's report JSON to the results repo.

        Args:
            run_dir: Local path to the run folder
                     (e.g. ``results/sld_asr_a1b2c3d4``).
            run_id:  The 8-char hex run identifier.

        Returns:
            URL of the uploaded file on the Hub.
        """
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
        """
        Upload a run's images folder to the images repo.

        Preserves subdirectory structure (e.g. target/, retain/ for ERR).

        Args:
            run_dir: Local path to the run folder.
            run_id:  The 8-char hex run identifier.

        Returns:
            URL of the uploaded folder on the Hub.
        """
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
        """
        Push both report and images for a completed run.

        Args:
            run_dir: Local path to the run folder.
            run_id:  The 8-char hex run identifier.

        Returns:
            Dict with ``report_url`` and ``images_url``.
        """
        report_url = self.push_report(run_dir, run_id)
        images_url = self.push_images(run_dir, run_id)
        return {"report_url": report_url, "images_url": images_url}

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------

    def pull_datasets(self, local_dir: str = "data") -> str:
        """
        Download all datasets from the HF datasets repo.

        Args:
            local_dir: Local directory to download into (default ``data/``).

        Returns:
            Path to the downloaded snapshot.
        """
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
        """
        Download all report JSONs from the results repo.

        Args:
            local_dir: Local directory to download into (default ``results/``).

        Returns:
            Path to the downloaded snapshot.
        """
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
        """
        Download images for a single run only.

        Args:
            folder_name: Run folder name (e.g. ``sld_asr_a1b2c3d4``).
            local_dir:   Local directory to download into.

        Returns:
            Path to the downloaded snapshot.
        """
        logger.info(
            "Pulling images for %s from %s", folder_name, self.images_repo
        )
        path = snapshot_download(
            repo_id=self.images_repo,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=f"{folder_name}/**",
            token=self.token,
        )
        logger.info("Images downloaded to %s", path)
        return path
