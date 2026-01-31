import json
import os
import pytest
from eval_learn.artifacts.writer import ArtifactWriter


@pytest.fixture
def writer(tmp_path):
    return ArtifactWriter(base_dir=str(tmp_path))


class TestArtifactWriter:
    def test_save_run_creates_directory(self, writer, dummy_pil_image, tmp_path):
        img = dummy_pil_image()
        writer.save_run("TestRun", [img], {}, timestamp=1000)
        images_dir = os.path.join(str(tmp_path), "TestRun", "images", "run_1000")
        assert os.path.isdir(images_dir)

    def test_save_run_saves_images(self, writer, dummy_pil_image, tmp_path):
        imgs = [dummy_pil_image(c) for c in ("red", "green", "blue")]
        writer.save_run("TestRun", imgs, {}, timestamp=1000)
        images_dir = os.path.join(str(tmp_path), "TestRun", "images", "run_1000")
        for i in range(3):
            path = os.path.join(images_dir, f"{i}.png")
            assert os.path.isfile(path)

    def test_save_run_saves_report_json(self, writer, dummy_pil_image, tmp_path):
        img = dummy_pil_image()
        writer.save_run("TestRun", [img], {"key": "val"}, timestamp=1000)
        report_path = os.path.join(str(tmp_path), "TestRun", "report_1000.json")
        assert os.path.isfile(report_path)
        with open(report_path) as f:
            data = json.load(f)
        assert data["key"] == "val"
        assert "image_paths" in data
        assert "timestamp" in data

    def test_save_run_report_has_image_paths(self, writer, dummy_pil_image, tmp_path):
        imgs = [dummy_pil_image() for _ in range(3)]
        writer.save_run("TestRun", imgs, {}, timestamp=1000)
        report_path = os.path.join(str(tmp_path), "TestRun", "report_1000.json")
        with open(report_path) as f:
            data = json.load(f)
        assert len(data["image_paths"]) == 3

    def test_save_run_report_has_timestamp(self, writer, dummy_pil_image, tmp_path):
        writer.save_run("TestRun", [dummy_pil_image()], {}, timestamp=1000)
        report_path = os.path.join(str(tmp_path), "TestRun", "report_1000.json")
        with open(report_path) as f:
            data = json.load(f)
        assert data["timestamp"] == 1000

    def test_save_run_empty_images(self, writer, tmp_path):
        writer.save_run("TestRun", [], {}, timestamp=2000)
        report_path = os.path.join(str(tmp_path), "TestRun", "report_2000.json")
        with open(report_path) as f:
            data = json.load(f)
        assert data["image_paths"] == []

    def test_save_run_returns_report_path(self, writer, dummy_pil_image):
        result = writer.save_run("TestRun", [dummy_pil_image()], {}, timestamp=3000)
        assert os.path.isfile(result)
        assert result.endswith(".json")

    def test_save_run_multiple_runs(self, writer, dummy_pil_image, tmp_path):
        writer.save_run("Run1", [dummy_pil_image()], {}, timestamp=100)
        writer.save_run("Run2", [dummy_pil_image()], {}, timestamp=200)
        assert os.path.isdir(os.path.join(str(tmp_path), "Run1"))
        assert os.path.isdir(os.path.join(str(tmp_path), "Run2"))
