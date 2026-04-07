"""Tests for ASRCustomMetric update and evaluation loop."""

import sys
from unittest.mock import MagicMock, patch
import pytest
import torch
from PIL import Image

# ring_a_bell is not installed in test env — stub it before the metric module loads
sys.modules.setdefault("ring_a_bell", MagicMock())
sys.modules.setdefault("ring_a_bell.encoder", MagicMock())

from eval_learn.metrics.asr_custom.metric import ASRCustomMetric  # noqa: E402
from eval_learn.types import MetricResult  # noqa: E402

DIM = 16  # small embedding dimension for test tensors


def _make_metric(concept="nudity", threshold=0.3, device="cpu"):
    """Instantiate ASRCustomMetric with mocked CLIP models, no discovery."""
    with (
        patch("eval_learn.metrics.asr_custom.metric.CLIPModel") as mock_model_cls,
        patch("eval_learn.metrics.asr_custom.metric.CLIPProcessor") as mock_proc_cls,
    ):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        import tempfile, csv, os
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        writer = csv.writer(tmp)
        writer.writerow(["prompt"])
        writer.writerow(["a test prompt"])
        tmp.close()

        metric = ASRCustomMetric(
            concept_name=concept,
            seed_prompts_csv=tmp.name,
            enable_discovery=False,
            similarity_threshold=threshold,
            device=device,
        )
        os.unlink(tmp.name)

    return metric


def _set_clip_response(metric, image_feat: torch.Tensor, text_feat: torch.Tensor):
    """Wire metric's CLIP model/processor to return specific feature tensors."""
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs

    metric.clip_processor.side_effect = None
    metric.clip_processor.return_value = mock_inputs
    metric.clip_model.get_image_features.return_value = image_feat
    metric.clip_model.get_text_features.return_value = text_feat


# ---------------------------------------------------------------------------
# _evaluate_batch_clip
# ---------------------------------------------------------------------------

class TestEvaluateBatchClip:
    """Tests for the CLIP similarity calculation in _evaluate_batch_clip."""

    def test_above_threshold_marked_unsafe(self):
        """Image with similarity > threshold should be flagged unsafe."""
        metric = _make_metric(threshold=0.3, device="cpu")

        # image and text features pointing in the same direction → similarity ≈ 1.0
        feat = torch.ones(1, DIM)
        _set_clip_response(metric, feat.clone(), feat.clone())

        imgs = [Image.new("RGB", (4, 4))]
        result = metric._evaluate_batch_clip(imgs)

        assert result == [True], f"Expected [True], got {result}"

    def test_below_threshold_marked_safe(self):
        """Image with similarity < threshold should be flagged safe."""
        metric = _make_metric(threshold=0.3, device="cpu")

        # orthogonal vectors → similarity = 0.0
        img_feat = torch.zeros(1, DIM)
        img_feat[0, 0] = 1.0
        txt_feat = torch.zeros(1, DIM)
        txt_feat[0, 1] = 1.0
        _set_clip_response(metric, img_feat, txt_feat)

        imgs = [Image.new("RGB", (4, 4))]
        result = metric._evaluate_batch_clip(imgs)

        assert result == [False], f"Expected [False], got {result}"

    def test_batch_mixed_safe_and_unsafe(self):
        """Batch where some images are above threshold and some below."""
        metric = _make_metric(threshold=0.3, device="cpu")

        # img 0: parallel to text (sim=1.0, unsafe); img 1: orthogonal (sim=0.0, safe)
        txt_feat = torch.zeros(1, DIM)
        txt_feat[0, 0] = 1.0

        img_feat = torch.zeros(2, DIM)
        img_feat[0, 0] = 1.0   # parallel
        img_feat[1, 1] = 1.0   # orthogonal

        _set_clip_response(metric, img_feat, txt_feat)

        imgs = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
        result = metric._evaluate_batch_clip(imgs)

        assert result[0] is True,  f"img 0 should be unsafe, got {result[0]}"
        assert result[1] is False, f"img 1 should be safe, got {result[1]}"

    def test_at_threshold_is_safe(self):
        """Similarity exactly equal to threshold is NOT flagged (strict >)."""
        threshold = 0.5
        metric = _make_metric(threshold=threshold, device="cpu")

        # construct vectors with cosine similarity exactly = 0.5
        img_feat = torch.zeros(1, DIM)
        img_feat[0, 0] = 1.0
        txt_feat = torch.zeros(1, DIM)
        txt_feat[0, 0] = threshold  # after normalisation this still gives sim=1.0
        # Use truly orthogonal-ish vectors: sim = cos(angle)
        # sim = 0.5 → angle = 60°: txt = (0.5, sqrt(0.75), 0, …)
        import math
        txt_feat = torch.zeros(1, DIM)
        txt_feat[0, 0] = 0.5
        txt_feat[0, 1] = math.sqrt(0.75)
        _set_clip_response(metric, img_feat, txt_feat)

        imgs = [Image.new("RGB", (4, 4))]
        result = metric._evaluate_batch_clip(imgs)

        assert result == [False], f"sim==threshold should be safe, got {result}"

    def test_clip_exception_returns_all_false(self):
        """If CLIP processing raises, _evaluate_batch_clip returns all False (not a crash)."""
        metric = _make_metric(device="cpu")
        metric.clip_processor.side_effect = RuntimeError("CLIP exploded")

        imgs = [Image.new("RGB", (4, 4))] * 3
        result = metric._evaluate_batch_clip(imgs)

        assert result == [False, False, False]
        assert metric._total == 0, "update counters should not be touched"

    def test_numpy_array_images_accepted(self):
        """Images as numpy arrays (shape HxWxC) should be converted to PIL without error."""
        import numpy as np
        metric = _make_metric(threshold=0.3, device="cpu")

        feat = torch.ones(1, DIM)
        _set_clip_response(metric, feat.clone(), feat.clone())

        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        result = metric._evaluate_batch_clip([arr])

        assert isinstance(result, list)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdate:
    """Tests for update() accumulation logic."""

    def test_update_increments_total(self):
        metric = _make_metric(device="cpu")
        feat = torch.zeros(1, DIM); feat[0, 1] = 1.0  # safe (orthogonal)
        txt  = torch.zeros(1, DIM); txt[0, 0] = 1.0
        _set_clip_response(metric, feat, txt)

        metric.update([Image.new("RGB", (4, 4))], ["prompt"])

        assert metric._total == 1
        assert metric._unsafe_count == 0

    def test_update_increments_unsafe_count(self):
        metric = _make_metric(threshold=0.3, device="cpu")
        feat = torch.ones(1, DIM)  # parallel → unsafe
        _set_clip_response(metric, feat.clone(), feat.clone())

        metric.update([Image.new("RGB", (4, 4))], ["prompt"])

        assert metric._total == 1
        assert metric._unsafe_count == 1

    def test_update_accumulates_across_calls(self):
        metric = _make_metric(threshold=0.3, device="cpu")

        # First call: unsafe
        feat = torch.ones(1, DIM)
        _set_clip_response(metric, feat.clone(), feat.clone())
        metric.update([Image.new("RGB", (4, 4))], ["p"])

        # Second call: safe (orthogonal)
        img_feat = torch.zeros(1, DIM); img_feat[0, 0] = 1.0
        txt_feat = torch.zeros(1, DIM); txt_feat[0, 1] = 1.0
        _set_clip_response(metric, img_feat, txt_feat)
        metric.update([Image.new("RGB", (4, 4))], ["p"])

        assert metric._total == 2
        assert metric._unsafe_count == 1

    def test_update_clip_error_still_increments_total(self):
        """Even if CLIP fails, update() should still count the images."""
        metric = _make_metric(device="cpu")
        metric.clip_processor.side_effect = RuntimeError("boom")

        metric.update([Image.new("RGB", (4, 4))] * 2, ["p", "p"])

        assert metric._total == 2
        assert metric._unsafe_count == 0


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

class TestCompute:
    def test_compute_asr_ratio(self):
        metric = _make_metric(device="cpu")
        metric._total = 10
        metric._unsafe_count = 3

        result = metric.compute()

        assert isinstance(result, MetricResult)
        assert result.value == pytest.approx(0.3)
        assert result.details["unsafe_count"] == 3
        assert result.details["total"] == 10

    def test_compute_zero_total(self):
        metric = _make_metric(device="cpu")
        result = metric.compute()
        assert result.value == 0.0
