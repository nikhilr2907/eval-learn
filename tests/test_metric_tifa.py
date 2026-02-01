import pytest
from unittest.mock import patch, MagicMock
from PIL import Image


@pytest.fixture
def mock_tifa_deps():
    """Patch all external deps for TIFA metric."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float16 = "float16"
    # torch.no_grad() used as decorator on _answer: must return a decorator
    # that returns the function unchanged
    def no_grad_decorator():
        def wrapper(fn):
            return fn
        wrapper.__enter__ = MagicMock()
        wrapper.__exit__ = MagicMock()
        return wrapper
    mock_torch.no_grad = no_grad_decorator

    mock_processor_cls = MagicMock()
    mock_gen_cls = MagicMock()

    mock_image = MagicMock()
    mock_image.Image = Image.Image

    patches = [
        patch("eval_learn.metrics.tifa.metric.torch", mock_torch),
        patch("eval_learn.metrics.tifa.metric.Blip2Processor", mock_processor_cls),
        patch("eval_learn.metrics.tifa.metric.Blip2ForConditionalGeneration", mock_gen_cls),
        patch("eval_learn.metrics.tifa.metric.Image", mock_image),
    ]
    for p in patches:
        p.start()
    yield {
        "torch": mock_torch,
        "processor_cls": mock_processor_cls,
        "gen_cls": mock_gen_cls,
    }
    for p in patches:
        p.stop()


@pytest.fixture
def tifa_metric(mock_tifa_deps):
    """Construct TIFAMetric with mocked deps, stub out lazy loading."""
    from eval_learn.metrics.tifa.metric import TIFAMetric
    metric = TIFAMetric(device="cpu")
    # Stub out lazy loading so tests don't try to load real models
    metric._ensure_vqa_loaded = MagicMock()
    return metric


class TestTIFAInit:
    def test_init_success(self, mock_tifa_deps):
        from eval_learn.metrics.tifa.metric import TIFAMetric
        metric = TIFAMetric(device="cpu")
        assert metric._model is None  # lazy loading

    def test_init_missing_torch(self):
        with patch("eval_learn.metrics.tifa.metric.torch", None):
            from eval_learn.metrics.tifa.metric import TIFAMetric
            with pytest.raises(RuntimeError, match="torch"):
                TIFAMetric()

    def test_init_missing_transformers(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch("eval_learn.metrics.tifa.metric.torch", mock_torch), \
             patch("eval_learn.metrics.tifa.metric.Blip2Processor", None):
            from eval_learn.metrics.tifa.metric import TIFAMetric
            with pytest.raises(RuntimeError, match="transformers"):
                TIFAMetric()


class TestTIFACompute:
    def test_compute_empty_images(self, tifa_metric):
        result = tifa_metric.compute([], [])
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_missing_qa_pairs(self, tifa_metric, dummy_pil_image):
        result = tifa_metric.compute([dummy_pil_image()], ["p"], metadata={})
        assert result.value == 0.0
        assert "qa_pairs" in result.details["error"]

    def test_compute_qa_length_mismatch(self, tifa_metric, dummy_pil_image):
        qa = [[{"question": "q?", "answer": "a"}]]
        result = tifa_metric.compute(
            [dummy_pil_image(), dummy_pil_image()], ["p1", "p2"],
            metadata={"qa_pairs": qa}
        )
        assert result.value == 0.0
        assert "length" in result.details["error"]

    def test_compute_all_correct(self, tifa_metric, dummy_pil_image):
        imgs = [dummy_pil_image(), dummy_pil_image()]
        qa = [
            [{"question": "Is there a dog?", "answer": "yes"},
             {"question": "Color?", "answer": "red"}],
            [{"question": "Is there a cat?", "answer": "yes"}],
        ]
        # Mock _answer to return the expected answers
        answers = iter(["yes", "red", "yes"])
        tifa_metric._answer = MagicMock(side_effect=answers)
        # Mock _load_pil to return the PIL images
        tifa_metric._load_pil = MagicMock(side_effect=lambda img: img if isinstance(img, Image.Image) else None)

        result = tifa_metric.compute(imgs, ["p1", "p2"], metadata={"qa_pairs": qa})
        assert result.value == 1.0
        assert result.details["correct"] == 3
        assert result.details["total_questions"] == 3

    def test_compute_all_wrong(self, tifa_metric, dummy_pil_image):
        imgs = [dummy_pil_image()]
        qa = [[{"question": "q?", "answer": "yes"}]]
        tifa_metric._answer = MagicMock(return_value="no")
        tifa_metric._load_pil = MagicMock(side_effect=lambda img: img if isinstance(img, Image.Image) else None)

        result = tifa_metric.compute(imgs, ["p"], metadata={"qa_pairs": qa})
        assert result.value == 0.0
        assert result.details["correct"] == 0

    def test_compute_mixed(self, tifa_metric, dummy_pil_image):
        imgs = [dummy_pil_image(), dummy_pil_image()]
        qa = [
            [{"question": "q1?", "answer": "yes"}],
            [{"question": "q2?", "answer": "yes"}],
        ]
        # First image correct, second wrong
        tifa_metric._answer = MagicMock(side_effect=["yes", "no"])
        tifa_metric._load_pil = MagicMock(side_effect=lambda img: img if isinstance(img, Image.Image) else None)

        result = tifa_metric.compute(imgs, ["p1", "p2"], metadata={"qa_pairs": qa})
        assert result.value == 0.5
        assert result.details["per_image_scores"] == [1.0, 0.0]

    def test_compute_empty_qa_for_image(self, tifa_metric, dummy_pil_image):
        imgs = [dummy_pil_image()]
        qa = [[]]  # empty QA list
        tifa_metric._load_pil = MagicMock(side_effect=lambda img: img if isinstance(img, Image.Image) else None)

        result = tifa_metric.compute(imgs, ["p"], metadata={"qa_pairs": qa})
        assert result.details["per_image_scores"][0] is None

    def test_lazy_load_triggered(self, mock_tifa_deps):
        from eval_learn.metrics.tifa.metric import TIFAMetric
        metric = TIFAMetric(device="cpu")
        spy = MagicMock()
        metric._ensure_vqa_loaded = spy
        metric._answer = MagicMock(return_value="yes")
        metric._load_pil = MagicMock(side_effect=lambda img: img if isinstance(img, Image.Image) else None)

        img = Image.new("RGB", (10, 10))
        metric.compute(
            [img], ["p"],
            metadata={"qa_pairs": [[{"question": "q?", "answer": "yes"}]]}
        )
        spy.assert_called_once()

    def test_result_has_config(self, tifa_metric, dummy_pil_image):
        tifa_metric._answer = MagicMock(return_value="yes")
        tifa_metric._load_pil = MagicMock(side_effect=lambda img: img if isinstance(img, Image.Image) else None)
        result = tifa_metric.compute(
            [dummy_pil_image()], ["p"],
            metadata={"qa_pairs": [[{"question": "q?", "answer": "yes"}]]}
        )
        assert "config" in result.details
