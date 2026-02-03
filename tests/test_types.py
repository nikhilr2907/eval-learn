from eval_learn.types import Dataset, MetricResult


class TestDataset:
    def test_dataset_basic_construction(self):
        ds = Dataset(prompts=["a", "b"], metadata={"k": "v"})
        assert ds.prompts == ["a", "b"]
        assert ds.metadata == {"k": "v"}

    def test_dataset_default_metadata(self):
        ds = Dataset(prompts=["a"])
        assert ds.metadata == {}

    def test_dataset_empty_prompts(self):
        ds = Dataset(prompts=[])
        assert len(ds.prompts) == 0
        assert ds.metadata == {}


class TestMetricResult:
    def test_metric_result_basic(self):
        mr = MetricResult(name="ASR", value=0.75, details={"k": 1})
        assert mr.name == "ASR"
        assert mr.value == 0.75
        assert mr.details == {"k": 1}

    def test_metric_result_default_details(self):
        mr = MetricResult(name="X", value=1.0)
        assert mr.details == {}

    def test_metric_result_special_values(self):
        for val in [0.0, float("inf"), -1.5]:
            mr = MetricResult(name="T", value=val)
            assert mr.value == val
