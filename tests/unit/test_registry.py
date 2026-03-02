import pytest
from eval_learn.registry.local import (
    register_technique, register_metric, register_dataset, register_benchmark,
    get_technique, get_metric, get_dataset, get_benchmark,
)


class TestRegistryTechniqueAndMetric:
    def test_register_technique_and_get(self, reset_registry):
        @register_technique("test_tech")
        class MyTech:
            pass
        assert get_technique("test_tech") is MyTech

    def test_register_metric_and_get(self, reset_registry):
        @register_metric("test_met")
        class MyMet:
            pass
        assert get_metric("test_met") is MyMet

    def test_register_dataset_and_get(self, reset_registry):
        @register_dataset("test_ds")
        class MyDS:
            pass
        assert get_dataset("test_ds") is MyDS

    def test_register_benchmark_and_get(self, reset_registry):
        @register_benchmark("test_bm")
        class MyBM:
            pass
        assert get_benchmark("test_bm") is MyBM


class TestRegistryNotFound:
    def test_get_technique_not_found(self, reset_registry):
        with pytest.raises(ValueError, match="not found"):
            get_technique("nonexistent")

    def test_get_metric_not_found(self, reset_registry):
        with pytest.raises(ValueError, match="not found"):
            get_metric("nonexistent")

    def test_get_dataset_not_found(self, reset_registry):
        with pytest.raises(ValueError, match="not found"):
            get_dataset("nonexistent")

    def test_get_benchmark_not_found(self, reset_registry):
        with pytest.raises(ValueError, match="not found"):
            get_benchmark("nonexistent")


class TestRegistryBehavior:
    def test_lowercase_normalization(self, reset_registry):
        @register_technique("MyTech")
        class T:
            pass
        assert get_technique("mytech") is T
        assert get_technique("MYTECH") is T

    def test_overwrite(self, reset_registry):
        @register_technique("dup")
        class First:
            pass

        @register_technique("dup")
        class Second:
            pass
        assert get_technique("dup") is Second

    def test_decorator_returns_original_class(self, reset_registry):
        class Foo:
            pass
        result = register_technique("foo_test")(Foo)
        assert result is Foo

    def test_error_message_lists_available(self, reset_registry):
        @register_technique("alpha")
        class A:
            pass
        with pytest.raises(ValueError, match="alpha"):
            get_technique("bad_key")

    def test_register_function_not_class(self, reset_registry):
        @register_dataset("func_ds")
        def my_loader():
            return "data"
        assert get_dataset("func_ds") is my_loader
