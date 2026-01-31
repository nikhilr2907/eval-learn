from dataclasses import dataclass
from eval_learn.configs.base import BaseConfig
from eval_learn.metrics.asr.config import ASRConfig
from eval_learn.metrics.fid.config import FIDConfig
from eval_learn.metrics.err.config import ERRConfig
from eval_learn.metrics.tifa.config import TIFAConfig
from eval_learn.metrics.clip_score.config import CLIPScoreConfig
from eval_learn.techniques.sld.config import SLDConfig


# ---------- BaseConfig ----------

class TestBaseConfig:
    def test_to_dict(self):
        @dataclass
        class MyConfig(BaseConfig):
            x: int = 1
            y: str = "hello"

        cfg = MyConfig(x=5, y="world")
        d = cfg.to_dict()
        assert d == {"x": 5, "y": "world"}

    def test_from_dict(self):
        @dataclass
        class MyConfig(BaseConfig):
            x: int = 1
            y: str = "hello"

        cfg = MyConfig.from_dict({"x": 10, "y": "test"})
        assert cfg.x == 10
        assert cfg.y == "test"

    def test_from_dict_ignores_unknown_keys(self):
        @dataclass
        class MyConfig(BaseConfig):
            x: int = 1

        cfg = MyConfig.from_dict({"x": 42, "garbage": "value"})
        assert cfg.x == 42
        assert not hasattr(cfg, "garbage")

    def test_from_dict_empty_uses_defaults(self):
        @dataclass
        class MyConfig(BaseConfig):
            x: int = 99

        cfg = MyConfig.from_dict({})
        assert cfg.x == 99


# ---------- ASRConfig ----------

class TestASRConfig:
    def test_defaults(self):
        cfg = ASRConfig()
        assert cfg.use_nudenet is True
        assert cfg.use_q16 is False
        assert cfg.device is None

    def test_from_dict(self):
        cfg = ASRConfig.from_dict({"use_nudenet": False})
        assert cfg.use_nudenet is False
        assert cfg.use_q16 is False

    def test_roundtrip(self):
        original = ASRConfig()
        restored = ASRConfig.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()


# ---------- FIDConfig ----------

class TestFIDConfig:
    def test_defaults(self):
        cfg = FIDConfig()
        assert cfg.real_images_dir == ""
        assert cfg.batch_size == 32
        assert cfg.device is None

    def test_from_dict_custom(self):
        cfg = FIDConfig.from_dict({"real_images_dir": "/tmp/real", "batch_size": 16})
        assert cfg.real_images_dir == "/tmp/real"
        assert cfg.batch_size == 16


# ---------- ERRConfig ----------

class TestERRConfig:
    def test_defaults(self):
        cfg = ERRConfig()
        assert cfg.clip_model_name == "openai/clip-vit-large-patch14"
        assert cfg.device is None

    def test_from_dict(self):
        cfg = ERRConfig.from_dict({"clip_model_name": "custom/model"})
        assert cfg.clip_model_name == "custom/model"


# ---------- TIFAConfig ----------

class TestTIFAConfig:
    def test_defaults(self):
        cfg = TIFAConfig()
        assert cfg.vqa_model_name == "Salesforce/blip2-flan-t5-xl"
        assert cfg.device is None

    def test_from_dict(self):
        cfg = TIFAConfig.from_dict({"device": "cpu"})
        assert cfg.device == "cpu"


# ---------- CLIPScoreConfig ----------

class TestCLIPScoreConfig:
    def test_defaults(self):
        cfg = CLIPScoreConfig()
        assert cfg.clip_model_name == "openai/clip-vit-base-patch32"
        assert cfg.device is None

    def test_from_dict(self):
        cfg = CLIPScoreConfig.from_dict({"clip_model_name": "custom/clip"})
        assert cfg.clip_model_name == "custom/clip"


# ---------- SLDConfig ----------

class TestSLDConfig:
    def test_defaults(self):
        cfg = SLDConfig()
        assert cfg.model_id == "AIML-TUDA/stable-diffusion-safe"
        assert cfg.sld_guidance_scale == 2000
        assert cfg.sld_warmup_steps == 7
        assert cfg.sld_threshold == 0.025
        assert cfg.sld_momentum_scale == 0.5
        assert cfg.sld_mom_beta == 0.7

    def test_from_dict(self):
        cfg = SLDConfig.from_dict({"sld_guidance_scale": 500})
        assert cfg.sld_guidance_scale == 500
        assert cfg.model_id == "AIML-TUDA/stable-diffusion-safe"

    def test_from_dict_ignores_extra(self):
        cfg = SLDConfig.from_dict({"model_id": "x", "unknown_key": True})
        assert cfg.model_id == "x"

    def test_preset_max(self):
        cfg = SLDConfig.from_dict({"preset": "MAX"})
        assert cfg.preset == "MAX"
        assert cfg.sld_guidance_scale == 5000
        assert cfg.sld_warmup_steps == 0
        assert cfg.sld_threshold == 1.0
        assert cfg.sld_momentum_scale == 0.5
        assert cfg.sld_mom_beta == 0.7

    def test_preset_none(self):
        cfg = SLDConfig.from_dict({"preset": "NONE"})
        assert cfg.sld_guidance_scale == 0
        assert cfg.sld_warmup_steps == 0

    def test_preset_weak(self):
        cfg = SLDConfig.from_dict({"preset": "weak"})
        assert cfg.sld_guidance_scale == 20
        assert cfg.sld_warmup_steps == 15
        assert cfg.sld_threshold == 0.0

    def test_preset_medium(self):
        cfg = SLDConfig.from_dict({"preset": "MEDIUM"})
        assert cfg.sld_guidance_scale == 1000
        assert cfg.sld_warmup_steps == 10
        assert cfg.sld_threshold == 0.01
        assert cfg.sld_momentum_scale == 0.3
        assert cfg.sld_mom_beta == 0.4

    def test_preset_strong(self):
        cfg = SLDConfig.from_dict({"preset": "Strong"})
        assert cfg.sld_guidance_scale == 2000
        assert cfg.sld_warmup_steps == 7

    def test_preset_with_override(self):
        cfg = SLDConfig.from_dict({"preset": "MAX", "sld_guidance_scale": 9999})
        assert cfg.sld_guidance_scale == 9999
        assert cfg.sld_warmup_steps == 0  # from MAX preset

    def test_preset_invalid(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown SLD preset"):
            SLDConfig.from_dict({"preset": "TURBO"})
