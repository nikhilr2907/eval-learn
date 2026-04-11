"""Unit tests for runner validation logic."""

import pytest

from eval_learn.runners.validation import (
    ValidationError,
    get_erase_concept,
    validate_nudity_metrics,
    validate_nudity_techniques,
    validate_uce_concept,
    validate_ua_ira_paths,
    validate_technique_metric_pair,
    validate_technique_metric_matrix,
)


# ---------------------------------------------------------------------------
# get_erase_concept
# ---------------------------------------------------------------------------

class TestGetEraseConcept:
    def test_uce_uses_preset(self):
        assert get_erase_concept("uce", {"preset": "Nudity"}) == "nudity"

    def test_uce_preset_none_returns_none(self):
        assert get_erase_concept("uce", {}) is None

    def test_other_technique_uses_erase_concept(self):
        assert get_erase_concept("esd", {"erase_concept": "Violence"}) == "violence"

    def test_other_technique_no_erase_concept_returns_none(self):
        assert get_erase_concept("esd", {}) is None

    def test_lowercases_value(self):
        assert get_erase_concept("mace", {"erase_concept": "DOG"}) == "dog"


# ---------------------------------------------------------------------------
# validate_nudity_metrics
# ---------------------------------------------------------------------------

class TestValidateNudityMetrics:
    def test_non_nudity_metric_always_passes(self):
        # fid, clip_score, ua_ira don't care about erase_concept
        validate_nudity_metrics("esd", "dog", "fid")
        validate_nudity_metrics("esd", None, "clip_score")

    def test_asr_with_nudity_passes(self):
        validate_nudity_metrics("esd", "nudity", "asr")

    def test_err_with_nudity_passes(self):
        validate_nudity_metrics("esd", "nudity", "err")

    def test_asr_with_non_nudity_concept_raises(self):
        with pytest.raises(ValidationError, match="nudity-specific"):
            validate_nudity_metrics("esd", "dog", "asr")

    def test_err_with_none_concept_raises(self):
        with pytest.raises(ValidationError, match="nudity-specific"):
            validate_nudity_metrics("esd", None, "err")

    def test_free_run_with_asr_passes_regardless_of_concept(self):
        validate_nudity_metrics("free_run", None, "asr")
        validate_nudity_metrics("free_run", "dog", "asr")


# ---------------------------------------------------------------------------
# validate_nudity_techniques
# ---------------------------------------------------------------------------

class TestValidateNudityTechniques:
    nudity_only = ["safree", "sld", "concept_steerers", "saeuron"]

    def test_non_nudity_technique_always_passes(self):
        validate_nudity_techniques("esd", "dog")
        validate_nudity_techniques("mace", None)

    @pytest.mark.parametrize("technique", nudity_only)
    def test_nudity_only_technique_with_nudity_passes(self, technique):
        validate_nudity_techniques(technique, "nudity")

    @pytest.mark.parametrize("technique", nudity_only)
    def test_nudity_only_technique_with_other_concept_raises(self, technique):
        with pytest.raises(ValidationError, match="only supports nudity"):
            validate_nudity_techniques(technique, "dog")

    @pytest.mark.parametrize("technique", nudity_only)
    def test_nudity_only_technique_with_none_concept_raises(self, technique):
        with pytest.raises(ValidationError, match="only supports nudity"):
            validate_nudity_techniques(technique, None)


# ---------------------------------------------------------------------------
# validate_uce_concept
# ---------------------------------------------------------------------------

class TestValidateUceConcept:
    def test_valid_presets_pass(self):
        for preset in ("nudity", "violence", "dog"):
            validate_uce_concept(preset)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValidationError, match="UCE only supports presets"):
            validate_uce_concept("cat")

    def test_none_preset_raises(self):
        with pytest.raises(ValidationError, match="UCE only supports presets"):
            validate_uce_concept(None)

    def test_error_message_lists_valid_presets(self):
        with pytest.raises(ValidationError, match="dog.*nudity.*violence"):
            validate_uce_concept("unknown")


# ---------------------------------------------------------------------------
# validate_ua_ira_paths
# ---------------------------------------------------------------------------

class TestValidateUaIraPaths:
    def test_both_paths_present_passes(self):
        validate_ua_ira_paths(
            {"target_prompts_path": "/data/target.csv", "retain_prompts_path": "/data/retain.csv"}
        )

    def test_missing_target_prompts_raises(self):
        with pytest.raises(ValidationError, match="target_prompts_path"):
            validate_ua_ira_paths({"retain_prompts_path": "/data/retain.csv"})

    def test_missing_retain_prompts_raises(self):
        with pytest.raises(ValidationError, match="retain_prompts_path"):
            validate_ua_ira_paths({"target_prompts_path": "/data/target.csv"})

    def test_empty_config_raises_on_target_first(self):
        with pytest.raises(ValidationError, match="target_prompts_path"):
            validate_ua_ira_paths({})


# ---------------------------------------------------------------------------
# validate_technique_metric_pair
# ---------------------------------------------------------------------------

class TestValidateTechniqueMetricPair:
    def test_valid_esd_fid_pair(self):
        validate_technique_metric_pair("esd", {"erase_concept": "nudity"}, "fid")

    def test_valid_esd_asr_nudity(self):
        validate_technique_metric_pair("esd", {"erase_concept": "nudity"}, "asr")

    def test_invalid_asr_with_dog_concept_raises(self):
        with pytest.raises(ValidationError):
            validate_technique_metric_pair("esd", {"erase_concept": "dog"}, "asr")

    def test_nudity_only_technique_with_wrong_concept_raises(self):
        with pytest.raises(ValidationError):
            validate_technique_metric_pair("sld", {"erase_concept": "dog"}, "fid")

    def test_uce_with_invalid_preset_raises(self):
        with pytest.raises(ValidationError, match="UCE only supports presets"):
            validate_technique_metric_pair("uce", {"preset": "cat"}, "fid")

    def test_uce_with_valid_preset_passes(self):
        validate_technique_metric_pair("uce", {"preset": "dog"}, "fid")

    def test_ua_ira_without_paths_raises(self):
        with pytest.raises(ValidationError, match="target_prompts_path"):
            validate_technique_metric_pair("esd", {"erase_concept": "dog"}, "ua_ira")

    def test_ua_ira_with_paths_passes(self):
        validate_technique_metric_pair(
            "esd",
            {"erase_concept": "dog"},
            "ua_ira",
            {"target_prompts_path": "/t.csv", "retain_prompts_path": "/r.csv"},
        )

    def test_metric_config_defaults_to_empty_dict(self):
        # Should not raise — metric_config=None is handled internally
        validate_technique_metric_pair("esd", {"erase_concept": "nudity"}, "fid", None)


# ---------------------------------------------------------------------------
# validate_technique_metric_matrix
# ---------------------------------------------------------------------------

class TestValidateTechniqueMetricMatrix:
    def test_valid_matrix_passes(self):
        validate_technique_metric_matrix(
            technique_names=["esd", "mace"],
            metric_names=["fid", "asr"],
            technique_configs={
                "esd": {"erase_concept": "nudity"},
                "mace": {"erase_concept": "nudity"},
            },
        )

    def test_invalid_pair_in_matrix_raises_with_context(self):
        with pytest.raises(ValidationError, match="technique='esd' × metric='asr'"):
            validate_technique_metric_matrix(
                technique_names=["esd"],
                metric_names=["asr"],
                technique_configs={"esd": {"erase_concept": "dog"}},
            )

    def test_error_includes_underlying_reason(self):
        with pytest.raises(ValidationError, match="nudity-specific"):
            validate_technique_metric_matrix(
                technique_names=["esd"],
                metric_names=["asr"],
                technique_configs={"esd": {"erase_concept": "dog"}},
            )

    def test_missing_technique_config_defaults_to_empty(self):
        # esd with no config → erase_concept=None → asr should raise
        with pytest.raises(ValidationError):
            validate_technique_metric_matrix(
                technique_names=["esd"],
                metric_names=["asr"],
                technique_configs={},
            )

    def test_metric_configs_passed_through(self):
        # ua_ira with valid paths should pass
        validate_technique_metric_matrix(
            technique_names=["esd"],
            metric_names=["ua_ira"],
            technique_configs={"esd": {"erase_concept": "dog"}},
            metric_configs={
                "ua_ira": {
                    "target_prompts_path": "/t.csv",
                    "retain_prompts_path": "/r.csv",
                }
            },
        )

    def test_stops_at_first_invalid_pair(self):
        # Only the first bad pair should be reported
        with pytest.raises(ValidationError, match="technique='esd' × metric='asr'"):
            validate_technique_metric_matrix(
                technique_names=["esd"],
                metric_names=["asr", "fid"],
                technique_configs={"esd": {"erase_concept": "dog"}},
            )