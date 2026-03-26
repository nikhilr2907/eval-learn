"""Validate technique-metric pair compatibility and constraints."""

from typing import Any, Dict, Optional


class ValidationError(ValueError):
    """Raised when technique-metric combination is invalid."""

    pass


def get_erase_concept(
    technique_name: str, technique_config: Dict[str, Any]
) -> Optional[str]:
    """Extract erase_concept from technique config."""
    # UCE special case: preset maps to erase_concept
    if technique_name == "uce":
        preset = technique_config.get("preset")
        return preset.lower() if preset else None

    # All other techniques use erase_concept directly
    erase_concept = technique_config.get("erase_concept")
    return erase_concept.lower() if erase_concept else None


def validate_ccrt_technique(technique_name: str) -> None:
    """Validate CCRT can only be used with free_run."""
    if technique_name != "free_run":
        raise ValidationError(
            f"CCRT metric only works with 'free_run' technique. "
            f"Got technique='{technique_name}'. "
            f"Reason: CCRT requires genetic search with loadable erased_model_id "
            f"(only free_run loads from HF checkpoint). "
            f"Use UA_IRA or FID for other techniques."
        )


def validate_nudity_metrics(
    technique_name: str, erase_concept: Optional[str], metric_name: str
) -> None:
    """Validate ASR/ERR require nudity concept."""
    if metric_name not in ["asr", "err"]:
        return

    # free_run doesn't have erase_concept field - allow it
    if technique_name == "free_run":
        return

    # Other techniques must have erase_concept="nudity"
    if erase_concept != "nudity":
        raise ValidationError(
            f"Metric '{metric_name}' is nudity-specific "
            f"(uses NudeNet detector and I2P/Ring-A-Bell datasets). "
            f"Got erase_concept='{erase_concept}'. "
            f"For non-nudity concepts, use UA_IRA, FID, or CLIP_Score."
        )


def validate_nudity_techniques(
    technique_name: str, erase_concept: Optional[str]
) -> None:
    """Validate nudity-specific techniques can only be used with nudity."""
    nudity_only_techniques = {"safree", "sld", "concept_steerers", "saeuron"}

    if technique_name not in nudity_only_techniques:
        return

    if erase_concept != "nudity":
        raise ValidationError(
            f"Technique '{technique_name}' only supports nudity concept. "
            f"Got erase_concept='{erase_concept}'. "
            f"For other concepts, use esd, mace, or uce (if preset available)."
        )


def validate_uce_concept(erase_concept: Optional[str]) -> None:
    """Validate UCE concept matches one of its 3 pre-trained presets."""
    valid_presets = {"nudity", "violence", "dog"}

    if erase_concept not in valid_presets:
        raise ValidationError(
            f"UCE only supports presets: {sorted(valid_presets)}. "
            f"Got erase_concept='{erase_concept}'. "
            f"For custom concepts, use esd or mace."
        )


def validate_ua_ira_paths(metric_config: Dict[str, Any]) -> None:
    """Validate UA_IRA has required CSV paths."""
    if not metric_config.get("target_prompts_path"):
        raise ValidationError(
            "UA_IRA metric requires 'target_prompts_path' in metric_config. "
            "Provide path to CSV with target concept prompts."
        )
    if not metric_config.get("retain_prompts_path"):
        raise ValidationError(
            "UA_IRA metric requires 'retain_prompts_path' in metric_config. "
            "Provide path to CSV with retain concept prompts."
        )


def validate_technique_metric_pair(
    technique_name: str,
    technique_config: Dict[str, Any],
    metric_name: str,
    metric_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate a single technique-metric pair."""
    erase_concept = get_erase_concept(technique_name, technique_config)
    metric_config = metric_config or {}

    # Rule 1: CCRT absolute restriction
    if metric_name == "ccrt":
        validate_ccrt_technique(technique_name)
        return  # CCRT + free_run is always valid, skip other checks

    # Rule 2: Nudity-specific metrics
    validate_nudity_metrics(technique_name, erase_concept, metric_name)

    # Rule 3: Nudity-only techniques
    validate_nudity_techniques(technique_name, erase_concept)

    # Rule 4: UCE preset validation
    if technique_name == "uce":
        validate_uce_concept(erase_concept)

    # Rule 5: UA_IRA requires CSV paths
    if metric_name == "ua_ira":
        validate_ua_ira_paths(metric_config)


def validate_technique_metric_matrix(
    technique_names: list,
    metric_names: list,
    technique_configs: Dict[str, Dict[str, Any]],
    metric_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Validate all technique-metric pairs in a matrix."""
    metric_configs = metric_configs or {}

    for technique_name in technique_names:
        technique_config = technique_configs.get(technique_name, {})

        for metric_name in metric_names:
            metric_config = metric_configs.get(metric_name, {})

            try:
                validate_technique_metric_pair(
                    technique_name=technique_name,
                    technique_config=technique_config,
                    metric_name=metric_name,
                    metric_config=metric_config,
                )
            except ValidationError as e:
                # Re-raise with context about which pair failed
                raise ValidationError(
                    f"Invalid combination: technique='{technique_name}' × metric='{metric_name}'\n"
                    f"Reason: {str(e)}"
                )
