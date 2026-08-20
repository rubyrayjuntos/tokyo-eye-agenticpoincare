"""Tokyo Eye MLflow-native governance package.

Runtime restore: ``models:/TokyoEye@champion`` / ``@experimental`` only.
Does not depend on filesystem HEALTHY_* seals.
"""

from science.tokyo_eye.governance.taxonomy import (
    DOMAINS,
    LINEAGE_ID,
    REGISTERED_MODEL_NAME,
    SUBSYSTEMS,
    experiment_path,
    mandatory_run_tags,
    validate_run_name,
)
from science.tokyo_eye.governance.registry import (
    ALIAS_CHAMPION,
    ALIAS_EXPERIMENTAL,
    get_model_by_alias,
    resolve_alias_uri,
    set_model_alias,
)
from science.tokyo_eye.governance.resolve import resolve_alias_checkpoint

__all__ = [
    "ALIAS_CHAMPION",
    "ALIAS_EXPERIMENTAL",
    "DOMAINS",
    "LINEAGE_ID",
    "REGISTERED_MODEL_NAME",
    "SUBSYSTEMS",
    "experiment_path",
    "get_model_by_alias",
    "mandatory_run_tags",
    "resolve_alias_checkpoint",
    "resolve_alias_uri",
    "set_model_alias",
    "validate_run_name",
]
