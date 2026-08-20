"""Guards for active TokyoEye MLflow SSOT naming."""

from __future__ import annotations

from scripts.lint_tokyoeye_mlflow_ssot import main


def test_active_surfaces_do_not_reintroduce_versioned_product_labels() -> None:
    assert main() == 0
