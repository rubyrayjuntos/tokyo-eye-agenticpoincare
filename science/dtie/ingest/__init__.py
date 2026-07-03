"""Structure ingest and onboard compute orchestration."""

from science.dtie.ingest.orchestrator import (
    ONBOARD_STAGE_IDS,
    run_onboard_compute,
)

__all__ = ["ONBOARD_STAGE_IDS", "run_onboard_compute"]
