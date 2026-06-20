"""Single point of control for SMD stub mode.

This module is the ONLY place that decides whether stub mode is active.
All SMD-related code must check `is_stub_mode()` rather than reading
environment variables directly.

Safety rules:
- Stub mode is NEVER enabled silently in non-test contexts.
- If the environment variable SMD_STUB_MODE=1 is set outside of a pytest
  session, the user MUST confirm via interactive prompt before proceeding.
- In pytest (detected via PYTEST_CURRENT_TEST env var), stub mode activates
  without prompting.
- All stub-mode executions are logged at WARNING level with a clear banner.

Usage:
    from science.dtie.cryptic.stub_control import require_stub_approval, is_stub_mode

    # At the start of any script that dispatches SMD jobs:
    require_stub_approval()

    # Before each SMD call:
    if is_stub_mode():
        return _run_stub(...)
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# The single environment variable controlling stub mode
_ENV_VAR = "SMD_STUB_MODE"

# Legacy env var — we still read it for backward compat but warn
_LEGACY_ENV_VAR = "SMD_RUNNER_USE_STUB"

# Module-level state: has the user approved stub mode this session?
_stub_approved: bool = False


def _is_pytest() -> bool:
    """Detect if we are running inside a pytest session."""
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def _env_requests_stub() -> bool:
    """Check if the environment variable requests stub mode."""
    val = os.environ.get(_ENV_VAR, "").lower()
    if val in ("1", "true", "yes"):
        return True
    # Check legacy variable
    legacy_val = os.environ.get(_LEGACY_ENV_VAR, "").lower()
    if legacy_val in ("1", "true", "yes"):
        logger.warning(
            "SMD_RUNNER_USE_STUB is deprecated — use SMD_STUB_MODE=1 instead."
        )
        return True
    return False


def is_stub_mode() -> bool:
    """Return True if stub mode is active AND approved.

    This is the single source of truth for whether SMD calls should be
    stubbed. Never check environment variables directly elsewhere.

    Returns False if:
    - No stub env var is set, OR
    - Env var is set but user has not approved (outside pytest)
    """
    if not _env_requests_stub():
        return False
    # In pytest, always allow without prompting
    if _is_pytest():
        return True
    # Outside pytest, only if explicitly approved
    return _stub_approved


def require_stub_approval() -> None:
    """Gate function: if stub mode is requested, require explicit user confirmation.

    Call this at the top of any script or entry point that dispatches SMD jobs
    (calibration scripts, pipeline runners, on-demand validation).

    Behavior:
    - If stub env var is NOT set → no-op (real mode, proceed).
    - If running inside pytest → auto-approve, log warning.
    - If interactive terminal → prompt user for confirmation.
    - If non-interactive (CI without pytest) → abort with clear error.

    Raises:
        SystemExit: If user declines or non-interactive without pytest.
    """
    global _stub_approved

    if not _env_requests_stub():
        # Real mode — nothing to approve
        return

    if _is_pytest():
        _stub_approved = True
        logger.warning(
            "╔══════════════════════════════════════════════════════════╗\n"
            "║  SMD STUB MODE ACTIVE (auto-approved: pytest session)   ║\n"
            "║  All MD results are SYNTHETIC — not real physics.       ║\n"
            "╚══════════════════════════════════════════════════════════╝"
        )
        return

    # Interactive check
    banner = (
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  ⚠️  SMD STUB MODE REQUESTED (SMD_STUB_MODE=1)                  ║\n"
        "║                                                                  ║\n"
        "║  All steered MD results will be SYNTHETIC placeholder values.    ║\n"
        "║  NO real physics simulation will be performed.                   ║\n"
        "║  Calibration thresholds and validation results will be INVALID.  ║\n"
        "║                                                                  ║\n"
        "║  This mode is intended for testing pipeline plumbing ONLY.       ║\n"
        "╚══════════════════════════════════════════════════════════════════╝\n"
    )

    if not sys.stdin.isatty():
        # Non-interactive (e.g., CI job without pytest)
        logger.error(
            "SMD_STUB_MODE=1 is set but running non-interactively outside pytest. "
            "Refusing to proceed — stub mode requires explicit user approval. "
            "Either unset SMD_STUB_MODE or run within pytest."
        )
        sys.exit(1)

    print(banner, file=sys.stderr)
    try:
        response = input("Continue in STUB mode? [yes/NO]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)

    if response not in ("yes", "y"):
        print("Aborted — unset SMD_STUB_MODE to use real MD.", file=sys.stderr)
        sys.exit(1)

    _stub_approved = True
    logger.warning("User approved SMD stub mode — proceeding with synthetic results.")


def reset_approval() -> None:
    """Reset the approval state. Used in tests to isolate state."""
    global _stub_approved
    _stub_approved = False
