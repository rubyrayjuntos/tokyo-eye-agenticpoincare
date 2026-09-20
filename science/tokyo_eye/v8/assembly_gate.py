"""Governed TokyoEye EQU assembly gate (Architecture SSOT).

Standing rule: prose without a failing test always loses to a convenient default.

SPECIFIED conditions (must all pass for governed / claim-bearing runs):
  1. Frontend is Equiformer pool — not SE(3)-lite / stub live_backbone.
  2. Live-forward pure_hyp_pass (tracer) succeeds.
  3. Claim-bearing biology (opt-in ``claim_bearing_biology=True``): no
     edge_type→target leakage, and biology_grad_by_source was logged.

ENFORCED: called from the training harness on the governed path, and covered by
``tests/v8/test_assembly_gate.py``. Off-path diagnostics must set
``allow_off_path_frontend=True`` explicitly.

Environment: pool constructibility deps are checked separately so missing
packages fail with a clear missing-deps message (not a silent architecture veto).

SE(3)-lite pilots: permitted only when pre-registered, disclaimed
(``do_not_promote`` / ``not_claims``), and framed as a cheap signal-existence
check before a pool-frontend confirmatory run — never as governed trunk
biology. See ``docs/TOKYOEYE_ARCHITECTURE_SSOT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# Packages required to import / construct EquiformerPoolFrontend in this repo.
POOL_FRONTEND_DEP_MODULES: tuple[str, ...] = (
    "ase",
    "e3nn",
    "lmdb",
    "torch_scatter",
    "torch_cluster",
)

_POOL_KIND = frozenset({"equiformer_pool", "equiformer_v3_pool", "pool"})
_POOL_BACKBONE = frozenset({"equiformer_v3_pool", "equiformer_pool", "pool"})


@dataclass
class AssemblyGateResult:
    """Outcome of ``assert_governed_assembly``.

    ``pure_hyp_ok`` is True only when the live tracer ran **and** passed.
    When the check was skipped (off-path allow, or no forward_fn), use
    ``pure_hyp_checked=False`` and ``pure_hyp_ok=False`` — never treat a skip
    as a clean geometry seal. Always read ``allow_off_path_frontend`` with the
    result; do not interpret ``pure_hyp_ok`` in isolation.
    """

    passed: bool
    frontend_ok: bool
    deps_ok: bool
    pure_hyp_ok: bool
    pure_hyp_checked: bool
    allow_off_path_frontend: bool
    frontend_kind: str
    claim_bearing_biology: bool = False
    claim_biology_ok: bool = True
    missing_deps: list[str] = field(default_factory=list)
    detail: str = ""
    pure_hyp_report: dict[str, Any] | None = None

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        raise AssemblyGateError(self.detail or self.summary())

    def summary(self) -> str:
        parts = [
            f"frontend_ok={self.frontend_ok}",
            f"deps_ok={self.deps_ok}",
            f"pure_hyp_checked={self.pure_hyp_checked}",
            f"pure_hyp_ok={self.pure_hyp_ok}",
            f"allow_off_path_frontend={self.allow_off_path_frontend}",
            f"frontend_kind={self.frontend_kind!r}",
        ]
        if self.claim_bearing_biology:
            parts.append(f"claim_biology_ok={self.claim_biology_ok}")
        if self.missing_deps:
            parts.append(f"missing_deps={self.missing_deps}")
        if self.detail:
            parts.append(self.detail)
        return "assembly_gate FAIL: " + "; ".join(parts)


class AssemblyGateError(RuntimeError):
    """Governed EQU run refused: assembly gate failed."""


def check_pool_frontend_dependencies(
    modules: Sequence[str] = POOL_FRONTEND_DEP_MODULES,
) -> tuple[bool, list[str]]:
    """Return (ok, missing_module_names)."""
    missing: list[str] = []
    for name in modules:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return (len(missing) == 0, missing)


def assert_pool_frontend_dependencies(
    modules: Sequence[str] = POOL_FRONTEND_DEP_MODULES,
) -> None:
    ok, missing = check_pool_frontend_dependencies(modules)
    if ok:
        return
    raise AssemblyGateError(
        "missing frontend dependencies for EquiformerPoolFrontend: "
        + ", ".join(missing)
        + ". Install these in the same environment as the governed harness/CI; "
        "this is an environment failure, not an architecture veto."
    )


def _frontend_is_pool(frontend_kind: str, load_info: dict[str, Any] | None) -> bool:
    """True only on direct evidence of Equiformer pool — never inferred from tags.

    ``forbid_se3_lite=True`` alone is **not** proof of pool construction
    (known-bad fixture: that flag with no pool kind/backbone must fail).
    """
    kind = str(frontend_kind or "").strip().lower()
    if kind in _POOL_KIND:
        return True
    info = load_info or {}
    mode = str(info.get("frontend_mode") or info.get("mode") or "").lower()
    if "equiformer" in mode and "pool" in mode:
        return True
    backbone = str(info.get("backbone_mode") or "").lower()
    if backbone in _POOL_BACKBONE:
        return True
    return False


def assert_governed_frontend(
    frontend_kind: str,
    *,
    load_info: dict[str, Any] | None = None,
    allow_off_path_frontend: bool = False,
) -> None:
    """Fail if governed run selects SE(3)-lite / stub without explicit allow."""
    if allow_off_path_frontend:
        return
    if _frontend_is_pool(frontend_kind, load_info):
        return
    raise AssemblyGateError(
        f"governed EQU requires --frontend equiformer_pool; got {frontend_kind!r} "
        f"(backbone_mode={((load_info or {}).get('backbone_mode'))!r}). "
        "SE(3)-lite / stub is forbidden on the production stack "
        "(docs/TOKYOEYE_ARCHITECTURE_SSOT.md). "
        "Pass --allow-off-path-frontend only for explicit non-claim diagnostics "
        "(pre-registered SE(3)-lite pilots as cheap signal-existence checks only)."
    )


def _as_numpy(t: Any) -> Any:
    import numpy as np

    if t is None:
        return None
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def detect_edge_type_label_leakage(batch: dict[str, Any]) -> dict[str, Any]:
    """Structural defect-B probe: do targets match reconstructions from edge_type?

    Returns a report with ``leaking`` True when ``edge_type`` is present as a
    batch field (model input channel) **and** dehydron labels and/or SDRP
    targets equal the loader reconstructions from that same ``edge_type``.
    A caller-asserted ``--biology-non-leaking-target`` flag is irrelevant here.
    """
    from science.tokyo_eye.v8.loader import (
        dehydron_labels_from_edges,
        sdrp_heuristic_from_edges,
    )

    edge_type = batch.get("edge_type")
    edge_index = batch.get("edge_index")
    report: dict[str, Any] = {
        "edge_type_present": edge_type is not None,
        "dehydron_matches_r2_incidence": False,
        "sdrp_matches_edge_histogram": False,
        "leaking": False,
    }
    if edge_type is None or edge_index is None:
        return report

    et = _as_numpy(edge_type)
    ei = _as_numpy(edge_index)
    if et is None or ei is None:
        return report
    # edge_index is [2, E]; num_nodes from labels or coords
    labels = batch.get("dehydron_labels")
    if labels is not None:
        lab = _as_numpy(labels).astype("float32").reshape(-1)
        n = int(lab.shape[0])
        recon = dehydron_labels_from_edges(n, ei, et).astype("float32").reshape(-1)
        report["dehydron_matches_r2_incidence"] = bool(
            lab.shape == recon.shape and (abs(lab - recon).max() < 1e-5)
        )
    sdrp = batch.get("sdrp_target")
    if sdrp is not None:
        st = _as_numpy(sdrp).astype("int64").reshape(-1)
        n = int(st.shape[0])
        recon_s = sdrp_heuristic_from_edges(n, ei, et).astype("int64").reshape(-1)
        report["sdrp_matches_edge_histogram"] = bool(
            st.shape == recon_s.shape and (st == recon_s).all()
        )
    report["leaking"] = bool(
        report["dehydron_matches_r2_incidence"]
        or report["sdrp_matches_edge_histogram"]
    )
    return report


def assert_claim_bearing_biology(
    *,
    biology_grad_sources_logged: bool,
    batch: dict[str, Any] | None = None,
    # Deprecated name-based args: ignored when ``batch`` is supplied.
    edge_type_is_model_input: bool | None = None,
    dehydron_or_sdrp_target_from_edge_type: bool | None = None,
) -> None:
    """Third assembly leg: claim-bearing biology must be structurally non-leaking.

    Leakage is verified by reconstructing dehydron/SDRP targets from the batch's
    ``edge_type`` (same functions as the loader). A CLI flag asserting
    non-leakage does **not** pass this check — known-bad fixture: flag true /
    ``dehydron_or_sdrp_target_from_edge_type=False`` while the batch still
    matches R2 incidence must fail.

    Also refuses when ``biology_grad_by_source`` was not logged.
    """
    if batch is None:
        # Fail closed: without a batch we cannot verify structure.
        # Legacy boolean path only when both name-based args are explicitly set
        # *and* no batch — still refuse if they claim leaking; if they claim
        # non-leaking without a batch, refuse (honest-system gap).
        if (
            edge_type_is_model_input is not None
            and dehydron_or_sdrp_target_from_edge_type is not None
        ):
            if edge_type_is_model_input and dehydron_or_sdrp_target_from_edge_type:
                raise AssemblyGateError(
                    "claim-bearing biology refused: defect B label leakage "
                    "(name-based args; prefer structural batch check)."
                )
            raise AssemblyGateError(
                "claim-bearing biology refused: structural batch required to "
                "verify non-leakage — a caller-asserted non-leaking flag is not "
                "sufficient (same class of gap as forbid_se3_lite-alone)."
            )
        raise AssemblyGateError(
            "claim-bearing biology refused: batch required for structural "
            "edge_type↔target leakage check."
        )

    leak = detect_edge_type_label_leakage(batch)
    if leak["leaking"]:
        raise AssemblyGateError(
            "claim-bearing biology refused: defect B label leakage — "
            "batch dehydron/SDRP targets reconstruct from edge_type while "
            "edge_type is present as a model input "
            f"(dehydron_match={leak['dehydron_matches_r2_incidence']}, "
            f"sdrp_match={leak['sdrp_matches_edge_histogram']}). "
            "A --biology-non-leaking-target flag does not override this check; "
            "see data/gates/tokyo_eye_equ_nonclaim_disposition.json."
        )
    if not biology_grad_sources_logged:
        raise AssemblyGateError(
            "claim-bearing biology refused: biology_grad_by_source was not "
            "logged. Pass --log-biology-grad-sources N (N>0) so hyp vs "
            "Euclidean-skip credit is measured before any biology claim."
        )


def run_pure_hyp_gate(
    system: Any,
    forward_fn: Callable[[], Any],
) -> tuple[bool, dict[str, Any]]:
    """Live-forward pure_hyp_pass tracer on ``forward_fn``."""
    from science.tokyo_eye.v8.pure_hyp_pass import scan_forward

    report = scan_forward(system, forward_fn)
    return bool(report.passed), report.as_dict()


def assert_governed_assembly(
    *,
    frontend_kind: str,
    load_info: dict[str, Any] | None = None,
    system: Any | None = None,
    forward_fn: Callable[[], Any] | None = None,
    allow_off_path_frontend: bool = False,
    require_pure_hyp: bool = True,
    check_deps: bool = True,
    claim_bearing_biology: bool = False,
    biology_grad_sources_logged: bool = False,
    claim_batch: dict[str, Any] | None = None,
    # Deprecated; structural ``claim_batch`` wins. Kept for call-site compat.
    edge_type_is_model_input: bool | None = None,
    dehydron_or_sdrp_target_from_edge_type: bool | None = None,
) -> AssemblyGateResult:
    """Full governed assembly check. Raises ``AssemblyGateError`` on failure."""
    missing: list[str] = []
    deps_ok = True
    if check_deps and not allow_off_path_frontend:
        deps_ok, missing = check_pool_frontend_dependencies()
        if not deps_ok:
            result = AssemblyGateResult(
                passed=False,
                frontend_ok=False,
                deps_ok=False,
                pure_hyp_ok=False,
                pure_hyp_checked=False,
                allow_off_path_frontend=allow_off_path_frontend,
                frontend_kind=frontend_kind,
                claim_bearing_biology=claim_bearing_biology,
                claim_biology_ok=False,
                missing_deps=missing,
                detail=(
                    "missing frontend dependencies for EquiformerPoolFrontend: "
                    + ", ".join(missing)
                ),
            )
            result.raise_if_failed()

    frontend_ok = True
    try:
        assert_governed_frontend(
            frontend_kind,
            load_info=load_info,
            allow_off_path_frontend=allow_off_path_frontend,
        )
    except AssemblyGateError as exc:
        frontend_ok = False
        result = AssemblyGateResult(
            passed=False,
            frontend_ok=False,
            deps_ok=deps_ok,
            pure_hyp_ok=False,
            pure_hyp_checked=False,
            allow_off_path_frontend=allow_off_path_frontend,
            frontend_kind=frontend_kind,
            claim_bearing_biology=claim_bearing_biology,
            claim_biology_ok=False,
            missing_deps=missing,
            detail=str(exc),
        )
        result.raise_if_failed()

    # pure_hyp: skip is not a pass — checked=False, ok=False until tracer runs.
    pure_hyp_checked = False
    pure_hyp_ok = False
    pure_report: dict[str, Any] | None = None
    should_run_pure_hyp = (
        require_pure_hyp
        and not allow_off_path_frontend
        and system is not None
        and forward_fn is not None
    )
    if should_run_pure_hyp:
        pure_hyp_checked = True
        pure_hyp_ok, pure_report = run_pure_hyp_gate(system, forward_fn)
        if not pure_hyp_ok:
            result = AssemblyGateResult(
                passed=False,
                frontend_ok=frontend_ok,
                deps_ok=deps_ok,
                pure_hyp_ok=False,
                pure_hyp_checked=True,
                allow_off_path_frontend=allow_off_path_frontend,
                frontend_kind=frontend_kind,
                claim_bearing_biology=claim_bearing_biology,
                claim_biology_ok=False,
                missing_deps=missing,
                detail="pure_hyp_pass failed on live forward (assembly gate)",
                pure_hyp_report=pure_report,
            )
            result.raise_if_failed()

    claim_biology_ok = True
    if claim_bearing_biology:
        if allow_off_path_frontend:
            raise AssemblyGateError(
                "claim-bearing biology refused: cannot combine with "
                "--allow-off-path-frontend / SE(3)-lite. Governed claims require "
                "Equiformer pool (Architecture SSOT)."
            )
        try:
            assert_claim_bearing_biology(
                batch=claim_batch,
                biology_grad_sources_logged=biology_grad_sources_logged,
                edge_type_is_model_input=edge_type_is_model_input,
                dehydron_or_sdrp_target_from_edge_type=(
                    dehydron_or_sdrp_target_from_edge_type
                ),
            )
        except AssemblyGateError as exc:
            claim_biology_ok = False
            result = AssemblyGateResult(
                passed=False,
                frontend_ok=frontend_ok,
                deps_ok=deps_ok,
                pure_hyp_ok=pure_hyp_ok,
                pure_hyp_checked=pure_hyp_checked,
                allow_off_path_frontend=allow_off_path_frontend,
                frontend_kind=frontend_kind,
                claim_bearing_biology=True,
                claim_biology_ok=False,
                missing_deps=missing,
                detail=str(exc),
                pure_hyp_report=pure_report,
            )
            result.raise_if_failed()

    return AssemblyGateResult(
        passed=True,
        frontend_ok=frontend_ok,
        deps_ok=deps_ok,
        pure_hyp_ok=pure_hyp_ok,
        pure_hyp_checked=pure_hyp_checked,
        allow_off_path_frontend=allow_off_path_frontend,
        frontend_kind=frontend_kind,
        claim_bearing_biology=claim_bearing_biology,
        claim_biology_ok=claim_biology_ok,
        missing_deps=missing,
        pure_hyp_report=pure_report,
        detail="assembly_gate PASS",
    )


__all__ = [
    "AssemblyGateError",
    "AssemblyGateResult",
    "POOL_FRONTEND_DEP_MODULES",
    "assert_claim_bearing_biology",
    "assert_governed_assembly",
    "assert_governed_frontend",
    "assert_pool_frontend_dependencies",
    "check_pool_frontend_dependencies",
    "detect_edge_type_label_leakage",
    "run_pure_hyp_gate",
]
