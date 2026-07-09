"""
auto_trainer.py — Governed autonomous v6 training loop (playbook-driven).

Runs allowlisted Makefile targets based on champion diagnostics, assesses
checkpoints, and promotes only when promotion_gate passes and score improves.

Usage:
    python -m experiments.training.v6.auto_trainer --dry-run
    python -m experiments.training.v6.auto_trainer --once
    make auto-train-v6-corpus25 DRY_RUN=1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PLAYBOOK = _REPO_ROOT / "manifests/auto_trainer/corpus25_playbook.yaml"
_AUTO_DIR = _REPO_ROOT / "checkpoints/v6/runs/auto_trainer"
_STATE_PATH = _AUTO_DIR / "state.json"
_JOURNAL_PATH = _AUTO_DIR / "journal.jsonl"
_DEFAULT_PDB_DIR = _REPO_ROOT / "pdb_cache"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("auto_trainer")


@dataclass
class TrainerState:
    champion_run_id: str
    champion_checkpoint: str
    gate_run_id: str
    iteration: int = 0
    failed_runs: int = 0
    failed_assessments: int = 0
    last_action_id: str | None = None
    last_run_id: str | None = None
    next_suggested: dict[str, Any] | None = None
    stalled: bool = False
    stall_reason: str | None = None
    unpromoted_same_action: int = 0
    program_status: str = "ok"
    program_block_reason: str | None = None
    program_escalation_action: str | None = None
    program_progress: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "champion_run_id": self.champion_run_id,
            "champion_checkpoint": self.champion_checkpoint,
            "gate_run_id": self.gate_run_id,
            "iteration": self.iteration,
            "failed_runs": self.failed_runs,
            "failed_assessments": self.failed_assessments,
            "last_action_id": self.last_action_id,
            "last_run_id": self.last_run_id,
            "next_suggested": self.next_suggested,
            "stalled": self.stalled,
            "stall_reason": self.stall_reason,
            "unpromoted_same_action": self.unpromoted_same_action,
            "program_status": self.program_status,
            "program_block_reason": self.program_block_reason,
            "program_escalation_action": self.program_escalation_action,
            "program_progress": self.program_progress,
        }
        return payload

    @classmethod
    def from_playbook(cls, playbook: dict[str, Any]) -> TrainerState:
        champ = playbook["champion"]
        return cls(
            champion_run_id=str(champ["run_id"]),
            champion_checkpoint=str(champ["checkpoint"]),
            gate_run_id=str(champ.get("gate_run_id", "")),
        )

    @classmethod
    def load(cls, path: Path, playbook: dict[str, Any]) -> TrainerState:
        if not path.is_file():
            return cls.from_playbook(playbook)
        raw = json.loads(path.read_text())
        base = cls.from_playbook(playbook)
        return cls(
            champion_run_id=str(raw.get("champion_run_id", base.champion_run_id)),
            champion_checkpoint=str(raw.get("champion_checkpoint", base.champion_checkpoint)),
            gate_run_id=str(raw.get("gate_run_id", base.gate_run_id)),
            iteration=int(raw.get("iteration", 0)),
            failed_runs=int(raw.get("failed_runs", 0)),
            failed_assessments=int(raw.get("failed_assessments", 0)),
            last_action_id=raw.get("last_action_id"),
            last_run_id=raw.get("last_run_id"),
            next_suggested=raw.get("next_suggested"),
            stalled=bool(raw.get("stalled", False)),
            stall_reason=raw.get("stall_reason"),
            unpromoted_same_action=int(raw.get("unpromoted_same_action", 0)),
            program_status=str(raw.get("program_status", "ok")),
            program_block_reason=raw.get("program_block_reason"),
            program_escalation_action=raw.get("program_escalation_action"),
            program_progress=raw.get("program_progress"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def champion_score_from_disk(self, repo_root: Path = _REPO_ROOT) -> float:
        return read_checkpoint_score(repo_root / self.champion_checkpoint)


@dataclass
class MetricSnapshot:
    focus: dict[str, Any]
    assess: dict[str, Any] | None
    champion_score: float
    routing_entropy: float | None
    probe_r_epi_sasa: float | None
    promotion_gate_passed: bool | None
    expert_starvation_count: int | None

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path,
        assess: dict[str, Any] | None = None,
        champion_checkpoint: Path | None = None,
    ) -> MetricSnapshot:
        focus_path = run_dir / "focus_summary.json"
        focus: dict[str, Any] = {}
        if focus_path.is_file():
            focus = json.loads(focus_path.read_text())

        ckpt = champion_checkpoint or resolve_run_checkpoint(run_dir.name, _REPO_ROOT)
        score = read_checkpoint_score(ckpt) if ckpt else float("-inf")

        routing_entropy = _metric_value(focus, "routing_entropy")
        probe_r_epi_sasa = _metric_value(focus, "probe_r_epi_sasa")
        if probe_r_epi_sasa is None and assess:
            geom = assess.get("geometry", {})
            probe_r_epi_sasa = _as_float(geom.get("probe_r_epi_sasa"))

        promotion_gate_passed = None
        expert_starvation_count = None
        if assess:
            gate = assess.get("promotion_gate", {})
            promotion_gate_passed = bool(gate.get("passed")) if gate else None
            moe = assess.get("moe", {})
            expert_starvation_count = int(moe.get("expert_starvation_count", 0))
            if routing_entropy is None:
                routing_entropy = _as_float(moe.get("routing_entropy_mean"))

        return cls(
            focus=focus,
            assess=assess,
            champion_score=score,
            routing_entropy=routing_entropy,
            probe_r_epi_sasa=probe_r_epi_sasa,
            promotion_gate_passed=promotion_gate_passed,
            expert_starvation_count=expert_starvation_count,
        )


def _metric_value(focus: dict[str, Any], name: str) -> float | None:
    for bucket in ("needs_work", "watch", "healthy"):
        for item in focus.get(bucket, []):
            if item.get("metric") == name:
                return _as_float(item.get("value"))
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_playbook(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid playbook (expected mapping): {path}")
    actions = raw.get("actions", {})
    for action_id in raw.get("priority", []):
        if action_id not in actions:
            raise ValueError(f"Priority references unknown action: {action_id}")
    return raw


def resolve_run_checkpoint(run_id: str, repo_root: Path = _REPO_ROOT) -> Path | None:
    run_dir = repo_root / "checkpoints/v6/runs" / run_id
    candidates = [
        run_dir / "v6_best.pt",
        run_dir / "v6_best_disc.pt",
        run_dir / "v6_phase4_23prot.pt",
        run_dir / "v6_phase4_12prot.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    phase_files = sorted(run_dir.glob("phase_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return phase_files[0] if phase_files else None


def read_checkpoint_score(checkpoint: Path | None) -> float:
    if checkpoint is None or not checkpoint.is_file():
        return float("-inf")
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return float("-inf")
    if isinstance(raw, dict):
        if "score" in raw:
            return float(raw["score"])
        metrics = raw.get("metrics", {})
        if isinstance(metrics, dict) and "score" in metrics:
            return float(metrics["score"])
    return float("-inf")


def _gate_checkpoint_exists(gate_run_id: str, repo_root: Path = _REPO_ROOT) -> bool:
    if not gate_run_id:
        return False
    path = repo_root / "checkpoints/v6/runs" / gate_run_id / "v6_best.pt"
    return path.is_file()


def conditions_match(when: dict[str, Any], snap: MetricSnapshot, state: TrainerState) -> bool:
    if not when:
        return True

    primary = when.get("primary_focus")
    if primary is not None:
        expected = {primary} if isinstance(primary, str) else set(primary)
        actual = set(snap.focus.get("primary_focus") or [])
        if not expected.intersection(actual):
            return False

    for metric in when.get("needs_work_any") or []:
        values = [
            item.get("metric")
            for item in snap.focus.get("needs_work", [])
            if item.get("matters", True)
        ]
        if metric not in values:
            return False

    gate = when.get("promotion_gate_passed")
    if gate is not None and snap.promotion_gate_passed is not None:
        if bool(gate) != bool(snap.promotion_gate_passed):
            return False

    if when.get("requires_gate_best") and not _gate_checkpoint_exists(state.gate_run_id):
        return False

    min_failed = when.get("min_failed_assessments")
    if min_failed is not None and state.failed_assessments < int(min_failed):
        return False

    entropy_above = when.get("routing_entropy_above")
    if entropy_above is not None:
        if snap.routing_entropy is None or snap.routing_entropy <= float(entropy_above):
            return False

    entropy_below = when.get("routing_entropy_below")
    if entropy_below is not None:
        if snap.routing_entropy is None or snap.routing_entropy >= float(entropy_below):
            return False

    epi_sasa_below = when.get("probe_r_epi_sasa_below")
    if epi_sasa_below is not None:
        if snap.probe_r_epi_sasa is None or snap.probe_r_epi_sasa >= float(epi_sasa_below):
            return False

    epi_sasa_above = when.get("probe_r_epi_sasa_above")
    if epi_sasa_above is not None:
        if snap.probe_r_epi_sasa is None or snap.probe_r_epi_sasa <= float(epi_sasa_above):
            return False

    starve_above = when.get("expert_starvation_count_above")
    if starve_above is not None:
        count = snap.expert_starvation_count if snap.expert_starvation_count is not None else 0
        if count <= int(starve_above):
            return False

    skip_last = when.get("skip_if_last_action")
    if skip_last is not None and state.last_action_id == skip_last:
        return False

    return True


def select_action(
    playbook: dict[str, Any],
    snap: MetricSnapshot,
    state: TrainerState,
) -> tuple[str, dict[str, Any]] | None:
    actions = playbook["actions"]
    for action_id in playbook.get("priority", []):
        action = actions[action_id]
        if conditions_match(action.get("when") or {}, snap, state):
            return action_id, action
    return None


def _format_env(
    env_template: dict[str, Any],
    *,
    state: TrainerState,
    prior_run_id: str,
) -> dict[str, str]:
    mapping = {
        "champion_run_id": state.champion_run_id,
        "champion_checkpoint": state.champion_checkpoint,
        "gate_run_id": state.gate_run_id,
        "prior_run_id": prior_run_id,
    }
    out: dict[str, str] = {}
    for key, template in env_template.items():
        value = str(template)
        for token, repl in mapping.items():
            value = value.replace("{" + token + "}", repl)
        out[str(key)] = value
    return out


def append_journal(entry: dict[str, Any], path: Path = _JOURNAL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {**entry, "ts": datetime.now(timezone.utc).isoformat()}
    with path.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def run_make_target(
    make_target: str,
    *,
    env: dict[str, str],
    repo_root: Path = _REPO_ROOT,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    cmd = ["make", make_target]
    merged = {**os.environ, **env}
    logger.info("Running: %s (env: %s)", " ".join(cmd), env)
    if dry_run:
        return None
    return subprocess.run(
        cmd,
        cwd=repo_root,
        env=merged,
        text=True,
        check=False,
    )


def run_assess(
    checkpoint: Path,
    *,
    corpus_manifest: str,
    max_proteins: int,
    device: str,
    output: Path,
    repo_root: Path = _REPO_ROOT,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    docker_ckpt = f"/app/{checkpoint.relative_to(repo_root)}"
    docker_out = f"/app/{output.relative_to(repo_root)}"
    cmd = [
        "make",
        "assess-v6",
        f"CHECKPOINT={docker_ckpt}",
        f"CORPUS={corpus_manifest}",
        f"MAX_PROTEINS={max_proteins}",
        f"DEVICE={device}",
        f"OUTPUT={docker_out}",
    ]
    logger.info("Assessing: %s", " ".join(cmd))
    if dry_run:
        return None
    proc = subprocess.run(cmd, cwd=repo_root, text=True, check=False)
    if proc.returncode != 0 and output.is_file():
        logger.warning("assess-v6 exited %d but report exists", proc.returncode)
    if output.is_file():
        return json.loads(output.read_text())
    return None


def build_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", prefix).strip("_")
    return f"{safe}_{stamp}"


def maybe_promote_champion(
    *,
    run_id: str,
    state: TrainerState,
    playbook: dict[str, Any],
    dry_run: bool,
) -> bool:
    run_dir = _REPO_ROOT / "checkpoints/v6/runs" / run_id
    ckpt = run_dir / "v6_best.pt"
    if not ckpt.is_file():
        logger.info("No v6_best.pt in %s — champion unchanged", run_id)
        return False

    corpus = playbook.get("corpus", {})
    assess_path = run_dir / "assess_report.json"
    report = run_assess(
        ckpt,
        corpus_manifest=str(corpus.get("manifest", "v6_corpus_stage_a_expand_v1.json")),
        max_proteins=int(corpus.get("max_proteins", 23)),
        device=str(playbook.get("budget", {}).get("device", "cuda")),
        output=assess_path,
        dry_run=dry_run,
    )
    if report is None:
        if dry_run:
            logger.info("[dry-run] would assess %s before promotion", ckpt)
        return False

    gate = report.get("promotion_gate", {})
    if not gate.get("passed"):
        logger.warning("Promotion gate failed for %s: %s", run_id, gate.get("failures"))
        return False

    new_score = read_checkpoint_score(ckpt)
    old_score = state.champion_score_from_disk()
    if new_score <= old_score:
        logger.info(
            "Score %.4f did not beat champion %.4f — no promotion",
            new_score,
            old_score,
        )
        return False

    rel = ckpt.relative_to(_REPO_ROOT)
    logger.info("Promoting champion → %s (score %.4f)", rel, new_score)
    if not dry_run:
        state.champion_run_id = run_id
        state.champion_checkpoint = str(rel)
        if state.last_action_id == "gate_promotion":
            state.gate_run_id = run_id
    return True


_DIAG_REPORT_PATH = _AUTO_DIR / "diagnostic_report.json"


def _plan_iteration(
    playbook: dict[str, Any],
    state: TrainerState,
    snap: MetricSnapshot,
    *,
    dry_run: bool,
    approve: bool = False,
    record_journal: bool = True,
) -> tuple[Any, Any | None]:
    """Collect diagnostics, evaluate program goals, and build a training plan."""
    from experiments.training.v6.auto_trainer_diagnostics import (
        collect_diagnostics,
        plan_training_iteration,
        write_diagnostic_report,
    )
    from experiments.training.v6.auto_trainer_program import (
        apply_program_evaluation,
        evaluate_program_goals,
        requires_program_approval,
    )

    bundle = collect_diagnostics(
        playbook,
        state,
        snap,
        repo_root=_REPO_ROOT,
        journal_path=_JOURNAL_PATH,
        skip_routing_eval=dry_run,
        routing_device="cpu",
    )
    program_eval = evaluate_program_goals(
        playbook, state, bundle, journal_path=_JOURNAL_PATH
    )
    if (
        program_eval.escalation_action
        and requires_program_approval(playbook, program_eval.escalation_action)
        and not approve
    ):
        program_eval.status = "blocked"
        program_eval.block_reason = (
            f"Approval required for escalation action '{program_eval.escalation_action}' "
            "(pass --approve or APPROVE=1)"
        )
    apply_program_evaluation(state, program_eval)
    if record_journal:
        append_journal({"event": "program_eval", **program_eval.to_dict()})

    for check in program_eval.checks:
        if check.status == "ok":
            continue
        logger.info("Program [%s] %s: %s", check.status, check.check_id, check.message)

    logger.info("Program status: %s", program_eval.status)
    if program_eval.escalation_action:
        logger.info("Program escalation action: %s", program_eval.escalation_action)

    plan = None
    if program_eval.status == "blocked" and not approve:
        logger.error("Program blocked: %s", program_eval.block_reason)
    else:
        forced = program_eval.escalation_action if program_eval.status == "warn" else None
        if program_eval.status == "blocked" and approve:
            forced = program_eval.escalation_action
        plan = plan_training_iteration(
            playbook,
            state,
            bundle,
            forced_action_id=forced,
        )

    program_payload = program_eval.to_dict()
    write_diagnostic_report(_DIAG_REPORT_PATH, bundle, plan, program=program_payload)
    (_AUTO_DIR / "program_evaluation.json").write_text(
        json.dumps(program_payload, indent=2) + "\n"
    )

    failing = [g for g in bundle.goals if not g.passing]
    if failing:
        summary = ", ".join(
            f"{g.goal_id}={g.value} (need "
            f"{'≥'+str(g.target_min) if g.target_min is not None else '≤'+str(g.target_max)})"
            for g in failing[:4]
        )
        logger.info("Benchmark gaps: %s", summary)
    logger.info(
        "Last run: action=%s promoted=%s run_id=%s",
        bundle.last_run.action_id,
        bundle.last_run.promoted,
        bundle.last_run.run_id,
    )
    if plan:
        logger.info(
            "Training plan: regime=%s goal=%s action=%s — %s",
            plan.regime_id,
            plan.primary_goal,
            plan.action_id,
            plan.rationale,
        )
    return bundle, plan


def print_program_status(
    playbook: dict[str, Any],
    state: TrainerState,
    *,
    dry_run: bool = True,
) -> None:
    """Print champion diagnostics, program checks, and next planned action."""
    snap = diagnose_champion(playbook, state, dry_run=dry_run)
    bundle, plan = _plan_iteration(
        playbook, state, snap, dry_run=dry_run, record_journal=False
    )
    progress = state.program_progress or {}
    print(json.dumps(
        {
            "champion": {
                "run_id": state.champion_run_id,
                "checkpoint": state.champion_checkpoint,
                "score": snap.champion_score,
                "routing_entropy": snap.routing_entropy,
                "probe_r_epi_sasa": snap.probe_r_epi_sasa,
            },
            "iteration": state.iteration,
            "program_status": state.program_status,
            "program_block_reason": state.program_block_reason,
            "program_escalation_action": state.program_escalation_action,
            "program_progress": progress,
            "failing_goals": [g.to_dict() for g in bundle.goals if not g.passing],
            "planned_training": plan.to_dict() if plan else None,
            "next_suggested": state.next_suggested,
            "stalled": state.stalled,
        },
        indent=2,
    ))


def _update_stall_counter(
    state: TrainerState,
    plan: Any,
    *,
    last_run_promoted: bool | None,
    iteration_promoted: bool | None,
) -> None:
    """Stall only when the same action repeats after consecutive unpromoted runs."""
    if plan.action_id != state.last_action_id:
        state.unpromoted_same_action = 0
        return
    was_promoted = iteration_promoted if iteration_promoted is not None else last_run_promoted
    if was_promoted is False:
        state.unpromoted_same_action += 1
    else:
        state.unpromoted_same_action = 0


def build_next_suggested_payload(
    action_id: str,
    action: dict[str, Any],
    snap: MetricSnapshot,
    *,
    state: TrainerState,
) -> dict[str, Any]:
    run_id_prefix = str(action.get("run_id_prefix", action_id))
    return {
        "action_id": action_id,
        "description": action.get("description"),
        "make_target": action.get("make_target"),
        "run_id_prefix": run_id_prefix,
        "example_run_id": build_run_id(run_id_prefix),
        "queued_for_iteration": state.iteration + 1,
        "champion_run_id": state.champion_run_id,
        "metrics_snapshot": {
            "champion_score": snap.champion_score,
            "routing_entropy": snap.routing_entropy,
            "probe_r_epi_sasa": snap.probe_r_epi_sasa,
            "promotion_gate_passed": snap.promotion_gate_passed,
            "primary_focus": snap.focus.get("primary_focus_str"),
        },
    }


def queue_next_suggested(
    playbook: dict[str, Any],
    state: TrainerState,
    *,
    promoted: bool | None = None,
    dry_run: bool = False,
    snap: MetricSnapshot | None = None,
    bundle: Any | None = None,
    plan: Any | None = None,
    approve: bool = False,
) -> None:
    """Preview the action that would run on the next iteration."""
    from experiments.training.v6.auto_trainer_diagnostics import build_next_suggested_from_plan

    if snap is None:
        snap = diagnose_champion(playbook, state, dry_run=dry_run)
    if bundle is None or plan is None:
        bundle, plan = _plan_iteration(
            playbook, state, snap, dry_run=dry_run, approve=approve
        )

    state.stalled = False
    state.stall_reason = None

    if plan is None:
        state.next_suggested = None
        append_journal(
            {
                "event": "next_suggested",
                "champion": state.champion_run_id,
                "action_id": None,
                "stalled": False,
                "diagnostics": bundle.to_dict(),
            }
        )
        logger.info("Next iteration: no playbook action matched — edit playbook or metrics")
        return

    _update_stall_counter(
        state,
        plan,
        last_run_promoted=bundle.last_run.promoted,
        iteration_promoted=promoted,
    )

    if state.unpromoted_same_action >= 2:
        state.stalled = True
        state.stall_reason = (
            f"Action '{plan.action_id}' queued {state.unpromoted_same_action} times "
            "without promotion — adjust playbook before next iteration"
        )

    state.next_suggested = build_next_suggested_from_plan(plan, state=state, bundle=bundle)
    append_journal(
        {
            "event": "next_suggested",
            "champion": state.champion_run_id,
            "promoted_last_iteration": promoted,
            "stalled": state.stalled,
            "next_suggested": state.next_suggested,
            "stall_reason": state.stall_reason,
            "diagnostics": bundle.to_dict(),
        }
    )

    suffix = " [STALLED — edit playbook before continuing]" if state.stalled else ""
    logger.info(
        "Next iteration (%d): %s [%s] — %s%s",
        state.iteration + 1,
        plan.action_id,
        plan.regime_id,
        plan.action.get("description"),
        suffix,
    )


def diagnose_champion(
    playbook: dict[str, Any],
    state: TrainerState,
    *,
    dry_run: bool = False,
) -> MetricSnapshot:
    ckpt = _REPO_ROOT / state.champion_checkpoint
    run_dir = _REPO_ROOT / "checkpoints/v6/runs" / state.champion_run_id
    assess_path = run_dir / "assess_report.json"

    assess: dict[str, Any] | None = None
    if assess_path.is_file():
        assess = json.loads(assess_path.read_text())
    elif ckpt.is_file() and not dry_run:
        corpus = playbook.get("corpus", {})
        assess = run_assess(
            ckpt,
            corpus_manifest=str(corpus.get("manifest", "v6_corpus_stage_a_expand_v1.json")),
            max_proteins=int(corpus.get("max_proteins", 23)),
            device=str(playbook.get("budget", {}).get("device", "cuda")),
            output=assess_path,
            dry_run=False,
        )

    snap = MetricSnapshot.from_run_dir(run_dir, assess=assess, champion_checkpoint=ckpt)
    logger.info(
        "Champion %s | score=%.4f route_H=%s r(epi,sasa)=%s gate=%s focus=%s",
        state.champion_run_id,
        snap.champion_score,
        f"{snap.routing_entropy:.3f}" if snap.routing_entropy is not None else "n/a",
        f"{snap.probe_r_epi_sasa:.3f}" if snap.probe_r_epi_sasa is not None else "n/a",
        snap.promotion_gate_passed,
        snap.focus.get("primary_focus_str"),
    )
    return snap


def run_iteration(
    playbook: dict[str, Any],
    state: TrainerState,
    *,
    dry_run: bool = False,
    approve: bool = False,
) -> bool:
    """Run one plan→train→assess cycle. Returns True if iteration ran."""
    budget = playbook.get("budget", {})
    program_budget = (playbook.get("program_goals") or {}).get("budget") or {}
    max_iter = int(budget.get("max_iterations", 5))
    max_program = program_budget.get("max_program_iterations")
    if max_program is not None:
        max_iter = max(max_iter, int(max_program))
    max_failed = int(budget.get("max_failed_runs", 3))

    if state.iteration >= max_iter:
        logger.info("Budget exhausted (%d iterations)", max_iter)
        return False
    if state.failed_runs >= max_failed:
        logger.error("Too many failed runs (%d) — stopping", max_failed)
        return False

    snap = diagnose_champion(playbook, state, dry_run=dry_run)
    if snap.promotion_gate_passed is False:
        state.failed_assessments += 1

    bundle, plan = _plan_iteration(playbook, state, snap, dry_run=dry_run, approve=approve)
    if plan is None and state.program_status == "blocked":
        logger.error("Stopping: program blocked — %s", state.program_block_reason)
        return False
    if plan is None:
        logger.info("No training plan matched current diagnostics")
        append_journal(
            {
                "event": "no_action",
                "champion": state.champion_run_id,
                "focus": snap.focus.get("primary_focus_str"),
                "diagnostics": bundle.to_dict(),
            }
        )
        state.iteration += 1
        queue_next_suggested(
            playbook, state, promoted=False, dry_run=dry_run, snap=snap, approve=approve
        )
        return True

    action_id = plan.action_id
    action = plan.action
    run_id = plan.run_id
    env = dict(plan.env)
    env["RUN_ID"] = run_id
    env.setdefault("DEVICE", str(budget.get("device", "cuda")))

    append_journal(
        {
            "event": "plan",
            "action_id": action_id,
            "run_id": run_id,
            "regime_id": plan.regime_id,
            "primary_goal": plan.primary_goal,
            "description": action.get("description"),
            "rationale": plan.rationale,
            "env": env,
            "diagnostics": bundle.to_dict(),
            "metrics": bundle.metrics,
        }
    )

    proc = run_make_target(
        str(action["make_target"]),
        env=env,
        dry_run=dry_run,
    )
    state.last_action_id = action_id
    state.last_run_id = run_id
    state.iteration += 1

    if dry_run:
        logger.info("[dry-run] planned %s → %s", action_id, run_id)
        queue_next_suggested(playbook, state, promoted=False, dry_run=True, approve=approve)
        return True

    if proc is None or proc.returncode != 0:
        state.failed_runs += 1
        append_journal({"event": "train_failed", "action_id": action_id, "run_id": run_id})
        logger.error("Training failed (exit %s)", proc.returncode if proc else "?")
        queue_next_suggested(playbook, state, promoted=False, dry_run=False, approve=approve)
        return True

    promote_run_id = str(plan.env.get("GATE_TOUCHUP_RUN") or run_id)
    if action_id == "full_push":
        promote_run_id = str(plan.env.get("GATE_TOUCHUP_RUN") or f"{run_id}_final")
    promoted = maybe_promote_champion(
        run_id=promote_run_id,
        state=state,
        playbook=playbook,
        dry_run=False,
    )
    from experiments.training.v6.auto_trainer_program import update_program_progress_after_iteration

    benchmark = playbook.get("benchmark_goals") or {}
    critical_ids = [
        gid for gid, cfg in benchmark.items() if str(cfg.get("tier", "")) == "critical"
    ]
    progress = state.program_progress or {}
    state.program_progress = update_program_progress_after_iteration(
        progress,
        bundle=bundle,
        plan_regime=plan.regime_id,
        promoted=promoted,
        critical_goal_ids=critical_ids,
    )
    append_journal(
        {
            "event": "train_complete",
            "action_id": action_id,
            "run_id": run_id,
            "regime_id": plan.regime_id,
            "primary_goal": plan.primary_goal,
            "promoted": promoted,
            "champion": state.champion_run_id,
            "metrics": bundle.metrics,
        }
    )
    queue_next_suggested(playbook, state, promoted=promoted, dry_run=False, approve=approve)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous v6 corpus-25 training loop")
    parser.add_argument("--playbook", type=Path, default=_DEFAULT_PLAYBOOK)
    parser.add_argument("--state", type=Path, default=_STATE_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not train")
    parser.add_argument("--once", action="store_true", help="Single iteration")
    parser.add_argument("--diagnose", action="store_true", help="Assess champion and exit")
    parser.add_argument("--status", action="store_true", help="Print program status JSON and exit")
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Allow blocked escalations (e.g. full_push) to run",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument(
        "--multi",
        type=int,
        default=None,
        metavar="N",
        help="Run up to N training iterations (sets budget + program_goals budget)",
    )
    args = parser.parse_args()

    playbook = load_playbook(args.playbook)
    state = TrainerState.load(args.state, playbook)
    if args.multi is not None:
        playbook.setdefault("budget", {})["max_iterations"] = args.multi
        playbook.setdefault("program_goals", {}).setdefault("budget", {})[
            "max_program_iterations"
        ] = args.multi
    if args.max_iterations is not None:
        playbook.setdefault("budget", {})["max_iterations"] = args.max_iterations

    _AUTO_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        print_program_status(playbook, state, dry_run=True)
        return

    if args.diagnose:
        snap = diagnose_champion(playbook, state, dry_run=args.dry_run)
        bundle, plan = _plan_iteration(
            playbook, state, snap, dry_run=args.dry_run, approve=args.approve
        )
        queue_next_suggested(
            playbook,
            state,
            promoted=bundle.last_run.promoted,
            dry_run=args.dry_run,
            snap=snap,
            bundle=bundle,
            plan=plan,
            approve=args.approve,
        )
        if not args.dry_run:
            state.save(args.state)
        if state.next_suggested:
            ns = state.next_suggested
            logger.info(
                "Queued for iteration %s: %s [%s] goal=%s — %s%s",
                ns.get("queued_for_iteration"),
                ns.get("action_id"),
                ns.get("regime_id"),
                ns.get("primary_goal"),
                ns.get("description"),
                " [STALLED]" if state.stalled else "",
            )
            if ns.get("rationale"):
                logger.info("Rationale: %s", ns.get("rationale"))
        return

    if state.stalled and not args.dry_run and not args.approve:
        logger.error(
            "Stalled: %s — edit playbook or run with --dry-run to replan",
            state.stall_reason,
        )
        return

    if state.program_status == "blocked" and not args.dry_run and not args.approve:
        logger.error(
            "Program blocked: %s — use --approve for escalation actions or edit playbook",
            state.program_block_reason,
        )
        return

    while run_iteration(playbook, state, dry_run=args.dry_run, approve=args.approve):
        if not args.dry_run:
            state.save(args.state)
        if args.once:
            break

    if not args.dry_run:
        state.save(args.state)
    logger.info("Auto-trainer finished (iteration=%d champion=%s)", state.iteration, state.champion_run_id)


if __name__ == "__main__":
    main()
