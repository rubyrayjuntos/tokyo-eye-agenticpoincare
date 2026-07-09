"""Program-level goals for multi-run auto-trainer progress (Phase A — journal + state)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from experiments.training.v6.auto_trainer import TrainerState
    from experiments.training.v6.auto_trainer_diagnostics import DiagnosticBundle


@dataclass
class ProgramCheck:
    check_id: str
    status: str  # ok | warn | blocked
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class ProgramEvaluation:
    status: str  # ok | warn | blocked
    block_reason: str | None = None
    escalation_action: str | None = None
    escalation_regime: str | None = None
    checks: list[ProgramCheck] = field(default_factory=list)
    progress: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "block_reason": self.block_reason,
            "escalation_action": self.escalation_action,
            "escalation_regime": self.escalation_regime,
            "checks": [c.to_dict() for c in self.checks],
            "progress": self.progress,
        }


def default_program_progress() -> dict[str, Any]:
    return {
        "best_metrics": {},
        "iterations_since_improvement": {},
        "consecutive_unpromoted": 0,
        "consecutive_critical_promotions": 0,
        "regime_attempts": {},
        "total_train_iterations": 0,
    }


def load_program_progress(state: TrainerState) -> dict[str, Any]:
    raw = getattr(state, "program_progress", None)
    if not isinstance(raw, dict):
        return default_program_progress()
    merged = default_program_progress()
    merged.update(raw)
    if not isinstance(merged.get("best_metrics"), dict):
        merged["best_metrics"] = {}
    if not isinstance(merged.get("iterations_since_improvement"), dict):
        merged["iterations_since_improvement"] = {}
    if not isinstance(merged.get("regime_attempts"), dict):
        merged["regime_attempts"] = {}
    return merged


def read_journal_events(journal_path: Path) -> list[dict[str, Any]]:
    if not journal_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in journal_path.read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _metric_from_plan_event(event: dict[str, Any], metric: str) -> float | None:
    for key in ("metrics", "diagnostics"):
        block = event.get(key)
        if isinstance(block, dict):
            metrics = block.get("metrics", block) if key == "diagnostics" else block
            if isinstance(metrics, dict) and metric in metrics:
                try:
                    return float(metrics[metric])
                except (TypeError, ValueError):
                    pass
    return None


def rebuild_progress_from_journal(
    events: list[dict[str, Any]],
    *,
    critical_goal_ids: list[str],
) -> dict[str, Any]:
    """Rebuild program progress counters from journal train_complete events."""
    progress = default_program_progress()
    consecutive_unpromoted = 0
    consecutive_critical = 0
    best: dict[str, float] = {}

    for event in events:
        if event.get("event") != "train_complete":
            continue
        progress["total_train_iterations"] += 1
        promoted = bool(event.get("promoted"))
        if promoted:
            consecutive_unpromoted = 0
            consecutive_critical += 1
        else:
            consecutive_unpromoted += 1
            consecutive_critical = 0

        plan_event = None
        run_id = event.get("run_id")
        for prior in reversed(events):
            if prior.get("event") == "plan" and prior.get("run_id") == run_id:
                plan_event = prior
                break

        if plan_event:
            regime = plan_event.get("regime_id")
            if regime:
                progress["regime_attempts"][regime] = int(progress["regime_attempts"].get(regime, 0)) + 1
            for metric in critical_goal_ids:
                value = _metric_from_plan_event(plan_event, metric)
                if value is None:
                    continue
                prev_best = best.get(metric)
                if prev_best is None or value > prev_best:
                    best[metric] = value
                    progress["iterations_since_improvement"][metric] = 0
                else:
                    progress["iterations_since_improvement"][metric] = int(
                        progress["iterations_since_improvement"].get(metric, 0)
                    ) + 1

    progress["consecutive_unpromoted"] = consecutive_unpromoted
    progress["consecutive_critical_promotions"] = consecutive_critical
    progress["best_metrics"] = best
    return progress


def update_program_progress_after_iteration(
    progress: dict[str, Any],
    *,
    bundle: DiagnosticBundle,
    plan_regime: str | None,
    promoted: bool,
    critical_goal_ids: list[str],
) -> dict[str, Any]:
    progress = {**default_program_progress(), **progress}
    progress["total_train_iterations"] = int(progress.get("total_train_iterations", 0)) + 1

    if promoted:
        progress["consecutive_unpromoted"] = 0
        progress["consecutive_critical_promotions"] = int(
            progress.get("consecutive_critical_promotions", 0)
        ) + 1
    else:
        progress["consecutive_unpromoted"] = int(progress.get("consecutive_unpromoted", 0)) + 1
        progress["consecutive_critical_promotions"] = 0

    if plan_regime:
        attempts = dict(progress.get("regime_attempts") or {})
        attempts[plan_regime] = int(attempts.get(plan_regime, 0)) + 1
        progress["regime_attempts"] = attempts

    best = dict(progress.get("best_metrics") or {})
    since = dict(progress.get("iterations_since_improvement") or {})
    for metric in critical_goal_ids:
        value = bundle.metrics.get(metric)
        if value is None:
            since[metric] = int(since.get(metric, 0)) + 1
            continue
        prev = best.get(metric)
        if prev is None or float(value) > float(prev):
            best[metric] = float(value)
            since[metric] = 0
        else:
            since[metric] = int(since.get(metric, 0)) + 1

    progress["best_metrics"] = best
    progress["iterations_since_improvement"] = since
    return progress


def evaluate_program_goals(
    playbook: dict[str, Any],
    state: TrainerState,
    bundle: DiagnosticBundle,
    *,
    journal_path: Path,
) -> ProgramEvaluation:
    program_cfg = playbook.get("program_goals") or {}
    checks: list[ProgramCheck] = []

    benchmark = playbook.get("benchmark_goals") or {}
    critical_ids = [
        gid for gid, cfg in benchmark.items() if str(cfg.get("tier", "")) == "critical"
    ]

    events = read_journal_events(journal_path)
    progress = load_program_progress(state)
    if int(progress.get("total_train_iterations", 0)) == 0 and events:
        progress = rebuild_progress_from_journal(events, critical_goal_ids=critical_ids)

    budget = program_cfg.get("budget") or {}
    max_unpromoted = budget.get("max_consecutive_unpromoted")
    if max_unpromoted is not None:
        count = int(progress.get("consecutive_unpromoted", 0))
        if count >= int(max_unpromoted):
            checks.append(
                ProgramCheck(
                    "budget.max_consecutive_unpromoted",
                    "blocked",
                    f"{count} consecutive unpromoted runs (max {max_unpromoted})",
                    {"count": count, "max": int(max_unpromoted)},
                )
            )
        else:
            checks.append(
                ProgramCheck(
                    "budget.max_consecutive_unpromoted",
                    "ok",
                    f"{count}/{max_unpromoted} consecutive unpromoted",
                    {"count": count, "max": int(max_unpromoted)},
                )
            )

    max_program_iterations = budget.get("max_program_iterations")
    if max_program_iterations is not None:
        total = int(state.iteration)
        if total >= int(max_program_iterations):
            checks.append(
                ProgramCheck(
                    "budget.max_program_iterations",
                    "blocked",
                    f"Program iteration budget exhausted ({total}/{max_program_iterations})",
                    {"iteration": total, "max": int(max_program_iterations)},
                )
            )
        else:
            checks.append(
                ProgramCheck(
                    "budget.max_program_iterations",
                    "ok",
                    f"Program iterations {total}/{max_program_iterations}",
                    {"iteration": total, "max": int(max_program_iterations)},
                )
            )

    metric_progress_cfg = program_cfg.get("metric_progress") or {}
    escalation_action: str | None = None
    for metric, cfg in metric_progress_cfg.items():
        if not isinstance(cfg, dict):
            continue
        current = bundle.metrics.get(metric)
        best = (progress.get("best_metrics") or {}).get(metric)
        since = int((progress.get("iterations_since_improvement") or {}).get(metric, 0))
        max_stall = cfg.get("max_iterations_without_improvement")
        target_min = cfg.get("target_min")
        min_delta = float(cfg.get("min_delta", 0.0))

        if target_min is not None and current is not None and float(current) >= float(target_min):
            checks.append(
                ProgramCheck(
                    f"metric_progress.{metric}",
                    "ok",
                    f"{metric} reached program target ({current:.3f} ≥ {target_min})",
                    {"value": current, "target_min": target_min},
                )
            )
            continue

        if max_stall is not None and since >= int(max_stall):
            on_stall = cfg.get("on_stall")
            checks.append(
                ProgramCheck(
                    f"metric_progress.{metric}",
                    "warn",
                    f"{metric} unchanged for {since} iterations (max {max_stall})",
                    {
                        "value": current,
                        "best": best,
                        "since": since,
                        "on_stall": on_stall,
                    },
                )
            )
            if on_stall:
                escalation_action = str(on_stall)
        elif current is not None and best is not None and float(current) - float(best) < min_delta:
            checks.append(
                ProgramCheck(
                    f"metric_progress.{metric}",
                    "warn",
                    f"{metric} below min_delta since best ({current:.3f} vs best {best:.3f})",
                    {"value": current, "best": best, "min_delta": min_delta, "since": since},
                )
            )
        else:
            checks.append(
                ProgramCheck(
                    f"metric_progress.{metric}",
                    "ok",
                    f"{metric} progressing (since={since})",
                    {"value": current, "best": best, "since": since},
                )
            )

    regime_limits = budget.get("max_regime_attempts") or {}
    if isinstance(regime_limits, dict):
        attempts = progress.get("regime_attempts") or {}
        for regime, limit in regime_limits.items():
            count = int(attempts.get(regime, 0))
            if count >= int(limit):
                checks.append(
                    ProgramCheck(
                        f"budget.max_regime_attempts.{regime}",
                        "warn",
                        f"Regime {regime} attempted {count}× (limit {limit})",
                        {"count": count, "limit": int(limit)},
                    )
                )

    corpus_cfg = program_cfg.get("corpus_stage") or {}
    unlock = corpus_cfg.get("unlock_when") or {}
    streak_needed = unlock.get("critical_goals_passing_streak")
    if streak_needed is not None:
        streak = int(progress.get("consecutive_critical_promotions", 0))
        critical_passing = all(
            g.passing for g in bundle.goals if g.tier == "critical" and g.value is not None
        )
        if streak >= int(streak_needed) and critical_passing:
            checks.append(
                ProgramCheck(
                    "corpus_stage.unlock",
                    "ok",
                    f"Corpus expand ready ({corpus_cfg.get('next')})",
                    {"streak": streak, "next": corpus_cfg.get("next")},
                )
            )
        else:
            checks.append(
                ProgramCheck(
                    "corpus_stage.unlock",
                    "warn",
                    f"Corpus locked at {corpus_cfg.get('current')} "
                    f"(need {streak_needed} promoted streak + critical pass)",
                    {"streak": streak, "critical_passing": critical_passing},
                )
            )

    status = "ok"
    block_reason: str | None = None
    if any(c.status == "blocked" for c in checks):
        status = "blocked"
        block_reason = next(c.message for c in checks if c.status == "blocked")
    elif any(c.status == "warn" for c in checks):
        status = "warn"

    return ProgramEvaluation(
        status=status,
        block_reason=block_reason,
        escalation_action=escalation_action,
        checks=checks,
        progress=progress,
    )


def apply_program_evaluation(state: TrainerState, evaluation: ProgramEvaluation) -> None:
    state.program_status = evaluation.status
    state.program_block_reason = evaluation.block_reason
    state.program_escalation_action = evaluation.escalation_action
    state.program_progress = evaluation.progress


def requires_program_approval(playbook: dict[str, Any], action_id: str | None) -> bool:
    if not action_id:
        return False
    required = playbook.get("program_goals", {}).get("approval_required_for") or []
    return action_id in required
