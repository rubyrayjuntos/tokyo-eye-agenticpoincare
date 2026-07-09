"""Diagnostic bundle + goal-driven training plan for the v6 auto-trainer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.training.v6.per_structure_routing_audit import (
    STARVE_THRESHOLD,
    eval_checkpoint_routing,
    summarize_routing_floor,
)

if TYPE_CHECKING:
    from experiments.training.v6.auto_trainer import MetricSnapshot, TrainerState

_TIER_ORDER = {"critical": 0, "important": 1, "stretch": 2, "monitor": 3}


@dataclass
class GoalStatus:
    goal_id: str
    tier: str
    value: float | None
    target_min: float | None
    target_max: float | None
    passing: bool
    gap: float | None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "tier": self.tier,
            "value": self.value,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "passing": self.passing,
            "gap": self.gap,
            "description": self.description,
        }


@dataclass
class LastRunContext:
    action_id: str | None = None
    run_id: str | None = None
    promoted: bool | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "run_id": self.run_id,
            "promoted": self.promoted,
            "description": self.description,
        }


@dataclass
class TrainingPlan:
    action_id: str
    action: dict[str, Any]
    regime_id: str
    primary_goal: str
    env: dict[str, str]
    run_id: str
    rationale: str
    goal_statuses: list[GoalStatus] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "regime_id": self.regime_id,
            "primary_goal": self.primary_goal,
            "make_target": self.action.get("make_target"),
            "description": self.action.get("description"),
            "run_id": self.run_id,
            "rationale": self.rationale,
            "env": self.env,
            "failing_goals": [g.to_dict() for g in self.goal_statuses if not g.passing],
        }


@dataclass
class DiagnosticBundle:
    champion_run_id: str
    champion_checkpoint: str
    metrics: dict[str, float | None]
    focus: dict[str, Any]
    assess: dict[str, Any] | None
    routing_floor: dict[str, Any]
    goals: list[GoalStatus]
    last_run: LastRunContext
    primary_failing_goal: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion_run_id": self.champion_run_id,
            "champion_checkpoint": self.champion_checkpoint,
            "metrics": self.metrics,
            "routing_floor": self.routing_floor,
            "goals": [g.to_dict() for g in self.goals],
            "last_run": self.last_run.to_dict(),
            "primary_failing_goal": self.primary_failing_goal,
            "focus_primary": self.focus.get("primary_focus_str"),
            "promotion_gate_passed": self.metrics.get("promotion_gate_passed"),
        }


def read_last_run_context(journal_path: Path) -> LastRunContext:
    if not journal_path.is_file():
        return LastRunContext()
    last_plan: dict[str, Any] | None = None
    last_complete: dict[str, Any] | None = None
    for line in journal_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        event = entry.get("event")
        if event == "plan":
            last_plan = entry
        elif event == "train_complete":
            last_complete = entry
    return LastRunContext(
        action_id=(last_complete or last_plan or {}).get("action_id"),
        run_id=(last_complete or last_plan or {}).get("run_id"),
        promoted=(last_complete or {}).get("promoted"),
        description=(last_plan or {}).get("description"),
    )


def evaluate_goals(
    playbook: dict[str, Any],
    metrics: dict[str, float | None],
) -> list[GoalStatus]:
    goals_cfg = playbook.get("benchmark_goals") or {}
    statuses: list[GoalStatus] = []
    for goal_id, cfg in goals_cfg.items():
        value = metrics.get(goal_id)
        target_min = cfg.get("target_min")
        target_max = cfg.get("target_max")
        tier = str(cfg.get("tier", "monitor"))
        passing = True
        gap: float | None = None
        if value is None:
            passing = False
        elif target_min is not None and value < float(target_min):
            passing = False
            gap = float(target_min) - value
        elif target_max is not None and value > float(target_max):
            passing = False
            gap = value - float(target_max)
        statuses.append(
            GoalStatus(
                goal_id=goal_id,
                tier=tier,
                value=value,
                target_min=float(target_min) if target_min is not None else None,
                target_max=float(target_max) if target_max is not None else None,
                passing=passing,
                gap=gap,
                description=str(cfg.get("description", "")),
            )
        )
    statuses.sort(key=lambda g: (_TIER_ORDER.get(g.tier, 99), -(g.gap or 0.0)))
    return statuses


def _metric_from_focus(focus: dict[str, Any], name: str) -> float | None:
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


def build_metrics_snapshot(
    snap: MetricSnapshot,
    routing_floor: dict[str, Any],
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "champion_score": snap.champion_score,
        "routing_entropy": snap.routing_entropy,
        "probe_r_epi_sasa": snap.probe_r_epi_sasa,
        "probe_r_depth_sasa": _metric_from_focus(snap.focus, "probe_r_depth_sasa"),
        "probe_r_epi_ale": _metric_from_focus(snap.focus, "probe_r_epi_ale"),
        "per_structure_min_routing": routing_floor.get("per_structure_min_routing"),
        "promotion_gate_passed": (
            1.0 if snap.promotion_gate_passed else 0.0 if snap.promotion_gate_passed is False else None
        ),
        "expert_starvation_count": (
            float(snap.expert_starvation_count)
            if snap.expert_starvation_count is not None
            else None
        ),
    }
    if snap.assess:
        geom = snap.assess.get("geometry", {})
        if metrics["probe_r_depth_sasa"] is None:
            metrics["probe_r_depth_sasa"] = _as_float(geom.get("probe_r_depth_sasa"))
        if metrics["probe_r_epi_sasa"] is None:
            metrics["probe_r_epi_sasa"] = _as_float(geom.get("probe_r_epi_sasa"))
    return metrics


def collect_diagnostics(
    playbook: dict[str, Any],
    state: TrainerState,
    snap: MetricSnapshot,
    *,
    repo_root: Path,
    journal_path: Path,
    skip_routing_eval: bool = False,
    routing_device: str = "cpu",
) -> DiagnosticBundle:
    routing_floor: dict[str, Any] = {
        "per_structure_min_routing": None,
        "structures_below_floor": [],
        "skipped": skip_routing_eval,
    }
    if not skip_routing_eval and playbook.get("benchmark_goals", {}).get("per_structure_min_routing"):
        ckpt = repo_root / state.champion_checkpoint
        if ckpt.is_file():
            from experiments.training.v6.corpus import load_training_proteins

            corpus = playbook.get("corpus", {})
            manifest = repo_root / "manifests" / str(corpus.get("manifest", "v6_corpus_stage_a_expand_v1.json"))
            proteins, _failed = load_training_proteins(
                repo_root / "pdb_cache",
                manifest,
                max_proteins=int(corpus.get("max_proteins", 23)),
                max_residues=650,
                use_cache=True,
            )
            if proteins:
                rows = eval_checkpoint_routing(ckpt, proteins, routing_device)
                routing_floor = summarize_routing_floor(
                    rows,
                    threshold=float(
                        playbook.get("benchmark_goals", {})
                        .get("per_structure_min_routing", {})
                        .get("target_min", STARVE_THRESHOLD)
                    ),
                )

    metrics = build_metrics_snapshot(snap, routing_floor)
    goals = evaluate_goals(playbook, metrics)
    primary_failing = next((g.goal_id for g in goals if not g.passing), None)
    last_run = read_last_run_context(journal_path)
    if state.last_action_id and not last_run.action_id:
        last_run.action_id = state.last_action_id
        last_run.run_id = state.last_run_id

    return DiagnosticBundle(
        champion_run_id=state.champion_run_id,
        champion_checkpoint=state.champion_checkpoint,
        metrics=metrics,
        focus=snap.focus,
        assess=snap.assess,
        routing_floor=routing_floor,
        goals=goals,
        last_run=last_run,
        primary_failing_goal=primary_failing,
    )


def _regime_for_goal(playbook: dict[str, Any], goal_id: str) -> tuple[str, dict[str, Any]] | None:
    regime_id = (playbook.get("goal_regimes") or {}).get(goal_id)
    if not regime_id:
        return None
    regime = (playbook.get("regimes") or {}).get(regime_id)
    if not regime:
        return None
    return regime_id, regime


def _action_override_for_history(
    regime: dict[str, Any],
    last_run: LastRunContext,
) -> str | None:
    if last_run.promoted is not False or not last_run.action_id:
        return None
    return (regime.get("after_unpromoted") or {}).get(last_run.action_id)


def _merge_training_env(
    action: dict[str, Any],
    *,
    state: TrainerState,
    prior_run_id: str,
    primary_goal: str,
    bundle: DiagnosticBundle,
) -> dict[str, str]:
    from experiments.training.v6.auto_trainer import _format_env

    env = _format_env(action.get("env") or {}, state=state, prior_run_id=prior_run_id)
    setup = (action.get("setup") or {}).get(primary_goal) or {}
    if isinstance(setup, dict):
        for key, value in setup.items():
            env[str(key)] = str(value)

    if primary_goal == "per_structure_min_routing":
        worst = bundle.routing_floor.get("worst_min_r")
        if worst is not None and float(worst) < 0.04:
            severe = (action.get("setup") or {}).get("per_structure_min_routing_severe") or {}
            for key, value in severe.items():
                env[str(key)] = str(value)

    return env


def plan_training_iteration(
    playbook: dict[str, Any],
    state: TrainerState,
    bundle: DiagnosticBundle,
    *,
    forced_action_id: str | None = None,
) -> TrainingPlan | None:
    from experiments.training.v6.auto_trainer import MetricSnapshot, conditions_match, select_action

    actions = playbook.get("actions") or {}
    if forced_action_id:
        if forced_action_id not in actions:
            return None
        action = actions[forced_action_id]
        goal = bundle.primary_failing_goal
        if not goal:
            failing = [g for g in bundle.goals if not g.passing]
            goal = failing[0].goal_id if failing else "program_escalation"
        mapped = _regime_for_goal(playbook, goal)
        regime_id = mapped[0] if mapped else "program_escalation"
        from experiments.training.v6.auto_trainer import build_run_id

        run_id = build_run_id(str(action.get("run_id_prefix", forced_action_id)))
        env = _merge_training_env(
            action,
            state=state,
            prior_run_id=run_id,
            primary_goal=goal,
            bundle=bundle,
        )
        rationale = (
            f"Program escalation to '{forced_action_id}' for goal '{goal}' "
            f"(last run={bundle.last_run.action_id} promoted={bundle.last_run.promoted})"
        )
        return TrainingPlan(
            action_id=forced_action_id,
            action=action,
            regime_id=regime_id,
            primary_goal=goal,
            env=env,
            run_id=run_id,
            rationale=rationale,
            goal_statuses=bundle.goals,
        )

    snap = MetricSnapshot(
        focus=bundle.focus,
        assess=bundle.assess,
        champion_score=float(bundle.metrics.get("champion_score") or float("-inf")),
        routing_entropy=bundle.metrics.get("routing_entropy"),
        probe_r_epi_sasa=bundle.metrics.get("probe_r_epi_sasa"),
        promotion_gate_passed=(
            bool(bundle.metrics.get("promotion_gate_passed"))
            if bundle.metrics.get("promotion_gate_passed") is not None
            else None
        ),
        expert_starvation_count=(
            int(bundle.metrics["expert_starvation_count"])
            if bundle.metrics.get("expert_starvation_count") is not None
            else None
        ),
    )

    if playbook.get("regimes") and playbook.get("benchmark_goals"):
        failing = [g for g in bundle.goals if not g.passing]
        candidate_goals = failing or [g for g in bundle.goals if g.tier == "stretch"]
        for goal in candidate_goals:
            mapped = _regime_for_goal(playbook, goal.goal_id)
            if not mapped:
                continue
            regime_id, regime = mapped
            forced = _action_override_for_history(regime, bundle.last_run)
            action_order = [forced] if forced else list(regime.get("action_order") or [])
            actions = playbook.get("actions") or {}
            for action_id in action_order:
                if not action_id or action_id not in actions:
                    continue
                action = actions[action_id]
                from experiments.training.v6.auto_trainer import build_run_id

                run_id = build_run_id(str(action.get("run_id_prefix", action_id)))
                env = _merge_training_env(
                    action,
                    state=state,
                    prior_run_id=run_id,
                    primary_goal=goal.goal_id,
                    bundle=bundle,
                )
                rationale = (
                    f"Goal '{goal.goal_id}' failing (value={goal.value}, "
                    f"target_min={goal.target_min}, target_max={goal.target_max}); "
                    f"regime '{regime_id}'; last run={bundle.last_run.action_id} "
                    f"promoted={bundle.last_run.promoted}"
                )
                return TrainingPlan(
                    action_id=action_id,
                    action=action,
                    regime_id=regime_id,
                    primary_goal=goal.goal_id,
                    env=env,
                    run_id=run_id,
                    rationale=rationale,
                    goal_statuses=bundle.goals,
                )

    selected = select_action(playbook, snap, state)
    if selected is None:
        return None
    action_id, action = selected
    from experiments.training.v6.auto_trainer import build_run_id

    run_id = build_run_id(str(action.get("run_id_prefix", action_id)))
    env = _merge_training_env(
        action,
        state=state,
        prior_run_id=run_id,
        primary_goal=bundle.primary_failing_goal or "legacy",
        bundle=bundle,
    )
    return TrainingPlan(
        action_id=action_id,
        action=action,
        regime_id="legacy_priority",
        primary_goal=bundle.primary_failing_goal or "unknown",
        env=env,
        run_id=run_id,
        rationale="Legacy playbook priority (no matching regime)",
        goal_statuses=bundle.goals,
    )


def write_diagnostic_report(
    path: Path,
    bundle: DiagnosticBundle,
    plan: TrainingPlan | None,
    *,
    program: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnostics": bundle.to_dict(),
        "planned_training": plan.to_dict() if plan else None,
        "program": program,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_next_suggested_from_plan(
    plan: TrainingPlan,
    *,
    state: TrainerState,
    bundle: DiagnosticBundle,
) -> dict[str, Any]:
    return {
        "action_id": plan.action_id,
        "description": plan.action.get("description"),
        "make_target": plan.action.get("make_target"),
        "run_id_prefix": plan.action.get("run_id_prefix", plan.action_id),
        "example_run_id": plan.run_id,
        "queued_for_iteration": state.iteration + 1,
        "champion_run_id": state.champion_run_id,
        "regime_id": plan.regime_id,
        "primary_goal": plan.primary_goal,
        "rationale": plan.rationale,
        "training_env": plan.env,
        "last_run": bundle.last_run.to_dict(),
        "metrics_snapshot": bundle.metrics,
        "failing_goals": [g.to_dict() for g in bundle.goals if not g.passing],
        "routing_floor": bundle.routing_floor,
        "program_status": state.program_status,
        "program_escalation_action": state.program_escalation_action,
        "program_block_reason": state.program_block_reason,
    }
