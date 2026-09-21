"""EQU agenda SSOT consistency — prose never closes; dangling edges never pass.

Standing rule: prose without a failing test always loses to a convenient default.

Enforces ``data/gates/tokyo_eye_equ_agenda.json``:
  * every DONE / WONT_DO / SUPERSEDED entry has nonempty evidence
  * every blocks / blocked_by string resolves to a real item (or closed) id
  * blocks and blocked_by are mutual (if A blocks B, B lists A in blocked_by)
  * related_lessons resolve to lesson ids; depends_on_read resolves to item ids
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AGENDA = REPO / "data" / "gates" / "tokyo_eye_equ_agenda.json"

ALLOWED_STATUS = frozenset(
    {"OPEN", "BLOCKED", "IN_FLIGHT", "DONE", "WONT_DO", "SUPERSEDED"}
)
CLOSED_STATUSES = frozenset({"DONE", "WONT_DO", "SUPERSEDED"})


@pytest.fixture(scope="module")
def agenda() -> dict:
    assert AGENDA.is_file(), f"missing agenda SSOT: {AGENDA}"
    return json.loads(AGENDA.read_text())


def _item_ids(agenda: dict) -> set[str]:
    ids = {str(i["id"]) for i in agenda.get("items", [])}
    ids |= {str(i["id"]) for i in agenda.get("closed_this_period", [])}
    return ids


def _lesson_ids(agenda: dict) -> set[str]:
    return {str(x["id"]) for x in agenda.get("lessons", [])}


def test_agenda_schema_basics(agenda: dict) -> None:
    assert agenda.get("gate_id") == "tokyo_eye_equ_agenda"
    assert agenda.get("schema_version") == 1
    assert agenda.get("items"), "items must be non-empty while LIVE"
    assert agenda.get("lessons"), "lessons must be non-empty (append-only)"


def test_unique_ids(agenda: dict) -> None:
    item_ids = [i["id"] for i in agenda["items"]]
    closed_ids = [i["id"] for i in agenda.get("closed_this_period", [])]
    lesson_ids = [x["id"] for x in agenda["lessons"]]
    for label, ids in (
        ("items", item_ids),
        ("closed_this_period", closed_ids),
        ("lessons", lesson_ids),
    ):
        assert len(ids) == len(set(ids)), f"duplicate ids in {label}: {ids}"
    overlap = set(item_ids) & set(closed_ids)
    assert not overlap, f"id in both items and closed_this_period: {overlap}"


def test_status_and_done_requires_evidence(agenda: dict) -> None:
    rows = list(agenda["items"]) + list(agenda.get("closed_this_period", []))
    for row in rows:
        status = str(row.get("status", ""))
        assert status in ALLOWED_STATUS, f"{row.get('id')}: bad status {status!r}"
        if status in CLOSED_STATUSES:
            ev = row.get("evidence")
            assert isinstance(ev, list) and len(ev) > 0, (
                f"{row.get('id')}: status={status} requires nonempty evidence[] "
                "(prose never closes)"
            )
            for path in ev:
                assert isinstance(path, str) and path.strip(), (
                    f"{row.get('id')}: empty evidence entry"
                )


def test_lesson_evidence_nonempty(agenda: dict) -> None:
    for lesson in agenda["lessons"]:
        ev = lesson.get("evidence")
        assert isinstance(ev, list) and len(ev) > 0, (
            f"lesson {lesson.get('id')}: nonempty evidence required"
        )


def test_edge_refs_resolve(agenda: dict) -> None:
    item_ids = _item_ids(agenda)
    lesson_ids = _lesson_ids(agenda)
    rows = list(agenda["items"]) + list(agenda.get("closed_this_period", []))
    for row in rows:
        rid = row["id"]
        for edge_key in ("blocks", "blocked_by", "depends_on_read"):
            refs = row.get(edge_key) or []
            assert isinstance(refs, list), f"{rid}.{edge_key} must be a list"
            for ref in refs:
                assert ref in item_ids, (
                    f"{rid}.{edge_key} -> {ref!r} is not a real item/closed id"
                )
        for ref in row.get("related_lessons") or []:
            assert ref in lesson_ids, (
                f"{rid}.related_lessons -> {ref!r} is not a real lesson id"
            )


def test_blocks_blocked_by_symmetric(agenda: dict) -> None:
    """If A.blocks contains B, then B.blocked_by must contain A (and converse)."""
    rows = {str(i["id"]): i for i in agenda["items"]}
    for closed in agenda.get("closed_this_period", []):
        rows[str(closed["id"])] = closed

    errors: list[str] = []
    for aid, a in rows.items():
        for bid in a.get("blocks") or []:
            b = rows.get(bid)
            if b is None:
                continue  # covered by test_edge_refs_resolve
            if aid not in (b.get("blocked_by") or []):
                errors.append(
                    f"{aid} blocks {bid}, but {bid}.blocked_by does not list {aid}"
                )
        for bid in a.get("blocked_by") or []:
            b = rows.get(bid)
            if b is None:
                continue
            if aid not in (b.get("blocks") or []):
                errors.append(
                    f"{aid} blocked_by {bid}, but {bid}.blocks does not list {aid}"
                )
    assert not errors, "asymmetric agenda edges:\n  " + "\n  ".join(errors)
