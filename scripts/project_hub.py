#!/usr/bin/env python3
"""Print Tokyo Eye Project Hub — live roadmap from phase status JSON.

SSOT status: data/gates/fix1_sparsity_biology_phase_status.json
Human hub:   docs/PROJECT_HUB.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = REPO / "data" / "gates" / "fix1_sparsity_biology_phase_status.json"
HUB = REPO / "docs" / "PROJECT_HUB.md"


def render(status_path: Path) -> str:
    d = json.loads(status_path.read_text())
    lines: list[str] = []
    lines.append("=== Tokyo Eye Project Hub (live) ===")
    lines.append(f"hub:    {HUB.relative_to(REPO)}")
    lines.append(f"status: {status_path.relative_to(REPO)}")
    lines.append(f"recorded_at: {d.get('recorded_at')}")
    lines.append(f"schema_version: {d.get('schema_version')}")
    lines.append(f"checkpoint_ssot: {d.get('checkpoint_ssot')}")
    champ = d.get("champion") or {}
    lines.append(
        f"champion: {champ.get('run_id')} ep{champ.get('epoch')} ({champ.get('verdict')})"
    )
    workstreams = d.get("workstreams") or {}
    if workstreams:
        lines.append("")
        lines.append("--- workstreams ---")
        for ws_id, ws in workstreams.items():
            lines.append(f"  [{ws.get('status', '?'):11}] {ws_id}: {ws.get('title')}")
            if ws.get("thesis"):
                lines.append(f"               {ws['thesis']}")
            meta = ws.get("meta") or {}
            model = meta.get("model") or {}
            if model.get("run_id") or model.get("epoch"):
                lines.append(
                    f"               model: {model.get('run_id')} ep{model.get('epoch')} "
                    f"({meta.get('gnn_input_mode', '')})"
                )
            if meta.get("updated_at"):
                lines.append(f"               updated: {meta['updated_at']}")
            for child in sorted(ws.get("children") or [], key=lambda c: int(c.get("order") or 99)):
                lines.append(
                    f"      {child.get('order')}. [{child.get('status', '?'):7}] "
                    f"{child.get('id')} — {child.get('title')}"
                )
                if child.get("note"):
                    lines.append(f"         {child['note']}")
                cm = child.get("meta") or {}
                dates = cm.get("dates") or {}
                outs = cm.get("outputs") or {}
                date_bits = []
                if dates.get("prereg_at"):
                    date_bits.append(f"prereg={dates['prereg_at'][:10]}")
                if dates.get("graded_at"):
                    date_bits.append(f"graded={dates['graded_at'][:10]}")
                if dates.get("closed_at"):
                    date_bits.append(f"closed={dates['closed_at'][:10]}")
                if date_bits:
                    lines.append(f"         dates: {', '.join(date_bits)}")
                if outs.get("verdict") is not None:
                    lines.append(f"         verdict: {outs['verdict']}")
                if outs.get("artifact"):
                    lines.append(f"         → {outs['artifact']}")
                elif (cm.get("inputs") or {}).get("roster"):
                    arms = list((cm["inputs"]["roster"]).keys())
                    lines.append(f"         inputs: arms={arms}")
            for defer in ws.get("deferred") or []:
                lines.append(f"      (deferred) {defer}")
    lines.append("")
    lines.append("--- ledger ---")
    for key, row in (d.get("ledger") or {}).items():
        st = row.get("status", "?")
        note = row.get("note", "")
        art = row.get("artifact", "")
        ws = row.get("workstream")
        lines.append(f"  [{st:7}] {key}" + (f"  ⌊{ws}" if ws else ""))
        if note:
            lines.append(f"           {note}")
        if art:
            lines.append(f"           → {art}")
    close = d.get("phase_4c_closeout") or {}
    if close:
        lines.append("")
        lines.append(f"--- phase_4c: {close.get('status')} ---")
        if close.get("claim_allowed"):
            lines.append(f"  allow: {close['claim_allowed']}")
        for forbid in close.get("claim_forbidden") or []:
            lines.append(f"  forbid: {forbid}")
    lines.append("")
    lines.append("--- next ---")
    for item in sorted(d.get("next") or [], key=lambda x: int(x.get("priority") or 99)):
        ws = item.get("workstream")
        suffix = f"  ⌊{ws}" if ws else ""
        lines.append(
            f"  {item.get('priority')}. [{item.get('id')}] {item.get('action')}{suffix}"
        )
        if item.get("make"):
            lines.append(f"      make: {item['make']}")
        if item.get("spec"):
            lines.append(f"      spec: {item['spec']}")
        if item.get("artifact"):
            lines.append(f"      artifact: {item['artifact']}")
        if item.get("prereg"):
            lines.append(f"      prereg: {item['prereg']}")
    lines.append("")
    lines.append("Open docs/PROJECT_HUB.md for the full doc map.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    args = p.parse_args(argv)
    if not args.status.is_file():
        raise SystemExit(f"missing status stamp: {args.status}")
    print(render(args.status), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
