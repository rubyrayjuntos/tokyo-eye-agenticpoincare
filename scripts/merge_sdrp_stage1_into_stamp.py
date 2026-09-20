"""Host agent: merge Stage 1 results into the operator-owned stamp."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = ROOT / "data" / "gates" / "tokyo_eye_equ_sdrp_target_wrap1.json"
RESULTS = ROOT / "data" / "gates" / "tokyo_eye_equ_sdrp_target_wrap1_stage1_results.json"


def main() -> None:
    r = json.loads(RESULTS.read_text())
    s = json.loads(STAMP.read_text())
    s["execution_state"] = "STAGE1_COMPLETE"
    s["candidates"] = {
        c: {
            "stage_1": v,
            "gates": v["gates"],
            "advances": v["passes_all_hard_gates"],
        }
        for c, v in r["candidates"].items()
        if c != "C0_status_quo"
    }
    s["candidates"]["C0_status_quo"] = {
        "stage_1": r["candidates"]["C0_status_quo"],
        "gates": r["candidates"]["C0_status_quo"]["gates"],
        "advances": False,
    }
    s["gates_stage_1"] = {c: v["gates"] for c, v in r["candidates"].items()}
    s["selected_candidate"] = r["selected_candidate"]
    s["stage_1_results_file"] = "data/gates/tokyo_eye_equ_sdrp_target_wrap1_stage1_results.json"
    s["stage_1_executed_at_utc"] = r["executed_at_utc"]
    s["stage_2_approved"] = False
    s["stage_1_caveats"] = r.get("caveats") or r.get("operator_notes") or [
        "C1 over-assigns rare relations toward incidence (salt 288/288, hydrophobe ~788/815)",
        "C1 class-1 is a strict subset of dehydron_labels (precision 1.0, G4 margin thin)",
        "C2 fails G2/G3 independently of G4 waiver",
    ]
    STAMP.write_text(json.dumps(s, indent=2, allow_nan=False) + "\n")
    print("merged into", STAMP)


if __name__ == "__main__":
    main()
