#!/usr/bin/env python3
"""Generate TypeScript types from onboard_contract.yaml api_surfaces."""

from __future__ import annotations

from pathlib import Path

from data.readiness import BINDING_SCAN_COMPLETE_STATUSES
from science.contracts.onboard_contract import (
    get_api_surface_fields,
    get_artifact_catalog,
    get_tier_artifact_keys,
    load_contract,
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "visualizer"
    / "frontend"
    / "src"
    / "lib"
    / "generated"
    / "onboard.ts"
)

ARTIFACT_CONTRACT_PATH = OUTPUT_PATH.parent / "artifactContract.generated.ts"

HEADER = """// AUTO-GENERATED from science/contracts/onboard_contract.yaml — do not edit by hand.
// Regenerate: python -m science.contracts.generate_typescript

"""

ARTIFACT_HEADER = """// AUTO-GENERATED from science/contracts/onboard_contract.yaml — do not edit by hand.
// Regenerate: make contract-sync

"""


def _pipeline_job_block() -> str:
    contract = load_contract()
    statuses = contract.get("api_surfaces", {}).get("PipelineJob", {}).get(
        "status_values",
        ["queued", "running", "complete", "failed", "timed_out", "skipped"],
    )
    status_union = " | ".join(f'"{value}"' for value in statuses)
    return f"""export interface PipelineJob {{
  job_id: string;
  structure_id: string;
  status: {status_union};
  current_step?: string | null;
  progress?: number | null;
  modules?: unknown[] | null;
  error?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}}
"""


def _ingest_response_block() -> str:
    coordinator_fields = get_api_surface_fields("IngestResponse", variant="coordinator")
    optional = {
        "title",
        "audit_only",
        "audit_run_id",
        "pipeline_status",
        "pipeline_job_id",
        "pipeline_status_url",
        "readiness_url",
        "source",
        "already_existed",
        "chain_count",
        "onboard_geometric_notes",
    }
    lines = ["export interface IngestResponse {"]
    for field in coordinator_fields:
        suffix = "?" if field in optional else ""
        if field in {"residues", "residue_count", "chain_count", "atoms"}:
            ts_type = "number"
        elif field == "onboard_geometric_notes":
            ts_type = "string[]"
        elif field == "chains":
            ts_type = "string[] | number"
        elif field in {"audit_only", "already_existed"}:
            ts_type = "boolean"
        elif field in {"audit_run_id", "pipeline_job_id", "pipeline_status_url", "readiness_url"}:
            ts_type = "string | null"
        elif field == "pipeline_status":
            ts_type = '"queued" | "running" | "complete" | "failed" | "skipped" | null'
        else:
            ts_type = "string"
        lines.append(f"  {field}{suffix}: {ts_type};")
    lines.append("}")
    return "\n".join(lines)


def _act_readiness_block() -> str:
    return """export interface ActReadiness {
  act_id: string;
  number: number;
  title: string;
  question: string;
  color: string;
  status: "running" | "complete" | "degraded" | "failed" | "pending" | "not_implemented" | string;
  jobs_complete: number;
  jobs_total: number;
  required_artifacts: Record<string, boolean>;
  optional_artifacts: Record<string, boolean>;
}
"""


def _structure_readiness_block() -> str:
    return """export interface StructureReadiness {
  structure_id: string;
  readiness_status: "running" | "ready" | "degraded" | "failed";
  computation_run_id?: string | null;
  pathway: string;
  current_act: number;
  foundation: Record<string, boolean>;
  acts: Record<string, ActReadiness>;
  artifacts: Record<string, boolean>;
  tier1: Record<string, boolean>;
  tier2: Record<string, boolean>;
  missing_artifacts: string[];
  degraded_reasons: string[];
  probe_errors?: Record<string, string>;
  pipeline_job?: PipelineJob | null;
  geometric_readiness?: {
    requires_hyperbolic: boolean;
    required_hyperbolic_artifacts: Record<string, boolean>;
    hyperbolic_ready: boolean;
    learned_curvature?: number | null;
    curvature_ready?: boolean | null;
  };
}
"""


def _hydrate_meta_block() -> str:
    return """export interface ArtifactAvailabilityEntry {
  present: boolean;
  tier: number | null;
  reason: string | null;
}

export interface HydrateMeta {
  contract_version: string;
  degraded: boolean;
  missing_tier1_count: number;
  missing_keys: string[];
}
"""


def _hydration_response_block() -> str:
    fields = get_api_surface_fields("HydrateBundle")
    lines = ["export interface HydrationResponse {"]
    for field in fields:
        if field == "structure_id":
            lines.append("  structure_id: string;")
        elif field == "artifact_availability":
            lines.append("  artifact_availability?: Record<string, ArtifactAvailabilityEntry>;")
        elif field == "hydrate_meta":
            lines.append("  hydrate_meta?: HydrateMeta;")
        else:
            lines.append(f"  {field}?: unknown | null;")
    lines.append("}")
    return "\n".join(lines)


def _artifact_contract_block() -> str:
    catalog = get_artifact_catalog()
    tier1 = list(get_tier_artifact_keys(1))
    tiers: dict[str, int | None] = {}
    for spec in catalog.values():
        key = spec["canonical_key"]
        tier = spec.get("tier")
        tiers[key] = tier if isinstance(tier, int) else None

    binding_statuses = ", ".join(
        f'"{status}"' for status in sorted(BINDING_SCAN_COMPLETE_STATUSES)
    )
    tier1_lines = ",\n  ".join(f'"{key}"' for key in tier1)
    tier_lines = ",\n  ".join(
        f'"{key}": {("null" if value is None else value)}'
        for key, value in sorted(tiers.items())
    )

    return f"""export const BINDING_SCAN_COMPLETE_STATUSES = new Set([
  {binding_statuses},
]);

export const TIER1_ARTIFACT_KEYS = [
  {tier1_lines},
] as const;

export type Tier1ArtifactKey = (typeof TIER1_ARTIFACT_KEYS)[number];

export const ARTIFACT_TIERS: Record<string, number | null> = {{
  {tier_lines},
}};
"""


def generate_artifact_contract_typescript() -> str:
    return ARTIFACT_HEADER + _artifact_contract_block()


def generate_typescript() -> str:
    parts = [
        HEADER,
        _pipeline_job_block(),
        _act_readiness_block(),
        _structure_readiness_block(),
        _ingest_response_block(),
        _hydrate_meta_block(),
        _hydration_response_block(),
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(generate_typescript(), encoding="utf-8")
    ARTIFACT_CONTRACT_PATH.write_text(
        generate_artifact_contract_typescript(),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {ARTIFACT_CONTRACT_PATH}")


if __name__ == "__main__":
    main()
