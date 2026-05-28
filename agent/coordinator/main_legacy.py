# Migrated from: SRC_AGENT/main.py on 2026-05-27
"""FastAPI entrypoint for the Tokyo Eyes Data Science Agent.

Serves the Coordinator agent via /chat and /health endpoints.
Deployed on ECS Fargate (or locally via uvicorn).
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

import bcrypt
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Standard Python logging (replaces google.cloud.logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="data_science",
    description="Tokyo Eyes Data Science Agent",
)


class ViewportState(BaseModel):
    """Current state of an open viewport, sent from frontend to agent."""

    viewport_type: str  # e.g. "poincare_disc"
    structure_id: str
    condition: str  # "gdp" | "gtp"
    mode: str  # "absolute" | "delta"
    color_by_metric: str  # "cone_depth" | "epistemic" | "aleatoric" | "total_uncertainty"
    zoom_level: float
    pan_offset: list[float]  # [x, y]
    focused_residue_id: str | None = None
    selected_residue_id: str | None = None
    visible_bounds: list[float] | None = None  # [min_x, min_y, max_x, max_y]
    mobius_enabled: bool = False


class HighlightGroup(BaseModel):
    """A named group of residues to highlight in the viewport."""

    group_name: str
    color: str  # CSS color
    residue_ids: list[str]


class ViewportDirective(BaseModel):
    """An instruction from the agent to the frontend to modify the viewport."""

    action: str  # "highlight" | "focus" | "set_metric" | "set_condition"
    highlight_groups: list[HighlightGroup] | None = None
    focus_residue_id: str | None = None
    metric: str | None = None
    condition: str | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    viewport_state: ViewportState | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    viewport_directives: list[ViewportDirective] | None = None


@app.get("/health")
def health():
    """Health check endpoint for ALB / ECS."""
    return {"status": "healthy"}


@app.post("/setup-schema")
def setup_schema():
    """Run the database schema SQL against Aurora.

    Executes both the core Tokyo Eyes schema and the GOSP pipeline
    star-schema.  Only callable from within the VPC.  Intended for
    initial setup.  Uses CREATE IF NOT EXISTS so it's safe to call
    multiple times.
    """
    import pathlib
    import pg8000

    schema_files = [
        pathlib.Path(__file__).parent / "scripts" / "create_schema.sql",
        pathlib.Path(__file__).parent / "scripts" / "gosp_schema.sql",
    ]

    missing = [str(f) for f in schema_files if not f.exists()]
    if missing:
        return {"status": "error", "message": f"Schema files not found: {missing}"}

    host = os.getenv("AURORA_HOST", "")
    port = int(os.getenv("AURORA_PORT", "5432"))
    database = os.getenv("AURORA_DATABASE", "tokyoeyes")
    user = os.getenv("AURORA_USER", "postgres")

    # Get password from Secrets Manager
    try:
        import boto3 as _boto3
        import json as _json

        rds_client = _boto3.client("rds", region_name=os.getenv("AWS_REGION", "us-east-1"))
        clusters = rds_client.describe_db_clusters(DBClusterIdentifier="tokyoeyes-aurora")
        secret_arn = clusters["DBClusters"][0]["MasterUserSecret"]["SecretArn"]

        sm_client = _boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION", "us-east-1"))
        secret = _json.loads(sm_client.get_secret_value(SecretId=secret_arn)["SecretString"])
        password = secret["password"]
    except Exception as e:
        return {"status": "error", "message": f"Could not retrieve Aurora password: {e}"}

    try:
        conn = pg8000.connect(host=host, port=port, database=database, user=user, password=password)
        conn.autocommit = True
        cur = conn.cursor()

        all_results = {}
        for schema_file in schema_files:
            sql = schema_file.read_text(encoding="utf-8")
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            # Filter out comment-only statements
            statements = [s for s in statements if not all(
                line.strip().startswith("--") or not line.strip()
                for line in s.splitlines()
            )]

            file_results = []
            for stmt in statements:
                try:
                    cur.execute(stmt)
                    file_results.append({"ok": stmt[:80]})
                except Exception as e:
                    file_results.append({"error": str(e), "stmt": stmt[:80]})

            all_results[schema_file.name] = {
                "statements": len(file_results),
                "results": file_results,
            }

        conn.close()
        return {"status": "ok", "schemas": all_results}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {e}"}


def _get_session_manager():
    """Return the appropriate session backend based on environment config.

    If SESSION_TABLE_NAME is set, use DynamoDB; otherwise use in-memory.
    """
    from data_science.session import DynamoDBSessionManager, SessionManager

    table_name = os.getenv("SESSION_TABLE_NAME")
    if table_name:
        return DynamoDBSessionManager(table_name)
    return SessionManager()


@app.get("/api/poincare-data")
def get_poincare_data(
    structure_id: str = Query(..., description="PDB ID or structure UUID"),
    run_id: str | None = Query(None, description="Specific DTIE run ID; defaults to latest"),
):
    """Return per-residue Poincaré disc data for both GDP and GTP conditions.

    Reads GNN node output from Aurora, projects 64-dim embeddings to 2D
    via PCA, and normalizes to the unit disc.

    When a DTIE run has an associated alignment_id, the response includes:
    - alignment_id, quality_class, common_core_size at the top level
    - per-residue alignment_deviation (Cα distance after superposition)
    - per-residue in_common_core flag and exclusion_reason annotation
    - Delta mode uses canonical_id matching for residue pairing

    Requirements: 10.1, 10.2, 10.3
    """
    from data_science.sub_agents.aurora.db import get_aurora_connection
    from data_science.sub_agents.dtie.store import read_alignment_metadata, read_poincare_data

    try:
        conn = get_aurora_connection()

        # Look up both GDP and GTP conditions via fact_dtie_run
        cursor = conn.cursor()
        alignment_id_from_run: str | None = None
        try:
            if run_id:
                cursor.execute(
                    "SELECT run_id, gdp_pdb_id, gtp_pdb_id, curvature_c, alignment_id "
                    "FROM fact_dtie_run WHERE run_id = %s LIMIT 1",
                    (run_id,),
                )
            else:
                cursor.execute(
                    "SELECT run_id, gdp_pdb_id, gtp_pdb_id, curvature_c, alignment_id "
                    "FROM fact_dtie_run "
                    "WHERE gdp_pdb_id = %s OR gtp_pdb_id = %s "
                    "ORDER BY computed_at DESC LIMIT 1",
                    (structure_id, structure_id),
                )
            dtie_row = cursor.fetchone()
        finally:
            cursor.close()

        conditions = []
        result: dict = {"structure_id": structure_id, "conditions": [], "curvature_c": 1.0}

        if dtie_row:
            _, gdp_pdb_id, gtp_pdb_id, curvature_c, alignment_id_from_run = dtie_row
            result["curvature_c"] = float(curvature_c) if curvature_c else 1.0

            gdp_data = read_poincare_data(conn, gdp_pdb_id, run_id=None)
            gtp_data = read_poincare_data(conn, gtp_pdb_id, run_id=None)

            if gdp_data:
                conditions.append("gdp")
                result["gdp"] = {"residues": gdp_data["residues"]}
                result["curvature_c"] = gdp_data["curvature_c"]
            if gtp_data:
                conditions.append("gtp")
                result["gtp"] = {"residues": gtp_data["residues"]}
                if not gdp_data:
                    result["curvature_c"] = gtp_data["curvature_c"]
        else:
            single = read_poincare_data(conn, structure_id, run_id=None)
            if single:
                conditions.append("gdp")
                result["gdp"] = {"residues": single["residues"]}
                result["curvature_c"] = single["curvature_c"]

        if not conditions:
            return JSONResponse(
                status_code=404,
                content={
                    "error": (
                        f"No GNN results for structure '{structure_id}'. "
                        "Run the DTIE pipeline first."
                    )
                },
            )

        result["conditions"] = conditions

        # Alignment metadata enrichment (Requirements 10.1, 10.2, 10.3)
        if alignment_id_from_run:
            alignment_meta = read_alignment_metadata(conn, alignment_id_from_run)
            if alignment_meta:
                result["alignment_id"] = alignment_meta["alignment_id"]
                result["quality_class"] = alignment_meta["quality_class"]
                result["common_core_size"] = alignment_meta["common_core_size"]

                residue_details = alignment_meta["residue_details"]
                for condition_key in ("gdp", "gtp"):
                    if condition_key not in result:
                        continue
                    for residue in result[condition_key]["residues"]:
                        rid = residue.get("residue_id", "")
                        detail = residue_details.get(rid)
                        if detail:
                            residue["alignment_deviation"] = detail["alignment_deviation"]
                            residue["in_common_core"] = detail["in_common_core"]
                            residue["exclusion_reason"] = detail["exclusion_reason"]
                            residue["canonical_id"] = detail["canonical_id"]
                        else:
                            residue["alignment_deviation"] = None
                            residue["in_common_core"] = None
                            residue["exclusion_reason"] = None
                            residue["canonical_id"] = None

                if "gdp" in result and "gtp" in result:
                    correspondence: dict[str, dict[str, str | None]] = {}
                    for residue in result["gdp"]["residues"]:
                        can_id = residue.get("canonical_id")
                        if can_id:
                            if can_id not in correspondence:
                                correspondence[can_id] = {"gdp_residue_id": None, "gtp_residue_id": None}
                            correspondence[can_id]["gdp_residue_id"] = residue.get("residue_id")
                    for residue in result["gtp"]["residues"]:
                        can_id = residue.get("canonical_id")
                        if can_id:
                            if can_id not in correspondence:
                                correspondence[can_id] = {"gdp_residue_id": None, "gtp_residue_id": None}
                            correspondence[can_id]["gtp_residue_id"] = residue.get("residue_id")
                    result["canonical_correspondence"] = correspondence

        return result

    except Exception as exc:
        logger.exception("poincare-data failed for %s", structure_id)
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
        )


def _build_agent_message(request: ChatRequest) -> str:
    """Build the message string sent to the agent, including viewport context.

    Pulls real-time viewport state from the WebSocket-based ViewportSession
    registry when available, falling back to the legacy REST-based
    ViewportState from the request body.

    Requirements: 9.1, 9.2, 9.3
    """
    from data_science.sub_agents.visualization.session import get_viewport_session

    # --- Try WebSocket-based viewport state first ---
    ws_context = _get_websocket_viewport_context(request.session_id)
    if ws_context:
        return (
            f"{request.message}\n\n"
            f"[The user has active viewers connected via the viewport protocol. "
            f"Use the viewport context below to ground your answer in what the "
            f"user is currently seeing. If you reference specific residues, they "
            f"will be highlighted automatically. Include ```viewport_directive "
            f"blocks to control the viewport.]\n"
            f"\n<VIEWPORT_CONTEXT>\n{ws_context}\n</VIEWPORT_CONTEXT>\n"
        )

    # --- Fallback to legacy REST-based viewport state ---
    if request.viewport_state is None:
        return request.message

    vs = request.viewport_state
    viewport_block = (
        "\n<VIEWPORT_STATE>\n"
        f'{vs.model_dump_json(indent=2)}\n'
        "</VIEWPORT_STATE>\n"
    )
    return (
        f"{request.message}\n\n"
        f"[The user currently has a {vs.viewport_type} viewport open for "
        f"structure {vs.structure_id} ({vs.condition}, {vs.color_by_metric}). "
        f"Use the viewport state below to contextualize your answer. "
        f"If you want to highlight residues or change the viewport, include a "
        f"JSON block tagged ```viewport_directive in your response.]\n"
        f"{viewport_block}"
    )


def _get_websocket_viewport_context(session_id: str | None) -> str | None:
    """Build a viewport context string from the WebSocket session registry.

    Returns None if no active WebSocket viewport session exists for the
    given session_id.

    Includes:
    - All active viewport states (viewer type, structures, camera, selections, coloring)
    - Recent event buffer summary (last 30 seconds)
    - Co-investigation status

    Requirements: 9.1, 9.2, 9.3
    """
    import time as _time

    from data_science.sub_agents.visualization.session import (
        VIEWPORT_GEOMETRY_CONTEXT,
        get_viewport_session,
    )

    if not session_id:
        return None

    vs = get_viewport_session(session_id)
    if vs is None or not vs.viewports:
        return None

    lines: list[str] = []

    # Active viewports
    lines.append(f"Active viewports: {len(vs.viewports)}")
    lines.append(f"Co-investigation: {'active' if vs.co_investigation_active else 'inactive'}")
    lines.append("")

    for vp in vs.viewports.values():
        lines.append(f"Viewport: {vp.viewport_id} ({vp.viewer_type})")
        if vp.mode:
            lines.append(f"  Mode: {vp.mode}")
        if vp.bloomed_hub:
            lines.append(f"  Bloomed hub (active pathway): {vp.bloomed_hub}")
        if vp.selected_pathways:
            lines.append(f"  Selected pathways: {', '.join(vp.selected_pathways)}")
        lines.append(f"  Structures: {', '.join(vp.structure_ids) if vp.structure_ids else 'none'}")
        if vp.camera_state:
            lines.append(f"  Camera: {json.dumps(vp.camera_state)}")
        if vp.active_selections:
            sel_summary = vp.active_selections[:10]
            suffix = f" (+{len(vp.active_selections) - 10} more)" if len(vp.active_selections) > 10 else ""
            lines.append(f"  Selections: {', '.join(sel_summary)}{suffix}")
        if vp.applied_coloring:
            lines.append(f"  Color-by: {vp.applied_coloring}")
        if vp.annotations:
            lines.append(f"  Annotations: {len(vp.annotations)} active")
        geometry = VIEWPORT_GEOMETRY_CONTEXT.get(vp.viewer_type)
        if geometry:
            lines.append(f"  [GEOMETRY CONTEXT FOR {vp.viewer_type}]")
            lines.append(geometry.strip())
        lines.append("")

    # Event buffer summary (last 30 seconds)
    now = _time.time()
    recent_events = vs.get_recent_events(now)
    if recent_events:
        # Summarize by event type
        event_counts: dict[str, int] = {}
        for evt in recent_events:
            event_counts[evt.event_type] = event_counts.get(evt.event_type, 0) + 1
        lines.append(f"Recent events (last 30s): {len(recent_events)} total")
        for etype, count in sorted(event_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {etype}: {count}")

        # Include the last few meaningful events for context
        meaningful_types = {"residue_click", "residue_hover", "mode_change", "metric_change"}
        meaningful = [e for e in recent_events if e.event_type in meaningful_types]
        if meaningful:
            last_events = meaningful[-5:]  # Last 5 meaningful events
            lines.append("  Last interactions:")
            for evt in last_events:
                payload_summary = ""
                if evt.payload:
                    # Extract key info from payload
                    rid = evt.payload.get("residue_id", "")
                    canonical = evt.payload.get("canonical_id", "")
                    if rid:
                        payload_summary = f" residue={rid}"
                    if canonical:
                        payload_summary += f" (canonical={canonical})"
                lines.append(f"    {evt.event_type}{payload_summary}")

    return "\n".join(lines)


def _parse_viewport_directives(response_text: str) -> list[ViewportDirective]:
    """Extract viewport directive JSON blocks from the agent response text.

    Looks for fenced code blocks tagged ``viewport_directive`` and parses
    each one as a ViewportDirective.
    """
    directives: list[ViewportDirective] = []
    pattern = re.compile(
        r"```viewport_directive\s*\n(.*?)\n\s*```",
        re.DOTALL,
    )
    for match in pattern.finditer(response_text):
        try:
            raw = json.loads(match.group(1))
            if isinstance(raw, list):
                directives.extend(ViewportDirective.model_validate(d) for d in raw)
            else:
                directives.append(ViewportDirective.model_validate(raw))
        except Exception:
            logger.warning(
                "Failed to parse viewport directive block",
                exc_info=True,
            )
    return directives


def _extract_residue_references(response_text: str) -> list[str]:
    """Parse agent response text for residue references.

    Detects patterns like:
    - Single-letter amino acid + position: G12, K16, L56, A146
    - Three-letter amino acid + position: Gly12, Lys16, Leu56
    - Residue ID patterns: residue 12, residue G12, Res12

    Returns a deduplicated list of residue identifiers found.

    Requirements: 9.2
    """
    residue_ids: list[str] = []
    seen: set[str] = set()

    # Pattern 1: Single-letter AA code + number (e.g., G12, K16, A146)
    # Must be preceded by whitespace/punctuation to avoid matching random text
    single_letter_pattern = re.compile(
        r'(?<![A-Za-z])([ACDEFGHIKLMNPQRSTVWY]\d{1,4})(?![A-Za-z\d])'
    )

    # Pattern 2: Three-letter AA code + number (e.g., Gly12, Lys16)
    three_letter_codes = (
        "Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|"
        "Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val"
    )
    three_letter_pattern = re.compile(
        rf'(?<![A-Za-z])({three_letter_codes})(\d{{1,4}})(?![A-Za-z\d])'
    )

    # Pattern 3: Explicit "residue X" references
    explicit_pattern = re.compile(
        r'[Rr]esidue[s]?\s+([A-Z]?\d{1,4}(?:\s*[-–,]\s*[A-Z]?\d{1,4})*)',
    )

    # Strip code blocks to avoid matching code/JSON content
    clean_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)

    for match in single_letter_pattern.finditer(clean_text):
        rid = match.group(1)
        if rid not in seen:
            seen.add(rid)
            residue_ids.append(rid)

    for match in three_letter_pattern.finditer(clean_text):
        # Normalize to single-letter + number format
        rid = match.group(0)
        if rid not in seen:
            seen.add(rid)
            residue_ids.append(rid)

    for match in explicit_pattern.finditer(clean_text):
        # Parse residue ranges/lists like "residues 12-18" or "residue G12, K16"
        ref_text = match.group(1)
        parts = re.split(r'[\s,]+', ref_text)
        for part in parts:
            part = part.strip('-–')
            if part and part not in seen:
                seen.add(part)
                residue_ids.append(part)

    return residue_ids


def _generate_highlight_directives(residue_ids: list[str]) -> list[ViewportDirective]:
    """Generate highlight directives for residues referenced in agent response.

    Requirements: 9.2
    """
    if not residue_ids:
        return []

    return [
        ViewportDirective(
            action="highlight",
            highlight_groups=[
                HighlightGroup(
                    group_name="agent_referenced",
                    color="#4A90D9",  # Soft blue for agent references
                    residue_ids=residue_ids,
                )
            ],
        )
    ]


def _strip_directive_blocks(response_text: str) -> str:
    """Remove viewport_directive fenced blocks from the user-facing response."""
    return re.sub(
        r"```viewport_directive\s*\n.*?\n\s*```",
        "",
        response_text,
        flags=re.DOTALL,
    ).strip()


# ---------------------------------------------------------------------------
# Phase 1: Proxy routes — replaces Express server.ts
# ---------------------------------------------------------------------------

# ── Shared config ─────────────────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).parent / "cache"
_CACHE_DIR.mkdir(exist_ok=True)

_AUTH_DB_PATH = _CACHE_DIR / "auth_db.json"

_JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "8"))

_STRING_SEMAPHORE = asyncio.Semaphore(3)  # max concurrent STRING requests
_STRING_CACHE_TTL = 86_400  # 24 h

_DEPMAP_URL = "https://depmap.org/portal/download/all?release=DepMap+Public+26Q1&file=CRISPRGeneDependency.csv"
_DEPMAP_CACHE_PATH = _CACHE_DIR / "CRISPRGeneDependency.csv"
_depmap_status: str = "fallback"
_depmap_centrality: dict = {}
_depmap_lock = asyncio.Lock()

PATHWAY_SEEDS: dict[str, list[str]] = {
    "RAS_MAPK":     ["KRAS","NRAS","HRAS","RAF1","MAP2K1","MAP2K2","MAPK1","MAPK3","EGFR","SOS1","GRB2","CDKN1A","CDKN2A"],
    "PI3K_AKT":     ["PIK3CA","AKT1","PTEN","MTOR","PIK3R1","KRAS","PDPK1","SOS1","GRB2"],
    "Cell_Cycle":   ["TP53","RB1","CDK4","CDK6","CCND1","CCNE1","E2F1","MDM2","CDKN1A","CDKN2A"],
    "Apoptosis":    ["TP53","BAX","BCL2","CASP3","MDM2","CDKN1A","CDKN2A"],
    "Angiogenesis": ["VEGFA","KDR","HIF1A","VHL","COL18A"],
    "WNT":          ["CTNNB1","APC","GSK3B","AXIN1","TCF7L2","MYC","CCND1","CDH1"],
    "NOTCH":        ["NOTCH1","HES1","MYC","DLL4","JAG1","CDKN1A","RBPJ"],
    "TGF_BETA":     ["TGFB1","SMAD2","SMAD3","SMAD4","CDKN1A","MYC","TP53"],
    "SRC_ABL":      ["SRC","ABL1","STAT3","PIK3CA","KRAS","PTK2","GRB2"],
    "MYC_Net":      ["MYC","MYCN","MAX","E2F1","BRD4","AURKA","CCND1","CDK4"],
    "AURORA":       ["AURKA","AURKB","TP53","MDM2","PLK1","KRAS","TPX2"],
}

REACTOME_IDS: dict[str, list[str]] = {
    "RAS_MAPK":     ["R-HSA-9649948"],
    "PI3K_AKT":     ["R-HSA-5218921"],
    "Cell_Cycle":   ["R-HSA-453276"],
    "Apoptosis":    ["R-HSA-109581"],
    "Angiogenesis": ["R-HSA-194138"],
    "WNT":          ["R-HSA-195721"],
    "NOTCH":        ["R-HSA-157118"],
    "TGF_BETA":     ["R-HSA-170834"],
    "SRC_ABL":      ["R-HSA-9006921"],
}

FALLBACK_CENTRALITY: dict[str, float] = {
    "RAS_MAPK": 0.95, "PI3K_AKT": 0.88, "Cell_Cycle": 0.72,
    "Apoptosis": 0.65, "Angiogenesis": 0.48,
}
_depmap_centrality = dict(FALLBACK_CENTRALITY)


# ── Auth helpers ──────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def _load_auth_db() -> dict:
    try:
        return json.loads(_AUTH_DB_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"users": [], "preferences": []}


def _save_auth_db(db: dict) -> None:
    _AUTH_DB_PATH.write_text(json.dumps(db, indent=2))


def _issue_jwt(user_id: int, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": int(time.time()) + _JWT_EXPIRY_HOURS * 3600,
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def _decode_jwt(token: str) -> dict:
    return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return _decode_jwt(creds.credentials)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ── DepMap helpers ─────────────────────────────────────────────────────────────

def _compute_centrality_from_csv(csv_path: Path) -> dict[str, float]:
    import csv as _csv
    all_genes = {g for genes in PATHWAY_SEEDS.values() for g in genes}
    gene_col: dict[str, list[int]] = {}
    gene_sums: dict[str, float] = {}
    gene_counts: dict[str, int] = {}

    with csv_path.open(newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
        for col_idx, col in enumerate(header):
            gene = col.strip('"').split(" ")[0].strip()
            if gene in all_genes:
                gene_col.setdefault(gene, []).append(col_idx)
                gene_sums[gene] = 0.0
                gene_counts[gene] = 0
        for row in reader:
            for gene, indices in gene_col.items():
                for idx in indices:
                    if idx < len(row):
                        try:
                            gene_sums[gene] += float(row[idx])
                            gene_counts[gene] += 1
                        except ValueError:
                            pass

    gene_means = {g: gene_sums[g] / gene_counts[g] for g in gene_sums if gene_counts[g] > 0}
    centrality: dict[str, float] = {}
    for pathway, genes in PATHWAY_SEEDS.items():
        available = [g for g in genes if g in gene_means]
        if available:
            mean = sum(gene_means[g] for g in available) / len(available)
            centrality[pathway] = min(0.98, max(0.1, -mean / 2.0))
        else:
            centrality[pathway] = FALLBACK_CENTRALITY.get(pathway, 0.5)
    return centrality


async def _download_depmap() -> None:
    global _depmap_status, _depmap_centrality
    async with _depmap_lock:
        _depmap_status = "downloading"
    try:
        import httpx as _httpx
        if not _DEPMAP_CACHE_PATH.exists():
            logger.info("Downloading DepMap CSV (~400 MB)...")
            async with _httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
                async with client.stream("GET", _DEPMAP_URL) as resp:
                    resp.raise_for_status()
                    with _DEPMAP_CACHE_PATH.open("wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
            logger.info("DepMap download complete.")
        centrality = await asyncio.get_event_loop().run_in_executor(
            None, _compute_centrality_from_csv, _DEPMAP_CACHE_PATH
        )
        async with _depmap_lock:
            _depmap_centrality = centrality
            _depmap_status = "ready"
        logger.info("DepMap centrality computed: %s", centrality)

        # Persist to Aurora under governance model
        _persist_depmap_to_aurora(centrality)

    except Exception as exc:
        logger.error("DepMap download/compute failed: %s", exc)
        if _DEPMAP_CACHE_PATH.exists():
            _DEPMAP_CACHE_PATH.unlink(missing_ok=True)
        async with _depmap_lock:
            _depmap_status = "failed"
            _depmap_centrality = dict(FALLBACK_CENTRALITY)


def _persist_depmap_to_aurora(centrality: dict) -> None:
    """Best-effort persistence of DepMap centrality to Aurora."""
    try:
        conn = _aurora_conn()
        from data_science.sub_agents.pipeline.network_store import (
            register_source,
            write_gene_essentiality,
        )
        source_id = register_source(
            conn,
            source_name="DepMap",
            source_version="26Q1",
            source_url=_DEPMAP_URL,
            record_count=len(centrality),
            ttl_hours=168,  # 7 days
        )
        write_gene_essentiality(conn, source_id, centrality)
    except Exception as e:
        logger.warning("DepMap Aurora persistence failed (non-critical): %s", e)


# Kick off DepMap bootstrap at startup (non-blocking)
@app.on_event("startup")
async def _startup_depmap():
    asyncio.create_task(_download_depmap())


# ── Server-side community detection (computed once, persisted to Aurora) ───────

# Cached community result (computed at startup, refreshed on STRING refresh)
_community_result: dict | None = None
_community_lock = asyncio.Lock()


def _louvain_python(
    nodes: list[str],
    edges: list[tuple[str, str]],
    edge_weights: dict[str, int],
    max_iterations: int = 20,
) -> dict[str, int]:
    """Louvain Phase 1 community detection (Python port of graph.ts)."""
    if not nodes:
        return {}

    adj: dict[str, list[tuple[str, float]]] = {n: [] for n in nodes}
    m = 0.0
    for u, v in edges:
        if u not in adj or v not in adj:
            continue
        w = edge_weights.get(f"{u}|{v}", edge_weights.get(f"{v}|{u}", 1))
        adj[u].append((v, w))
        adj[v].append((u, w))
        m += w
    if m == 0:
        m = 1.0
    two_m = 2.0 * m

    k = {n: sum(w for _, w in adj[n]) for n in nodes}
    comm = {n: i for i, n in enumerate(nodes)}
    comm_k: dict[int, float] = {i: k[n] for i, n in enumerate(nodes)}

    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for u in nodes:
            cu = comm[u]
            k_to_comm: dict[int, float] = {}
            for v, w in adj[u]:
                cv = comm[v]
                k_to_comm[cv] = k_to_comm.get(cv, 0.0) + w

            ku = k[u]
            comm_k[cu] -= ku
            k_u_cu = k_to_comm.get(cu, 0.0)

            best_comm = cu
            best_gain = 0.0
            for c_str, k_uc in k_to_comm.items():
                c = c_str
                if c == cu:
                    continue
                sigma_c = comm_k.get(c, 0.0)
                gain = (k_uc - k_u_cu) / m - (ku * (sigma_c - comm_k.get(cu, 0.0))) / (two_m * two_m)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = c

            comm_k[cu] += ku
            if best_comm != cu:
                comm_k[cu] -= ku
                comm_k[best_comm] = comm_k.get(best_comm, 0.0) + ku
                comm[u] = best_comm
                improved = True

    # Remap to contiguous IDs
    id_map: dict[int, int] = {}
    next_id = 0
    result: dict[str, int] = {}
    for n in nodes:
        c = comm[n]
        if c not in id_map:
            id_map[c] = next_id
            next_id += 1
        result[n] = id_map[c]
    return result


def _annotate_communities(
    communities: dict[str, int],
    reference_seeds: dict[str, list[str]],
    min_score: float = 0.15,
) -> list[dict]:
    """Annotate communities with best-matching pathway label (port of reactome.ts)."""
    groups: dict[int, list[str]] = {}
    for protein, c_id in communities.items():
        groups.setdefault(c_id, []).append(protein)

    ref_sets = {pw: set(g.upper() for g in genes) for pw, genes in reference_seeds.items()}

    labels = []
    for c_id, members in groups.items():
        member_upper = set(g.upper() for g in members)
        best_label = f"Cluster {c_id}"
        best_score = 0.0
        best_shared: list[str] = []

        for pw, ref_set in ref_sets.items():
            shared = [g for g in members if g.upper() in ref_set]
            overlap = len(shared) / min(len(member_upper), len(ref_set)) if min(len(member_upper), len(ref_set)) > 0 else 0
            if overlap > best_score:
                best_score = overlap
                best_label = pw if overlap >= min_score else f"Cluster {c_id}"
                best_shared = shared

        labels.append({
            "communityId": c_id,
            "label": best_label,
            "score": best_score,
            "topGenes": best_shared[:5],
            "size": len(members),
        })

    labels.sort(key=lambda x: x["size"], reverse=True)
    return labels


async def _compute_communities() -> dict | None:
    """Fetch CGC STRING network, run Louvain, annotate, persist to Aurora."""
    global _community_result

    try:
        # Fetch the full CGC network from STRING (uses cache)
        from poincare_frontend_src_lib_cosmicGenes import COSMIC_CGC_GENES
    except ImportError:
        # Fall back: read the gene list from the TS file
        cosmic_file = Path(__file__).parent / "poincare-frontend" / "src" / "lib" / "cosmicGenes.ts"
        if not cosmic_file.exists():
            logger.warning("cosmicGenes.ts not found — skipping community computation")
            return None
        content = cosmic_file.read_text()
        # Parse the array from TS
        import re as _re
        match = _re.search(r"export const COSMIC_CGC_GENES.*?=\s*\[(.*?)\]", content, _re.DOTALL)
        if not match:
            return None
        genes_raw = match.group(1)
        COSMIC_CGC_GENES = [g.strip().strip('"').strip("'") for g in genes_raw.split(",") if g.strip().strip('"').strip("'")]

    if not COSMIC_CGC_GENES:
        return None

    logger.info("Computing CGC communities (%d genes)...", len(COSMIC_CGC_GENES))

    # Fetch STRING data
    string_data = await _fetch_string(",".join(COSMIC_CGC_GENES))

    # Build graph
    nodes_set: set[str] = set()
    edges: list[tuple[str, str]] = []
    edge_weights: dict[str, int] = {}
    for row in string_data:
        u = row.get("preferredName_A")
        v = row.get("preferredName_B")
        if not u or not v:
            continue
        nodes_set.add(u)
        nodes_set.add(v)
        score = row.get("combined_score") or row.get("score", 0)
        if isinstance(score, float) and score <= 1.0:
            score = int(score * 1000)
        edges.append((u, v))
        edge_weights[f"{u}|{v}"] = int(score)
        edge_weights[f"{v}|{u}"] = int(score)

    nodes = list(nodes_set)
    if not nodes:
        return None

    # Run Louvain (CPU-bound, run in executor)
    communities = await asyncio.get_event_loop().run_in_executor(
        None, _louvain_python, nodes, edges, edge_weights, 20
    )

    # Annotate with pathway labels
    labels = _annotate_communities(communities, PATHWAY_SEEDS)

    result = {
        "communities": communities,
        "labels": labels,
        "nodes": nodes,
        "edges": [[u, v] for u, v in edges],
        "edgeWeights": edge_weights,
    }

    async with _community_lock:
        _community_result = result

    # Persist to Aurora (best-effort)
    try:
        conn = _aurora_conn()
        from data_science.sub_agents.pipeline.network_store import (
            register_source,
            write_community_detection,
        )
        source_id = register_source(
            conn,
            source_name="STRING_CGC_Louvain",
            source_version="12.0",
            source_url="https://string-db.org/api/json/network",
            record_count=len(nodes),
            parameters={"algorithm": "louvain", "n_genes": len(COSMIC_CGC_GENES), "n_edges": len(edges)},
            ttl_hours=24,
        )
        write_community_detection(
            conn, source_id, communities, labels,
            input_nodes=len(nodes), input_edges=len(edges),
        )
        logger.info("Persisted %d communities to Aurora", len(set(communities.values())))
    except Exception as e:
        logger.warning("Community Aurora persistence failed (non-critical): %s", e)

    logger.info(
        "Community detection complete: %d nodes, %d edges, %d communities",
        len(nodes), len(edges), len(set(communities.values())),
    )
    return result


@app.on_event("startup")
async def _startup_communities():
    """Compute CGC communities after a short delay (let STRING cache warm up)."""
    async def _delayed():
        await asyncio.sleep(5)  # Let STRING cache populate first
        await _compute_communities()
    asyncio.create_task(_delayed())


@app.get("/api/communities")
async def get_communities():
    """Return pre-computed CGC community detection results.

    The frontend can use this instead of running Louvain client-side.
    Returns communities, labels, and the full graph for rendering.
    """
    if _community_result is None:
        return JSONResponse(
            status_code=202,
            content={"status": "computing", "message": "Community detection in progress"},
        )
    return _community_result


@app.post("/api/communities/refresh")
async def refresh_communities(_: dict = Depends(get_current_user)):
    """Force recomputation of CGC communities (e.g. after STRING cache refresh)."""
    asyncio.create_task(_compute_communities())
    return {"status": "recomputing"}


# ── STRING proxy (1.1) ────────────────────────────────────────────────────────

async def _fetch_string(proteins: str) -> list:
    """Core STRING fetch logic shared by GET and POST handlers.

    Always uses POST to STRING to avoid URL length limits with large gene lists
    (Reactome pathways can return 400+ genes). Cache key is a stable MD5 hash
    of the sorted protein list so it survives container restarts.
    """
    import hashlib
    import httpx as _httpx

    protein_list = [p.strip() for p in proteins.replace("\r", ",").split(",") if p.strip()]
    if not protein_list:
        return []

    cache_key = ",".join(sorted(protein_list))
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_file = _CACHE_DIR / f"string_{cache_hash}.json"

    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < _STRING_CACHE_TTL:
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    async with _STRING_SEMAPHORE:
        # Re-check cache after acquiring semaphore — another request may have
        # populated it while we were waiting.
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < _STRING_CACHE_TTL:
            try:
                return json.loads(cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        try:
            async with _httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://string-db.org/api/json/network",
                    data={
                        "identifiers": "\r".join(protein_list),
                        "species": "9606",
                        "required_score": "700",
                        "network_type": "functional",
                        "caller_identity": "tokyo_eyes",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("STRING API error: %s", exc)
            raise HTTPException(status_code=502, detail="STRING API unavailable")

    try:
        cache_file.write_text(json.dumps(data))
    except OSError:
        pass

    # Persist to Aurora under governance model (best-effort, non-blocking)
    _persist_string_to_aurora(data, protein_list)

    return data


def _persist_string_to_aurora(data: list, protein_list: list) -> None:
    """Best-effort persistence of STRING interactions to Aurora."""
    try:
        conn = _aurora_conn()
        from data_science.sub_agents.pipeline.network_store import (
            register_source,
            write_string_interactions,
        )
        source_id = register_source(
            conn,
            source_name="STRING",
            source_version="12.0",
            source_url="https://string-db.org/api/json/network",
            record_count=len(data),
            parameters={"species": "9606", "required_score": "700", "n_proteins": len(protein_list)},
            ttl_hours=24,
        )
        write_string_interactions(conn, source_id, data)
    except Exception as e:
        logger.warning("STRING Aurora persistence failed (non-critical): %s", e)


class _StringPostBody(BaseModel):
    proteins: str


@app.get("/api/string")
async def proxy_string_get(proteins: str = Query(...)):
    """STRING proxy — GET for small lists (≤50 proteins, backward-compatible)."""
    return await _fetch_string(proteins)


@app.post("/api/string")
async def proxy_string_post(body: _StringPostBody):
    """STRING proxy — POST for large lists (>50 proteins, avoids URL length limits)."""
    return await _fetch_string(body.proteins)


# ── Reactome proxy (1.2) ──────────────────────────────────────────────────────

@app.get("/api/reactome")
async def proxy_reactome(pathway: str = Query(...)):
    """Return deduplicated HGNC gene symbols for a named cancer pathway.

    Falls back to PATHWAY_SEEDS when Reactome is unreachable or the pathway
    has no REACTOME_IDS mapping. Results are cached to disk indefinitely
    (pathway gene membership changes rarely).
    """
    import httpx as _httpx

    pw = pathway.strip()
    ids = REACTOME_IDS.get(pw)
    if not ids:
        return PATHWAY_SEEDS.get(pw, [])

    cache_file = _CACHE_DIR / f"reactome_{pw}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    gene_set: set[str] = set()
    async with _httpx.AsyncClient(timeout=20) as client:
        results = await asyncio.gather(
            *[
                client.get(
                    f"https://reactome.org/ContentService/data/participants/{rid}/referenceEntities"
                )
                for rid in ids
            ],
            return_exceptions=True,
        )
    for resp in results:
        if isinstance(resp, Exception) or resp.status_code != 200:
            continue
        try:
            for entity in resp.json():
                if entity.get("className", "").startswith("Reference") or \
                   entity.get("schemaClass", "").startswith("Reference"):
                    for g in entity.get("geneName", []):
                        gene_set.add(g)
        except Exception:
            pass

    genes = list(gene_set) if gene_set else PATHWAY_SEEDS.get(pw, [])
    try:
        cache_file.write_text(json.dumps(genes))
    except OSError:
        pass

    # Persist pathway membership to Aurora (best-effort)
    _persist_reactome_to_aurora(pw, genes)

    return genes


def _persist_reactome_to_aurora(pathway_name: str, genes: list) -> None:
    """Best-effort persistence of Reactome pathway membership to Aurora."""
    try:
        conn = _aurora_conn()
        from data_science.sub_agents.pipeline.network_store import (
            register_source,
            write_pathway_membership,
        )
        source_id = register_source(
            conn,
            source_name="Reactome",
            source_version="89",
            source_url="https://reactome.org/ContentService/data/participants",
            record_count=len(genes),
            parameters={"pathway": pathway_name},
            ttl_hours=720,  # 30 days — pathway membership changes rarely
        )
        write_pathway_membership(conn, source_id, {pathway_name: genes})
    except Exception as e:
        logger.warning("Reactome Aurora persistence failed (non-critical): %s", e)


# ── Reactome cache refresh (break-glass) ─────────────────────────────────────

@app.post("/api/reactome/refresh")
async def refresh_reactome(pathway: str = Query(...), _: dict = Depends(get_current_user)):
    """Delete the cached Reactome gene list for a pathway and force re-fetch.

    Use when Reactome returns stale or malformed data that got written to disk.
    Subsequent GET /api/reactome?pathway=<name> will re-fetch from Reactome.
    Pass pathway=* to clear all cached Reactome files.
    """
    if pathway == "*":
        deleted = [f.name for f in _CACHE_DIR.glob("reactome_*.json")]
        for f in _CACHE_DIR.glob("reactome_*.json"):
            f.unlink(missing_ok=True)
        return {"deleted": deleted}

    pw = pathway.strip()
    cache_file = _CACHE_DIR / f"reactome_{pw}.json"
    if cache_file.exists():
        cache_file.unlink()
        return {"deleted": [cache_file.name]}
    return {"deleted": []}


# ── UniProt binding proxy (1.3) ───────────────────────────────────────────────

@app.get("/api/uniprot/binding")
async def proxy_uniprot_binding(gene: str = Query(...)):
    """Return binding sites and active site flag for a human gene from UniProt.

    Queries reviewed (Swiss-Prot) entries only. Falls back to empty result on
    any error so the disc renders without blocking.
    """
    import httpx as _httpx

    gene = gene.strip().upper()
    if not gene:
        return {"bindingLigands": [], "hasActiveSite": False}

    cache_file = _CACHE_DIR / f"uniprot_binding_{gene}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    try:
        async with _httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
                    "fields": "ft_binding,ft_act_site",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("UniProt binding error for %s: %s", gene, exc)
        return {"bindingLigands": [], "hasActiveSite": False}

    entry = (data.get("results") or [None])[0] or {}
    features = entry.get("features", [])
    ligands: set[str] = set()
    has_active_site = False
    for f in features:
        if f.get("type") == "Binding site":
            name = f.get("ligand", {}).get("name") or f.get("description", "")
            if name:
                ligands.add(name)
        if f.get("type") == "Active site":
            has_active_site = True

    result = {"bindingLigands": sorted(ligands), "hasActiveSite": has_active_site}
    try:
        cache_file.write_text(json.dumps(result))
    except OSError:
        pass
    return result


# ── OncoKB genes (1.3b) ───────────────────────────────────────────────────────

@app.get("/api/oncokb/genes")
async def get_oncokb_genes():
    """Return oncogene/TSG classifications from OncoKB.

    Requires ONCOKB_TOKEN env var. Returns {available: false} silently when
    the token is absent — client falls back to hardcoded PROTEIN_ROLES.
    """
    import httpx as _httpx

    token = os.getenv("ONCOKB_TOKEN")
    if not token:
        return {"available": False, "genes": {}}

    cache_file = _CACHE_DIR / "oncokb_genes.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    try:
        async with _httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://www.oncokb.org/api/v1/genes",
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            raw: list[dict] = resp.json()
    except Exception as exc:
        logger.warning("OncoKB error: %s", exc)
        return {"available": False, "genes": {}}

    genes: dict[str, dict] = {}
    for g in raw:
        sym = g.get("hugoSymbol")
        if not sym:
            continue
        if g.get("oncogene"):
            genes[sym] = {"role": "oncogene", "druggable": True}
        elif g.get("tsg"):
            genes[sym] = {"role": "tumor_suppressor", "druggable": False}

    payload = {"available": True, "genes": genes}
    try:
        cache_file.write_text(json.dumps(payload))
    except OSError:
        pass
    return payload


# ── DepMap routes (1.4) ───────────────────────────────────────────────────────

@app.get("/api/depmap")
async def get_depmap():
    """Return current DepMap status and per-pathway centrality scores."""
    return {"status": _depmap_status, "centrality": _depmap_centrality}


@app.post("/api/depmap/refresh")
async def refresh_depmap(_: dict = Depends(get_current_user)):
    """Delete cached DepMap CSV and re-download in the background."""
    _DEPMAP_CACHE_PATH.unlink(missing_ok=True)
    asyncio.create_task(_download_depmap())
    return {"status": "downloading"}


# ── Phase 5: DTIE Aurora endpoints ───────────────────────────────────────────
#
# Graceful degradation pattern: all three endpoints return aurora_available=False
# and empty/null data rather than 503 when Aurora is unreachable.

def _aurora_conn():
    """Return an Aurora connection or raise. Caller wraps with try/except."""
    from data_science.sub_agents.aurora.db import get_aurora_connection
    return get_aurora_connection()


@app.get("/api/dtie-status")
def get_dtie_status(
    structure_id: str = Query(..., description="PDB ID to check (e.g. '6GOF_GDP')"),
    _user: dict = Depends(get_current_user),
):
    """Return pipeline phase completion for a structure.

    Queries dim_ingestion_status for completed phases.
    """
    try:
        conn = _aurora_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT phase_name, status FROM dim_ingestion_status WHERE structure_id = %s",
            (structure_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
    except Exception:
        logger.warning("Aurora unreachable — returning empty dtie-status for %s", structure_id)
        return {"available": False, "phases_complete": [], "aurora_available": False}

    phases_complete = [row[0] for row in rows if row[1] == "complete"]
    return {
        "available": len(phases_complete) > 0,
        "phases_complete": phases_complete,
        "aurora_available": True,
    }


@app.get("/api/dtie/run")
def get_dtie_run(
    manifestPath: str = Query(None, description="Optional manifest path filter"),
    _user: dict = Depends(get_current_user),
):
    """Return DTIE pipeline run results.

    Queries fact_dtie_run for the latest (or specified) run and returns
    aggregate statistics and per-structure results for the React frontend.
    """
    try:
        conn = _aurora_conn()
        cursor = conn.cursor()

        if manifestPath:
            cursor.execute(
                "SELECT run_id, gdp_pdb_id, gtp_pdb_id, pipeline_mode, "
                "curvature_c, n_doorways, n_pathways, n_pharmacophores, "
                "runtime_sec, results_json, computed_at "
                "FROM fact_dtie_run WHERE run_id = %s",
                (manifestPath,),
            )
        else:
            cursor.execute(
                "SELECT run_id, gdp_pdb_id, gtp_pdb_id, pipeline_mode, "
                "curvature_c, n_doorways, n_pathways, n_pharmacophores, "
                "runtime_sec, results_json, computed_at "
                "FROM fact_dtie_run ORDER BY computed_at DESC LIMIT 20"
            )

        rows = cursor.fetchall()
        cursor.close()
    except Exception as e:
        logger.warning("Aurora unreachable for /api/dtie/run: %s", e)
        return None

    if not rows:
        return None

    import json as _json

    results = []
    for row in rows:
        results_json = row[9]
        parsed = _json.loads(results_json) if isinstance(results_json, str) else (results_json or {})
        results.append({
            "pdbId": row[1],
            "pdbPath": row[1],
            "nodeCount": parsed.get("phase1", {}).get("n_witnesses", 0),
            "edgeCount": parsed.get("phase4", {}).get("n_conductance_paths", 0),
            "dehydronScoreMean": 0.0,
            "dehydronScoreMax": 0.0,
            "coneDepthMean": 0.0,
            "coneWidthMean": 0.0,
            "uncertainty": {
                "epistemic": 0.0,
                "aleatoric": 0.0,
                "total": 0.0,
            },
            "expertTop1Distribution": [],
            "expertDominance": 0.0,
            "vulnerabilityScore": parsed.get("phase2", {}).get("n_doorways", 0),
        })

    # Aggregate stats
    n_total = len(results)
    n_success = sum(1 for r in results if r["vulnerabilityScore"] > 0)

    return {
        "source": "aurora",
        "manifestPath": rows[0][0] if manifestPath else "latest",
        "schema": "fact_dtie_run",
        "generatedAtUtc": str(rows[0][10]) if rows[0][10] else "",
        "countTotal": n_total,
        "countSuccess": n_success,
        "countFailed": n_total - n_success,
        "aggregate": {
            "dehydronMean": 0.0,
            "coneDepthMean": 0.0,
            "coneWidthMean": 0.0,
            "uncertaintyTotalMean": 0.0,
        },
        "topStructures": [
            {
                "pdbId": r["pdbId"],
                "vulnerabilityScore": r["vulnerabilityScore"],
                "uncertaintyTotal": r["uncertainty"]["total"],
                "expertDominance": r["expertDominance"],
            }
            for r in sorted(results, key=lambda x: x["vulnerabilityScore"], reverse=True)[:5]
        ],
        "results": results,
    }


@app.get("/api/protein-structure")
def get_protein_structure(
    gene: str = Query(..., description="Gene symbol (e.g. 'KRAS')"),
    _user: dict = Depends(get_current_user),
):
    """Return structure IDs with DTIE data available for a gene symbol.

    Queries fact_dtie_run filtered by gene_symbol (migration 010), intersected
    with dim_ingestion_status to return only structures with completed phases.
    """
    try:
        conn = _aurora_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT r.gdp_pdb_id
            FROM fact_dtie_run r
            JOIN dim_ingestion_status dis
                ON dis.structure_id = r.gdp_pdb_id
               AND dis.status = 'complete'
            WHERE r.gene_symbol = %s
            """,
            (gene,),
        )
        rows = cursor.fetchall()
        cursor.close()
    except Exception:
        logger.warning("Aurora unreachable — returning empty protein-structure for %s", gene)
        return {"gene": gene, "structure_ids": [], "aurora_available": False}

    structure_ids = [row[0] for row in rows]
    return {
        "gene": gene,
        "structure_ids": structure_ids,
        "aurora_available": True,
    }


@app.get("/api/protein-scores")
def get_protein_scores(_user: dict = Depends(get_current_user)):
    """Return DTIE druggability scores for all structures with Aurora data.

    Aggregates fact_phase5_pharmacophore.druggability_score per run, keyed by
    gdp_pdb_id. When Aurora is unreachable, returns empty scores without 503.
    """
    try:
        conn = _aurora_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT r.gdp_pdb_id,
                   AVG(p.druggability_score) AS score_mean,
                   MAX(p.druggability_score) AS score_max,
                   COUNT(p.pharmacophore_id)  AS n_pharmacophores
            FROM fact_dtie_run r
            JOIN fact_phase5_pharmacophore p ON p.run_id = r.run_id
            GROUP BY r.gdp_pdb_id
            ORDER BY score_mean DESC
            """,
        )
        rows = cursor.fetchall()
        cursor.close()
    except Exception:
        logger.warning("Aurora unreachable — returning null protein scores")
        return {"scores": {}, "aurora_available": False}

    scores = {
        row[0]: {
            "score_mean": float(row[1]) if row[1] is not None else None,
            "score_max": float(row[2]) if row[2] is not None else None,
            "n_pharmacophores": int(row[3]),
        }
        for row in rows
    }
    return {"scores": scores, "aurora_available": True}


# ── Auth routes (1.5) ─────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def auth_login(body: LoginRequest):
    """Validate credentials and return a JWT.

    JWT payload: {user_id, username, exp} — no session_id.
    session_id is assigned server-side on WebSocket registration.
    """
    db = _load_auth_db()
    username = body.username.strip()
    user = next((u for u in db["users"] if u["username"] == username), None)
    if not user or not bcrypt.checkpw(body.password.encode(), user["password"].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = _issue_jwt(user["id"], user["username"])
    return {"token": token, "user": {"id": user["id"], "username": user["username"]}}


@app.post("/api/auth/signup")
def auth_signup(body: SignupRequest):
    """Create a new user account and return a JWT."""
    db = _load_auth_db()
    username = body.username.strip()
    if not username or not body.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    if any(u["username"] == username for u in db["users"]):
        raise HTTPException(status_code=400, detail="Username already exists")
    new_id = max((u["id"] for u in db["users"]), default=0) + 1
    hashed = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    db["users"].append({"id": new_id, "username": username, "password": hashed})
    db.setdefault("preferences", []).append({
        "user_id": new_id,
        "zeta": 1.0,
        "bloomScale": 1.8,
        "selectedPathways": list(PATHWAY_SEEDS.keys()),
    })
    _save_auth_db(db)
    token = _issue_jwt(new_id, username)
    return {"token": token, "user": {"id": new_id, "username": username}}


@app.post("/api/auth/logout")
def auth_logout():
    """Client should discard its token; no server-side state to clear."""
    return {"success": True}


@app.get("/api/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    """Return the authenticated user's identity from the JWT."""
    return {"id": user["user_id"], "username": user["username"]}


# ── User preferences routes ───────────────────────────────────────────────────

class PreferencesUpdate(BaseModel):
    zeta: float
    bloomScale: float
    selectedPathways: list[str]


@app.get("/api/preferences")
def get_preferences(user: dict = Depends(get_current_user)):
    db = _load_auth_db()
    prefs = next((p for p in db.get("preferences", []) if p["user_id"] == user["user_id"]), None)
    if not prefs:
        raise HTTPException(status_code=404, detail="Preferences not found")
    return {"zeta": prefs["zeta"], "bloomScale": prefs["bloomScale"],
            "selectedPathways": prefs["selectedPathways"]}


@app.post("/api/preferences")
def save_preferences(body: PreferencesUpdate, user: dict = Depends(get_current_user)):
    valid_pathways = [p for p in body.selectedPathways if p in PATHWAY_SEEDS]
    if not valid_pathways:
        raise HTTPException(status_code=400, detail="At least one valid pathway required")
    db = _load_auth_db()
    prefs = db.setdefault("preferences", [])
    idx = next((i for i, p in enumerate(prefs) if p["user_id"] == user["user_id"]), None)
    entry = {"user_id": user["user_id"], "zeta": body.zeta,
             "bloomScale": body.bloomScale, "selectedPathways": sorted(set(valid_pathways))}
    if idx is not None:
        prefs[idx] = entry
    else:
        prefs.append(entry)
    _save_auth_db(db)
    return {"success": True}



def _directive_to_semantic_command(directive: "ViewportDirective") -> dict | None:
    """Convert a ViewportDirective to a semantic_command WS message dict."""
    if directive.action == "highlight" and directive.highlight_groups:
        proteins = [
            rid
            for g in directive.highlight_groups
            for rid in g.residue_ids
        ]
        color = directive.highlight_groups[0].color
        return {
            "type": "semantic_command",
            "viewport_id": "disc-main",
            "command": "highlight_proteins",
            "proteins": proteins,
            "color": color,
        }
    if directive.action == "focus" and directive.focus_residue_id:
        return {
            "type": "semantic_command",
            "viewport_id": "disc-main",
            "command": "camera_animate_to",
            "target": directive.focus_residue_id,
        }
    return None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the Coordinator agent and get a response."""
    # Lazy import to avoid heavy init at module load for health checks
    from data_science.agent import root_agent
    from data_science.session import get_session, set_session

    session_id = request.session_id or str(uuid.uuid4())

    session = _get_session_manager()

    try:
        if hasattr(session, "load"):
            session.load(session_id)
    except Exception:
        logger.warning("Failed to load session %s", session_id, exc_info=True)

    prev_db_settings = get_session().get("database_settings")
    set_session(session)
    if prev_db_settings and session.get("database_settings") is None:
        session.set("database_settings", prev_db_settings)

    agent_message = _build_agent_message(request)

    try:
        result = await asyncio.to_thread(root_agent, agent_message)
        response_text = str(result)
    except Exception as e:
        logger.exception("Error processing chat request")
        response_text = f"Error: {e}"

    try:
        if hasattr(session, "save"):
            session.save(session_id)
    except Exception:
        logger.warning("Failed to save session %s", session_id, exc_info=True)

    directives = _parse_viewport_directives(response_text)
    clean_response = _strip_directive_blocks(response_text) if directives else response_text

    referenced_residues = _extract_residue_references(clean_response)
    if referenced_residues:
        directives.extend(_generate_highlight_directives(referenced_residues))

    # Push directives as semantic_command messages over WS so onCommand fires
    if directives and request.session_id:
        for directive in directives:
            msg = _directive_to_semantic_command(directive)
            if msg:
                await viewport_connection_manager.send_to_session(session_id, msg)

    return ChatResponse(
        response=clean_response,
        session_id=session_id,
        viewport_directives=directives if directives else None,
    )


# ---------------------------------------------------------------------------
# WebSocket Viewport Protocol (Requirements: 1.1, 1.5, 1.6, 11.1, 11.2)
# ---------------------------------------------------------------------------
from data_science.sub_agents.visualization.connection import (
    ConnectionManager,
    authenticate_token,
)
from data_science.sub_agents.visualization.protocol import (
    ViewportRegistration,
    SemanticCommand,
    ViewportEvent as ProtocolViewportEvent,
    check_command_support,
    negotiate_capabilities,
)
from data_science.sub_agents.visualization.session import (
    ViewportState as SessionViewportState,
    create_viewport_session,
    get_viewport_session as get_vs,
)

viewport_connection_manager = ConnectionManager()


@app.websocket("/ws/viewport")
async def websocket_viewport(websocket: WebSocket):
    """WebSocket endpoint for the unified Viewport Protocol.

    Connection lifecycle:
    1. Client connects with token query param for authentication
    2. Server validates JWT and accepts or rejects
    3. Client sends viewport_register messages for each viewer
    4. Bidirectional event/command flow
    5. On disconnect, session state is preserved for reconnection

    Requirements: 1.1, 1.6, 11.1, 11.2
    """
    # Authenticate via query parameter token
    token = websocket.query_params.get("token", "")
    session_id = authenticate_token(token)

    if session_id is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    conn = await viewport_connection_manager.connect(websocket, session_id)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "viewport_register":
                # Handle viewer registration (Requirement 1.2)
                viewport_id = data.get("viewport_id", "")
                viewer_type = data.get("viewer_type", "")
                declared_caps = data.get("capabilities", [])
                initial_state = data.get("initial_state", {})
                structure_ids = data.get("structure_ids", [])

                # Negotiate capabilities (Requirement 1.7)
                negotiated = negotiate_capabilities(viewer_type, declared_caps)
                viewport_connection_manager.register_viewport(conn, viewport_id)

                # Populate (or update) the ViewportSession so the agent's
                # VIEWPORT_CONTEXT reflects current viewer state.
                # Uses dict assignment — safe for re-registration on mode switch.
                vs = create_viewport_session(session_id)
                vs.register_viewport(SessionViewportState(
                    viewport_id=viewport_id,
                    viewer_type=viewer_type,
                    capabilities=negotiated,
                    structure_ids=structure_ids,
                    camera_state=initial_state.get("camera_state"),
                    active_selections=initial_state.get("active_selections", []),
                    applied_coloring=initial_state.get("applied_coloring"),
                ))

                # Acknowledge registration
                await websocket.send_json({
                    "type": "register_ack",
                    "viewport_id": viewport_id,
                    "negotiated_capabilities": negotiated,
                    "session_id": session_id,
                })
                logger.info(
                    "Viewport registered: id=%s type=%s caps=%s",
                    viewport_id,
                    viewer_type,
                    negotiated,
                )

            elif msg_type == "viewport_event":
                # Handle viewport events from frontend (Requirement 1.3)
                viewport_id = data.get("viewport_id", "")
                event_type = data.get("event_type", "")
                timestamp = data.get("timestamp", 0.0)
                payload = data.get("payload", {})

                logger.debug(
                    "Viewport event: viewport=%s type=%s payload=%s",
                    viewport_id,
                    event_type,
                    payload,
                )

                # Update viewport state and buffer the event
                _vs = get_vs(session_id)
                if _vs:
                    _vp = _vs.get_viewport(viewport_id)
                    if _vp:
                        if event_type == "hub_click":
                            hub_id = payload.get("hub_id", "")
                            is_bloomed = payload.get("is_bloomed", False)
                            _vp.bloomed_hub = hub_id if is_bloomed else None
                        elif event_type == "pathway_filter":
                            _vp.selected_pathways = list(payload.get("pathways", []))
                    _vs.add_event(ProtocolViewportEvent(
                        viewport_id=viewport_id,
                        event_type=event_type,
                        timestamp=timestamp,
                        payload=payload,
                    ))

            elif msg_type == "command_response":
                # Handle command acknowledgments from frontend
                logger.debug(
                    "Command response: viewport=%s status=%s",
                    data.get("viewport_id", ""),
                    data.get("status", ""),
                )

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        # Session state is preserved — client can reconnect (Requirement 1.6)
        logger.info(
            "WebSocket disconnected (session preserved): session=%s",
            session_id,
        )
    except Exception:
        logger.exception("WebSocket error for session %s", session_id)
    finally:
        await viewport_connection_manager.disconnect(conn)


# ---------------------------------------------------------------------------
# Lightweight web console for interacting with the agent
# ---------------------------------------------------------------------------
_FRONTEND_DIR = Path(__file__).parent / "frontend"

if _FRONTEND_DIR.exists():
    app.mount(
        "/ui",
        StaticFiles(directory=_FRONTEND_DIR, html=True),
        name="ui",
    )


@app.get("/", response_class=HTMLResponse)
def serve_root():
    """Serve the console if present, otherwise show a simple status message."""
    index_file = _FRONTEND_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return HTMLResponse("Tokyo Eyes Data Science Agent is running.")


if __name__ == "__main__":
    # Use the PORT environment variable provided by ECS/Fargate, defaulting to 8080
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
