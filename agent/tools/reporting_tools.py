"""Grounded manuscript/report drafting tools."""

from __future__ import annotations

from typing import Any

from agent.tools.dtie.tools import ToolDB, ToolResult
from agent.tools.literature_tools import search_literature


def _fmt_float(value: Any) -> str:
    return f"{float(value):.3f}" if value is not None else "n/a"


async def draft_paper_section(
    structure_id: str,
    section: str = "results",
    run_id: str | None = None,
    literature_query: str | None = None,
    db: Any = None,
) -> ToolResult:
    """Draft a grounded paper/report section from governed DB facts."""
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)
    structure_row = await tool_db.fetch_one(
        """
        SELECT COUNT(*) AS residue_count,
               AVG(cone_depth) AS avg_cone_depth,
               AVG(epistemic_uncertainty) AS avg_epistemic_uncertainty,
               MAX(computed_at) AS last_embedding_at
        FROM fact_gnn_node_embedding
        WHERE structure_id = :structure_id
          AND ((:run_id)::text IS NULL OR run_id = :run_id)
        """,
        {"structure_id": structure_id, "run_id": run_id},
    )
    leak_row = await tool_db.fetch_one(
        """
        SELECT COUNT(*) AS leak_count,
               MAX(leak_score) AS max_leak_score,
               AVG(leak_score) AS avg_leak_score
        FROM fact_source_leak
        WHERE structure_id = :structure_id
        """,
        {"structure_id": structure_id},
    )
    pocket_row = await tool_db.fetch_one(
        """
        SELECT pocket_index, druggability_score, residue_count, allosteric_coupling
        FROM fact_pharmacophore
        WHERE structure_id = :structure_id
        ORDER BY druggability_score DESC NULLS LAST
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    candidate_row = await tool_db.fetch_one(
        """
        SELECT pocket_index, combined_druggability, accessibility_score,
               binding_potential, selectivity_ratio, admet_pass
        FROM fact_drug_candidate
        WHERE structure_id = :structure_id
        ORDER BY combined_druggability DESC NULLS LAST
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )

    if not structure_row or not structure_row.get("residue_count"):
        return ToolResult(success=False, message=f"No governed structure data found for {structure_id}")

    grounded_facts = {
        "structure_id": structure_id,
        "run_id": run_id,
        "residue_count": structure_row.get("residue_count"),
        "avg_cone_depth": structure_row.get("avg_cone_depth"),
        "avg_epistemic_uncertainty": structure_row.get("avg_epistemic_uncertainty"),
        "leak_count": leak_row.get("leak_count") if leak_row else 0,
        "max_leak_score": leak_row.get("max_leak_score") if leak_row else None,
        "top_pocket": dict(pocket_row) if pocket_row else None,
        "top_candidate": dict(candidate_row) if candidate_row else None,
    }

    literature = None
    warnings: list[str] = []
    if literature_query:
        literature_result = await search_literature(query=literature_query, max_results=3)
        if literature_result.success:
            literature = literature_result.data
        else:
            warnings.append(literature_result.message)

    section_key = section.lower().strip()
    top_pocket_text = (
        f"Pocket {pocket_row['pocket_index']} was the highest-scoring pharmacophore site "
        f"(druggability={_fmt_float(pocket_row.get('druggability_score'))}, residues={pocket_row['residue_count']}, "
        f"allosteric_coupling={_fmt_float(pocket_row.get('allosteric_coupling'))})."
        if pocket_row and pocket_row.get("druggability_score") is not None
        else "No persisted pharmacophore pocket was available for this section."
    )
    top_candidate_text = (
        f"The top candidate mapped to pocket {candidate_row['pocket_index']} with "
        f"combined_druggability={_fmt_float(candidate_row.get('combined_druggability'))}, "
        f"binding_potential={_fmt_float(candidate_row.get('binding_potential'))}, "
        f"accessibility_score={_fmt_float(candidate_row.get('accessibility_score'))}, "
        f"selectivity_ratio={_fmt_float(candidate_row.get('selectivity_ratio'))}, "
        f"admet_pass={candidate_row['admet_pass']}."
        if candidate_row and candidate_row.get("combined_druggability") is not None
        else "No persisted Phase 6 candidate ranking was available."
    )

    templates = {
        "abstract": (
            f"We analyzed {structure_id} with the Tokyo Eye DTIE pipeline, covering "
            f"{structure_row['residue_count']} residues. The structure showed {leak_row.get('leak_count', 0) if leak_row else 0} "
            f"source-leak candidates with a maximum leak score of "
            f"{(leak_row.get('max_leak_score') if leak_row else None)!s}. {top_pocket_text} {top_candidate_text}"
        ),
        "results": (
            f"Results for {structure_id} show a mean cone depth of "
            f"{(structure_row.get('avg_cone_depth') or 0):.3f} and mean epistemic uncertainty of "
            f"{(structure_row.get('avg_epistemic_uncertainty') or 0):.3f}. "
            f"{top_pocket_text} {top_candidate_text}"
        ),
        "discussion": (
            f"The governed DTIE outputs for {structure_id} indicate that the strongest intervention surface is "
            f"associated with {top_pocket_text.lower()} {top_candidate_text} "
            f"These findings should be interpreted as hypothesis-generating until externally validated."
        ),
        "methods": (
            f"{structure_id} was evaluated with the Tokyo Eye DTIE pipeline using governed provenance-tracked "
            f"hyperbolic embeddings, graph metrics, source-leak analysis, pharmacophore extraction, and "
            f"drug-candidate ranking. This section is grounded in persisted database facts"
            f"{' for run ' + run_id if run_id else ''}."
        ),
        "figure_legend": (
            f"Figure: DTIE-derived overview for {structure_id}. Bars/points summarize "
            f"{structure_row['residue_count']} residues, highlight "
            f"{leak_row.get('leak_count', 0) if leak_row else 0} source-leak candidates, and annotate the "
            f"top pharmacophore pocket and top-ranked candidate."
        ),
    }
    draft_text = templates.get(section_key, templates["results"])

    if literature and literature.get("papers"):
        citations = []
        for paper in literature["papers"][:3]:
            title = paper.get("title") or "Untitled"
            year = paper.get("year") or "n.d."
            journal = paper.get("journal") or "Unknown journal"
            citations.append(f"{title} ({journal}, {year})")
        draft_text += " Related literature surfaced by the system includes: " + "; ".join(citations) + "."

    return ToolResult(
        success=True,
        data={
            "structure_id": structure_id,
            "run_id": run_id,
            "section": section_key,
            "grounded_facts": grounded_facts,
            "literature": literature,
            "draft_text": draft_text,
        },
        message=f"Drafted {section_key} section for {structure_id}",
        warnings=warnings,
    )
