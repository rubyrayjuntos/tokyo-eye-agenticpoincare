"""Free literature-research tools grounded in public APIs."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlencode, quote_plus
from urllib.request import urlopen

from agent.tools.dtie.tools import ToolResult


def _fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


async def search_literature(
    query: str,
    max_results: int = 5,
    open_access_only: bool = False,
) -> ToolResult:
    """Search Europe PMC for academic literature relevant to a query."""
    effective_query = query.strip()
    if not effective_query:
        return ToolResult(success=False, message="Query cannot be empty")

    if open_access_only:
        effective_query = f"({effective_query}) AND OPEN_ACCESS:y"

    params = urlencode(
        {
            "query": effective_query,
            "format": "json",
            "resultType": "core",
            "pageSize": max(1, min(max_results, 25)),
        },
        quote_via=quote_plus,
    )
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}"

    try:
        payload = await asyncio.to_thread(_fetch_json, url)
    except Exception as exc:
        return ToolResult(success=False, message=f"Literature search failed: {exc}")

    raw_results = payload.get("resultList", {}).get("result", [])
    papers: list[dict[str, Any]] = []
    for item in raw_results:
        papers.append(
            {
                "title": item.get("title"),
                "authors": item.get("authorString"),
                "journal": item.get("journalTitle"),
                "year": item.get("pubYear"),
                "doi": item.get("doi"),
                "pmid": item.get("pmid"),
                "pmcid": item.get("pmcid"),
                "source": item.get("source"),
                "is_open_access": item.get("isOpenAccess") == "Y",
                "abstract": item.get("abstractText"),
                "url": item.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url")
                if item.get("fullTextUrlList")
                else None,
            }
        )

    return ToolResult(
        success=True,
        data={
            "query": query,
            "source": "Europe PMC",
            "open_access_only": open_access_only,
            "count": len(papers),
            "papers": papers,
        },
        message=f"Found {len(papers)} literature matches for '{query}'",
    )
