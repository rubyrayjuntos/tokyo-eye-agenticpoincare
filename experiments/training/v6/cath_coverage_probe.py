#!/usr/bin/env python3
"""Read-only CATH coverage probe for corpus manifest entries (Stage A de-risk).

Fetches PDBe SIFTS CATH mappings, caches responses locally, and reports
topology vs architecture vs unverified coverage before building redundancy matrix.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST = _REPO / "manifests" / "v6_corpus_120.json"
_CACHE_DIR = _REPO / "manifests" / "cache" / "cath_pdbe"
_PDEB_API = "https://www.ebi.ac.uk/pdbe/api/mappings/cath/{pdb_id}"


def _enabled_entries(manifest_path: Path) -> list[dict[str, Any]]:
    data = json.loads(manifest_path.read_text())
    return [e for e in data["proteins"] if e.get("enabled", True)]


def _fetch_cath(pdb_id: str, *, cache_dir: Path, refresh: bool = False) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{pdb_id.lower()}.json"
    if cache_file.is_file() and not refresh:
        return json.loads(cache_file.read_text())

    url = _PDEB_API.format(pdb_id=pdb_id.lower())
    req = urllib.request.Request(url, headers={"User-Agent": "tokyo-eye-cath-probe/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        payload = {"error": f"HTTP {exc.code}", "pdb_id": pdb_id}
    except urllib.error.URLError as exc:
        payload = {"error": str(exc.reason), "pdb_id": pdb_id}

    cache_file.write_text(json.dumps(payload, indent=2))
    time.sleep(0.15)  # gentle rate limit
    return payload


def _chain_mappings(payload: dict[str, Any], pdb_id: str, chain: str) -> list[dict[str, Any]]:
    if "error" in payload:
        return []
    pdb_key = pdb_id.lower()
    block = payload.get(pdb_key, {}).get("CATH", {})
    if not isinstance(block, dict):
        return []
    chain_u = chain.upper()
    hits: list[dict[str, Any]] = []
    for cath_code, info in block.items():
        if not isinstance(info, dict):
            continue
        for m in info.get("mappings", []) or []:
            m_chain = str(m.get("chain_id", "")).upper()
            m_asym = str(m.get("struct_asym_id", "")).upper()
            if m_chain == chain_u or m_asym == chain_u:
                hits.append(
                    {
                        "cath_code": cath_code,
                        "parts": cath_code.split("."),
                        "domain": m.get("domain"),
                        "start": m.get("start", {}).get("author_residue_number"),
                        "end": m.get("end", {}).get("author_residue_number"),
                        "topology_name": info.get("topology"),
                        "architecture_name": info.get("architecture"),
                        "class_name": info.get("class"),
                    }
                )
    return hits


def _resolve_fold_id(hits: list[dict[str, Any]]) -> dict[str, Any]:
    if not hits:
        return {
            "fold_id": None,
            "fold_id_tier": "unverified",
            "status": "FOLD_UNVERIFIED",
        }
    # Prefer longest span on chain; topology code has 4 dot parts (C.A.T.S)
    best = max(
        hits,
        key=lambda h: (int(h.get("end") or 0) - int(h.get("start") or 0), len(h.get("parts", []))),
    )
    parts = best["parts"]
    if len(parts) >= 4:
        tier = "topology"
        fold_id = best["cath_code"]
        status = "OK_TOPOLOGY"
    elif len(parts) == 3:
        tier = "architecture"
        fold_id = best["cath_code"]
        status = "OK_ARCHITECTURE_ONLY"
    else:
        tier = "unverified"
        fold_id = best["cath_code"]
        status = "FOLD_PARTIAL"
    return {
        "fold_id": fold_id,
        "fold_id_tier": tier,
        "status": status,
        "domain": best.get("domain"),
        "topology_name": best.get("topology_name"),
        "architecture_name": best.get("architecture_name"),
        "all_hits": hits,
    }


def author_chain_from_cache(
    pdb_id: str,
    manifest_chain: str,
    *,
    cache_dir: Path = _CACHE_DIR,
) -> str | None:
    """Map manifest chain (often struct_asym_id) to author chain_id in the PDB file."""
    cache_file = cache_dir / f"{pdb_id.lower()}.json"
    if not cache_file.is_file():
        return None
    payload = json.loads(cache_file.read_text())
    block = payload.get(pdb_id.lower(), {}).get("CATH", {})
    if not isinstance(block, dict):
        return None
    chain_u = manifest_chain.upper()
    for info in block.values():
        if not isinstance(info, dict):
            continue
        for m in info.get("mappings", []) or []:
            m_chain = str(m.get("chain_id", "")).upper()
            m_asym = str(m.get("struct_asym_id", "")).upper()
            if m_asym == chain_u or m_chain == chain_u:
                return str(m.get("chain_id", ""))
    return None


def resolve_fold_from_cache(
    pdb_id: str,
    chain: str,
    *,
    cache_dir: Path = _CACHE_DIR,
) -> dict[str, Any]:
    """Resolve fold_id from frozen PDBe cache (no network)."""
    cache_file = cache_dir / f"{pdb_id.lower()}.json"
    if not cache_file.is_file():
        return {
            "fold_id": None,
            "fold_id_tier": "unverified",
            "status": "FOLD_UNVERIFIED",
            "fetch_error": "cache missing",
        }
    payload = json.loads(cache_file.read_text())
    resolved = _resolve_fold_id(_chain_mappings(payload, pdb_id, chain))
    resolved["fetch_error"] = payload.get("error")
    return resolved


def probe_manifest(
    manifest_path: Path,
    *,
    cache_dir: Path,
    refresh: bool = False,
) -> dict[str, Any]:
    entries = _enabled_entries(manifest_path)
    structures: list[dict[str, Any]] = []
    for entry in entries:
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry.get("chain", "A"))
        payload = _fetch_cath(pdb_id, cache_dir=cache_dir, refresh=refresh)
        hits = _chain_mappings(payload, pdb_id, chain)
        resolved = _resolve_fold_id(hits)
        structures.append(
            {
                "pdb_id": pdb_id,
                "chain": chain,
                "gene": entry.get("gene"),
                "manifest_fold_class": entry.get("fold_class"),
                **resolved,
                "fetch_error": payload.get("error"),
            }
        )

    n = len(structures)
    topo = sum(1 for s in structures if s["fold_id_tier"] == "topology")
    arch = sum(1 for s in structures if s["fold_id_tier"] == "architecture")
    unverified = sum(1 for s in structures if s["fold_id_tier"] == "unverified")
    mixed_tier = topo > 0 and arch > 0

    return {
        "spec_version": "cath-coverage-probe:2026-07-03",
        "input_manifest": str(manifest_path.relative_to(_REPO)),
        "cache_dir": str(cache_dir.relative_to(_REPO)),
        "api": _PDEB_API,
        "enabled_count": n,
        "coverage": {
            "topology": topo,
            "architecture_only": arch,
            "unverified": unverified,
            "topology_pct": round(100.0 * topo / n, 1) if n else 0.0,
            "any_cath_pct": round(100.0 * (topo + arch) / n, 1) if n else 0.0,
            "mixed_tier_vocabulary": mixed_tier,
        },
        "structures": structures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CATH coverage probe for corpus manifest")
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=_REPO / "manifests" / "cache" / "cath_coverage_report.json")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    report = probe_manifest(args.manifest, cache_dir=args.cache_dir, refresh=args.refresh)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    cov = report["coverage"]
    print(f"Enabled structures: {report['enabled_count']}")
    print(f"Topology fold_id:   {cov['topology']} ({cov['topology_pct']}%)")
    print(f"Architecture only:  {cov['architecture_only']}")
    print(f"Unverified:         {cov['unverified']}")
    print(f"Any CATH:           {cov['any_cath_pct']}%")
    print(f"Mixed tier vocab:   {cov['mixed_tier_vocabulary']}")
    print(f"Report: {args.out}")
    if cov["unverified"]:
        print("\nUnverified entries:")
        for s in report["structures"]:
            if s["fold_id_tier"] == "unverified":
                print(f"  {s['pdb_id']}:{s['chain']} gene={s.get('gene')} err={s.get('fetch_error')}")


if __name__ == "__main__":
    main()
