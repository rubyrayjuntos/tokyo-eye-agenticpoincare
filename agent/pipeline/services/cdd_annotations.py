"""CDD annotation service stubs."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import requests

from gosp.models.data_models import (
    CddAccession,
    CddAnnotation,
    CddArchitecture,
    CddDomain,
    CddHierarchy,
    CddInterface,
    CddLinker,
    CddResidueConfidence,
    ResidueRange,
    CddRigidBody,
    RigidBodySphere,
)
from gosp.services.structure_ingestion import fetch_structure_from_rcsb

logger = logging.getLogger(__name__)

_CDD_CACHE: Dict[Tuple[str, str], Tuple[float, CddAnnotation]] = {}
_CACHE_TTL_SECONDS = 3600
_CACHE_DIR = Path(os.getenv("GOSP_CDD_CACHE_DIR", "/tmp/gosp_cdd"))


async def fetch_cdd_annotations(pdb_id: str, chain_id: str) -> CddAnnotation:
    """Fetch and normalize CDD annotations for a PDB chain."""
    cached = get_cached_cdd_annotations(pdb_id, chain_id)
    if cached:
        return cached

    structure = await asyncio.to_thread(
        fetch_structure_from_rcsb,
        pdb_id,
        "cif",
        chain_id,
    )
    chain_sequence = ""
    for chain in structure.chains:
        if chain.chain_id == chain_id:
            chain_sequence = chain.sequence
            break
    if not chain_sequence:
        raise ValueError(f"Chain '{chain_id}' not found in structure {pdb_id}")

    annotation = CddAnnotation(
        pdb_id=pdb_id,
        chain_id=chain_id,
        source="NCBI_CDD",
        fetch_date=time.strftime("%Y-%m-%d"),
        cdd_version="unknown",
        rpsblast_version=None,
        full_sequence=chain_sequence,
        architecture=None,
        domains=[],
        linkers=[],
        interfaces=[],
        rigid_bodies=[],
    )

    cd_search_url = os.getenv(
        "GOSP_CDD_API_URL",
        "https://www.ncbi.nlm.nih.gov/Structure/bwrpsb/bwrpsb.cgi",
    )
    raw_response = await asyncio.to_thread(
        _attempt_cd_search_fetch,
        cd_search_url,
        pdb_id,
        chain_id,
        chain_sequence,
    )
    if raw_response:
        annotation = _parse_cd_search_response(raw_response, annotation)

    if annotation.domains:
        annotation.rigid_bodies = _build_rigid_bodies(structure, annotation.domains, chain_id)

    _store_cache(pdb_id, chain_id, annotation)
    _store_cache_to_disk(pdb_id, chain_id, annotation, raw_response)

    return annotation


def get_cached_cdd_annotations(pdb_id: str, chain_id: str) -> Optional[CddAnnotation]:
    """Return cached CDD annotations if available."""
    cache_key = (pdb_id.upper(), chain_id)
    cached = _CDD_CACHE.get(cache_key)
    if not cached:
        disk_cached = _load_cache_from_disk(pdb_id, chain_id)
        if disk_cached:
            _CDD_CACHE[cache_key] = (time.time(), disk_cached)
        return disk_cached
    cached_at, annotation = cached
    if time.time() - cached_at > _CACHE_TTL_SECONDS:
        _CDD_CACHE.pop(cache_key, None)
        return None
    return annotation


def _store_cache(pdb_id: str, chain_id: str, annotation: CddAnnotation) -> None:
    cache_key = (pdb_id.upper(), chain_id)
    _CDD_CACHE[cache_key] = (time.time(), annotation)


def _cache_paths(pdb_id: str, chain_id: str) -> Tuple[Path, Path]:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{pdb_id.upper()}_{chain_id}"
    return (
        _CACHE_DIR / f"{safe_name}.annotation.json",
        _CACHE_DIR / f"{safe_name}.raw.json",
    )


def _store_cache_to_disk(
    pdb_id: str,
    chain_id: str,
    annotation: CddAnnotation,
    raw_response: Optional[dict],
) -> None:
    annotation_path, raw_path = _cache_paths(pdb_id, chain_id)
    annotation_path.write_text(annotation.model_dump_json(indent=2), encoding="utf-8")
    if raw_response is not None:
        raw_path.write_text(json.dumps(raw_response, indent=2), encoding="utf-8")


def _load_cache_from_disk(pdb_id: str, chain_id: str) -> Optional[CddAnnotation]:
    annotation_path, _ = _cache_paths(pdb_id, chain_id)
    if not annotation_path.exists():
        return None
    try:
        return CddAnnotation.model_validate_json(annotation_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load cached CDD annotation: %s", exc)
        return None


def _attempt_cd_search_fetch(
    cd_search_url: str,
    pdb_id: str,
    chain_id: str,
    sequence: str,
) -> Optional[dict]:
    """Attempt a CD-Search fetch and return parsed JSON when possible."""
    fasta = f">{pdb_id.upper()}_{chain_id}\n{sequence}\n"
    payload = {
        "queries": fasta,
        "db": "cdd",
        "evalue": "0.01",
        "output": "JSON",
    }
    try:
        response = requests.post(cd_search_url, data=payload, timeout=20)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            return json.loads(response.text)
        return json.loads(response.text)
    except Exception as exc:
        logger.warning("CDD CD-Search fetch failed: %s", exc)
        return None


def _parse_cd_search_response(response: dict, annotation: CddAnnotation) -> CddAnnotation:
    """Best-effort parsing of CD-Search JSON into domain annotations."""
    annotation.cdd_version = response.get("cdd_version", annotation.cdd_version)
    hits = _extract_hits(response)
    domains: list[CddDomain] = []
    for hit in hits:
        cdd_id = hit.get("accession") or hit.get("acc") or hit.get("id")
        name = hit.get("title") or hit.get("name") or cdd_id
        start = hit.get("from") or hit.get("start") or hit.get("pos_from")
        end = hit.get("to") or hit.get("end") or hit.get("pos_to")
        if not cdd_id or not name or not start or not end:
            continue
        try:
            range_obj = ResidueRange(start=int(start), end=int(end))
        except Exception:
            continue

        accession = CddAccession(
            id=str(cdd_id),
            url=f"https://www.ncbi.nlm.nih.gov/Structure/cdd/cddsrv.cgi?uid={cdd_id}",
        )

        domain = CddDomain(
            cdd_id=str(cdd_id),
            name=str(name),
            range=range_obj,
            evalue=_safe_float(hit.get("evalue") or hit.get("expect")),
            bit_score=_safe_float(hit.get("bitscore") or hit.get("bit_score")),
            confidence=hit.get("confidence"),
            accession=accession,
            hierarchy=_parse_hierarchy(hit),
            residue_confidence=_parse_residue_confidence(hit),
        )
        domains.append(domain)

    if domains:
        annotation.domains = domains
        annotation.architecture = _build_architecture(domains)

    linkers = _extract_linkers(response)
    if linkers:
        annotation.linkers = linkers

    interfaces = _extract_interfaces(response)
    if interfaces:
        annotation.interfaces = interfaces
    return annotation


def _extract_hits(response: dict) -> Iterable[dict]:
    """Extract hit lists from common CD-Search JSON shapes."""
    if "hits" in response and isinstance(response["hits"], list):
        return response["hits"]
    results = response.get("results")
    if isinstance(results, list):
        hits: list[dict] = []
        for result in results:
            if isinstance(result, dict) and isinstance(result.get("hits"), list):
                hits.extend(result["hits"])
        return hits
    return []


def _parse_hierarchy(hit: dict) -> Optional[CddHierarchy]:
    hierarchy = hit.get("hierarchy")
    if not isinstance(hierarchy, dict):
        return None
    root = hierarchy.get("root") or hierarchy.get("root_id")
    if not root:
        return None
    return CddHierarchy(
        root=str(root),
        superfamily=hierarchy.get("superfamily"),
        family=hierarchy.get("family"),
        lineage=hierarchy.get("lineage") or [],
    )


def _parse_residue_confidence(hit: dict) -> list[CddResidueConfidence]:
    residue_scores = hit.get("residue_confidence")
    if not isinstance(residue_scores, list):
        return []
    parsed: list[CddResidueConfidence] = []
    for entry in residue_scores:
        if not isinstance(entry, dict):
            continue
        pos = entry.get("pos")
        score = entry.get("score")
        if pos is None or score is None:
            continue
        try:
            parsed.append(CddResidueConfidence(pos=int(pos), score=float(score)))
        except Exception:
            continue
    return parsed


def _build_architecture(domains: list[CddDomain]) -> CddArchitecture:
    ordered = sorted(domains, key=lambda d: d.range.start)
    return CddArchitecture(
        arch_id="|".join([d.cdd_id for d in ordered]),
        order=[d.cdd_id for d in ordered],
    )


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_linkers(response: dict) -> list[CddLinker]:
    raw_linkers = response.get("linkers")
    if not isinstance(raw_linkers, list):
        return []
    linkers: list[CddLinker] = []
    for entry in raw_linkers:
        if not isinstance(entry, dict):
            continue
        start = entry.get("start") or entry.get("from")
        end = entry.get("end") or entry.get("to")
        if start is None or end is None:
            continue
        linkers.append(
            CddLinker(
                id=str(entry.get("id") or f"linker_{start}_{end}"),
                from_domain=entry.get("from_domain"),
                to_domain=entry.get("to_domain"),
                range=ResidueRange(start=int(start), end=int(end)),
                sequence=entry.get("sequence"),
                type=entry.get("type"),
                hinge_prior=entry.get("hinge_prior"),
            )
        )
    return linkers


def _extract_interfaces(response: dict) -> list[CddInterface]:
    raw_interfaces = response.get("interfaces")
    if not isinstance(raw_interfaces, list):
        return []
    interfaces: list[CddInterface] = []
    for entry in raw_interfaces:
        if not isinstance(entry, dict):
            continue
        domains = entry.get("domains")
        if not isinstance(domains, list) or len(domains) < 2:
            continue
        interfaces.append(
            CddInterface(
                id=str(entry.get("id") or f"interface_{domains[0]}_{domains[1]}"),
                domains=[str(d) for d in domains],
                residue_pairs=entry.get("residue_pairs") or [],
                distance_average_A=_safe_float(entry.get("distance_average_A")),
                interaction_types=entry.get("interaction_types") or [],
                dehydron_count=entry.get("dehydron_count"),
                glue_priority=entry.get("glue_priority"),
                ui=entry.get("ui"),
            )
        )
    return interfaces


def _build_rigid_bodies(structure, domains: list[CddDomain], chain_id: str) -> list[CddRigidBody]:
    rigid_bodies: list[CddRigidBody] = []
    chain = next((c for c in structure.chains if c.chain_id == chain_id), None)
    if chain is None:
        return rigid_bodies

    for domain in domains:
        atoms = []
        for residue in chain.residues:
            if domain.range.start <= residue.residue_id <= domain.range.end:
                atoms.extend(residue.atoms)
        if not atoms:
            continue

        coords = [(atom.x, atom.y, atom.z) for atom in atoms]
        center = [
            sum(p[i] for p in coords) / len(coords)
            for i in range(3)
        ]
        radius = max(
            ((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2 + (p[2] - center[2]) ** 2) ** 0.5
            for p in coords
        )

        rigid_bodies.append(
            CddRigidBody(
                id=f"RB_{domain.cdd_id}",
                domains=[domain.cdd_id],
                residue_range=[domain.range.start, domain.range.end],
                sphere=RigidBodySphere(center=center, radius=radius),
                constraint=None,
                ui=domain.ui,
            )
        )

    return rigid_bodies
