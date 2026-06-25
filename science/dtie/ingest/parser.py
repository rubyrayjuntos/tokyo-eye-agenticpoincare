"""BinaryCIF parser — biotite-based extraction into structured dataclasses.

Extracts all chains, residues (including unresolved), atoms, modified residues,
covalent bonds, and quality flags from a BinaryCIF file.

Requirements: 1.2, 1.9, 1.10, 1.11, 8.1, 8.3, 8.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Backbone atoms required to be "complete"
_BACKBONE_ATOMS = frozenset({"N", "CA", "C"})

# Standard amino acid 3-letter to 1-letter mapping
_AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common modified residues → map to parent 1-letter
    "MSE": "M", "SEP": "S", "TPO": "T", "CSO": "C", "PTR": "Y",
    "HYP": "P", "MLY": "K", "CSD": "C", "OCS": "C",
}


# ---------------------------------------------------------------------------
# Dataclasses (design spec §2)
# ---------------------------------------------------------------------------


@dataclass
class ParsedAtom:
    """A single atom from the structure."""

    atom_name: str
    element: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    altloc: str | None
    is_hetero: bool
    model_id: int


@dataclass
class ParsedResidue:
    """A single residue (resolved or unresolved)."""

    auth_seq_id: int
    label_seq_id: int | None
    insertion_code: str | None
    residue_name: str       # 1-letter code
    residue_name_3: str     # 3-letter code
    comp_id: str            # mmCIF comp_id
    parent_comp_id: str | None  # for modified residues (MSE→MET, etc.)
    sse_code: str | None
    is_resolved: bool
    is_modified: bool
    partial_backbone: bool
    max_b_factor: float | None
    low_confidence_coords: bool
    atoms: list[ParsedAtom] = field(default_factory=list)


@dataclass
class ParsedChain:
    """A single polymer chain."""

    auth_asym_id: str
    label_asym_id: str
    entity_id: str
    entity_type: str  # protein, nucleic_acid, etc.
    residues: list[ParsedResidue] = field(default_factory=list)


@dataclass
class CovalentBond:
    """A covalent bond from _struct_conn."""

    chain_1: str
    res_seq_1: int
    ins_code_1: str | None
    atom_1: str
    chain_2: str
    res_seq_2: int
    ins_code_2: str | None
    atom_2: str
    bond_type: str  # disulf, covale, etc.


@dataclass
class ParsedStructure:
    """Complete parsed representation of a BinaryCIF structure."""

    pdb_id: str
    method: str
    resolution: float | None
    r_factor: float | None
    r_free: float | None
    title: str
    organism: str | None
    release_date: str | None
    polymer_composition: str  # protein, protein/na, etc.
    model_count: int
    assembly_id: str | None
    chains: list[ParsedChain] = field(default_factory=list)
    covalent_bonds: list[CovalentBond] = field(default_factory=list)
    modified_residues: dict[str, str] = field(default_factory=dict)  # comp_id → parent_comp_id



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_bcif(file_path: Path) -> ParsedStructure:
    """Parse BinaryCIF using biotite into structured dataclasses.

    Extracts:
    - All chains (from _atom_site + _pdbx_poly_seq_scheme)
    - All residues including unresolved (is_resolved=False from seq scheme)
    - All atoms with altlocs preserved
    - Modified residues from _pdbx_struct_mod_residue
    - Covalent bonds from _struct_conn (disulf, covale)
    - Quality flags: partial_backbone, max_b_factor, low_confidence_coords

    For NMR structures, defaults to model 1 and stores model_count.

    Args:
        file_path: Path to decompressed .cif file.

    Returns:
        ParsedStructure with all extracted data.
    """
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    # Read the mmCIF file (text format from files.rcsb.org)
    cif_file = pdbx.CIFFile.read(str(file_path))
    block = cif_file.block

    # Extract entry-level metadata
    pdb_id = _extract_pdb_id(block)
    method = _extract_method(block)
    resolution = _extract_resolution(block)
    r_factor, r_free = _extract_r_values(block)
    title = _extract_title(block)
    release_date = _extract_release_date(block)

    # Determine model count (NMR structures have multiple models)
    model_count = _extract_model_count(block)

    # Parse atom_site — use model 1 (default for NMR)
    # Use altloc="all" to preserve all alternate locations (we handle filtering ourselves)
    try:
        atom_array = pdbx.get_structure(cif_file, model=1, altloc="all")
    except Exception:
        # Fallback for altloc encoding issues
        atom_array = pdbx.get_structure(cif_file, model=1, altloc="first")

    # Parse modified residues from _pdbx_struct_mod_residue
    modified_residues = _extract_modified_residues(block)

    # Parse covalent bonds from _struct_conn
    covalent_bonds = _extract_covalent_bonds(block)

    # Build chains with residues and atoms
    chains = _build_chains(block, atom_array, modified_residues)

    # Determine polymer composition
    polymer_composition = _determine_polymer_composition(chains)

    return ParsedStructure(
        pdb_id=pdb_id,
        method=method,
        resolution=resolution,
        r_factor=r_factor,
        r_free=r_free,
        title=title,
        organism=None,  # Populated by metadata enricher
        release_date=release_date,
        polymer_composition=polymer_composition,
        model_count=model_count,
        assembly_id=None,  # Populated by metadata enricher
        chains=chains,
        covalent_bonds=covalent_bonds,
        modified_residues=modified_residues,
    )


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------


def _extract_pdb_id(block) -> str:
    """Extract PDB ID from _entry.id."""
    try:
        entry = block["entry"]
        return str(entry["id"].as_array()[0]).strip().lower()
    except (KeyError, IndexError):
        return "unknown"


def _extract_method(block) -> str:
    """Extract experimental method from _exptl.method."""
    try:
        exptl = block["exptl"]
        return str(exptl["method"].as_array()[0]).strip()
    except (KeyError, IndexError):
        return "UNKNOWN"


def _extract_resolution(block) -> float | None:
    """Extract resolution from _refine.ls_d_res_high or _reflns.d_resolution_high."""
    try:
        refine = block["refine"]
        val = refine["ls_d_res_high"].as_array()[0]
        return float(val) if val is not None else None
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    try:
        reflns = block["reflns"]
        val = reflns["d_resolution_high"].as_array()[0]
        return float(val) if val is not None else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _extract_r_values(block) -> tuple[float | None, float | None]:
    """Extract R-factor and R-free from _refine."""
    r_factor = None
    r_free = None
    try:
        refine = block["refine"]
        r_val = refine["ls_R_factor_R_work"].as_array()[0]
        r_factor = float(r_val) if r_val is not None else None
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    try:
        refine = block["refine"]
        rf_val = refine["ls_R_factor_R_free"].as_array()[0]
        r_free = float(rf_val) if rf_val is not None else None
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return r_factor, r_free


def _extract_title(block) -> str:
    """Extract title from _struct.title."""
    try:
        struct = block["struct"]
        return str(struct["title"].as_array()[0]).strip()
    except (KeyError, IndexError):
        return ""


def _extract_release_date(block) -> str | None:
    """Extract release date from _pdbx_database_status.recvd_initial_deposition_date."""
    try:
        status = block["pdbx_database_status"]
        val = status["recvd_initial_deposition_date"].as_array()[0]
        return str(val).strip() if val else None
    except (KeyError, IndexError):
        return None


def _extract_model_count(block) -> int:
    """Count number of models from _atom_site.pdbx_PDB_model_num."""
    try:
        atom_site = block["atom_site"]
        model_nums = atom_site["pdbx_PDB_model_num"].as_array()
        unique_models = set(model_nums)
        return len(unique_models)
    except (KeyError, IndexError):
        return 1


def _extract_modified_residues(block) -> dict[str, str]:
    """Extract modified residue mappings from _pdbx_struct_mod_residue.

    Returns dict: comp_id → parent_comp_id (e.g., MSE → MET)
    """
    mapping: dict[str, str] = {}
    try:
        mod_res = block["pdbx_struct_mod_residue"]
        labels = mod_res["label_comp_id"].as_array()
        parents = mod_res["parent_comp_id"].as_array()
        for label, parent in zip(labels, parents):
            label_str = str(label).strip()
            parent_str = str(parent).strip()
            if label_str and parent_str and parent_str != "?":
                mapping[label_str] = parent_str
    except (KeyError, IndexError):
        pass
    return mapping


def _extract_covalent_bonds(block) -> list[CovalentBond]:
    """Extract covalent bonds from _struct_conn (disulf, covale types)."""
    bonds: list[CovalentBond] = []
    try:
        conn = block["struct_conn"]
        conn_types = conn["conn_type_id"].as_array()

        # Partner 1
        chain_1_arr = conn["ptnr1_auth_asym_id"].as_array()
        seq_1_arr = conn["ptnr1_auth_seq_id"].as_array()
        atom_1_arr = conn["ptnr1_label_atom_id"].as_array()

        # Partner 2
        chain_2_arr = conn["ptnr2_auth_asym_id"].as_array()
        seq_2_arr = conn["ptnr2_auth_seq_id"].as_array()
        atom_2_arr = conn["ptnr2_label_atom_id"].as_array()

        # Insertion codes (optional)
        ins_1_arr = _safe_array(conn, "pdbx_ptnr1_PDB_ins_code")
        ins_2_arr = _safe_array(conn, "pdbx_ptnr2_PDB_ins_code")

        for i, conn_type in enumerate(conn_types):
            conn_type_str = str(conn_type).strip().lower()
            if conn_type_str not in ("disulf", "covale"):
                continue

            ins_1 = _clean_ins_code(ins_1_arr[i] if ins_1_arr is not None else None)
            ins_2 = _clean_ins_code(ins_2_arr[i] if ins_2_arr is not None else None)

            bonds.append(CovalentBond(
                chain_1=str(chain_1_arr[i]).strip(),
                res_seq_1=int(seq_1_arr[i]),
                ins_code_1=ins_1,
                atom_1=str(atom_1_arr[i]).strip(),
                chain_2=str(chain_2_arr[i]).strip(),
                res_seq_2=int(seq_2_arr[i]),
                ins_code_2=ins_2,
                atom_2=str(atom_2_arr[i]).strip(),
                bond_type=conn_type_str,
            ))
    except (KeyError, IndexError) as e:
        logger.debug("No struct_conn data or parse error: %s", e)
    return bonds



def _build_chains(
    block,
    atom_array,
    modified_residues: dict[str, str],
) -> list[ParsedChain]:
    """Build chain/residue/atom hierarchy from atom_site + pdbx_poly_seq_scheme.

    Includes unresolved residues from the sequence scheme.
    """
    import biotite.structure as struc

    # Build mapping of (auth_asym_id) → (label_asym_id, entity_id)
    chain_meta = _extract_chain_metadata(block)

    # Get entity types from _entity_poly or _entity
    entity_types = _extract_entity_types(block)

    # Group atoms by chain
    unique_chains = sorted(set(atom_array.chain_id))

    chains: list[ParsedChain] = []

    for auth_chain in unique_chains:
        chain_mask = atom_array.chain_id == auth_chain
        chain_atoms = atom_array[chain_mask]

        # Look up label_asym_id and entity_id
        label_asym_id = chain_meta.get(auth_chain, {}).get("label_asym_id", auth_chain)
        entity_id = chain_meta.get(auth_chain, {}).get("entity_id", "1")
        entity_type = entity_types.get(entity_id, "polymer")

        # Build residues from atoms
        residues = _build_residues_from_atoms(
            chain_atoms, auth_chain, modified_residues
        )

        # Add unresolved residues from pdbx_poly_seq_scheme
        resolved_seq_ids = {r.auth_seq_id for r in residues}
        unresolved = _extract_unresolved_residues(
            block, auth_chain, label_asym_id, resolved_seq_ids, modified_residues
        )
        residues.extend(unresolved)

        # Sort by auth_seq_id
        residues.sort(key=lambda r: (r.auth_seq_id, r.insertion_code or ""))

        chains.append(ParsedChain(
            auth_asym_id=auth_chain,
            label_asym_id=label_asym_id,
            entity_id=entity_id,
            entity_type=entity_type,
            residues=residues,
        ))

    return chains


def _build_residues_from_atoms(
    chain_atoms,
    auth_chain: str,
    modified_residues: dict[str, str],
) -> list[ParsedResidue]:
    """Build ParsedResidue list from a chain's AtomArray."""
    import biotite.structure as struc

    residues: list[ParsedResidue] = []
    residue_starts = struc.get_residue_starts(chain_atoms)

    for i, start_idx in enumerate(residue_starts):
        # Determine end index for this residue
        if i + 1 < len(residue_starts):
            end_idx = residue_starts[i + 1]
        else:
            end_idx = len(chain_atoms)

        res_atoms_slice = chain_atoms[start_idx:end_idx]

        auth_seq_id = int(chain_atoms.res_id[start_idx])
        res_name_3 = str(chain_atoms.res_name[start_idx]).strip()
        comp_id = res_name_3

        # Insertion code
        ins_code = None
        if hasattr(chain_atoms, "ins_code"):
            raw_ins = str(chain_atoms.ins_code[start_idx]).strip()
            if raw_ins and raw_ins not in ("?", ".", ""):
                ins_code = raw_ins

        # label_seq_id (if available)
        label_seq_id = None
        if hasattr(chain_atoms, "label_seq_id"):
            try:
                lsid = chain_atoms.label_seq_id[start_idx]
                label_seq_id = int(lsid) if lsid is not None else None
            except (TypeError, ValueError):
                pass

        # Modified residue?
        parent_comp_id = modified_residues.get(comp_id)
        is_modified = parent_comp_id is not None

        # 1-letter code
        residue_name = _three_to_one(parent_comp_id if parent_comp_id else comp_id)

        # Extract atoms
        parsed_atoms = _extract_atoms(res_atoms_slice)

        # Backbone completeness check
        atom_names_in_residue = {a.atom_name for a in parsed_atoms}
        partial_backbone = not _BACKBONE_ATOMS.issubset(atom_names_in_residue)

        # B-factor stats
        max_b_factor = None
        low_confidence_coords = False
        if parsed_atoms:
            b_factors = [a.b_factor for a in parsed_atoms]
            max_b_factor = max(b_factors)
            low_confidence_coords = max_b_factor > 100.0

        residues.append(ParsedResidue(
            auth_seq_id=auth_seq_id,
            label_seq_id=label_seq_id,
            insertion_code=ins_code,
            residue_name=residue_name,
            residue_name_3=res_name_3,
            comp_id=comp_id,
            parent_comp_id=parent_comp_id,
            sse_code=None,  # SSE assigned separately if needed
            is_resolved=True,
            is_modified=is_modified,
            partial_backbone=partial_backbone,
            max_b_factor=max_b_factor,
            low_confidence_coords=low_confidence_coords,
            atoms=parsed_atoms,
        ))

    return residues


def _extract_atoms(res_atoms) -> list[ParsedAtom]:
    """Extract ParsedAtom list from a residue's AtomArray slice."""
    atoms: list[ParsedAtom] = []
    for i in range(len(res_atoms)):
        atom_name = str(res_atoms.atom_name[i]).strip()
        element = str(res_atoms.element[i]).strip()
        x, y, z = float(res_atoms.coord[i, 0]), float(res_atoms.coord[i, 1]), float(res_atoms.coord[i, 2])

        occupancy = 1.0
        if hasattr(res_atoms, "occupancy"):
            occupancy = float(res_atoms.occupancy[i])

        b_factor = 0.0
        if hasattr(res_atoms, "b_factor"):
            b_factor = float(res_atoms.b_factor[i])

        altloc = None
        if hasattr(res_atoms, "altloc_id"):
            raw_alt = str(res_atoms.altloc_id[i]).strip()
            if raw_alt and raw_alt not in (".", "?", ""):
                altloc = raw_alt

        is_hetero = bool(res_atoms.hetero[i]) if hasattr(res_atoms, "hetero") else False

        model_id = 1  # Always model 1 (we filter at parse time)

        atoms.append(ParsedAtom(
            atom_name=atom_name,
            element=element,
            x=x, y=y, z=z,
            occupancy=occupancy,
            b_factor=b_factor,
            altloc=altloc,
            is_hetero=is_hetero,
            model_id=model_id,
        ))
    return atoms


def _extract_unresolved_residues(
    block,
    auth_chain: str,
    label_asym_id: str,
    resolved_seq_ids: set[int],
    modified_residues: dict[str, str],
) -> list[ParsedResidue]:
    """Extract unresolved residues from _pdbx_poly_seq_scheme.

    These are residues in the sequence that have no atoms in the model.
    """
    unresolved: list[ParsedResidue] = []
    try:
        seq_scheme = block["pdbx_poly_seq_scheme"]
        asym_ids = seq_scheme["asym_id"].as_array()
        auth_seq_ids = seq_scheme["auth_seq_num"].as_array()
        comp_ids = seq_scheme["mon_id"].as_array()
        pdb_seq_nums = seq_scheme["pdb_seq_num"].as_array()

        for i in range(len(asym_ids)):
            asym = str(asym_ids[i]).strip()
            if asym != label_asym_id:
                continue

            # Get auth_seq_id from pdb_seq_num
            raw_pdb_seq = str(pdb_seq_nums[i]).strip()
            if not raw_pdb_seq or raw_pdb_seq in ("?", "."):
                continue
            try:
                seq_id = int(raw_pdb_seq)
            except ValueError:
                continue

            if seq_id in resolved_seq_ids:
                continue

            comp = str(comp_ids[i]).strip()
            parent = modified_residues.get(comp)
            is_mod = parent is not None
            residue_name = _three_to_one(parent if parent else comp)

            unresolved.append(ParsedResidue(
                auth_seq_id=seq_id,
                label_seq_id=None,
                insertion_code=None,
                residue_name=residue_name,
                residue_name_3=comp,
                comp_id=comp,
                parent_comp_id=parent,
                sse_code=None,
                is_resolved=False,
                is_modified=is_mod,
                partial_backbone=True,  # No atoms at all → missing backbone
                max_b_factor=None,
                low_confidence_coords=False,
                atoms=[],
            ))
    except (KeyError, IndexError) as e:
        logger.debug("No pdbx_poly_seq_scheme for chain %s: %s", auth_chain, e)

    return unresolved


def _extract_chain_metadata(block) -> dict[str, dict[str, str]]:
    """Map auth_asym_id → {label_asym_id, entity_id} from _atom_site or _struct_asym."""
    mapping: dict[str, dict[str, str]] = {}

    # Try _struct_asym first (cleaner)
    try:
        struct_asym = block["struct_asym"]
        asym_ids = struct_asym["id"].as_array()
        entity_ids = struct_asym["entity_id"].as_array()
        for asym_id, entity_id in zip(asym_ids, entity_ids):
            # _struct_asym uses label_asym_id as the id
            mapping[str(asym_id).strip()] = {
                "label_asym_id": str(asym_id).strip(),
                "entity_id": str(entity_id).strip(),
            }
    except (KeyError, IndexError):
        pass

    # Also build auth→label mapping from _atom_site if available
    try:
        atom_site = block["atom_site"]
        auth_ids = atom_site["auth_asym_id"].as_array()
        label_ids = atom_site["label_asym_id"].as_array()
        entity_ids = atom_site["label_entity_id"].as_array()
        for auth, label, eid in zip(auth_ids, label_ids, entity_ids):
            auth_str = str(auth).strip()
            if auth_str not in mapping:
                mapping[auth_str] = {
                    "label_asym_id": str(label).strip(),
                    "entity_id": str(eid).strip(),
                }
    except (KeyError, IndexError):
        pass

    return mapping


def _extract_entity_types(block) -> dict[str, str]:
    """Map entity_id → entity_type from _entity_poly or _entity."""
    types: dict[str, str] = {}
    try:
        entity = block["entity"]
        ids = entity["id"].as_array()
        etypes = entity["type"].as_array()
        for eid, etype in zip(ids, etypes):
            types[str(eid).strip()] = str(etype).strip().lower()
    except (KeyError, IndexError):
        pass
    return types


def _determine_polymer_composition(chains: list[ParsedChain]) -> str:
    """Determine overall polymer composition (protein, na, protein/na, etc.)."""
    has_protein = any(c.entity_type in ("polymer", "polypeptide(l)", "protein") for c in chains)
    has_na = any(
        c.entity_type in ("polyribonucleotide", "polydeoxyribonucleotide", "nucleic_acid")
        for c in chains
    )
    if has_protein and has_na:
        return "protein/na"
    elif has_na:
        return "nucleic_acid"
    elif has_protein:
        return "protein"
    return "other"


def _safe_array(category, column_name):
    """Safely get a column array, returning None if not present."""
    try:
        return category[column_name].as_array()
    except (KeyError, IndexError):
        return None


def _clean_ins_code(val) -> str | None:
    """Clean insertion code value — return None if empty/missing."""
    if val is None:
        return None
    s = str(val).strip()
    if s in ("", "?", "."):
        return None
    return s


def _three_to_one(code: str) -> str:
    """Convert 3-letter amino acid code to 1-letter. Unknown → 'X'."""
    return _AA_3TO1.get(code.upper(), "X")
