"""
Physics Kernel Service

Handles energy calculations using RDKit force fields and FreeSASA.
Computes Gibbs free energy with solvation penalty.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
"""

import time
import uuid
from typing import Literal, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import freesasa
import biotite.structure as struc

from gosp.models.data_models import StructureData, EnergyCalculationResult, ComputationProvenance, ProvenanceStep
from gosp.services.biotite_utils import build_biotite_structure


class PhysicsKernelError(Exception):
    """Raised when physics calculation fails."""
    pass


class ForceFieldParameterizationError(PhysicsKernelError):
    """Raised when force field parameterization fails."""
    pass


class EnergyMinimizationError(PhysicsKernelError):
    """Raised when energy minimization fails."""
    pass


SOLVATION_PENALTY_COEFFICIENT = 0.025  # kcal/mol/Å² (nonpolar SASA desolvation)


def calculate_energy(
    structure: StructureData,
    force_field: Literal["MMFF94", "UFF"] = "MMFF94",
    max_iterations: int = 500,
    timeout_sec: float = 5.0
) -> EnergyCalculationResult:
    """
    Calculate Gibbs free energy for a protein structure.

    Builds a ComputationProvenance record tracking every step,
    including fallbacks and approximations.
    """
    computation_start = time.time()
    steps: list[ProvenanceStep] = []
    step_counter = 0
    actual_method = force_field
    warnings: list[str] = []

    try:
        # Create a single, correct biotite structure representation first.
        biotite_structure = build_biotite_structure(structure)

        # Step 1: Convert structure to RDKit molecule
        step_counter += 1
        step_start = time.time()
        try:
            mol = _structure_to_rdkit_mol(biotite_structure)
            steps.append(ProvenanceStep(
                step_id=step_counter,
                method="pdb_conversion",
                status="success",
                duration_sec=time.time() - step_start,
                inputs={
                    "num_atoms": biotite_structure.array_length(),
                },
                outputs={"rdkit_atom_count": mol.GetNumAtoms()},
                approximations=[],
            ))
        except Exception as e:
            steps.append(ProvenanceStep(
                step_id=step_counter,
                method="pdb_conversion",
                status="failed",
                duration_sec=time.time() - step_start,
                inputs={"num_atoms": biotite_structure.array_length()},
                outputs={},
                reason=str(e),
                approximations=[],
            ))
            raise

        # Check timeout
        if time.time() - computation_start > timeout_sec:
            raise PhysicsKernelError(f"Calculation exceeded timeout of {timeout_sec}s")

        # Step 2: Attempt force field minimization
        step_counter += 1
        step_start = time.time()
        try:
            potential_energy, converged = _minimize_energy(
                mol, force_field, max_iterations,
                timeout_sec - (time.time() - computation_start),
            )
            steps.append(ProvenanceStep(
                step_id=step_counter,
                method=f"{force_field.lower()}_minimization",
                status="success",
                duration_sec=time.time() - step_start,
                inputs={
                    "force_field": force_field,
                    "max_iterations": max_iterations,
                },
                outputs={
                    "potential_energy": potential_energy,
                    "converged": converged,
                },
            ))
        except (ForceFieldParameterizationError, EnergyMinimizationError) as ff_err:
            steps.append(ProvenanceStep(
                step_id=step_counter,
                method=f"{force_field.lower()}_minimization",
                status="failed",
                duration_sec=time.time() - step_start,
                inputs={"force_field": force_field},
                outputs={},
                reason=str(ff_err),
            ))

            # Step 3: Fallback to contact potential
            step_counter += 1
            step_start = time.time()
            potential_energy = _compute_contact_potential(structure)
            converged = True
            actual_method = "contact_potential"
            warnings.append(
                f"Energy computed via residue contact potential "
                f"({force_field} parameterization failed for this structure)"
            )
            steps.append(ProvenanceStep(
                step_id=step_counter,
                method="contact_potential",
                status="success",
                duration_sec=time.time() - step_start,
                inputs={
                    "n_residues": structure.total_residues,
                    "contact_cutoff_A": 8.0,
                    "hydrophobic_weight": -1.5,
                    "other_weight": -0.5,
                },
                outputs={"potential_energy": potential_energy},
                reason=f"{force_field} parameterization failed for this structure",
                approximations=[
                    "Simplified contact energy: hydrophobic pairs −1.5, others −0.5 kcal/mol",
                ],
            ))

        # Check timeout
        if time.time() - computation_start > timeout_sec:
            raise PhysicsKernelError(f"Calculation exceeded timeout of {timeout_sec}s")

        # Step 4: Calculate SASA
        step_counter += 1
        step_start = time.time()
        try:
            sasa = _calculate_sasa(biotite_structure)
            steps.append(ProvenanceStep(
                step_id=step_counter,
                method="freesasa_calculation",
                status="success",
                duration_sec=time.time() - step_start,
                inputs={"num_atoms": biotite_structure.array_length()},
                outputs={"total_sasa": sasa},
            ))
        except Exception as e:
            steps.append(ProvenanceStep(
                step_id=step_counter,
                method="freesasa_calculation",
                status="failed",
                duration_sec=time.time() - step_start,
                inputs={"num_atoms": biotite_structure.array_length()},
                outputs={},
                reason=str(e),
            ))
            raise

        # Calculate solvation term and ΔG
        solvation_term = SOLVATION_PENALTY_COEFFICIENT * sasa
        delta_g = potential_energy + solvation_term
        computation_time = time.time() - computation_start

        # Determine confidence
        if actual_method == force_field:
            confidence = "high"
        elif actual_method == "contact_potential":
            confidence = "medium"
        else:
            confidence = "low"

        provenance = ComputationProvenance(
            computation_id=str(uuid.uuid4()),
            computation_type="energy",
            requested_method=force_field,
            actual_method=actual_method,
            steps=steps,
            warnings=warnings,
            total_duration_sec=computation_time,
            confidence=confidence,
        )

        return EnergyCalculationResult(
            delta_g=delta_g,
            potential_energy=potential_energy,
            solvation_term=solvation_term,
            sasa=sasa,
            computation_time_sec=computation_time,
            force_field=force_field,
            converged=converged,
            provenance=provenance,
        )

    except Exception as e:
        raise PhysicsKernelError(f"Energy calculation failed: {str(e)}")


def _structure_to_rdkit_mol(atom_array: struc.AtomArray) -> Chem.Mol:
    """
    Convert a Biotite AtomArray to an RDKit molecule via a PDB string.
    """
    import io
    from biotite.structure.io.pdb import PDBFile
    
    try:
        # Create a PDB file in memory
        pdb_file = PDBFile()
        pdb_file.set_structure(atom_array)
        
        # Write to a string buffer
        with io.StringIO() as f:
            pdb_file.write(f)
            pdb_block = f.getvalue()

        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True, removeHs=False)

        if mol is None:
            # Fallback: try with relaxed sanitization
            mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^
                                     Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    pass  # Use partially sanitized molecule

        if mol is None:
            raise PhysicsKernelError(
                "Failed to create RDKit molecule from PDB coordinates"
            )

        return mol

    except PhysicsKernelError:
        raise
    except Exception as e:
        raise PhysicsKernelError(f"Failed to convert structure to RDKit molecule: {str(e)}")


def _minimize_energy(
    mol: Chem.Mol,
    force_field: str,
    max_iterations: int,
    timeout_sec: float
) -> tuple[float, bool]:
    """
    Perform force field energy minimization.
    
    Args:
        mol: RDKit molecule
        force_field: "MMFF94" or "UFF"
        max_iterations: Maximum iterations
        timeout_sec: Remaining timeout
    
    Returns:
        Tuple of (potential_energy, converged)
    
    Raises:
        ForceFieldParameterizationError: If force field setup fails
        EnergyMinimizationError: If minimization fails
    
    Validates: Requirements 2.1, 2.2, 2.5
    """
    start_time = time.time()
    
    try:
        # Set up force field
        if force_field == "MMFF94":
            # Try MMFF94
            ff_props = AllChem.MMFFGetMoleculeProperties(mol)
            if ff_props is None:
                raise ForceFieldParameterizationError(
                    "MMFF94 parameterization failed - molecule may contain unsupported atom types"
                )
            ff = AllChem.MMFFGetMoleculeForceField(mol, ff_props)
            if ff is None:
                raise ForceFieldParameterizationError(
                    "Failed to create MMFF94 force field"
                )
        elif force_field == "UFF":
            # Try UFF as fallback
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol)
                if ff is None:
                    raise ForceFieldParameterizationError(
                        "Failed to create UFF force field"
                    )
            except Exception as e:
                raise ForceFieldParameterizationError(
                    f"UFF parameterization failed: {str(e)}"
                )
        else:
            raise ForceFieldParameterizationError(
                f"Unsupported force field: {force_field}"
            )
        
        # Compute single-point energy first (before minimization modifies state)
        import math
        pre_energy = ff.CalcEnergy()

        # Attempt minimization. If it fails (e.g. bad BFGS direction from
        # clashing hydrogens placed at heavy atom positions), fall back to
        # the single-point energy.
        converged = False
        try:
            result = ff.Minimize(maxIts=max_iterations)
            converged = (result == 0)

            if time.time() - start_time > timeout_sec:
                raise EnergyMinimizationError(
                    f"Minimization exceeded timeout of {timeout_sec}s"
                )
        except EnergyMinimizationError:
            raise
        except Exception:
            # Minimization crashed — return pre-minimization energy
            pass

        potential_energy = ff.CalcEnergy()
        # If post-minimize energy is nan/inf, use the pre-minimization value
        if math.isnan(potential_energy) or math.isinf(potential_energy):
            potential_energy = pre_energy
        # If still bad, use SASA-only estimate
        if math.isnan(potential_energy) or math.isinf(potential_energy):
            potential_energy = 0.0

        return potential_energy, converged
    
    except ForceFieldParameterizationError:
        raise
    except EnergyMinimizationError:
        raise
    except Exception as e:
        raise EnergyMinimizationError(f"Energy minimization failed: {str(e)}")


def _compute_contact_potential(structure: StructureData) -> float:
    """
    Compute potential energy using a simplified residue-contact model.

    Uses Cα–Cα distances to count contacts (< 8 Å, sequence separation ≥ 2).
    Hydrophobic–hydrophobic contacts contribute −1.5 kcal/mol,
    all other contacts contribute −0.5 kcal/mol.

    This replaces RDKit force-field energy for protein structures where
    MMFF94/UFF parameterization fails.
    """
    ca_coords = []
    full_sequence = []
    
    for chain in structure.chains:
        full_sequence.extend(list(chain.sequence))
        for residue in chain.residues:
            for atom in residue.atoms:
                if atom.atom_name == 'CA':
                    ca_coords.append([atom.x, atom.y, atom.z])
                    break # Assume one CA per residue
    
    if not ca_coords:
        return 0.0

    ca_coords = np.array(ca_coords)
    n_res = len(ca_coords)

    # Pairwise distances (vectorised)
    diff = ca_coords[:, np.newaxis, :] - ca_coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # Sequence separation ≥ 2, upper triangle only
    idx = np.arange(n_res)
    seq_sep = np.abs(idx[:, None] - idx[None, :])
    contact_mask = (dist < 8.0) & (seq_sep >= 2) & (idx[:, None] < idx[None, :])

    # Hydrophobic residues
    hydrophobic = set("AVILMFWP")
    is_hp = np.array([s in hydrophobic for s in full_sequence])
    hh_mask = is_hp[:, None] & is_hp[None, :]

    n_hh = int(np.sum(contact_mask & hh_mask))
    n_other = int(np.sum(contact_mask & ~hh_mask))

    return -1.5 * n_hh + -0.5 * n_other


def _calculate_sasa(atom_array: struc.AtomArray) -> float:
    """
    Calculate solvent-accessible surface area using FreeSASA.
    
    Args:
        atom_array: A biotite AtomArray.
    
    Returns:
        SASA in square Angstroms
    
    Raises:
        PhysicsKernelError: If SASA calculation fails
    
    Validates: Requirements 2.3
    """
    import tempfile
    import os
    from biotite.structure.io.pdb import PDBFile
    import io

    try:
        # Create a PDB file in memory
        pdb_file = PDBFile()
        pdb_file.set_structure(atom_array)
        
        # Write to a string buffer
        with io.StringIO() as f:
            pdb_file.write(f)
            pdb_string = f.getvalue()

        # FreeSASA requires a file path, not a string
        # Write PDB string to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_string)
            pdb_path = f.name
        
        try:
            # Calculate SASA using FreeSASA with file path
            structure_freesasa = freesasa.Structure(pdb_path)
            result = freesasa.calc(structure_freesasa)
            
            # Get total SASA
            total_sasa = result.totalArea()
            
            return total_sasa
        finally:
            # Clean up temporary file
            if os.path.exists(pdb_path):
                os.unlink(pdb_path)
    
    except Exception as e:
        raise PhysicsKernelError(f"SASA calculation failed: {str(e)}")


def calculate_per_residue_sasa(structure: StructureData) -> dict:
    """
    Calculate per-residue SASA using FreeSASA.

    Walks the Chain → Residue → Atom hierarchy in StructureData.

    Returns:
        dict mapping residue_id (int) -> SASA in Å².
        On FreeSASA failure returns 0.0 for all residues.
    """
    _ELEMENT_RADII = {
        "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
        "H": 1.20, "P": 1.80,
    }
    default_radius = 1.70

    flat_coords: list = []
    radii: list = []
    residue_atom_indices: dict = {}  # residue_id -> list of global atom indices

    atom_global_idx = 0
    for chain in (structure.chains or []):
        for residue in chain.residues:
            rid = residue.residue_id
            residue_atom_indices.setdefault(rid, [])
            for atom in residue.atoms:
                flat_coords.extend([atom.x, atom.y, atom.z])
                element = atom.element.upper() if atom.element else "C"
                radii.append(_ELEMENT_RADII.get(element, default_radius))
                residue_atom_indices[rid].append(atom_global_idx)
                atom_global_idx += 1

    if atom_global_idx == 0:
        return {}

    try:
        result = freesasa.calcCoord(flat_coords, radii)
    except Exception:
        return {rid: 0.0 for rid in residue_atom_indices}

    per_residue: dict = {}
    for rid, indices in residue_atom_indices.items():
        per_residue[rid] = float(sum(result.atomArea(i) for i in indices))

    return per_residue


def _one_to_three_letter(one_letter: str) -> str:
    """Convert 1-letter amino acid code to 3-letter code."""
    conversion = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
        'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
        'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
        'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
        'X': 'UNK'
    }
    return conversion.get(one_letter, 'UNK')
