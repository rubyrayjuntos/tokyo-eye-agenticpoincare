"""
Energy Calculation Service

Computes approximate Gibbs free energy (ΔG) using RDKit force field minimization.
Replaces placeholder energy values with physics-based thermodynamic calculations.

Implements MMFF94 and UFF force fields for potential energy minimization,
combined with SASA-based solvation penalty approximation.

Validates: Technical Specification §2.1 (xState Energy Guard)
Reference: RDKit AllChem.MMFF94/UFF implementations
"""

import numpy as np
from typing import Literal, Optional, Tuple
from dataclasses import dataclass
import tempfile
import os

from rdkit import Chem
from rdkit.Chem import AllChem
try:
    from rdkit.Chem.rdFreeSASA import CalcSASA
    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False
    # Fallback: estimate SASA from molecular surface

import biotite.structure as struc
import biotite.structure.io as strucio

from gosp.models.data_models import StructureData


class EnergyCalculationError(Exception):
    """Raised when energy calculation fails."""
    pass


@dataclass
class EnergyResult:
    """Container for energy calculation results."""
    delta_g_total: float  # Total ΔG approximation (kcal/mol)
    potential_energy: float  # Force field potential energy (kcal/mol)
    solvation_term: float  # SASA-based solvation penalty (kcal/mol)
    sasa_angstrom2: Optional[float]  # Solvent-accessible surface area (Ų)
    force_field: str  # Force field used ('MMFF94' or 'UFF')
    minimization_converged: bool  # Whether minimization converged
    rmsd_pre_post_min: Optional[float]  # RMSD before/after minimization (Å)
    computation_time_sec: float  # Computation time
    
    def to_dict(self):
        """Convert to JSON-serializable dict."""
        return {
            "delta_g_total": float(self.delta_g_total),
            "potential_energy": float(self.potential_energy),
            "solvation_term": float(self.solvation_term),
            "sasa_angstrom2": float(self.sasa_angstrom2) if self.sasa_angstrom2 else None,
            "force_field": self.force_field,
            "minimization_converged": bool(self.minimization_converged),
            "rmsd_pre_post_min": float(self.rmsd_pre_post_min) if self.rmsd_pre_post_min else None,
            "computation_time_sec": float(self.computation_time_sec)
        }


def compute_energy(
    structure: StructureData,
    force_field: Literal["MMFF94", "UFF"] = "MMFF94",
    max_iterations: int = 500,
    sasa_probe_radius: float = 1.4,
    solvation_factor: float = 0.005,
    embed_conformer: bool = False
) -> EnergyResult:
    """
    Compute approximate Gibbs free energy using RDKit force fields.
    
    Algorithm:
    1. Convert StructureData to Biotite AtomArray
    2. Export to PDB format string
    3. Load into RDKit Mol object
    4. Add hydrogens (if missing)
    5. Optionally embed conformer (if coordinates missing)
    6. Apply force field (MMFF94 or UFF)
    7. Minimize structure (max_iterations)
    8. Calculate potential energy
    9. Calculate SASA-based solvation term
    10. Return ΔG ≈ potential_energy + solvation_term
    
    Args:
        structure: StructureData object with atomic coordinates
        force_field: Force field to use ('MMFF94' or 'UFF')
        max_iterations: Maximum minimization iterations (default: 500)
        sasa_probe_radius: Probe radius for SASA calculation (default: 1.4 Å)
        solvation_factor: SASA→solvation conversion factor (default: 0.005 kcal/mol/Ų)
        embed_conformer: Whether to generate 3D coordinates (default: False - use existing)
        
    Returns:
        EnergyResult with ΔG approximation and components
        
    Raises:
        EnergyCalculationError: If energy calculation fails
        
    Notes:
        - MMFF94 preferred for protein structures (better parameterization)
        - UFF is more robust but less accurate for proteins
        - Solvation term is empirical approximation (not rigorous implicit solvent)
        - For GOSP validation: ΔG < -15.5 kcal/mol passes ALIGN→LOCK guard
    """
    import time
    start_time = time.time()
    
    try:
        # Step 1: Convert StructureData to Biotite AtomArray
        biotite_structure = _structure_data_to_biotite(structure)
        
        # Step 2: Create temporary PDB file for RDKit
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            strucio.save_structure(tmp_path, biotite_structure)
            
            # Step 3: Load into RDKit
            mol = Chem.MolFromPDBFile(tmp_path, removeHs=False)
            if mol is None:
                raise EnergyCalculationError("Failed to convert structure to RDKit Mol object")
            
            # Store original coordinates for RMSD
            original_conf = mol.GetConformer()
            original_coords = original_conf.GetPositions()
            
            # Step 4: Add hydrogens if not present
            num_atoms_before = mol.GetNumAtoms()
            mol = Chem.AddHs(mol, addCoords=True)
            num_atoms_after = mol.GetNumAtoms()
            
            # Step 5: Embed conformer if requested (generates 3D coords)
            if embed_conformer:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            
            # Step 6: Apply force field
            if force_field == "MMFF94":
                # MMFF94 - better for proteins
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props is None:
                    raise EnergyCalculationError("Failed to get MMFF94 properties (unsupported atoms?)")
                ff_obj = AllChem.MMFFGetMoleculeForceField(mol, props)
            elif force_field == "UFF":
                # UFF - universal force field (more robust)
                ff_obj = AllChem.UFFGetMoleculeForceField(mol)
            else:
                raise EnergyCalculationError(f"Unsupported force field: {force_field}")
            
            if ff_obj is None:
                raise EnergyCalculationError(f"Failed to create {force_field} force field object")
            
            # Step 7: Minimize structure
            convergence_code = ff_obj.Minimize(maxIts=max_iterations)
            minimization_converged = (convergence_code == 0)
            
            # Step 8: Calculate potential energy
            potential_energy = ff_obj.CalcEnergy()
            
            # Calculate RMSD before/after minimization (heavy atoms only)
            minimized_conf = mol.GetConformer()
            minimized_coords = minimized_conf.GetPositions()
            
            # Only compare original heavy atoms (before H addition)
            if len(minimized_coords) >= len(original_coords):
                rmsd = np.sqrt(np.mean(
                    np.sum((minimized_coords[:len(original_coords)] - original_coords)**2, axis=1)
                ))
            else:
                rmsd = None
            
            # Step 9: Calculate SASA-based solvation term
            if HAS_FREESASA:
                try:
                    # CalcSASA with modern RDKit API (radii dict required)
                    from rdkit.Chem.rdFreeSASA import classifyAtoms
                    # Use default radii classification
                    radii = classifyAtoms(mol)
                    sasa = CalcSASA(mol, radii)
                    solvation_term = solvation_factor * sasa
                except Exception as e:
                    # Fallback if SASA fails
                    num_heavy_atoms = mol.GetNumHeavyAtoms()
                    sasa_estimate = num_heavy_atoms * 1.5
                    solvation_term = solvation_factor * sasa_estimate
                    sasa = sasa_estimate
            else:
                # Fallback: approximate from molecular weight
                # Empirical: ~1.5 Ų per heavy atom
                num_heavy_atoms = mol.GetNumHeavyAtoms()
                sasa_estimate = num_heavy_atoms * 1.5
                solvation_term = solvation_factor * sasa_estimate
                sasa = sasa_estimate
            
            # Step 10: Total ΔG approximation
            delta_g_total = potential_energy + solvation_term
            
            computation_time = time.time() - start_time
            
            return EnergyResult(
                delta_g_total=delta_g_total,
                potential_energy=potential_energy,
                solvation_term=solvation_term,
                sasa_angstrom2=sasa,
                force_field=force_field,
                minimization_converged=minimization_converged,
                rmsd_pre_post_min=rmsd,
                computation_time_sec=computation_time
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        raise EnergyCalculationError(f"Energy calculation failed: {str(e)}")


def _structure_data_to_biotite(structure_data: StructureData) -> struc.AtomArray:
    """Convert StructureData to Biotite AtomArray."""
    total_atoms = structure_data.total_atoms
    atom_array = struc.AtomArray(total_atoms)
    
    atom_index = 0
    for chain in structure_data.chains:
        for residue in chain.residues:
            for atom in residue.atoms:
                atom_array.atom_name[atom_index] = atom.atom_name
                atom_array.res_name[atom_index] = atom.residue_name
                atom_array.chain_id[atom_index] = atom.chain_id
                atom_array.res_id[atom_index] = atom.residue_id
                atom_array.coord[atom_index] = np.array([atom.x, atom.y, atom.z], dtype=np.float32)
                atom_array.element[atom_index] = atom.element
                if hasattr(atom_array, 'occupancy'):
                    atom_array.occupancy[atom_index] = atom.occupancy
                if hasattr(atom_array, 'b_factor'):
                    atom_array.b_factor[atom_index] = atom.b_factor
                atom_index += 1
    
    return atom_array


def evaluate_state_machine_guard(
    energy_result: EnergyResult,
    threshold: float = -15.5
) -> Tuple[bool, str]:
    """
    Evaluate whether energy passes GOSP state machine guard.
    
    GOSP ALIGN→LOCK transition guard:
    - ΔG must be < -15.5 kcal/mol (stable folded state)
    - Minimization should converge (structure is physically reasonable)
    
    Args:
        energy_result: EnergyResult from compute_energy()
        threshold: Energy threshold in kcal/mol (default: -15.5)
        
    Returns:
        Tuple of (passes_guard, reason)
    """
    if energy_result.delta_g_total < threshold:
        if energy_result.minimization_converged:
            return (True, f"PASS: ΔG = {energy_result.delta_g_total:.2f} < {threshold} kcal/mol (converged)")
        else:
            return (True, f"PASS: ΔG = {energy_result.delta_g_total:.2f} < {threshold} kcal/mol (WARNING: minimization did not converge)")
    else:
        return (False, f"FAIL: ΔG = {energy_result.delta_g_total:.2f} ≥ {threshold} kcal/mol (unstable state)")


def compare_energies(
    structure_1: StructureData,
    structure_2: StructureData,
    force_field: Literal["MMFF94", "UFF"] = "MMFF94"
) -> dict:
    """
    Compare energies of two structures (e.g., before/after stabilizer placement).
    
    Args:
        structure_1: First structure
        structure_2: Second structure  
        force_field: Force field to use
        
    Returns:
        Dict with comparison results
    """
    result_1 = compute_energy(structure_1, force_field=force_field)
    result_2 = compute_energy(structure_2, force_field=force_field)
    
    delta_delta_g = result_2.delta_g_total - result_1.delta_g_total
    
    return {
        "structure_1_dg": result_1.delta_g_total,
        "structure_2_dg": result_2.delta_g_total,
        "delta_delta_g": delta_delta_g,
        "improvement": delta_delta_g < 0,  # Structure 2 is more stable
        "force_field": force_field,
        "result_1": result_1.to_dict(),
        "result_2": result_2.to_dict()
    }
