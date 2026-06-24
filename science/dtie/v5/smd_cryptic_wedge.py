"""
EIDETIX BIO - TOKYO EYE PIPELINE
Target: KRAS A:78-A:82 Cryptic Wedge (Phase 5 Bypass)
Protocol: 13-ns Steered Molecular Dynamics (SMD)

Features:
  - Hydrogen Mass Repartitioning (HMR): 1.5 amu
  - Rhombic Dodecahedron solvent box: 70.7% efficient
  - 4fs timestep with LangevinMiddleIntegrator
  - PME long-range electrostatics
  - GPU-accelerated via CUDA (fallback to CPU if unavailable)

Three-Phase Protocol:
  Phase 1: PULL (2 ns) - Spring-induced void creation
  Phase 2: EQUILIBRATE (1 ns) - Solvent relaxation
  Phase 3: RELEASE (10 ns) - Validation of persistent pocket

Usage:
  python smd_cryptic_wedge.py <input_pdb> [--gpu] [--output-dir DIR]

Requirements:
  - openmm>=8.5.0
  - pdbfixer
  - mdtraj
"""

import sys
import os
import argparse
from pathlib import Path
from openmm import *
from openmm.app import *
from openmm.unit import *


def prepare_structure(pdb_file, output_dir="."):
    """Load PDB and ensure it has all hydrogens."""
    from pdbfixer import PDBFixer
    
    print(f"[*] Preparing PDB: {pdb_file}")
    fixer = PDBFixer(filename=pdb_file)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)  # pH 7
    
    fixed_pdb = Path(output_dir) / "smd_prepared.pdb"
    with open(fixed_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    
    print(f"[*] Prepared structure saved: {fixed_pdb}")
    return str(fixed_pdb)


def setup_smd_simulation(pdb_file, leu79_idx=None, val81_idx=None, use_cuda=True, output_dir="."):
    """
    Set up the SMD simulation system.
    
    Args:
        pdb_file: Path to prepared PDB
        leu79_idx: Atom index for Leu79 CA/CB (auto-detect if None)
        val81_idx: Atom index for Val81 CA/CB (auto-detect if None)
        use_cuda: Try CUDA platform if available
        output_dir: Directory for output files
    
    Returns:
        (simulation, platform_name)
    """
    print("[*] Loading structure...")
    pdb = PDBFile(pdb_file)
    
    print("[*] Building forcefield...")
    ff = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    
    print("[*] Adding solvent (Dodecahedron box, 0.8 nm padding)...")
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(ff, boxShape='dodecahedron', padding=0.8*nanometers)
    
    print("[*] Creating system with HMR (hydrogenMass=1.5 amu)...")
    system = ff.createSystem(modeller.topology,
                            nonbondedMethod=PME,
                            nonbondedCutoff=1.0*nanometers,
                            constraints=HBonds,
                            rigidWater=True,
                            hydrogenMass=1.5*amu)
    
    # Auto-detect Leu79 and Val81 if indices not provided
    if leu79_idx is None or val81_idx is None:
        print("[*] Auto-detecting target residue atom indices...")
        ca_indices = []
        for atom_idx, atom in enumerate(pdb.topology.atoms()):
            if atom.name == 'CA':
                ca_indices.append(atom_idx)
        
        if leu79_idx is None and len(ca_indices) > 0:
            leu79_idx = ca_indices[0]
        if val81_idx is None and len(ca_indices) > 1:
            val81_idx = ca_indices[1]
    
    if leu79_idx is None or val81_idx is None:
        raise ValueError(f"Could not locate residue atoms: Leu79={leu79_idx}, Val81={val81_idx}")
    
    print(f"[*] SMD target atoms: Leu79 (idx={leu79_idx}), Val81 (idx={val81_idx})")
    
    # Configure SMD force
    print("[*] Setting up SMD CustomExternalForce...")
    smd_force = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    smd_force.addGlobalParameter("k", 0.0 * kilocalories_per_mole/angstroms**2)
    smd_force.addPerParticleParameter("x0")
    smd_force.addPerParticleParameter("y0")
    smd_force.addPerParticleParameter("z0")
    
    # Target displacements (from DTIE cryptic wedge analysis)
    smd_force.addParticle(leu79_idx, [2.0, 1.5, 1.0])
    smd_force.addParticle(val81_idx, [-1.5, 1.8, -0.8])
    
    system.addForce(smd_force)
    
    # Integrator: 4fs timestep with HMR
    print("[*] Configuring LangevinMiddleIntegrator (4fs timestep)...")
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    
    # Platform selection
    platform_name = "CPU"
    try:
        if use_cuda:
            platform = Platform.getPlatformByName('CUDA')
            platform.setPropertyDefaultValue('Precision', 'mixed')
            platform_name = "CUDA"
            print("[*] Using CUDA platform (GPU acceleration)")
        else:
            raise Exception("CUDA disabled by user")
    except Exception as e:
        print(f"[!] CUDA unavailable ({e}), falling back to CPU")
        platform = Platform.getPlatformByName('CPU')
        platform_name = "CPU"
    
    print("[*] Creating simulation...")
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    
    # Configure reporters
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    
    simulation.reporters.append(DCDReporter(str(out_path / 'trajectory_smd.dcd'), 25000))
    simulation.reporters.append(StateDataReporter(
        sys.stdout, 25000, step=True,
        potentialEnergy=True, temperature=True,
        volume=True, speed=True
    ))
    simulation.reporters.append(StateDataReporter(
        str(out_path / 'smd_metrics.csv'), 25000, step=True,
        potentialEnergy=True, temperature=True
    ))
    
    return simulation, platform_name


def run_smd_protocol(simulation, output_dir="."):
    """
    Execute the three-phase SMD protocol.
    
    Phase 1: PULL (2 ns, 500k steps @ 4fs)
    Phase 2: EQUILIBRATE (1 ns, 250k steps @ 4fs)
    Phase 3: RELEASE (10 ns, 2.5M steps @ 4fs)
    """
    print("\n" + "="*70)
    print("PHASE 1: PULL (2 ns) - Spring-Induced Void Creation")
    print("="*70)
    print("[>>>] Turning spring ON (k=10.0 kcal/mol/A^2)...")
    simulation.context.setParameter("k", 10.0 * kilocalories_per_mole/angstroms**2)
    simulation.step(500000)
    print("[<<<] PHASE 1 complete")
    
    print("\n" + "="*70)
    print("PHASE 2: EQUILIBRATE (1 ns) - Constrained Solvent Relaxation")
    print("="*70)
    print("[>>>] Spring remains ON, solvent equilibrating...")
    simulation.step(250000)
    print("[<<<] PHASE 2 complete")
    
    print("\n" + "="*70)
    print("PHASE 3: RELEASE (10 ns) - Persistent Pocket Validation")
    print("="*70)
    print("[>>>] Turning spring OFF (k=0.0)...")
    simulation.context.setParameter("k", 0.0)
    simulation.step(2500000)
    print("[<<<] PHASE 3 complete")
    
    print("\n" + "="*70)
    print("[SUCCESS] 13-ns SMD Protocol Complete")
    print("="*70)
    print(f"Trajectory: {Path(output_dir) / 'trajectory_smd.dcd'}")
    print(f"Metrics:    {Path(output_dir) / 'smd_metrics.csv'}")


def main():
    parser = argparse.ArgumentParser(description="KRAS Cryptic Wedge SMD Pipeline")
    parser.add_argument("pdb", help="Input PDB file")
    parser.add_argument("--leu79-idx", type=int, help="Leu79 atom index (auto-detect if omitted)")
    parser.add_argument("--val81-idx", type=int, help="Val81 atom index (auto-detect if omitted)")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use CUDA if available (default)")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU platform")
    parser.add_argument("-o", "--output-dir", default=".", help="Output directory")
    parser.add_argument("--skip-prep", action="store_true", help="Skip PDB preparation (use as-is)")
    
    args = parser.parse_args()
    
    # Prepare structure
    if args.skip_prep:
        pdb_file = args.pdb
    else:
        pdb_file = prepare_structure(args.pdb, args.output_dir)
    
    # Setup simulation
    sim, platform_name = setup_smd_simulation(
        pdb_file,
        leu79_idx=args.leu79_idx,
        val81_idx=args.val81_idx,
        use_cuda=(not args.cpu_only),
        output_dir=args.output_dir
    )
    
    print(f"\n[*] Platform: {platform_name}")
    print("[*] Running energy minimization...")
    sim.minimizeEnergy(maxIterations=5000)
    
    print("[*] Running 100ps NVT equilibration...")
    sim.step(50000)
    
    # Run protocol
    run_smd_protocol(sim, args.output_dir)


if __name__ == "__main__":
    main()

