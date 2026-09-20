"""Smoke: PDB CA atoms as C -> EquiformerV3_OC forward cut -> (s,v) -> geoopt projector."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from Bio.PDB import PDBParser

sys.path.insert(0, "/app/science/third_party")

from equiformer_v3_model.equiformer_v3 import EquiformerV3_OC
from science.tokyo_eye.v8.projector import RadialAngularProjector
from science.tokyo_eye.v8.loader import ensure_pdb_cached

MPTRJ_KWARGS = dict(
    use_pbc=False,
    use_pbc_single=True,
    otf_graph=True,
    regress_forces=False,
    regress_stress=False,
    direct_prediction=False,
    max_neighbors=50,  # smoke: smaller than 300 for 4GB GPU
    max_radius=8.0,
    num_radial_basis=10,
    num_layers=7,
    num_channels=128,
    attn_hidden_channels=32,
    num_heads=8,
    attn_alpha_channels=64,
    attn_value_channels=16,
    ffn_hidden_channels=512,
    norm_type="merge_layer_norm",
    lmax=4,
    mmax=2,
    attn_grid_resolution_list=[14, 8],
    ffn_grid_resolution_list=[14, 14],
    edge_channels=128,
    use_grid_mlp=True,
)


def pdb_ca_to_data(pdb_path: Path, chain: str, device: torch.device):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    pos = []
    for res in model.get_residues():
        if res.parent.id != chain or res.get_id()[0] != " ":
            continue
        if "CA" not in res:
            continue
        pos.append(res["CA"].coord.tolist())
    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
    n = pos_t.shape[0]
    # CA treated as carbon (Z=6) for smoke — full atomization is next hardening
    atomic_numbers = torch.full((n,), 6, dtype=torch.long, device=device)
    batch = torch.zeros(n, dtype=torch.long, device=device)
    natoms = torch.tensor([n], dtype=torch.long, device=device)
    # non-PBC dummy cell
    cell = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0) * 1000.0
    pbc = torch.zeros(1, 3, dtype=torch.bool, device=device)
    return SimpleNamespace(
        pos=pos_t,
        atomic_numbers=atomic_numbers,
        batch=batch,
        natoms=natoms,
        cell=cell,
        pbc=pbc,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    pdb_dir = Path("/tmp/dtie_pdb_cache")
    pdb_path = ensure_pdb_cached("4OBE", pdb_dir)
    data = pdb_ca_to_data(pdb_path, "A", device)
    print("N_atoms", int(data.natoms[0]))

    print("build+load model...")
    model = EquiformerV3_OC(**MPTRJ_KWARGS).to(device)
    ckpt = Path("checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    if any(str(k).startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("load missing", len(missing), "unexpected", len(unexpected))
    model.eval()

    print("forward cut...")
    with torch.no_grad():
        model.device = data.pos.device
        model.dtype = data.pos.dtype
        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            *_rest,
        ) = model.generate_graph(
            data,
            enforce_max_neighbors_strictly=True,
            use_pbc_single=True,
        )
        print("edges", edge_index.shape[1])
        atomic_numbers = data.atomic_numbers.long()
        src_z = atomic_numbers[edge_index[0]]
        tgt_z = atomic_numbers[edge_index[1]]
        edge_distance, edge_envelope_weight = model._forward_edge(
            edge_distance, edge_distance_vec
        )
        x = model._forward_embedding(
            atomic_numbers, edge_distance, edge_index, edge_envelope_weight
        )
        x_scalar, x = model._forward_blocks(
            x, src_z, tgt_z, edge_distance, edge_index, edge_envelope_weight, data.batch
        )
        print("x_scalar", tuple(x_scalar.shape), "x", tuple(x.shape))
        s = x_scalar
        # x layout: [N, spheres, C] — L=0 is index 0; L=1 occupies 1:4
        if x.ndim == 3 and x.shape[1] >= 4:
            v = x[:, 1:4, :].mean(dim=-1)
        else:
            v = torch.zeros(s.shape[0], 3, device=s.device, dtype=s.dtype)
        print("s", tuple(s.shape), "v", tuple(v.shape))
        proj = RadialAngularProjector(s.shape[-1], v.shape[-1], 128, c=1.0).to(device)
        z = proj(s, v, tau_ceiling=0.8)
        print(
            "z",
            tuple(z.shape),
            "r_mean",
            float(z.norm(dim=-1).mean()),
            "r_max",
            float(z.norm(dim=-1).max()),
            "finite",
            bool(torch.isfinite(z).all()),
        )
    print("SMOKE_PASS")


if __name__ == "__main__":
    main()
