import torch
from pathlib import Path
from equiformer_v3_model.equiformer_v3 import EquiformerV3_OC

kwargs = dict(
    use_pbc=False,
    use_pbc_single=True,  # MPtrj often True
    otf_graph=True,
    regress_forces=False,
    regress_stress=False,
    direct_prediction=False,  # gradient finetune
    max_neighbors=300,
    max_radius=8.0,  # check yaml
    num_radial_basis=10,
    num_layers=7,
    num_channels=128,
    attn_hidden_channels=32,
    num_heads=8,
    attn_alpha_channels=64,
    ffn_hidden_channels=512,
    norm_type="merge_layer_norm",
    lmax=4,
    mmax=2,
    attn_grid_resolution_list=[14, 8],
    ffn_grid_resolution_list=[14, 14],
)
print("building with MPtrj YAML dims...")
model = EquiformerV3_OC(**kwargs)
print("params M", sum(p.numel() for p in model.parameters())/1e6)
ckpt = Path("checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt")
blob = torch.load(ckpt, map_location="cpu", weights_only=False)
state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
if any(str(k).startswith("module.") for k in state):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print("missing", len(missing), "unexpected", len(unexpected))
print("missing sample", missing[:12])
print("unexpected sample", unexpected[:12])
# count size mismatches by trying strict
bad = 0
for k, v in state.items():
    if k in model.state_dict() and tuple(model.state_dict()[k].shape) != tuple(v.shape):
        bad += 1
        if bad <= 5:
            print("mismatch", k, tuple(v.shape), "vs", tuple(model.state_dict()[k].shape))
print("n_shape_mismatch", bad)
