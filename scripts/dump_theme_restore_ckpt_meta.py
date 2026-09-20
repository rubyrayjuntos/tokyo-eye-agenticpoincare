import json
from pathlib import Path
import torch
p = Path("checkpoints/tokyoeye/runs/eqf_equ_theme_restore_20260916/tokyoeye_best.pt")
blob = torch.load(p, map_location="cpu", weights_only=False)
meta = {k: blob[k] for k in blob if k not in ("model", "optimizer")}
def conv(x):
    import torch as T
    if isinstance(x, T.Tensor):
        if x.numel() == 1:
            return x.item()
        return f"Tensor{tuple(x.shape)}"
    if isinstance(x, dict):
        return {kk: conv(vv) for kk, vv in x.items()}
    if isinstance(x, (list, tuple)):
        return [conv(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)
print(json.dumps(conv(meta), indent=2))
