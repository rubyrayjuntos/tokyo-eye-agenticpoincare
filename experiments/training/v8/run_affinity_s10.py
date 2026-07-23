#!/usr/bin/env python3
"""TokyoEye-v8 Sprint 10 — protein-only affinity training.

Modes
-----
``head_only``: freeze frontend + hyp trunk + MoE; train PocketGatedAffinityHead.
``finetune_hyp`` (rematch): freeze Equiformer / SE(3) frontend; train projector,
  HypGraphAttention, MoE, mechanism head (aux), and affinity head.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from science.tokyo_eye.v8.affinity_head import (
    JointPocketAffinityHead,
    PocketGatedAffinityHead,
)
from science.tokyo_eye.v8.biophysics import parse_residue_records_from_pdb_chain
from science.tokyo_eye.v8.equiformer_frontend import (
    StubEquiformerFrontend,
    TokyoEyeV8WithFrontend,
    load_weight_map,
)
from science.tokyo_eye.v8.ligand_interface import (
    build_r6_edges,
    extract_ligand_hetatm,
    ligand_feature_matrix,
    residue_proxy_coords,
)
from science.tokyo_eye.v8.loader import DEFAULT_PDB_DIR, ensure_pdb_cached, load_structure_batch
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
from science.tokyo_eye.v8.pdbbind_loader import entries_from_split_manifest
from science.tokyo_eye.v8.seq_cluster import assert_no_core_leak

AffinityHead = PocketGatedAffinityHead | JointPocketAffinityHead

DEFAULT_INIT = Path(
    "checkpoints/v8/runs/tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt"
)
DEFAULT_SPLITS = Path("manifests/v8_pdbbind_refined_cluster30_v1.json")
DEFAULT_WEIGHT_MAP = Path("science/tokyo_eye/v8/configs/equiformer_v3_weight_map.json")
HEAD_ONLY_CORE_BASELINE = 0.054  # subset head_only Core Pearson floor for rematch saves


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def freeze_geometry_banks(system: TokyoEyeV8WithFrontend) -> dict[str, int]:
    """Strict head_only isolation: zero grad on frontend + spine geometry/MoE."""
    frozen = 0
    targets: list[nn.Module] = [system.frontend]
    bank = getattr(system.frontend, "backbone", None)
    if isinstance(bank, nn.Module):
        targets.append(bank)
    spine = system.spine
    for name in (
        "projector",
        "attn_layers",
        "moe",
        "euc_skip",
        "sdrp_head",
        "evidential_head",
        "mechanism_head",
    ):
        mod = getattr(spine, name, None)
        if isinstance(mod, nn.Module):
            targets.append(mod)
    for mod in targets:
        for param in mod.parameters():
            param.requires_grad = False
            frozen += 1
    return {"n_frozen_tensors": frozen, "mode": "head_only"}


def configure_finetune_hyp(system: TokyoEyeV8WithFrontend) -> dict[str, Any]:
    """Rematch: freeze physical frontend; unfreeze hyp trunk + MoE + mech head."""
    frozen = 0
    trainable = 0
    # Entire frontend (SE(3)-lite + Equiformer bank) stays frozen
    for param in system.frontend.parameters():
        param.requires_grad = False
        frozen += 1
    spine = system.spine
    train_names = ("projector", "attn_layers", "moe", "euc_skip", "mechanism_head")
    freeze_names = ("sdrp_head", "evidential_head")
    for name in train_names:
        mod = getattr(spine, name, None)
        if isinstance(mod, nn.Module):
            for param in mod.parameters():
                param.requires_grad = True
                trainable += 1
    for name in freeze_names:
        mod = getattr(spine, name, None)
        if isinstance(mod, nn.Module):
            for param in mod.parameters():
                param.requires_grad = False
                frozen += 1
    return {
        "mode": "finetune_hyp",
        "n_frozen_tensors": frozen,
        "n_trainable_spine_tensors": trainable,
        "train_modules": list(train_names),
    }


class AffinityEntryDataset(Dataset):
    """Lazy PDB-Bind protein batches; soft-skips unloadable structures."""

    def __init__(
        self,
        entries: list[dict[str, Any]],
        *,
        pdb_dir: Path,
        device: str = "cpu",
        use_graph_cache: bool = True,
        max_entries: int | None = None,
        require_ligand: bool = False,
    ) -> None:
        self.entries = list(entries)
        if max_entries is not None:
            self.entries = self.entries[: int(max_entries)]
        self.pdb_dir = Path(pdb_dir)
        self.device = device
        self.use_graph_cache = bool(use_graph_cache)
        self.require_ligand = bool(require_ligand)
        self._ok_cache: dict[int, dict[str, Any] | None] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _attach_ligand_r6(self, batch: dict[str, Any], pdb_id: str, chain: str) -> dict[str, Any] | None:
        """On-the-fly ligand + R6 (never writes protein graph cache)."""
        pdb_path = ensure_pdb_cached(pdb_id, self.pdb_dir)
        lig = extract_ligand_hetatm(pdb_path)
        if lig is None:
            if self.require_ligand:
                raise ValueError("no ligand after HETATM sanitize")
            batch["lig_feat"] = torch.zeros(0, 10, dtype=torch.float32)
            batch["edge_index_r6"] = torch.zeros(2, 0, dtype=torch.long)
            batch["r6_empty"] = 1
            batch["n_lig"] = 0
            batch["n_r6_edges"] = 0
            return batch
        records = parse_residue_records_from_pdb_chain(pdb_path, chain)
        records = [r for r in records if r.get_atom("CA") is not None]
        if len(records) != int(batch["num_nodes"]):
            # Align to CA-complete count used by protein batch
            if len(records) < 1:
                raise ValueError("no CA residues for R6 proxy")
        proxy = residue_proxy_coords(records)
        if proxy.shape[0] != int(batch["num_nodes"]):
            # Fallback: use CA coords already in batch (cache-aligned)
            proxy = batch["x"].detach().cpu().numpy().astype(np.float32)
        feats = ligand_feature_matrix(lig)
        ei, meta = build_r6_edges(proxy, lig.coords, cutoff=4.5)
        batch["lig_feat"] = torch.from_numpy(feats)
        batch["edge_index_r6"] = torch.from_numpy(ei.astype(np.int64))
        batch["r6_empty"] = int(meta["r6_empty"])
        batch["n_lig"] = int(lig.n_atoms)
        batch["n_r6_edges"] = int(meta["n_edges"])
        return batch

    def _load(self, idx: int) -> dict[str, Any] | None:
        if idx in self._ok_cache:
            return self._ok_cache[idx]
        row = self.entries[idx]
        pdb_id = str(row["pdb_id"]).upper()
        chain = str(row.get("chain") or "A")
        try:
            batch = load_structure_batch(
                pdb_id,
                chain,
                pdb_dir=self.pdb_dir,
                device="cpu",
                use_graph_cache=self.use_graph_cache,
            )
            batch["affinity"] = float(row["affinity"])
            batch["pdb_id"] = pdb_id
            if self.require_ligand:
                batch = self._attach_ligand_r6(batch, pdb_id, chain)
                assert batch is not None
            else:
                batch["lig_feat"] = torch.zeros(0, 10, dtype=torch.float32)
                batch["edge_index_r6"] = torch.zeros(2, 0, dtype=torch.long)
                batch["r6_empty"] = 1
                batch["n_lig"] = 0
                batch["n_r6_edges"] = 0
            self._ok_cache[idx] = batch
            return batch
        except Exception as exc:  # noqa: BLE001 — soft skip missing/corrupt PDB
            print(f"[s10] skip {pdb_id}:{chain} ({exc})")
            self._ok_cache[idx] = None
            return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        batch = self._load(idx)
        if batch is None:
            return {"_skip": True, "pdb_id": self.entries[idx]["pdb_id"]}
        return batch


def _run_affinity_head(
    head: AffinityHead,
    z_hyp: torch.Tensor,
    *,
    mechanism_score: torch.Tensor,
    dehydron_labels: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
    joint_head: bool,
) -> dict[str, torch.Tensor]:
    if joint_head:
        lig = batch["lig_feat"].to(device)
        r6 = batch["edge_index_r6"].to(device)
        return head(
            z_hyp,
            mechanism_score=mechanism_score,
            dehydron_labels=dehydron_labels,
            lig_feat=lig,
            edge_index_r6=r6,
        )
    return head(
        z_hyp,
        mechanism_score=mechanism_score,
        dehydron_labels=dehydron_labels,
    )


def _collate_one(batch_list: list[dict[str, Any]]) -> dict[str, Any] | None:
    assert len(batch_list) == 1
    item = batch_list[0]
    if item.get("_skip"):
        return None
    return item


def build_system(cfg: dict[str, Any], device: torch.device) -> TokyoEyeV8WithFrontend:
    frontend = StubEquiformerFrontend(
        in_dim=3,
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        num_backbone_blocks=int(cfg.get("num_backbone_blocks", 7)),
        live_backbone=True,
    )
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_attn_layers=int(cfg.get("num_attn_layers", 2)),
        num_relations=int(cfg.get("num_relations", 6)),
        num_sdrp_classes=int(cfg.get("num_sdrp_classes", 5)),
        gate_hidden=int(cfg.get("gate_hidden", 16)),
        c=float(cfg.get("curvature_c", cfg.get("c", 1.0))),
        eps=float(cfg.get("eps", 1e-5)),
        moe_temperature=float(cfg.get("gumbel_tau_start", 1.0)),
    )
    return TokyoEyeV8WithFrontend(frontend, spine).to(device)


def pearson_spearman_rmse(
    pred: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if pred.size < 2:
        return {"pearson": float("nan"), "spearman": float("nan"), "rmse": float("nan")}
    p = pred - pred.mean()
    t = target - target.mean()
    denom = float(np.linalg.norm(p) * np.linalg.norm(t))
    pearson = float((p * t).sum() / denom) if denom > 0 else float("nan")
    pr = pred.argsort().argsort().astype(np.float64)
    tr = target.argsort().argsort().astype(np.float64)
    pr -= pr.mean()
    tr -= tr.mean()
    denom_s = float(np.linalg.norm(pr) * np.linalg.norm(tr))
    spearman = float((pr * tr).sum() / denom_s) if denom_s > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    return {"pearson": pearson, "spearman": spearman, "rmse": rmse}


@torch.no_grad()
def evaluate_split(
    system: TokyoEyeV8WithFrontend,
    head: AffinityHead,
    loader: DataLoader,
    device: torch.device,
    *,
    tau_ceiling: float = 0.995,
    joint_head: bool = False,
) -> dict[str, float]:
    system.eval()
    head.eval()
    preds: list[float] = []
    targets: list[float] = []
    n_skip = 0
    r6_empty_count = 0
    n_lig_atoms = 0
    n_r6_edges = 0
    for batch in loader:
        if batch is None:
            n_skip += 1
            continue
        x = batch["x"].to(device)
        ei = batch["edge_index"].to(device)
        et = batch["edge_type"].to(device)
        dehyd = batch["dehydron_labels"].to(device)
        y = float(batch["affinity"])
        out = system(x, ei, et, tau_ceiling=tau_ceiling)
        aff = _run_affinity_head(
            head,
            out["z_hyp"],
            mechanism_score=out["mechanism_score"],
            dehydron_labels=dehyd,
            batch=batch,
            device=device,
            joint_head=joint_head,
        )
        preds.append(float(aff["affinity_pred"].detach().cpu()))
        targets.append(y)
        r6_empty_count += int(batch.get("r6_empty", 0))
        n_lig_atoms += int(batch.get("n_lig", 0))
        n_r6_edges += int(batch.get("n_r6_edges", 0))
    metrics = pearson_spearman_rmse(np.array(preds), np.array(targets))
    metrics["n"] = float(len(preds))
    metrics["n_skip"] = float(n_skip)
    metrics["r6_empty_count"] = float(r6_empty_count)
    metrics["lig_atoms_mean"] = float(n_lig_atoms) / max(len(preds), 1)
    metrics["r6_edges_mean"] = float(n_r6_edges) / max(len(preds), 1)
    return metrics


def _dehydron_aux_loss(
    mechanism_score: torch.Tensor, dehydron_labels: torch.Tensor
) -> torch.Tensor:
    """BCE-with-logits: mechanism score ↔ dehydron incidence (λ_aux target)."""
    return F.binary_cross_entropy_with_logits(
        mechanism_score.reshape(-1),
        dehydron_labels.reshape(-1).to(dtype=mechanism_score.dtype),
    )


def train_epoch_head_only(
    system: TokyoEyeV8WithFrontend,
    head: AffinityHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    tau_ceiling: float = 0.995,
    joint_head: bool = False,
) -> dict[str, float]:
    system.eval()
    head.train()
    total = 0.0
    n = 0
    for batch in loader:
        if batch is None:
            continue
        x = batch["x"].to(device)
        ei = batch["edge_index"].to(device)
        et = batch["edge_type"].to(device)
        dehyd = batch["dehydron_labels"].to(device)
        y = torch.tensor(float(batch["affinity"]), device=device)
        with torch.no_grad():
            out = system(x, ei, et, tau_ceiling=tau_ceiling)
        aff = _run_affinity_head(
            head,
            out["z_hyp"].detach(),
            mechanism_score=out["mechanism_score"].detach(),
            dehydron_labels=dehyd,
            batch=batch,
            device=device,
            joint_head=joint_head,
        )
        loss = F.mse_loss(aff["affinity_pred"], y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        n += 1
    return {"train_mse": total / max(n, 1), "n_train": float(n), "train_aux": 0.0}


def train_epoch_finetune_hyp(
    system: TokyoEyeV8WithFrontend,
    head: AffinityHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    tau_ceiling: float = 0.995,
    aux_coeff: float = 0.10,
    cv_coeff: float = 1.0,
    moe_quota_coeff: float = 1.0,
    max_grad_norm: float = 1.0,
    joint_head: bool = False,
) -> dict[str, float]:
    system.train()
    system.frontend.eval()
    head.train()
    total = 0.0
    total_aux = 0.0
    total_mse = 0.0
    n = 0
    for batch in loader:
        if batch is None:
            continue
        x = batch["x"].to(device)
        ei = batch["edge_index"].to(device)
        et = batch["edge_type"].to(device)
        dehyd = batch["dehydron_labels"].to(device)
        y = torch.tensor(float(batch["affinity"]), device=device)
        out = system(x, ei, et, tau_ceiling=tau_ceiling)
        aff = _run_affinity_head(
            head,
            out["z_hyp"],
            mechanism_score=out["mechanism_score"],
            dehydron_labels=dehyd,
            batch=batch,
            device=device,
            joint_head=joint_head,
        )
        loss_mse = F.mse_loss(aff["affinity_pred"], y)
        loss_aux = _dehydron_aux_loss(out["mechanism_score"], dehyd)
        moe_aux = out.get("moe_aux") or {}
        loss_cv = moe_aux.get("cv_loss")
        loss_quota = moe_aux.get("quota_loss")
        if loss_cv is None:
            loss_cv = torch.zeros((), device=device)
        if loss_quota is None:
            loss_quota = torch.zeros((), device=device)
        loss = (
            loss_mse
            + float(aux_coeff) * loss_aux
            + float(cv_coeff) * loss_cv
            + float(moe_quota_coeff) * loss_quota
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in list(system.parameters()) + list(head.parameters()) if p.requires_grad],
            max_norm=float(max_grad_norm),
        )
        optimizer.step()
        total += float(loss.detach())
        total_aux += float(loss_aux.detach())
        total_mse += float(loss_mse.detach())
        n += 1
    return {
        "train_loss": total / max(n, 1),
        "train_mse": total_mse / max(n, 1),
        "train_aux": total_aux / max(n, 1),
        "n_train": float(n),
    }


def build_optimizer(
    mode: str,
    system: TokyoEyeV8WithFrontend,
    head: AffinityHead,
    *,
    lr_head: float,
    lr_hyperbolic: float,
) -> torch.optim.Optimizer:
    if mode == "head_only":
        params = [p for p in head.parameters() if p.requires_grad]
        assert params, "affinity head has no trainable params"
        for p in system.parameters():
            assert not p.requires_grad, "geometry bank still requires_grad"
        return torch.optim.Adam(params, lr=float(lr_head))

    hyp_params = [p for p in system.spine.parameters() if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]
    assert hyp_params, "finetune_hyp: no trainable spine params"
    assert head_params, "finetune_hyp: no trainable head params"
    for p in system.frontend.parameters():
        assert not p.requires_grad, "frontend must stay frozen in finetune_hyp"
    return torch.optim.Adam(
        [
            {"params": hyp_params, "lr": float(lr_hyperbolic), "name": "hyperbolic"},
            {"params": head_params, "lr": float(lr_head), "name": "affinity_head"},
        ]
    )


def save_checkpoint(
    path: Path,
    *,
    system: TokyoEyeV8WithFrontend,
    head: AffinityHead,
    epoch: int,
    val: dict[str, float],
    mode: str,
    splits: str,
    init_ckpt: str,
    tag: str,
) -> None:
    torch.save(
        {
            "model": system.state_dict(),
            "affinity_head": head.state_dict(),
            "epoch": epoch,
            "val": val,
            "mode": mode,
            "splits": splits,
            "init_ckpt": init_ckpt,
            "tag": tag,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sprint 10 affinity train / rematch")
    p.add_argument(
        "--mode",
        type=str,
        default="head_only",
        choices=["head_only", "finetune_hyp"],
    )
    p.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--init-ckpt", type=Path, default=DEFAULT_INIT)
    p.add_argument("--affinity-init", type=Path, default=None, help="Optional affinity head warm-start")
    p.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    p.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/v8/runs"))
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3, help="Affinity head LR")
    p.add_argument("--lr-hyperbolic", type=float, default=3e-4)
    p.add_argument("--aux-coeff", type=float, default=0.10, help="λ_aux dehydron BCE")
    p.add_argument("--cv-coeff", type=float, default=1.0)
    p.add_argument("--moe-quota-coeff", type=float, default=1.0)
    p.add_argument(
        "--ckpt-baseline",
        type=float,
        default=HEAD_ONLY_CORE_BASELINE,
        help="Save incremental improve-* ckpts when val_pearson exceeds this",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-core", type=int, default=None)
    p.add_argument("--cached-only", action="store_true")
    p.add_argument(
        "--joint-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use JointPocketAffinityHead with on-the-fly R6 (Sprint 10.1)",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-graph-cache", action="store_true")
    p.add_argument("--tau-ceiling", type=float, default=0.995)
    return p.parse_args()


def _filter_cached(entries: list[dict[str, Any]], pdb_dir: Path) -> list[dict[str, Any]]:
    out = []
    for row in entries:
        pid = str(row["pdb_id"]).upper()
        path = Path(pdb_dir) / f"{pid}.pdb"
        if path.is_file() and path.stat().st_size > 0:
            out.append(row)
    return out


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.max_train = args.max_train or 8
        args.max_val = args.max_val or 4
        args.max_core = args.max_core or 8
    _set_seed(args.seed)
    mode = str(args.mode)

    splits = entries_from_split_manifest(args.splits)
    assert_no_core_leak(
        [r["pdb_id"] for r in splits["train"]],
        [r["pdb_id"] for r in splits["val"]],
        [r["pdb_id"] for r in splits["core_test"]],
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[s10] WARN: CUDA unavailable — cpu")
        args.device = "cpu"
    device = torch.device(args.device)
    cfg = load_weight_map(args.weight_map)
    system = build_system(cfg, device)
    if args.init_ckpt.is_file():
        blob = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        missing, unexpected = system.load_state_dict(state, strict=False)
        print(
            f"[s10] init_ckpt={args.init_ckpt} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    else:
        print(f"[s10] WARN: init ckpt missing ({args.init_ckpt}) — random spine")

    if mode == "head_only":
        freeze_info = freeze_geometry_banks(system)
    else:
        freeze_info = configure_finetune_hyp(system)
    print(f"[s10] freeze_config={json.dumps(freeze_info)}")

    head = (
        JointPocketAffinityHead(
            int(cfg["hidden_dim"]),
            gate_hidden=32,
            ffn_hidden=64,
            c=float(cfg.get("curvature_c", cfg.get("c", 1.0))),
            eps=float(cfg.get("eps", 1e-5)),
        )
        if bool(args.joint_head)
        else PocketGatedAffinityHead(
            int(cfg["hidden_dim"]),
            gate_hidden=32,
            ffn_hidden=64,
            c=float(cfg.get("curvature_c", cfg.get("c", 1.0))),
            eps=float(cfg.get("eps", 1e-5)),
        )
    ).to(device)
    print(f"[s10] joint_head={bool(args.joint_head)} head={type(head).__name__}")
    if args.affinity_init is not None and Path(args.affinity_init).is_file():
        ablob = torch.load(args.affinity_init, map_location=device, weights_only=False)
        hstate = ablob.get("affinity_head", ablob)
        missing, unexpected = head.load_state_dict(hstate, strict=False)
        print(
            f"[s10] affinity_init={args.affinity_init} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    optimizer = build_optimizer(
        mode,
        system,
        head,
        lr_head=float(args.lr),
        lr_hyperbolic=float(args.lr_hyperbolic),
    )
    print(
        f"[s10] mode={mode} lr_head={args.lr} lr_hyp={args.lr_hyperbolic} "
        f"aux_coeff={args.aux_coeff}"
    )

    use_cache = not args.no_graph_cache
    train_rows, val_rows, core_rows = (
        splits["train"],
        splits["val"],
        splits["core_test"],
    )
    if args.cached_only:
        train_rows = _filter_cached(train_rows, args.pdb_dir)
        val_rows = _filter_cached(val_rows, args.pdb_dir)
        core_rows = _filter_cached(core_rows, args.pdb_dir)
        print(
            f"[s10] cached-only train={len(train_rows)} "
            f"val={len(val_rows)} core={len(core_rows)}"
        )
    train_ds = AffinityEntryDataset(
        train_rows,
        pdb_dir=args.pdb_dir,
        use_graph_cache=use_cache,
        max_entries=args.max_train,
        require_ligand=bool(args.joint_head),
    )
    val_ds = AffinityEntryDataset(
        val_rows,
        pdb_dir=args.pdb_dir,
        use_graph_cache=use_cache,
        max_entries=args.max_val,
        require_ligand=bool(args.joint_head),
    )
    core_ds = AffinityEntryDataset(
        core_rows,
        pdb_dir=args.pdb_dir,
        use_graph_cache=use_cache,
        max_entries=args.max_core,
        require_ligand=bool(args.joint_head),
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=_collate_one
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=_collate_one
    )
    core_loader = DataLoader(
        core_ds, batch_size=1, shuffle=False, collate_fn=_collate_one
    )

    run_name = args.run_name or (
        f"tokyo_eye_v8_affinity_s10_{mode}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    improve_dir = out_dir / "improve_ckpts"
    improve_dir.mkdir(exist_ok=True)

    history: list[dict[str, Any]] = []
    best_val = -math.inf
    best_path = out_dir / "v8_affinity_best.pt"
    baseline = float(args.ckpt_baseline)
    n_improve_saves = 0

    for epoch in range(int(args.epochs)):
        if mode == "head_only":
            tr = train_epoch_head_only(
                system,
                head,
                train_loader,
                optimizer,
                device,
                tau_ceiling=float(args.tau_ceiling),
                joint_head=bool(args.joint_head),
            )
        else:
            tr = train_epoch_finetune_hyp(
                system,
                head,
                train_loader,
                optimizer,
                device,
                tau_ceiling=float(args.tau_ceiling),
                aux_coeff=float(args.aux_coeff),
                cv_coeff=float(args.cv_coeff),
                moe_quota_coeff=float(args.moe_quota_coeff),
                joint_head=bool(args.joint_head),
            )
        val = evaluate_split(
            system,
            head,
            val_loader,
            device,
            tau_ceiling=float(args.tau_ceiling),
            joint_head=bool(args.joint_head),
        )
        row = {"epoch": epoch, **tr, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(json.dumps(row, indent=None), flush=True)
        score = val.get("pearson", float("nan"))
        if score == score and score > best_val:
            best_val = score
            save_checkpoint(
                best_path,
                system=system,
                head=head,
                epoch=epoch,
                val=val,
                mode=mode,
                splits=str(args.splits),
                init_ckpt=str(args.init_ckpt),
                tag="best_val_pearson",
            )
            # Incremental rematch artifacts once past head_only Core baseline
            if score > baseline:
                n_improve_saves += 1
                tag = f"e{epoch:03d}_r{score:.4f}"
                improve_path = improve_dir / f"v8_affinity_improve_{tag}.pt"
                save_checkpoint(
                    improve_path,
                    system=system,
                    head=head,
                    epoch=epoch,
                    val=val,
                    mode=mode,
                    splits=str(args.splits),
                    init_ckpt=str(args.init_ckpt),
                    tag=tag,
                )
                # Keep a rolling "latest improve over baseline" pointer
                latest = out_dir / "v8_affinity_improve_latest.pt"
                shutil.copy2(improve_path, latest)
                print(
                    f"[s10] ckpt improve over baseline={baseline:.3f}: "
                    f"{improve_path.name} (n={n_improve_saves})",
                    flush=True,
                )

    if best_path.is_file():
        blob = torch.load(best_path, map_location=device, weights_only=False)
        system.load_state_dict(blob["model"], strict=False)
        head.load_state_dict(blob["affinity_head"])
    core = evaluate_split(
        system,
        head,
        core_loader,
        device,
        tau_ceiling=float(args.tau_ceiling),
        joint_head=bool(args.joint_head),
    )
    summary = {
        "run_name": run_name,
        "mode": mode,
        "joint_head": bool(args.joint_head),
        "best_val_pearson": best_val,
        "ckpt_baseline": baseline,
        "n_improve_saves": n_improve_saves,
        "core": core,
        "gate_core_pearson_ge_0_40": bool(
            core.get("pearson", float("nan")) >= 0.40
            if core.get("pearson") == core.get("pearson")
            else False
        ),
        "n_history": len(history),
        "aux_coeff": float(args.aux_coeff) if mode == "finetune_hyp" else 0.0,
        "lr_hyperbolic": float(args.lr_hyperbolic) if mode == "finetune_hyp" else 0.0,
        "lr_head": float(args.lr),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save_checkpoint(
        out_dir / "v8_affinity_last.pt",
        system=system,
        head=head,
        epoch=int(args.epochs) - 1,
        val=core,
        mode=mode,
        splits=str(args.splits),
        init_ckpt=str(args.init_ckpt),
        tag="last",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
