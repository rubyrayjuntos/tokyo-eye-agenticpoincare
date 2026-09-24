"""Pool-frontend backbone train mode: the known override, and the card's fix.

Standing rule: prose without a failing test always loses to a convenient default.

``EquiformerPoolFrontend.train()`` ends with ``self._backbone.eval()``
unconditionally, so even with ``freeze_backbone=False`` the EquiformerV3
backbone's own regularization (7 attention-weight ``Dropout`` p=0.1 + 7
``GraphDropPath`` 0.05) is OFF during "training". The first 12-fold pool card
(``lr_1e-4`` arm, FAIL_NO_SIGNAL) ran that way. See lesson
``L-backbone-eval-mode-during-train`` in ``data/gates/tokyo_eye_equ_agenda.json``.

Builds the (cold-init) model on CPU; skipped if the pool-frontend extras
(``requirements-diagnostics-pool-frontend.txt`` + torch-cluster) are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch = pytest.importorskip("torch")
for _mod in ("torch_cluster", "torch_scatter", "ase", "lmdb", "e3nn"):
    pytest.importorskip(_mod)

from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system  # noqa: E402
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map  # noqa: E402

import wrap1_zhyp_m2_pool_lr as R  # noqa: E402

N_REG_MODULES = 14  # 7 attention Dropout (p=0.1) + 7 GraphDropPath (0.05)


def _build():
    cfg = load_weight_map(REPO / DEFAULT_WEIGHT_MAP)
    system, _ = build_equiformer_pool_system(
        cfg, equiformer_ckpt=None, device=torch.device("cpu"),
        freeze_backbone=False, max_neighbors=16, cold_init=True,
    )
    return system


def test_default_pool_frontend_train_leaves_backbone_regularization_off() -> None:
    """KNOWN behavior, pinned on purpose.

    If this starts failing because ``EquiformerPoolFrontend.train()`` was fixed,
    that is good -- but the sealed ``lr_1e-4`` / ``sealed`` arms were run with
    regularization off, so update their scoped verdict lines and this test together.
    """
    system = _build()
    system.train()
    live, total = R.backbone_reg_modules_live(system)
    assert total == N_REG_MODULES
    assert live == 0, "backbone regularization unexpectedly live under default train()"
    with pytest.raises(RuntimeError, match="backbone regularization not active"):
        R.assert_backbone_regularization_live(system)


def test_apply_backbone_train_mode_activates_all_and_survives_retrain() -> None:
    system = _build()
    R.apply_backbone_train_mode(system)

    system.train()
    assert R.backbone_reg_modules_live(system) == (N_REG_MODULES, N_REG_MODULES)
    assert R.assert_backbone_regularization_live(system) == (N_REG_MODULES, N_REG_MODULES)

    system.eval()  # evaluation must stay deterministic
    assert R.backbone_reg_modules_live(system) == (0, N_REG_MODULES)

    # what the runner does around the assembly gate (scan_forward -> eval) and every step (train)
    system.train()
    assert R.backbone_reg_modules_live(system)[0] == N_REG_MODULES
    system.frontend.train(True)
    system.train()
    system.train()
    assert R.backbone_reg_modules_live(system)[0] == N_REG_MODULES


def test_only_bbtrain_arm_enables_backbone_train_mode_and_nothing_else_differs() -> None:
    base, arm = R.ARMS["lr_1e-4"], R.ARMS["lr_1e-4_bbtrain"]
    assert base["backbone_train_mode"] is False
    assert R.ARMS["sealed"]["backbone_train_mode"] is False
    assert arm["backbone_train_mode"] is True
    differing = {k for k in base if base[k] != arm[k]}
    assert differing == {"backbone_train_mode", "prereg"}, (
        "single-variable isolation broken: arms differ in " + str(sorted(differing))
    )
    # the two original arms share one prereg; the new card has its own
    assert R.ARMS["sealed"]["prereg"] == base["prereg"] != arm["prereg"]
