"""Trunk-wide pure-hyp veto (freeze reconciliation addendum §2.4).

This is a **live forward tracer**, not a module-tree enumeration:

1. ``nn.Linear`` forward hooks (covers Linear inside ``nn.Sequential``).
2. Functional patches for ``F.linear`` / ``torch.matmul`` / ``torch.mm`` /
   ``torch.addmm`` — the addendum's separate "Euclidean matmul" case.
   A bare ``F.linear(z, W)`` or ``z @ W`` on an ``nn.Parameter`` with no
   ``nn.Linear`` wrapper is a violation. ``isinstance(..., nn.Linear)``
   alone is **not** sufficient.
3. ``log₀ → Linear/matmul → exp₀`` sandwiches, regardless of filename.

Allowed post-lift multiply: origin-fixing Poincaré isometries — square
``O(d)`` / ``SO(d)`` acting on ball coordinates (``GyroOrthogonalMap``:
QR → ``SO(d)``, then ``z @ R``). Orthogonality is ``||RᵀR − I|| ≤ atol``
with ``atol=1e-4`` (QR residuals are ~1e-6; a smuggled shear at 1e-3
fails). Unconstrained ``z @ W`` is ``euclidean_matmul_on_ball``.

A gate-weighted ambient sum of two ball expert outputs is
``euclidean_mix_on_ball``. Scaling a **single** selected expert by a
non-{0,1} confidence (``z * 0.87``) is ``euclidean_scale_on_ball``.
Hard one-hot ``{0,1}`` masks are identity-or-zero, not a scale.

Pre-lift Equiformer / projector / ``euc_skip`` Linears are skipped by
module name (Euclidean frontend). Nested ops *inside the live*
``torch.linalg.qr`` call are ignored **only when the left operand is
not an open-ball batch**. Listed Möbius / Einstein / ball-projection
kernels are skipped by function-object identity (``__code__``) — an
**opt-in allowlist**. Unlisted functions are scanned normally.

**Ball detection is a numeric heuristic, not a type check.**
``_is_open_ball_batch`` returns True iff every row's ‖·‖ lies in
``(_BALL_MIN_SIGNAL, _BALL_MAX)`` = ``(1e-4, 0.999)``. There is no
``ManifoldTensor`` / provenance tag. Consequences:

- **Near-origin false negative (known limitation):** a genuine ball batch
  with all radii ≤ 1e-4 (fresh lift, small init) is *not* classified as
  ball, so ``Linear``-on-ball at that moment is invisible. Documented by
  ``test_near_origin_linear_on_ball_is_known_limitation``.
- **Euclidean-feature false positive:** a gate / feature vector whose
  components all happen to fall in that band can be mis-flagged as ball.
  Fail-safe (blocks rather than passes bad ops). **Author convention:**
  concatenate a constant marker column ``2.0`` (norm ≫ ``_BALL_MAX``)
  onto Euclidean gate features before any ``nn.Linear`` — see
  ``moe.topology_gate_features`` / ``TopologyAwareHardMoE.forward``.

Filename greps are forbidden as a substitute (addendum §2.4 / §2.5).

Do not attach this check to MLflow promote stamps until
``tests/v8/test_pure_hyp_pass.py`` shows the known-bad fixtures fail and
the known-good gyro fixture passes.
"""

from __future__ import annotations

import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from science.tokyo_eye.v8 import attention as attn_mod
from science.tokyo_eye.v8.freeze_reconciliation import PURE_HYP_PASS_VERSION

_BALL_MAX = 1.0 - 1e-3
_BALL_MIN_SIGNAL = 1e-4
# Above float32 QR residual (~1e-6); below a visible shear (~1e-3).
_ORTH_ATOL = 1e-4
_skip_linalg = ContextVar("pure_hyp_skip_linalg", default=0)
_skip_prelift = ContextVar("pure_hyp_skip_prelift", default=0)
_skip_manifold = ContextVar("pure_hyp_skip_manifold", default=0)
_skip_isometry = ContextVar("pure_hyp_skip_isometry", default=0)

# Opt-in: only these ``attn_mod`` callables wrap with ``_skip_manifold``.
# Unlisted functions are traced like any other Python.
MANIFOLD_FORMULA_NAMES: tuple[str, ...] = (
    "mobius_add",
    "gyroscalar_mul",
    "einstein_midpoint",
    "sparse_einstein_klein_aggregate",
    "project_to_ball",
    "poincare_to_klein",
    "klein_to_poincare",
    "lorentz_factor",
    "poincare_dist",
    "clamp_ball_radius",
    "exp_map_zero",
    "log_map_zero",
)


@dataclass(frozen=True)
class PureHypViolation:
    kind: str
    module_name: str
    detail: str


@dataclass
class PureHypReport:
    passed: bool
    violations: list[PureHypViolation] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pure_hyp_pass": bool(self.passed),
            "pure_hyp_violations": [
                {
                    "kind": v.kind,
                    "module": v.module_name,
                    "detail": v.detail,
                }
                for v in self.violations
            ]
            or ["none"],
        }


def _is_open_ball_batch(x: torch.Tensor) -> bool:
    """True when every row-vector looks like an open unit-ball point.

    Numeric heuristic only (see module docstring): all radii in
    ``(_BALL_MIN_SIGNAL, _BALL_MAX)``. Near-origin batches and
    Euclidean features that happen to land in-band are the documented
    edge cases; gate authors use a ``2.0`` marker column to stay outside.
    """
    if not torch.is_tensor(x) or x.ndim < 1 or x.numel() == 0:
        return False
    if not torch.isfinite(x).all():
        return False
    vec = x.reshape(-1, x.shape[-1]) if x.ndim >= 2 else x.reshape(1, -1)
    if vec.shape[-1] < 2:
        return False
    radii = torch.linalg.vector_norm(vec.float(), dim=-1)
    return bool(
        (radii < _BALL_MAX).all().item()
        and (radii.max() > _BALL_MIN_SIGNAL).item()
    )


def _is_pre_lift_linear(name: str) -> bool:
    """Equiformer / lift Linears act on Euclidean tensors (addendum §2.4).

    Match both nested children (``.projector.``) and the projector module
    itself when it is a trailing segment (``spine.projector``) — full-spine
    wraps name the lift that way; middle-only matching false-positived the
    pre-``exp₀`` tangent mix at ``projector.py`` (see
    ``tokyo_eye_equ_pure_hyp_full_spine_scan.json`` root_cause_analysis).
    """
    n = name or ""
    return (
        n == "euc_skip"
        or n.endswith(".euc_skip")
        or n == "projector"
        or n.startswith("projector.")
        or n.endswith(".projector")
        or ".projector." in n
        or n == "frontend"
        or n.startswith("frontend.")
        or n.endswith(".frontend")
        or ".frontend." in n
        or n.startswith("equiformer")
        or ".equiformer" in n
    )


def _is_pre_lift_scope(name: str) -> bool:
    """Skip functional Euclidean ops while the Equiformer / projector runs."""
    return _is_pre_lift_linear(name)


def _is_origin_isometry_matrix(m: torch.Tensor, *, atol: float | None = None) -> bool:
    """Square real matrix with ``MᵀM ≈ I`` (origin-fixing ball isometry)."""
    if not torch.is_tensor(m) or m.ndim != 2:
        return False
    if int(m.shape[0]) != int(m.shape[1]) or int(m.shape[0]) < 2:
        return False
    if not torch.isfinite(m).all():
        return False
    tol = _ORTH_ATOL if atol is None else float(atol)
    eye = torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
    tok = _skip_isometry.set(_skip_isometry.get() + 1)
    try:
        gram = m.transpose(0, 1) @ m
    finally:
        _skip_isometry.reset(tok)
    return bool(torch.allclose(gram, eye, atol=tol, rtol=0.0))


def _gram_inf(m: torch.Tensor) -> float:
    eye = torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
    return float((m.transpose(0, 1) @ m - eye).abs().max())


def _is_soft_expert_stack(t: torch.Tensor) -> bool:
    """``[N, E, d]`` ball rows with ≥2 nonzero experts — ambient mix, not Top-1."""
    if not torch.is_tensor(t) or t.ndim != 3:
        return False
    _n, n_exp, dim = t.shape
    if n_exp < 2 or dim < 2:
        return False
    radii = torch.linalg.vector_norm(t.float(), dim=-1)
    if not bool((radii < _BALL_MAX).all().item()):
        return False
    support = (radii > _BALL_MIN_SIGNAL).sum(dim=-1)
    return bool((support > 1).any().item())


def _is_hard_01_scale(scale: Any) -> bool:
    """True for a 0/1 mask (identity or zero), not a softmax confidence."""
    if not torch.is_tensor(scale):
        try:
            v = float(scale)
        except (TypeError, ValueError):
            return False
        return abs(v) <= 1e-5 or abs(v - 1.0) <= 1e-5
    t = scale.detach().float()
    if t.numel() == 0:
        return False
    near0 = t.abs() <= 1e-5
    near1 = (t - 1.0).abs() <= 1e-5
    return bool((near0 | near1).all().item())


def _is_broadcast_scale(scale: Any, ball: torch.Tensor) -> bool:
    """Scalar / ``[..., 1]`` / per-row multiplier, not an embedding-shaped mix."""
    if not torch.is_tensor(scale):
        try:
            float(scale)
            return True
        except (TypeError, ValueError):
            return False
    if scale.ndim == 0 or scale.numel() == 1:
        return True
    if scale.shape[-1] == 1:
        return True
    if scale.ndim + 1 == ball.ndim and scale.shape == ball.shape[:-1]:
        return True
    return False


def _unwrap_input(inp: Any) -> torch.Tensor | None:
    if isinstance(inp, (tuple, list)):
        if not inp:
            return None
        return inp[0] if torch.is_tensor(inp[0]) else None
    return inp if torch.is_tensor(inp) else None


class PureHypTracer:
    """Trace Linear modules **and** functional Euclidean maps for one forward."""

    def __init__(self) -> None:
        self.violations: list[PureHypViolation] = []
        self._log0_tensors: list[torch.Tensor] = []
        self._linear_from_log0: list[torch.Tensor] = []
        self._hooks: list[Any] = []
        self._patches: list[tuple[Any, str, Any]] = []

    def _same_storage(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        try:
            return a.data_ptr() == b.data_ptr() and a.shape == b.shape
        except RuntimeError:
            return False

    def _is_log0_output(self, x: torch.Tensor) -> bool:
        return any(self._same_storage(x, t) for t in self._log0_tensors)

    def _is_linear_from_log0(self, x: torch.Tensor) -> bool:
        return any(self._same_storage(x, t) for t in self._linear_from_log0)

    def _note(
        self, kind: str, module_name: str, detail: str, *, out: torch.Tensor | None = None
    ) -> None:
        self.violations.append(
            PureHypViolation(kind=kind, module_name=module_name, detail=detail)
        )
        if out is not None and torch.is_tensor(out) and kind in {
            "linear_on_log0",
            "euclidean_matmul_on_log0",
        }:
            self._linear_from_log0.append(out)

    def _on_linear(
        self, name: str, module: nn.Module, inp: Any, out: Any
    ) -> None:
        x = _unwrap_input(inp)
        if x is None:
            return
        if self._is_log0_output(x):
            self._note(
                "linear_on_log0",
                name or module.__class__.__name__,
                "nn.Linear applied to log_map_zero output (tangent workspace)",
                out=out if torch.is_tensor(out) else None,
            )
            return
        if _is_open_ball_batch(x):
            self._note(
                "linear_on_ball",
                name or module.__class__.__name__,
                (
                    f"{type(module).__name__} applied to open-ball tensor "
                    f"shape={tuple(x.shape)} max_radius="
                    f"{float(torch.linalg.vector_norm(x.reshape(-1, x.shape[-1]), dim=-1).max()):.4f}"
                ),
            )

    def _flag_functional_left(
        self,
        x: torch.Tensor,
        *,
        op: str,
        right: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> None:
        if _skip_prelift.get() > 0:
            return
        if self._is_log0_output(x):
            self._note(
                "euclidean_matmul_on_log0"
                if "matmul" in op or op in {"mm", "addmm", "Tensor.matmul"}
                else "linear_on_log0",
                op,
                f"{op} applied to log_map_zero output",
                out=out,
            )
            return
        if not _is_open_ball_batch(x):
            return
        if right is not None:
            tok = _skip_linalg.set(_skip_linalg.get() + 1)
            try:
                isometric = _is_origin_isometry_matrix(right)
            finally:
                _skip_linalg.reset(tok)
            if isometric:
                return
        kind = "linear_on_ball" if op == "F.linear" else "euclidean_matmul_on_ball"
        self._note(
            kind,
            op,
            f"{op} applied to open-ball tensor shape={tuple(x.shape)}",
        )

    def _wrap_log0(self, orig: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        tracer = self

        def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
            out = orig(*args, **kwargs)
            if torch.is_tensor(out):
                tracer._log0_tensors.append(out)
            return out

        return wrapped

    def _wrap_exp0(self, orig: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        tracer = self

        def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
            x = args[0] if args else kwargs.get("x")
            if torch.is_tensor(x) and tracer._is_linear_from_log0(x):
                tracer._note(
                    "log0_linear_exp0_sandwich",
                    "exp_map_zero",
                    "exp_map_zero consumed a Linear/matmul(log_map_zero(.)) tensor",
                )
            return orig(*args, **kwargs)

        return wrapped

    def _wrap_f_linear(self, orig: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        tracer = self

        def wrapped(input: torch.Tensor, weight: torch.Tensor, bias: Any = None) -> torch.Tensor:
            out = orig(input, weight, bias)
            if _skip_linalg.get() > 0 and not _is_open_ball_batch(input):
                return out
            tracer._flag_functional_left(input, op="F.linear", out=out)
            return out

        return wrapped

    def _wrap_matmul(self, orig: Callable[..., torch.Tensor], *, op: str) -> Callable[..., Any]:
        tracer = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _skip_isometry.get() > 0:
                return orig(*args, **kwargs)
            out = orig(*args, **kwargs)
            left = args[0] if args else kwargs.get("input", kwargs.get("mat1"))
            right = args[1] if len(args) > 1 else kwargs.get("mat2", kwargs.get("other"))
            if _skip_linalg.get() > 0 and not (
                torch.is_tensor(left) and _is_open_ball_batch(left)
            ):
                return out
            if torch.is_tensor(left) and torch.is_tensor(right):
                # Node map: last dim of left contracts with rows of right.
                # ``__matmul__`` is bound: left is self, right is the other matrix.
                if right.ndim == 2 and left.ndim >= 2 and int(right.shape[0]) == int(left.shape[-1]):
                    tracer._flag_functional_left(
                        left, op=op, right=right, out=out if torch.is_tensor(out) else None
                    )
            return out

        return wrapped

    def _wrap_addmm(self, orig: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        tracer = self

        def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
            out = orig(*args, **kwargs)
            mat1 = args[1] if len(args) > 1 else kwargs.get("mat1")
            mat2 = args[2] if len(args) > 2 else kwargs.get("mat2")
            if _skip_linalg.get() > 0 and not (
                torch.is_tensor(mat1) and _is_open_ball_batch(mat1)
            ):
                return out
            if torch.is_tensor(mat1) and torch.is_tensor(mat2):
                if mat2.ndim == 2 and mat1.ndim >= 2 and int(mat2.shape[0]) == int(mat1.shape[-1]):
                    tracer._flag_functional_left(mat1, op="addmm", right=None, out=out)
            return out

        return wrapped

    def _wrap_qr(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        """Skip nested linalg on the QR factor only — not a name match on 'qr'."""

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            tok = _skip_linalg.set(_skip_linalg.get() + 1)
            try:
                return orig(*args, **kwargs)
            finally:
                _skip_linalg.reset(tok)

        return wrapped

    def _wrap_manifold_formula(
        self, orig: Callable[..., torch.Tensor]
    ) -> Callable[..., torch.Tensor]:
        def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
            tok = _skip_manifold.set(_skip_manifold.get() + 1)
            try:
                return orig(*args, **kwargs)
            finally:
                _skip_manifold.reset(tok)

        return wrapped

    def _wrap_add(self, orig: Callable[..., Any], *, op: str) -> Callable[..., Any]:
        tracer = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = orig(*args, **kwargs)
            if _skip_manifold.get() > 0 or _skip_prelift.get() > 0:
                return out
            a = args[0] if args else None
            b = args[1] if len(args) > 1 else kwargs.get("other")
            if torch.is_tensor(a) and torch.is_tensor(b):
                if _is_open_ball_batch(a) and _is_open_ball_batch(b):
                    tracer._note(
                        "euclidean_mix_on_ball",
                        op,
                        "ambient add of two open-ball tensors",
                    )
            return out

        return wrapped

    def _wrap_sum(self, orig: Callable[..., Any]) -> Callable[..., Any]:
        tracer = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = orig(*args, **kwargs)
            if _skip_manifold.get() > 0 or _skip_prelift.get() > 0:
                return out
            x = args[0] if args else kwargs.get("input")
            dim = args[1] if len(args) > 1 else kwargs.get("dim")
            if torch.is_tensor(x) and dim == 1 and _is_soft_expert_stack(x):
                tracer._note(
                    "euclidean_mix_on_ball",
                    "sum",
                    "ambient weighted sum over expert stack [N, E, d]",
                )
            return out

        return wrapped

    def _wrap_mul(self, orig: Callable[..., Any], *, op: str) -> Callable[..., Any]:
        tracer = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = orig(*args, **kwargs)
            if _skip_manifold.get() > 0 or _skip_prelift.get() > 0:
                return out
            a = args[0] if args else None
            b = args[1] if len(args) > 1 else kwargs.get("other")
            ball: torch.Tensor | None = None
            scale: Any = None
            if torch.is_tensor(a) and _is_open_ball_batch(a) and _is_broadcast_scale(b, a):
                ball, scale = a, b
            elif torch.is_tensor(b) and _is_open_ball_batch(b) and _is_broadcast_scale(a, b):
                ball, scale = b, a
            if ball is None:
                return out
            if _is_hard_01_scale(scale):
                return out
            tracer._note(
                "euclidean_scale_on_ball",
                op,
                "ambient scalar multiply on an open-ball tensor (not a 0/1 mask)",
            )
            return out

        return wrapped

    def _install_code_wrap(self, attr: str, orig: Any, wrapped: Any) -> None:
        code = getattr(orig, "__code__", None)
        if code is None:
            return
        for _mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            fn = getattr(mod, attr, None)
            if callable(fn) and getattr(fn, "__code__", None) is code:
                self._patches.append((mod, attr, fn))
                setattr(mod, attr, wrapped)

    def _patch_attr(self, obj: Any, name: str, wrapper_fn: Callable[[Any], Any]) -> None:
        orig = getattr(obj, name, None)
        if orig is None or not callable(orig):
            return
        wrapped = wrapper_fn(orig)
        self._patches.append((obj, name, orig))
        setattr(obj, name, wrapped)

    def install(self, model: nn.Module) -> None:
        def _prelift_pre(_m: nn.Module, _inp: Any) -> None:
            _skip_prelift.set(_skip_prelift.get() + 1)

        def _prelift_post(_m: nn.Module, _inp: Any, _out: Any) -> None:
            cur = _skip_prelift.get()
            _skip_prelift.set(cur - 1 if cur > 0 else 0)

        for name, mod in model.named_modules():
            if _is_pre_lift_scope(name) and mod is not model:
                self._hooks.append(mod.register_forward_pre_hook(_prelift_pre))
                self._hooks.append(mod.register_forward_hook(_prelift_post))
            if isinstance(mod, nn.Linear):
                if _is_pre_lift_linear(name):
                    continue
                self._hooks.append(
                    mod.register_forward_hook(
                        lambda m, i, o, n=name: self._on_linear(n, m, i, o)
                    )
                )

        orig_log = attn_mod.log_map_zero
        orig_exp = attn_mod.exp_map_zero
        wrapped_log = self._wrap_log0(self._wrap_manifold_formula(orig_log))
        wrapped_exp = self._wrap_exp0(self._wrap_manifold_formula(orig_exp))
        self._install_code_wrap("log_map_zero", orig_log, wrapped_log)
        self._install_code_wrap("exp_map_zero", orig_exp, wrapped_exp)

        for mname in MANIFOLD_FORMULA_NAMES:
            orig_fn = getattr(attn_mod, mname)
            self._install_code_wrap(
                mname, orig_fn, self._wrap_manifold_formula(orig_fn)
            )

        self._patch_attr(F, "linear", self._wrap_f_linear)
        if torch.nn.functional is not F:
            self._patch_attr(torch.nn.functional, "linear", self._wrap_f_linear)
        self._patch_attr(torch, "matmul", lambda o: self._wrap_matmul(o, op="torch.matmul"))
        self._patch_attr(torch.Tensor, "matmul", lambda o: self._wrap_matmul(o, op="Tensor.matmul"))
        # ``z @ W`` does not go through ``torch.matmul`` in current PyTorch.
        self._patch_attr(
            torch.Tensor, "__matmul__", lambda o: self._wrap_matmul(o, op="Tensor.__matmul__")
        )
        self._patch_attr(torch, "mm", lambda o: self._wrap_matmul(o, op="torch.mm"))
        self._patch_attr(torch, "addmm", self._wrap_addmm)
        self._patch_attr(torch.linalg, "qr", self._wrap_qr)
        self._patch_attr(torch, "add", lambda o: self._wrap_add(o, op="torch.add"))
        self._patch_attr(torch.Tensor, "__add__", lambda o: self._wrap_add(o, op="Tensor.__add__"))
        self._patch_attr(torch.Tensor, "sum", self._wrap_sum)
        self._patch_attr(torch, "sum", self._wrap_sum)
        self._patch_attr(torch, "mul", lambda o: self._wrap_mul(o, op="torch.mul"))
        self._patch_attr(torch.Tensor, "__mul__", lambda o: self._wrap_mul(o, op="Tensor.__mul__"))
        self._patch_attr(torch.Tensor, "__rmul__", lambda o: self._wrap_mul(o, op="Tensor.__rmul__"))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for mod, attr, orig in reversed(self._patches):
            setattr(mod, attr, orig)
        self._patches.clear()
        _skip_linalg.set(0)
        _skip_prelift.set(0)
        _skip_manifold.set(0)
        _skip_isometry.set(0)


def scan_forward(
    model: nn.Module,
    forward_fn: Callable[[], Any],
) -> PureHypReport:
    """Run ``forward_fn`` under the tracer. ``forward_fn`` should invoke ``model``."""
    tracer = PureHypTracer()
    model.eval()
    try:
        tracer.install(model)
        with torch.no_grad():
            forward_fn()
    finally:
        tracer.remove()
    seen: set[tuple[str, str, str]] = set()
    uniq: list[PureHypViolation] = []
    for v in tracer.violations:
        key = (v.kind, v.module_name, v.detail)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(v)
    return PureHypReport(passed=not uniq, violations=uniq)


__all__ = [
    "PureHypReport",
    "PureHypTracer",
    "PureHypViolation",
    "_ORTH_ATOL",
    "_gram_inf",
    "_is_origin_isometry_matrix",
    "MANIFOLD_FORMULA_NAMES",
    "PURE_HYP_PASS_VERSION",
    "scan_forward",
]
