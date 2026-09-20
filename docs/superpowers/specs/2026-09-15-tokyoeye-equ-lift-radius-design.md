# Tokyo Eye EQU — Learnable Lift Radius Design

**Gate ID:** `tokyo_eye_equ_lift_radius`  
**Status:** APPROVED_LOCKED  
**Date:** 2026-09-15  
**Approver:** Ray Swan (proceed as deemed correct)  
**Predecessor:** `tokyo_eye_equ_shell_unpack` FAILED (MLflow `f37bd3afa156426c995f14311d4f8c4d`)  
**Experiment:** `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`

## 1. Root cause (sealed)

Shell packing survived rim_volume and shell_unpack. Equivariance/MoE/purity pass. Telemetry: `r ≈ τ` always.

Invariant projector used `‖t‖ → artanh(τ)·tanh(‖u‖)`, which **saturates the lift onto the τ shell** whenever the MLP fires large. Mean-radius loss and `final_only` clamp cannot create interior mass if the lift contract forbids it.

## 2. Fix

### 2.1 Learnable radius α ∈ (0, τ)

$$
\alpha = \tau\cdot\sigma(\mathrm{MLP}_\alpha(s_{\mathrm{inv}})),\quad
\hat{u}=\mathrm{MLP}_u(s_{\mathrm{inv}})/\|\cdot\|,\quad
z_0=\exp_0(\mathrm{artanh}(\alpha)\,\hat{u})
$$

No `artanh(τ)` ceiling on magnitude except through learned α < τ.

### 2.2 Retain

- Invariant features only (no â)
- `tau_clamp_mode=final_only`
- `L_rad` with μ*=0.35, σ*=0.20, λ_mean=3, λ_spread=1, λ_bar=0.5
- Same 6 gates; lift+spine Δ_equiv
- Fresh spine; forbid shell_unpack / rim / correct_start / champion θ
- MPtrj pin; 20 epochs; τ 0.60→0.95

## 3. Exit

QUALIFIED → `tokyo_eye_equ_affinity_coupling`. Else FAILED; no patch of Fail θ.
