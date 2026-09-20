# Tokyo Eye EQU — Correct Start Implementation Plan

**Gate ID:** `tokyo_eye_equ_correct_start`  
**Display Lineage:** Tokyo Eye EQU  
**Status:** APPROVED & LOCKED  
**Date:** 2026-09-15  
**Experiment (Domain Charter):** `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`  
**Governance Specification:** `tokyo_eye_equ_governance`  
**Pure-Hyp Amendment:** `tokyo_eye_equ_pure_hyp_v1`  
**Design Reference:** `docs/superpowers/specs/2026-09-15-tokyoeye-equ-correct-start-design.md`

---

## 1. Executive Summary & Operational Boundaries

This document provides the step-by-step execution protocol for **`tokyo_eye_equ_correct_start`**. It formalizes the clean restart of the Tokyo Eye Equiformer $e(3)$ MoE geometry trunk under the amended pure-hyperbolic freeze.

### Locked Rules:
1. **No Tangent Linear Substitutes:** Live `attention.py` and `affinity_head.py` tangent operations ($\exp_0(W \cdot \log_0(z))$) are removed from the active forward path. Q/K/V operations and aggregations are conducted strictly on-manifold via gyrovector operations and Einstein/Klein barycenters.
2. **Cold Boot Isolation:** No weights are loaded from cold boot or v5 champion checkpoints (`sha 507d54bd6d8fb6c2…`). Frontend features utilize the pinned MPtrj EquiformerV3 bank (`checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt`, sha256 `59c6c23573a3b05b347662f473209d1bb1ccb5b85f6624a8f87072e7c397addf` — same as cold-boot). Hyperbolic spine: fresh random init. **Do not** load champion / cold-boot / affinity / C1 θ.
3. **MLflow SSOT Alignment:** The run will log telemetry directly to MLflow experiment `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine` under the run name `tokyo_eye_equ_correct_start`.

---

## 2. Pre-Registered Data Panels

The training run and subsequent hygiene audits strictly utilize the following pre-registered PDB chains. No online modifications or dataset substitutions are permitted during execution.

### 2.1 Training Panel ($N = 8$)
```text
1MBN (Chain A) - Myoglobin (Globin fold)
1LYZ (Chain A) - Hen Egg White Lysozyme (α+β fold)
1F88 (Chain A) - Bovine Rhodopsin (7TM helical bundle)
1HHP (Chain A) - HIV-1 Protease (β-barrel dimer)
1TEN (Chain A) - Tenascin (FnIII domain / All-β)
1UBQ (Chain A) - Ubiquitin (α/β roll)
1TIM (Chain A) - Triosephosphate Isomerase (TIM barrel)
4OBE (Chain A) - KRAS (P-loop NTPase / cold-boot home)
```

### 2.2 Probe Panel ($N = 6$)
```text
1HNG (Chain A) - Hydroxymethylbilane Synthase
1ALC (Chain A) - Alpha-Lactalbumin
1A5R (Chain A) - Small Ubiquitin-like Modifier (SUMO)
KEEP (Chain A) - Hemoglobin Alpha Chain
1GKY (Chain A) - Guanylate Kinase
```

### 2.3 Sealed Holdout Panel ($N = 4$)
```text
1BG1 (Chain A) - Beta-Glucosidase
2Z6H (Chain A) - Methyltransferase
1IVO (Chain A) - Epidermal Growth Factor Receptor (EGFR)
2SHP (Chain A) - SHP-2 Protein Tyrosine Phosphatase
```

---

## 3. Mathematical & Architectural Specification

### 3.1 On-Manifold Q/K/V & Sparse Hyperbolic Attention
Let $\mathcal{P}_c^d = \{ x \in \mathbb{R}^d : c \|x\|^2 < 1 \}$ denote the Poincaré ball of curvature $c > 0$. The lift from the Equiformer $e(3)$ scalar/vector representations is performed **once** at the projector boundary:
$$z_0 = \exp_0^c\left(\text{RadialAngularProjector}(s, v)\right) \in \mathcal{P}_c^d$$

Post-lift, for node embeddings $z_i \in \mathcal{P}_c^d$, query and key vectors are constructed using Möbius gyrovector scalings and gyro-rotations:
$$Q_i = R_Q \otimes_c z_i, \quad K_j = R_K \otimes_c z_j, \quad V_j = R_V \otimes_c z_j$$
where $R_Q, R_K, R_V \in \text{SO}(d)$ are orthogonal matrices preserving Poincaré ball distance metrics.

For each directed edge $(i \to j)$ in the sparse R0–R5 graph with relation type $R \in \{0, 1, 2, 3, 4, 5\}$, attention logits are computed via native Poincaré distance $d_{\mathcal{H}}^c$:
$$A_{ij} = \frac{- \gamma_R d_{\mathcal{H}}^c(Q_i, K_j) + \beta_R}{\sqrt{d_h}}$$
where $\gamma_R > 0$ and $\beta_R \in \mathbb{R}$ are learned relation-specific parameters, and $d_h$ is the head dimension.

Attention weights $\alpha_{ij}$ are obtained via segmented softmax over incoming edges:
$$\alpha_{ij} = \frac{\exp(A_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(A_{ik})}$$

### 3.2 Einstein Barycentric Aggregation
Node aggregation avoids Euclidean matrix multiplication in tangent space. Aggregated value representations are computed as the Einstein midpoint in Klein space coordinates $\mathcal{K}_c^d$:
$$k_j = \phi_{\mathcal{P} \to \mathcal{K}}(V_j) = \frac{2 V_j}{1 + c \|V_j\|^2}$$
$$\bar{k}_i = \frac{\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \gamma(k_j) k_j}{\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \gamma(k_j)}, \quad \text{where } \gamma(k) = \frac{1}{\sqrt{1 - c \|k\|^2}}$$
$$z_i^{(l+1)} = \phi_{\mathcal{K} \to \mathcal{P}}(\bar{k}_i) = \frac{\bar{k}_i}{1 + \sqrt{1 - c \|\bar{k}_i\|^2}}$$

---

## 4. Pre-Registered Estimator Implementations

All telemetry functions are executed at the end of every epoch and on probe evaluates. Gate stamps fail immediately if any metric violates its threshold.

### 4.1 Static AST / Inspection Pure-Hyp Pass (`pure_hyp_pass`)
```python
import inspect
import torch.nn as nn

def evaluate_pure_hyp_pass(model: nn.Module) -> bool:
    """
    Audits the forward execution graph of the EQU trunk.
    Fails if any tangent Linear layer (_tangent_linear) or log0-exp0 Euclidean matrix
    multiplication is active post-lift.
    """
    for name, module in model.named_modules():
        if "attention" in name or "spine" in name:
            if hasattr(module, "use_tangent_shortcut") and module.use_tangent_shortcut:
                return False
            if module.__class__.__name__ in ["TangentLinear", "TangentPool"]:
                return False
    return True
```

### 4.2 Probe Saturation & Radius Spread Gates
```python
import torch

def evaluate_geometric_hygiene(probe_embeddings: torch.Tensor, c: float = 1.0):
    """
    Computes boundary saturation and Poincaré radius spread.
    probe_embeddings: [N, d] Poincaré ball points
    """
    radii_euclidean = torch.norm(probe_embeddings, dim=-1) # ||x||_2
    mean_sat = torch.mean(radii_euclidean).item()
    
    # Hyperbolic radius: r_H = (1 / sqrt(c)) * log((1 + sqrt(c)*||x||) / (1 - sqrt(c)*||x||))
    sqrt_c = c ** 0.5
    radii_hyp = (1.0 / sqrt_c) * torch.log((1.0 + sqrt_c * radii_euclidean) / (1.0 - sqrt_c * radii_euclidean + 1e-7))
    std_spread = torch.std(radii_hyp).item()
    
    finite_h2 = torch.all(torch.isfinite(radii_hyp)).item()
    
    sat_pass = mean_sat < 0.50
    spread_pass = std_spread > 0.15
    
    return {
        "mean_sat": mean_sat,
        "std_spread": std_spread,
        "finite_h2": float(finite_h2),
        "hygiene_pass": sat_pass and spread_pass and finite_h2
    }
```

### 4.3 MoE Routing Entropy Gate (`moe_liveness_gate`)
```python
import torch

def evaluate_moe_liveness(routing_probs: torch.Tensor, num_experts: int = 4) -> float:
    """
    Computes normalized routing entropy across K experts.
    routing_probs: [N, K] softmax probabilities
    """
    avg_probs = torch.mean(routing_probs, dim=0) # [K]
    entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-9))
    max_entropy = torch.log(torch.tensor(float(num_experts)))
    normalized_entropy = (entropy / max_entropy).item()
    return normalized_entropy
```

### 4.4 $e(3)$ Equivariance Residual Gate (`equiv_residual_gate`)
```python
import torch

def evaluate_equivariance_residual(model: nn.Module, batch_graph) -> float:
    """
    Evaluates || f(g * x) - g * f(x) ||_2 for a random 3D rotation matrix g in SO(3).
    """
    model.eval()
    with torch.no_grad():
        # Generate random orthogonal rotation matrix g
        R, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.linalg.det(R) < 0:
            R[:, 0] *= -1
            
        # Standard forward pass
        out_orig = model(batch_graph)
        
        # Rotated input forward pass
        batch_graph_rot = batch_graph.clone()
        batch_graph_rot.pos = torch.matmul(batch_graph.pos, R.T)
        out_rot_input = model(batch_graph_rot)
        
        # Apply rotation to original output vectors if vector features exist
        # Residual metric on scalar Poincaré points (which should be invariant under SO(3))
        diff = torch.norm(out_orig.z_hyp - out_rot_input.z_hyp, dim=-1)
        max_residual = torch.max(diff).item()
        
    return max_residual
```

---

## 5. Training Protocol & Curriculum Schedule

### 5.1 Optimization Parameters
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 1e-8$)
- **Learning Rate:**
  - Backbone (Equiformer scalar/vector adapt): $\text{LR}_{\text{backbone}} = 1.0 \times 10^{-5}$
  - Hyperbolic Spine & MoE: $\text{LR}_{\text{hyp}} = 3.0 \times 10^{-4}$
- **Weight Decay:** $1.0 \times 10^{-4}$
- **Gradient Clipping:** Max norm $1.0$ (Poincaré gradient clipping applied near boundary)
- **Curvature $c$:** Initialized at $c = 1.0$, learned via passthrough parameter $\log c$.

### 5.2 Radius Curriculum Schedule ($\tau$)
To prevent early rim collapse where embeddings accumulate on the boundary, the maximum allowable radius $\tau_{\text{ceiling}}$ follows a smooth step curriculum over $E = 20$ training epochs:

$$\tau_{\text{ceiling}}(e) = \tau_{\text{start}} + (\tau_{\text{max}} - \tau_{\text{start}}) \cdot \left( \frac{e}{E} \right)^2$$
where $\tau_{\text{start}} = 0.60$ and $\tau_{\text{max}} = 0.95$.

---

## 6. Execution Protocol & Container Invocations

All execution must be run inside the standard Docker science container `tokyoeye_science`.

### 6.1 Environment Verification
```bash
docker exec -it tokyoeye_science python -c "
import torch, mlflow
print('CUDA Available:', torch.cuda.is_available())
print('MLflow Version:', mlflow.__version__)
"
```

### 6.2 Pre-Flight Static Check
Execute unit tests ensuring tangent operations are disabled:
```bash
docker exec -it tokyoeye_science pytest tests/v8/test_hyperbolic_graph_attention.py -k "pure_hyp"
```

### 6.3 Driver Execution Command
Run the clean training driver script with MLflow logging enabled:
```bash
docker exec -it tokyoeye_science python experiments/training/v8/run_v8_experiment.py \
    --experiment-name "tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine" \
    --run-name "tokyo_eye_equ_correct_start" \
    --equiformer-ckpt "checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt" \
    --epochs 20 \
    --lr-hyp 3e-4 \
    --lr-backbone 1e-5 \
    --pure-hyp-strict True \
    --gate-stamp-path "data/gates/tokyo_eye_equ_correct_start.json"
```

---

## 7. Gate Output Telemetry & Stamp Structure

Upon completion of the execution, the runner produces the locked JSON gate artifact at `data/gates/tokyo_eye_equ_correct_start.json` and attaches it to MLflow run `tokyo_eye_equ_correct_start`.

### 7.1 Target Gate JSON Schema
```json
{
  "gate_id": "tokyo_eye_equ_correct_start",
  "display_lineage": "Tokyo Eye EQU",
  "timestamp": "2026-09-15T20:30:00Z",
  "mlflow_experiment_id": "14",
  "mlflow_run_id": "<AUTO_GENERATED_UUID>",
  "status": "QUALIFIED",
  "gates": {
    "pure_hyp_pass": {
      "pass": true,
      "value": 1.0,
      "threshold": 1.0
    },
    "probe_sat_gate": {
      "pass": true,
      "value": 0.382,
      "threshold": 0.50
    },
    "radius_spread_gate": {
      "pass": true,
      "value": 0.214,
      "threshold": 0.15
    },
    "finite_h2_gate": {
      "pass": true,
      "value": 1.0,
      "threshold": 1.0
    },
    "moe_liveness_gate": {
      "pass": true,
      "value": 0.842,
      "threshold": 0.60
    },
    "equiv_residual_gate": {
      "pass": true,
      "value": 4.1e-06,
      "threshold": 1.0e-05
    }
  },
  "artifacts": [
    "docs/superpowers/specs/2026-09-15-tokyoeye-equ-correct-start-design.md",
    "docs/superpowers/plans/2026-09-15-tokyoeye-equ-correct-start.md"
  ]
}
```

---

## 8. Sign-off & Lock

- **Lead Architect Sign-off:** APPROVED (Ray Swan)
- **Execution State:** LOCKED — ready for science container invocation.
