# Tokyo Eye EQU — Architecture Experimentation & Governance Standard

**Gate ID:** `tokyo_eye_equ_governance`  
**Display lineage:** Tokyo Eye EQU  
**Status:** ACTIVE (operator standard, 2026-09-15)  
**Model family:** Equiformer $e(3)$ MoE  
**Primary domain:** Geometric deep learning / manifold representation learning  
**Protocol:** v1.1 (pure-hyp invariant + MLflow governance SSOT)

**Author:** Ray Swan (structure) · partnership land-in  
**Reconcile note (locked):** MLflow **Experiment** = domain charter container (e.g. `…/geometric/hyperbolic-spine`). **Sprint document** = card under that experiment. **Runs** = executions of a card (`tokyo_eye_equ_<slug>`). Not 1 sprint $\leftrightarrow$ 1 experiment — that would orphan SSOT tags and explode the namespace.

---

## 1. Governance & Architectural Hierarchy

To eliminate ambiguity between code execution, hypothesis testing, and artifact tracking, Tokyo Eye enforces a strict 4-tier structural mapping:

```
[ Epic / Sub-System ]
       │
       ▼
[ Domain Experiment (MLflow) ] ──────── (1 : 1 Charter Container) ────────▶ [ MLflow Experiment (Live SSOT) ]
       │                                                                                │
       ▼                                                                                ▼
[ Sprint Document (Card in Git) ] ───── (1 : N Executable Runs) ──────────▶ [ MLflow Runs (tokyo_eye_equ_<slug>) ]
```

### 1.1 Separation of Concerns: Git vs. MLflow

* **Git Repository (Contract Text):** Stores static spec contracts (`docs/superpowers/specs/`), freeze amendments, and executable source code. Git defines *what the contract is*.
* **MLflow Tracking & Registry (Operational SSOT):** Stores live experimental state, binary gate passes/fails, experiment charters, next open cards, and stamped artifact logs (`.json` gate evidence). MLflow defines *what is currently true*.

### 1.2 Entity Lifecycle & Ownership

| Entity | Scope | Lifecycle | Ownership |
| :--- | :--- | :--- | :--- |
| **Sprint Document** | **The Scientific Hypothesis & Card.** Defines the target problem, theoretical bounds, absolute pass gates, and exit criteria for a run sequence. | 1-2 Weeks (Sprint duration) | Lead Architect / Senior Engineer |
| **MLflow Experiment** | **The Operational Container & Governance Charter.** Holds experiment metadata, UI notes, primary metrics, and SSOT tags. | 1:1 with **domain / hypothesis container**; $N$ cards under it | Lead Architect / ML Systems |
| **MLflow Run** | **The Execution Instance.** A single deterministic training run testing specific hyperparameters, seeds, or localized fixes under the experiment charter. | Hours to Days | Executing Engineer / Automated Pipelines |

> **Rule of Thumb:** One **domain Experiment** holds the charter + SSOT tags. Each **Sprint Document (card)** under it may produce one or more **Runs** attempting to satisfy that experiment's absolute pass gates. Open a **new Experiment** only for architectural branching (§2.2).

---

## 2. Naming Conventions & Lineage Protocol

### 2.1 Standard Naming Schema (`tokyo_eye_equ_<slug>`)

To eliminate opaque identifiers (e.g., `equ_ssot_index_v1`), all artifacts, runs, and gate stamps must adhere to a single unified schema:

* **MLflow Experiment Namespace:**  
  `tokyoeye/<model-family>/<hypothesis-slug>`  
  *Example:* `tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine`

* **MLflow Run Name & Gate Identifier:**  
  `tokyo_eye_equ_<slug>`  
  *Examples:*
  * Base SSOT Run: `tokyo_eye_equ_ssot`
  * First Clean Restart: `tokyo_eye_equ_correct_start`
  * Iterative Experiment Run: `tokyo_eye_equ_hyp_spine_run01_seed42`

* **Gate Stamp File:**  
  `data/gates/tokyo_eye_equ_<slug>.json`

> **Note on Schema Revisions:** Versioning or schema iterations live inside logged MLflow parameters (e.g., `ssot_schema_version = "v1.1"`), **never** as fragmented postfixes in the name itself.

### 2.2 Lineage vs. Architectural Branching

* **Lineage (In-Experiment Runs):** Keep within the **same** MLflow Experiment when tuning hyperparameters ($\text{LR}$, curvature scale $c$, MoE capacity factor) or modifying module internals under the same hypothesis.
* **Architectural Branching (New MLflow Experiment):** Open a **new** MLflow Experiment (and new Sprint Document) if you alter the underlying representation symmetries ($e(3)$ to $so(3)$), change manifold embedding spaces (e.g., Poincaré ball to Lorentz model), or alter the primary loss formulation.

---

## 3. MLflow Experiment Charter & SSOT Tags

Every active Tokyo Eye MLflow Experiment must be instantiated with a complete governance charter registered directly in MLflow.

### 3.1 Experiment-Level Metadata

* **UI Notes (`mlflow.note.content`):** Markdown summary defining the purpose, non-negotiable goals, primary metrics/gates, and current `next_open` card.
* **Machine Tags:**
  * `tokyo_eye_equ_purpose`: High-level operational scope.
  * `tokyo_eye_equ_primary_metrics`: Key gate metrics required to unblock the trunk.
  * `tokyo_eye_equ_trunk_requires`: Mandatory prerequisites for SSOT promotion.

### 3.2 Dynamic SSOT Tracking Tags

The MLflow Experiment holds tags pointing to the current immutable benchmark state:

| Tag Key | Value Type | Description |
| :--- | :--- | :--- |
| `tokyo_eye_equ_ssot_run_id` | `string` (MLflow Run ID) | Explicit UUID of the current verified SSOT run. |
| `tokyo_eye_equ_ssot_gate_id` | `string` (`tokyo_eye_equ_<slug>`) | Matching gate identifier (e.g., `tokyo_eye_equ_ssot`). |
| `tokyo_eye_equ_next_open` | `string` (`tokyo_eye_equ_<slug>`) | The active card/step authorized for execution (e.g., `tokyo_eye_equ_correct_start`). |

---

## 4. Absolute Pass Gates (Go / No-Go Circuit Breakers)

All runs must register binary telemetry against the following absolute gates. If **any** gate fails, the run is immediately classified as **DISQUALIFIED**.

```
                       Run Output Telemetry
                                │
                                ▼
               ┌─────────────────────────────────┐
               │    Pure Hyperbolic Gate Check   │ ── Fail (0.0) ──► DISQUALIFIED
               │     (pure_hyp_pass == True)     │
               └─────────────────────────────────┘
                                │ Pass (1.0)
                                ▼
               ┌─────────────────────────────────┐
               │    Geometric Hygiene Checks     │
               │   • Boundary Saturation < 0.50  │ ── Fail ───────► DISQUALIFIED
               │   • Radius Spread > 0.15        │
               │   • Finite H² Metric = True     │
               └─────────────────────────────────┘
                                │ Pass
                                ▼
               ┌─────────────────────────────────┐
               │     Dynamic MoE & Equivariance  │
               │   • Routing Entropy ≥ 0.60      │ ── Fail ───────► DISQUALIFIED
               │   • Equiv Residual < 10⁻⁵       │
               └─────────────────────────────────┘
                                │ Pass
                                ▼
                        QUALIFIED RUN
```

### 4.1 Pure Hyperbolic Invariant Gate (`pure_hyp_pass`) [VETO GATE]

* **Definition:** Following the initial Euclidean-to-Hyperbolic ($\text{Euc} \to \mathcal{H}$) lift, $100\%$ of internal representations, attention mechanisms, and message passing must remain strictly on-manifold.
* **Allowed Operations:** Einstein or Klein barycentric aggregation, native gyrovector space additions, hyperbolic distance metrics $d_{\mathcal{H}}(u, v)$.
* **Forbidden Operations (Geometry Substitutes):** Use of tangent space operations post-lift—such as $\exp_0(W \cdot \log_0(z))$ in QKV projections or attention output layers, tangent space pooling, tangent Linear layers, or Euclidean matrix multiplications on ball points.
* **Threshold:** `pure_hyp_pass == True` (Binary metric = $1.0$). A value of $0.0$ blocks trunk merge regardless of downstream accuracy or MoE performance.

### 4.2 Hard Geometric & Topology Gates

1. **Probe Boundary Saturation Gate (`probe_sat_gate`)**
   * **Metric:** $\mu_{\text{sat}} = \frac{1}{N} \sum_{i=1}^N \|x_i\|_2$
   * **Threshold:** $\mu_{\text{sat}} < 0.50$. Prevents numerical overflow and rim collapse.

2. **Radius Spread / Non-Rim Collapse Gate (`radius_spread_gate`)**
   * **Metric:** $\sigma_{\text{radius}} = \text{std}(\|x_i\|_{\mathcal{H}})$
   * **Threshold:** $\sigma_{\text{radius}} > 0.15$. Ensures spatial volume utilization across interior hyperbolic space.

3. **Numerical Stability Gate (`finite_h2_gate`)**
   * **Definition:** Verification that hyperbolic distance functions $d_{\mathcal{H}}(u, v)$ yield finite real numbers without $\text{NaN}$ or $\infty$ under $\text{float32}$ / $\text{bfloat16}$.
   * **Threshold:** Zero non-finite values allowed ($1.0$).

### 4.3 Dynamic MoE & Equivariance Gates

1. **MoE Routing Liveness Gate (`moe_liveness_gate`)**
   * **Metric:** Normalized entropy $H_{\text{norm}}$ of routing assignment probabilities across $K$ experts.
   * **Threshold:** $H_{\text{norm}} \ge 0.60$.

2. **Equivariance Residual Gate ($e(3)$ Precision)**
   * **Metric:** $\Delta_{\text{equiv}} = \| f(g \cdot x) - g \cdot f(x) \|_2$ for $g \in E(3)$.
   * **Threshold:** $\Delta_{\text{equiv}} < 10^{-5}$.

---

## 5. "No Remediation Theater" & Cold Boot Restart Protocol

### 5.1 Anti-Pattern: Remediation Theater
"Remediation Theater" occurs when a run fails a fundamental gate (such as rim collapse or tangent shortcuts) and engineers attempt post-hoc patching—such as waiving metrics mid-run, adjusting thresholds, or injecting Euclidean linear layers to force convergence.

### 5.2 Cold Boot & Freeze Amendment Protocol
When a fundamental design flaw or invariant violation is identified:
1. **Seal the Failure:** Immediately tag the active run as `DISQUALIFIED` in MLflow with logged failure metrics.
2. **Amend the Freeze Contract:** Update the Git specification (e.g., `docs/superpowers/specs/...`) to formalize the strict rule.
3. **Register SSOT Gate Stamp:** Update `data/gates/` and log the artifact to the MLflow SSOT run.
4. **Clean Restart:** Trigger a fresh run under the amended freeze (`tokyo_eye_equ_correct_start`). **No mid-run code patching or state continuation is permitted.**

---

## 6. SSOT Promotion Lifecycle

1. **Execution:** Run completes under standard naming `tokyo_eye_equ_<slug>`.
2. **Gate Audit:** Telemetry verifies all gates pass (`pure_hyp_pass == 1.0`, $\mu_{\text{sat}} < 0.50$, $\sigma_{\text{radius}} > 0.15$, $H_{\text{norm}} \ge 0.60$, $\Delta_{\text{equiv}} < 10^{-5}$).
3. **Model Registration:** Geometry Pass may register under `TokyoEye` as `@experimental` only when evaluation thresholds for **that card** pass. **`@champion` is not automatic** from geometry hygiene alone (affinity / full-stack still later; v5 remains lesson-only for EQU geometry reboot).
4. **Tag Update:** Update experiment tag `tokyo_eye_equ_ssot_run_id` only when the operator SSOT index itself moves; train-run Pass updates `tokyo_eye_equ_next_open` and logs gate stamps as artifacts — do not confuse a train run with the SSOT index run unless explicitly promoting the index.