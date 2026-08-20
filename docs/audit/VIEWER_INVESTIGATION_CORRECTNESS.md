# Viewer Investigation correctness (CLOSED — 2026-07-16)

**Status:** `CLOSED` — Tokyo Eye **product-correctness** finding, not a GNN routing research question.  
**Separate from:** Gram/purity/width ablation thread; gate-geometry track.  
**Related research:** G5b ρ-proxy / OOD inversion in `GNNV7_SUCCESS_CRITERIA.md` and `EVIDENTIAL_UNCERTAINTY.md`.

---

## Verdict (plain)

1. **Render bug:** Split / disc canvas `colorScale` was inverted relative to NGL `RdYlBu` + `colorReverse: true` (disc: high→blue; 3D: high→red). Dual-panel screenshots silently disagreed on color direction for every shared metric.
2. **Signal bug:** Default “Investigation” = `aleatoric × (1 − epistemic)` on near-flat evidential heads. On 4OBE (`v66_best_disc.pt`): `corr(inv, ρ)=+0.83`, `corr(inv, τ)=−0.60`, **0/25** top-ranked residues τ=1, head spans Δale≈0.0035 / Δepi≈0.0115. Ranking is min-max dressing of noise correlated with **well-wrapped core** — the opposite of dehydron / cryptic-pocket priority.

**Product rule:** Evidential Investigation must not be the trusted default. Default Investigation coloring is physics-layer underwrap (`physics_investigation_scores` from ρ/τ). Evidential formula remains an explicitly labeled experimental overlay.

---

## Evidence (4OBE split viewer)

| Check | Result |
| ----- | ------ |
| Colormap direction | Disc `t=0→red`, `t=1→blue` vs NGL high→red — **mismatch** (fixed) |
| Top-25 evidential inv τ=1 | **0/25** |
| Top-25 mean ρ vs TAU=13 | **27.2** (well-wrapped) |
| `corr(inv, ρ)` / `corr(inv, τ)` | **+0.83** / **−0.60** |
| Ale / epi dynamic range | **0.0035** / **0.0115** |

Artifact path used: `data/local_objects/gnn_viewer/4obe/4obe_split_viewer.html` + physics from `build_from_pdb_chain(4OBE, A)`.

---

## Code changes (this close-out)

| Item | Location |
| ---- | -------- |
| Shared disc scale (low=blue, high=red) | `_DISC_COLOR_SCALE_JS` in `interactive_viewer.py` |
| Default metric | `physics_investigation` (ρ/τ underwrap) |
| Evidential overlay | option `investigation` — labeled experimental |
| PDB B-factor default | physics map (not evidential) |

Sweep: only `science/dtie/v6/visualization/interactive_viewer.py` defines dual-panel canvas-vs-NGL continuous coloring. No other React/frontend dual-panel `colorScale` found. Expert route uses discrete NGL bands vs continuous disc `expert/3` (pre-existing; same direction after scale fix).

---

## Consumer sweep (`investigation_scores`)

| Consumer | Role | Action |
| -------- | ---- | ------ |
| `interactive_viewer.py` | UI default / B-factor | **Fixed** — physics default; evidential experimental |
| `shell_signal_gate.py` | Head flatness health (`investigation_std`) | Keep on evidential — intentional head monitor |
| `flag_investigation_sites` (`aleatoric_residue_diagnostics.py`) | Training triage (ale + rim + clustering) | Separate rule; not viewer paint |
| Hypothesis confidence (`agent/tools/hypothesis/`) | Evidence-based cards | **Does not** call `investigation_scores` |
| Agent tools (`get_source_leaks`, uncertainty search) | **Fixed 2026-07-16** — default `physics_rim` (τ + cone_depth); evidential ranking opt-in + warned |

v5 `export_for_viewer.py` uses a similar ale/epi “outlier” heuristic for `isOutlier` flags — legacy path; not the live v6 split viewer default.

---

## Uncertainty-track reopen (narrow)

**Trigger satisfied:** downstream deliverable (viewer Investigation) required a trustworthy “look here” map.

**Scope:** product default grounded in stable physics; **not** “fix DER in general.” Evidential heads remain parked for research until G5b-class OOD / ρ-proxy is independently resolved; experimental overlay may stay for monitoring.

---

## Do not

- Cite evidential Investigation screenshots as scientific priority maps without the experimental label.
- Re-default the viewer to `ale × (1−epi)` without a new closed audit that overturns G5b + this 4OBE measurement.
- Confuse disc radius vs PDB burial: `shell_corr_depth_sasa_weight=0` means those axes are unaligned by design — orthogonal to this finding.

---

## Follow-on: τ vs cone_depth majority-color paradox (4OBE, 2026-07-16)

Flip-through on regenerated `…/4d_seed1_v1/viewers/4obe/` (κ=0.704):

| Check | Result |
| ----- | ------ |
| τ color path | **Same** continuous `scaleMetric` + RdYlBu+reverse as depth (only `expert` is special-cased) — **not** a second render bug |
| `corr(cone_depth, τ)` this chain | Pearson **+0.66** / Spearman **+0.72** — **not** 1.000 |
| τ=1 fraction | **60%** → paints **60% red** under binary min-max |
| cone_depth red bin (t>0.66) | **11%** — long right tail; bulk stays blue after min-max |
| Among τ=1, depth looks red | only **18%**; **31%** of τ=1 still sit in the depth-blue bin |
| `corr(ρ, τ)` | **−0.80** (ρ mean 7.1 vs 20.3) — physics threshold identity holds |
| `corr(ale, epi)` | **+0.996**; both track ρ (**~+0.74**) and anti-track depth (**~−0.95**) |

**Read:** majority-red τ vs majority-blue depth is mostly **binary full-strength red vs continuous long-tail min-max**, plus a real but moderate depth↔τ coupling on this structure — not an inverted colormap. Corpus/training `r(depth,τ)≈1` claims must not be treated as a per-structure spatial identity.

Reusable check: `python -m experiments.diagnostics.viewer_channel_flipthrough <split_viewer.html>`.

### Corpus aggregate on same checkpoint (Stage A-12 viewers)

Artifact: `checkpoints/v66/diagnostics/three_vector_stack_battery_reverify/4d_seed1_corpus_channel_flipthrough.json`
(`fix1_s4_stack_initseed_controlled_4d_seed1_v1`, all 12 regenerated structures).

| Metric | Result |
| ------ | ------ |
| `r(cone_depth, τ)` Pearson | mean **0.561**, median **0.563**, range **[0.365, 0.668]** (1TEN → 1UBQ) |
| 4OBE | **0.662** — rank **11/12** (near the top), z=**+1.25** — **not** a low outlier |
| `r(ale, epi)` | mean **0.994**, min **0.990**, max **0.996** — near-duplication on **every** structure |
| `r(ale, ρ)` / `r(epi, ρ)` | corpus mean **0.60** / **0.62** |

**Reconcile with SSOT `r≈1.000`:** `GNNV7` already flags `P_DEHYDRON_CONE_01` r=1.000 as **loss-wiring / gate-sanity** on master-cold (`cone_target_mode=tau_dehydron_rim` minimizes `1−Pearson`). That framing is about the *contract being active*, not about achieved per-structure geometry on later feeler-stack checkpoints. On this controlled 4-D seed1 lineage the objective is **only partially satisfied**: typical structure sits near **r≈0.56**, with real spread (~0.08 std). 4OBE’s 0.66 is typical-to-strong for this run — the gap from “1.000” is **corpus-wide slack**, not a 4OBE-specific failure.

**G5b generalization (explicit):** ale↔epi ≈ **0.994** across all 12 structures on this **4-D / seed1** checkpoint confirms uncertainty-head collapse is live on this lineage, not only wherever it was first diagnosed. Strengthens (does not change) the shipped product default: physics Investigation trusted; evidential experimental.