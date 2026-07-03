# Briefing Panel Specification — Discovery Cockpit Left Rail

## Overview

The briefing panel is the left rail of the Discovery Cockpit. It functions as a **phase table of contents** — a contextual instrument gauge that tells the scientist what's important NOW as they navigate drug discovery phases. It is NOT a static sidebar; its content, metrics, color modes, and LLM insights swap entirely when the discovery phase changes.

The panel occupies the left 320px column of the cockpit grid. It is collapsible to a 28px vertical strip showing "Briefing" in rotated text.

## Data Dependencies

All metrics are derived from the hydration response (`GET /api/structures/{id}/hydrate`):
- `embeddings` → per-residue x, y, cone_depth, epistemic_uncertainty, aleatoric_uncertainty
- `graph_metrics` → degree, betweenness, clustering_coefficient, closeness, eigenvector_centrality, is_bridge, conductance
- `allosteric_sites` → site_id, confidence, residue_ids
- `source_leaks` → residue_id, leak_score, source
- `resistance_data` → sensitivity_score, coupling_count, is_hinge, lambda_2
- `pharmacophore_pockets` → pocket_index, druggability_score, residue_count, volume_estimate, allosteric_coupling
- `drug_candidates` → combined_druggability, accessibility_score, binding_potential, admet_pass, selectivity_ratio
- `hypotheses` → statement, status, confidence, evidence_supporting, evidence_contradicting

---

## Phase 1: RESIDUE

**Question this phase answers:** Where are the unusual residues, and what patterns emerge from the hyperbolic embedding?

### Hero Metrics (3)

| Metric | Data Source | Format | Color Rule | Why It Matters |
|--------|-------------|--------|------------|----------------|
| Mean Epistemic σ | `mean(embeddings.residues[*].epistemic_uncertainty)` | `0.XX` (2 decimal) | Teal ≤0.15, Amber 0.15–0.40, Magenta >0.40 | Global model confidence — low means the GNN is certain about this fold; high means structural ambiguity worth investigating |
| Spike Count | `count(residues where epistemic_uncertainty > mean + 2σ)` | Integer | Teal 0–2, Amber 3–5, Magenta >5 | Local uncertainty spikes are the learned fingerprint of cryptic pockets — more spikes = more discovery potential |
| Max Cone Depth | `max(embeddings.residues[*].cone_depth)` | `X.XX` | Teal ≤1.5, Amber 1.5–2.5, Magenta >2.5 | Deepest point in the hyperbolic hierarchy — extreme depth signals a residue buried far from the exposed fringe |

**Good vs Anomalous:**
- Good: Low mean σ (model confident), few spikes (fold well-characterized), moderate depth range
- Anomalous (discovery-rich): High spike count with clustered residues (cryptic pocket signature), extreme depth outliers

### LLM Briefing Prompt

```json
{
  "system": "You are a structural biology instrument producing cockpit readouts.",
  "prompt": "Structure {pdb_id}. Mean epistemic σ={mean_epistemic:.3f}, spike_count={spike_count}, max_cone_depth={max_cone_depth:.2f}, residue_count={residue_count}. Spikes at: {spike_residue_list}. Classify the embedding pattern in exactly 2-3 words (e.g. 'Cryptic-pocket-primed', 'Fold-confident', 'Boundary-ambiguous'). Then give exactly one sentence interpreting what this means for drug discovery. Return JSON: {\"classification\": \"...\", \"interpretation\": \"...\"}",
  "max_tokens": 80
}
```

### Recommended Viewport Color Mode
`epistemic_uncertainty` — highlights where the model is uncertain, revealing cryptic signals

### Recommended Dock Tab
**Motifs/Clusters** — Poincaré disc clusters and co-embedding groupings are the star of Phase 1

### Interactive Traces
- Click spike chip → highlights that residue on Poincaré disc + 3D viewer, emits selection_sync
- Click "Max Depth" metric → highlights the deepest residue, triggers Möbius focus on it
- Click cluster badge (from dock) → highlights all residues in that angular sector

---

## Phase 2: TOPOLOGY

**Question this phase answers:** How does force and signal flow through the fold, and where are the choke points?

### Hero Metrics (3)

| Metric | Data Source | Format | Color Rule | Why It Matters |
|--------|-------------|--------|------------|----------------|
| λ₂ (Fiedler Value) | `resistance_data.spectral.lambda_2` | `0.XXX` (3 decimal) | Teal ≥0.05, Amber 0.02–0.05, Magenta <0.02 | Algebraic connectivity — how easily the fold can be split into disconnected modules. Low λ₂ = fragile, allosterically exploitable |
| Bridge Count | `count(graph_metrics.metrics where is_bridge=true)` | Integer | Teal 0–1, Amber 2–3, Magenta >3 | Bridges are single points of failure in signal flow — removing them disconnects the graph. Drug targets. |
| Max Betweenness | `max(graph_metrics.metrics[*].betweenness)` | `0.XXX` | Teal ≤0.08, Amber 0.08–0.20, Magenta >0.20 | The most central residue in signal routing — a choke point where allosteric information must pass |

**Good vs Anomalous:**
- Good: High λ₂ (robust connectivity), no bridges (redundant paths), moderate betweenness (distributed flow)
- Anomalous (discovery-rich): Low λ₂ with multiple bridges (fragile fold, allosteric leverage points), concentrated betweenness (single choke point = drug target)

### LLM Briefing Prompt

```json
{
  "system": "You are a structural biology instrument producing cockpit readouts.",
  "prompt": "Structure {pdb_id}. lambda_2={lambda_2:.4f}, bridge_count={bridge_count}, max_betweenness={max_betweenness:.3f}, hinge_count={hinge_count}. Top centrality residues: {top_centrality_list}. Bridges at: {bridge_residue_list}. Classify the topology in 2-3 words (e.g. 'Fragile-hub-dependent', 'Distributed-resilient', 'Hinge-exploitable'). One sentence on allosteric implications. JSON: {\"classification\": \"...\", \"interpretation\": \"...\"}",
  "max_tokens": 80
}
```

### Recommended Viewport Color Mode
`resistance` — visualizes sensitivity scores and hinge residues across the fold

### Recommended Dock Tab
**Motifs/Clusters** — topology clusters reveal force-flow modules

### Interactive Traces
- Click centrality table row → highlights that residue, shows its betweenness neighborhood (connected residues glow)
- Click "Bridge Count" metric → highlights all bridge residues simultaneously
- Click λ₂ metric → triggers viewport_directive to show spectral partitioning (two-color split at the Fiedler cut)

---

## Phase 3: STRUCTURE

**Question this phase answers:** How does mutation reshape the landscape compared to wild-type, and what structural consequences emerge?

### Hero Metrics (2)

| Metric | Data Source | Format | Color Rule | Why It Matters |
|--------|-------------|--------|------------|----------------|
| Active Hypotheses | `count(hypotheses where status in ['proposed','gathering'])` | Integer | Teal 0, Amber 1–2, Magenta >2 | Scientific claims currently under investigation — more = active structural questioning |
| Mean Displacement | `compare.top_movers.mean(displacement)` (if compare active) OR `source_leaks.count` | `0.XX Å` or Integer | Teal ≤0.5Å / 0–2 leaks, Amber 0.5–1.5Å / 3–5, Magenta >1.5Å / >5 | Quantifies how much the mutation moved residues from WT positions, or how many energy leaks exist |

**Good vs Anomalous:**
- Good: Minimal displacement (mutation is silent structurally), few source leaks
- Anomalous (discovery-rich): Large displacements clustered near active site (mutation reshapes the druggable landscape), many source leaks (permanently open dehydrons = energy inefficiency → cancer mechanism)

### LLM Briefing Prompt

```json
{
  "system": "You are a structural biology instrument producing cockpit readouts.",
  "prompt": "Structure {pdb_id}. hypothesis_count={active_hypotheses}, source_leak_count={source_leak_count}, top_leaks={top_leak_residues}. Hypothesis statements: {hypothesis_summaries}. Classify the structural state in 2-3 words (e.g. 'Mutation-destabilized', 'Leak-cluster-active', 'Conformationally-rigid'). One sentence on what the structural evidence suggests. JSON: {\"classification\": \"...\", \"interpretation\": \"...\"}",
  "max_tokens": 80
}
```

### Recommended Viewport Color Mode
`allosteric` — shows coupling patterns that reveal how mutation effects propagate

### Recommended Dock Tab
**Hypotheses** — Phase 3 star; scientific claims being tested against structural evidence

### Interactive Traces
- Click hypothesis card → highlights all residues referenced in that hypothesis's evidence
- Click source leak chip → highlights that residue, shows leak_score in tooltip
- Click displacement metric (compare mode) → highlights top movers with displacement magnitude as glow intensity

---

## Phase 4: POCKET

**Question this phase answers:** Where can a drug actually bind, and which sites are druggable?

### Hero Metrics (3)

| Metric | Data Source | Format | Color Rule | Why It Matters |
|--------|-------------|--------|------------|----------------|
| Pocket Count | `pharmacophore_pockets.count` | Integer | Teal 1–2, Amber 3–4, Magenta >4 or 0 | How many distinct binding sites exist — zero is bad (no targets), too many dilutes focus |
| Top Druggability | `max(pharmacophore_pockets.pockets[*].druggability_score)` | `0.XX` (0–1 scale) | Teal ≥0.7, Amber 0.4–0.7, Magenta <0.4 | Best pocket's likelihood of binding a drug-like molecule — above 0.7 is actionable |
| Cryptic Pocket Ratio | `count(pockets where allosteric_coupling > 0.5) / total_pockets` | `X/Y` | Teal >50%, Amber 25–50%, Magenta <25% | Fraction of pockets that are allosterically coupled (not just surface cavities) — cryptic > canonical for this platform |

**Good vs Anomalous:**
- Good: 1–3 pockets with high druggability (>0.7), at least one cryptic (allosteric_coupling > 0.5)
- Anomalous: Zero pockets (undruggable), or many low-druggability pockets (broad but shallow landscape)

### LLM Briefing Prompt

```json
{
  "system": "You are a structural biology instrument producing cockpit readouts.",
  "prompt": "Structure {pdb_id}. pocket_count={pocket_count}, top_druggability={top_druggability:.2f}, cryptic_ratio={cryptic_count}/{total_pockets}. Pockets: {pocket_summary_list}. Best pocket residues: {best_pocket_residues}. Classify pocket landscape in 2-3 words (e.g. 'Single-cryptic-target', 'Multi-site-druggable', 'Shallow-undruggable'). One sentence on which pocket to prioritize and why. JSON: {\"classification\": \"...\", \"interpretation\": \"...\"}",
  "max_tokens": 80
}
```

### Recommended Viewport Color Mode
`plasticity` — reveals conformational flexibility that defines pocket boundaries

### Recommended Dock Tab
**Pockets** — Phase 4 star; shows pocket cards with druggability scores, volumes, and residue lists

### Interactive Traces
- Click pocket card (dock) → highlights all residues in that pocket on both viewers, 3D camera centers on pocket centroid
- Click "Top Druggability" metric → highlights the best-scoring pocket's residues
- Click cryptic ratio → highlights only allosterically-coupled pockets (allosteric_coupling > 0.5), dims canonical ones

---

## Phase 5: SCREENING

**Question this phase answers:** Which molecular fragments fit these pockets, and what's the binding potential?

### Hero Metrics (2)

| Metric | Data Source | Format | Color Rule | Why It Matters |
|--------|-------------|--------|------------|----------------|
| ADMET Pass Rate | `drug_candidates.admet_passed_count / drug_candidates.count` | `XX%` | Teal ≥60%, Amber 30–60%, Magenta <30% | Fraction of candidates passing absorption/distribution/metabolism/excretion/toxicity filters — high = viable leads |
| State-Selective Hits | `drug_candidates.state_selective_count` | Integer | Teal ≥2, Amber 1, Magenta 0 | Candidates that preferentially bind the mutant over wild-type — the goal for precision oncology |

**Good vs Anomalous:**
- Good: ADMET pass rate >60%, at least 2 state-selective candidates (precision medicine viable)
- Anomalous: Zero state-selective hits (no specificity), low ADMET (toxic or non-bioavailable candidates)

### LLM Briefing Prompt

```json
{
  "system": "You are a structural biology instrument producing cockpit readouts.",
  "prompt": "Structure {pdb_id}. candidate_count={candidate_count}, admet_pass_rate={admet_rate:.0%}, state_selective_count={state_selective}. Top candidate: pocket_{best_pocket_idx} druggability={best_druggability:.2f} selectivity_ratio={best_selectivity:.1f}. Classify screening outcome in 2-3 words (e.g. 'Selective-lead-found', 'Broad-non-selective', 'Toxic-no-leads'). One sentence on next experimental step. JSON: {\"classification\": \"...\", \"interpretation\": \"...\"}",
  "max_tokens": 80
}
```

### Recommended Viewport Color Mode
`allosteric` — shows coupling landscape relevant to selective binding

### Recommended Dock Tab
**Fragments** — Phase 5 star; molecular fragments screened against identified pockets

### Interactive Traces
- Click fragment card → highlights the target pocket residues, shows binding pose overlay in 3D viewer
- Click "State-Selective Hits" metric → highlights pockets of state-selective candidates
- Click ADMET rate → shows pass/fail breakdown tooltip with per-candidate status

---

## Phase 6: REPORT

**Question this phase answers:** What's the full evidence chain, and how confident are we in the findings?

### Hero Metrics (3)

| Metric | Data Source | Format | Color Rule | Why It Matters |
|--------|-------------|--------|------------|----------------|
| Supported Hypotheses | `count(hypotheses where status='supported')` | `X/Y` (supported/total) | Teal ≥50% supported, Amber 25–50%, Magenta <25% | Fraction of scientific claims with positive evidence — measures overall discovery confidence |
| Evidence Depth | `sum(hypotheses[*].evidence_supporting + hypotheses[*].evidence_contradicting)` | Integer | Teal ≥10, Amber 5–10, Magenta <5 | Total evidence gathered — more evidence = more robust conclusions regardless of direction |
| Pipeline Completeness | phases computed (from `persistence_status` booleans) | `X/6 phases` | Teal 6/6, Amber 4–5/6, Magenta <4/6 | How much of the analysis pipeline has run — incomplete phases mean gaps in the evidence chain |

**Good vs Anomalous:**
- Good: High hypothesis support rate, deep evidence, full pipeline completion (ready to publish/act)
- Anomalous: Many contradicted hypotheses (theory doesn't match data — revise), shallow evidence (need more tool runs)

### LLM Briefing Prompt

```json
{
  "system": "You are a structural biology instrument producing cockpit readouts.",
  "prompt": "Structure {pdb_id}. supported={supported_count}/{total_hypotheses}, evidence_depth={evidence_depth}, pipeline={phases_complete}/6. Top supported hypothesis: \"{top_hypothesis_statement}\". Contradicted: {contradicted_count}. Classify report readiness in 2-3 words (e.g. 'Publication-ready', 'Evidence-gaps-remain', 'Theory-revision-needed'). One sentence summarizing the state of discovery. JSON: {\"classification\": \"...\", \"interpretation\": \"...\"}",
  "max_tokens": 80
}
```

### Recommended Viewport Color Mode
`cone_depth` — neutral, comprehensive view showing the full embedding landscape for final review

### Recommended Dock Tab
**Hypotheses** — review all hypotheses and their evidence chains for the final report

### Interactive Traces
- Click supported hypothesis → highlights its evidence residues with green glow
- Click contradicted hypothesis → highlights contradicting residues with magenta glow
- Click "Pipeline Completeness" → shows checklist of which phases have/haven't run with links to trigger missing ones

---

## Color Decision Table

Thresholds determine which of the three signal colors (teal/amber/magenta) to apply to each metric value.

| Metric | Teal (healthy/expected) | Amber (noteworthy) | Magenta (anomalous/discovery) |
|--------|------------------------|--------------------|-----------------------------|
| Mean Epistemic σ | ≤0.15 | 0.15–0.40 | >0.40 |
| Spike Count | 0–2 | 3–5 | >5 |
| Max Cone Depth | ≤1.5 | 1.5–2.5 | >2.5 |
| λ₂ (Fiedler) | ≥0.05 | 0.02–0.05 | <0.02 |
| Bridge Count | 0–1 | 2–3 | >3 |
| Max Betweenness | ≤0.08 | 0.08–0.20 | >0.20 |
| Active Hypotheses | 0 | 1–2 | >2 |
| Mean Displacement | ≤0.5 Å | 0.5–1.5 Å | >1.5 Å |
| Source Leak Count | 0–2 | 3–5 | >5 |
| Pocket Count | 1–2 | 3–4 | >4 or 0 |
| Top Druggability | ≥0.7 | 0.4–0.7 | <0.4 |
| Cryptic Pocket Ratio | >50% | 25–50% | <25% |
| ADMET Pass Rate | ≥60% | 30–60% | <30% |
| State-Selective Hits | ≥2 | 1 | 0 |
| Supported Hypotheses | ≥50% | 25–50% | <25% |
| Evidence Depth | ≥10 | 5–10 | <5 |
| Pipeline Completeness | 6/6 | 4–5/6 | <4/6 |

**Color semantics reminder:**
- **Teal (#5DDBC2)** — within expected range, structurally healthy, connected
- **Amber (#ECA53A)** — transitional, moderate, warrants attention
- **Magenta (#C026D3)** — anomalous signal, NOT an error — this is where discovery happens

**Implementation note:** Color is applied via CSS classes `text-teal`, `text-warning` (amber), `text-magenta`. The metric card's top-right corner accent uses the same color at reduced opacity (the corner-bracket decorative element from the prototype).

---

## Spike Chips Format

When displaying "local epistemic spikes", "top centrality residues", "source leaks", or any residue-metric list:

### Format Specification

```
[3-letter residue name] [residue number] · [metric abbreviation] [value]
```

**Examples:**
- `CYS 12 · σ 0.92` (epistemic spike)
- `GLY 60 · bc 0.31` (betweenness centrality)
- `ASP 119 · lk 0.87` (leak score)
- `VAL 44 · cd 2.41` (cone depth)

### Chip Styling

```tsx
// Chip component structure
<span className={`
  inline-flex items-center gap-1.5 px-2 py-1
  rounded-[var(--radius-badge)]
  font-mono text-[10px] tabular-nums
  cursor-pointer border transition-colors
  hover:border-${chipColor}-dim
  ${isHighlighted ? 'bg-${chipColor}-dim/30 border-${chipColor}-dim text-${chipColor}' : 'bg-bg-elevated border-slate-light text-text-secondary'}
`}>
  <span className="font-semibold">{residueName} {residueNumber}</span>
  <span className="text-text-muted">·</span>
  <span className={`text-${chipColor}`}>{metricAbbrev} {metricValue}</span>
</span>
```

### Color Assignment for Chips

The chip color follows the same threshold table:
- Value in teal range → chip text/border uses teal
- Value in amber range → chip text/border uses amber/warning
- Value in magenta range → chip text/border uses magenta

### Click Behavior

Clicking a spike chip:
1. Emits `selection_sync` with the residue's canonical ID (`{chain_id}:{residue_number}`)
2. Highlights the residue on both Poincaré disc and 3D molecular viewer
3. 3D camera animates to center on the residue
4. Chip enters "highlighted" visual state (filled background)
5. If Möbius focus is enabled, disc recenters on the clicked residue

### Ordering

Chips are sorted by metric value descending (most anomalous first). Maximum 7 chips displayed; if more exist, show "+N more" overflow that expands on click.

---

## Phase Transition Behavior

When the user changes discovery phase (via the phase stepper in NavBar or agent-driven phase_transition):

### 1. Left Panel Content Swap

The briefing panel performs a cross-fade transition (150ms opacity) and renders:
- New phase's hero metrics (fetched from hydration data, already in memory)
- New phase's LLM classification + interpretation (request fires on transition)
- New phase's spike chips / residue lists
- New phase's interpretation paragraph

### 2. Viewport Color Mode Change

The Poincaré disc and 3D viewer switch to the phase-recommended color mode:

| Phase | Poincaré Color Mode | 3D Viewer Color Mode |
|-------|--------------------|--------------------|
| RESIDUE | `epistemic_uncertainty` | `epistemic` |
| TOPOLOGY | `resistance` | `resistance` |
| STRUCTURE | `allosteric` | `allosteric` |
| POCKET | `plasticity` | `pockets` |
| SCREENING | `allosteric` | `drug_candidates` |
| REPORT | `cone_depth` | `cone_depth` |

**Implementation:** Emit `viewport_directive { action: 'set_metric', metric: '<mode>' }` on phase change.

### 3. Dock Tab Switch

The bottom dock foregrounds the phase-star tab:

| Phase | Star Tab |
|-------|----------|
| RESIDUE | Motifs/Clusters |
| TOPOLOGY | Motifs/Clusters |
| STRUCTURE | Hypotheses |
| POCKET | Pockets |
| SCREENING | Fragments |
| REPORT | Hypotheses |

### 4. LLM Insight Refresh

On phase transition:
1. Show shimmer placeholder in the classification area (2-3 word slot + 1-sentence slot)
2. Fire the phase-appropriate LLM briefing prompt with current metric values
3. On response, cross-fade shimmer → classification text + interpretation
4. Cache the response for this phase + structure (invalidate on new pipeline run)

### 5. Clear Previous Selection

Phase transitions clear the current residue selection and highlights to avoid confusion. The scientist starts fresh in each phase's context.

---

## Topology Table Fix

The "Top Centrality" table (visible in phases TOPOLOGY and RESIDUE) must display real computed data:

### Data Source

```typescript
// From hydration response: graph_metrics.metrics[]
interface TopologyRow {
  residue_id: string;       // e.g. "A:60"
  residue_name: string;     // e.g. "GLY" — from embeddings.residues matching by residue_id
  betweenness: number;      // actual betweenness centrality value
  is_bridge: boolean;       // from graph_metrics
  degree: number;           // connection count
  region?: string;          // derived: "Switch II", "P-loop", etc. (from known KRAS regions, or "—" if unknown)
}
```

### Table Rendering Rules

1. **Sort by betweenness descending** — top 7 residues shown
2. **Residue label format:** `{3-letter name} {number}` (e.g. "GLY 60", not "RES" or "A:60")
3. **Centrality value:** shown as `0.XXX` with a proportional bar (width = value / max_value * 100%)
4. **Bridge badge:** Only shown when `is_bridge === true`. Badge text: "BRIDGE" in amber (#ECA53A) with amber border
5. **Region label:** Optional secondary text showing structural region. If region mapping unavailable, show "—" in muted text
6. **Click behavior:** Click row → emits selection for that residue → highlights on all viewers

### Loading State

If graph metrics haven't been computed yet (graph_metrics is null in hydration response):

```tsx
<div className="flex flex-col gap-3 py-4">
  <div className="flex items-center gap-2">
    <div className="w-4 h-4 rounded-full bg-slate-light animate-pulse" />
    <span className="text-xs text-text-muted">Hydrating topology…</span>
  </div>
  {/* 5 shimmer rows */}
  {Array.from({ length: 5 }).map((_, i) => (
    <div key={i} className="h-6 rounded bg-slate-light/50 animate-pulse" style={{ width: `${85 - i * 8}%` }} />
  ))}
</div>
```

### Row Component

```tsx
<div
  onClick={() => onResidueClick(row.residue_id)}
  className="flex items-center justify-between px-2 py-1.5 rounded cursor-pointer
             hover:bg-slate/50 transition-colors group"
>
  <div className="flex items-center gap-2 min-w-0">
    <span className="font-mono text-xs text-teal w-14 shrink-0">
      {row.residue_name} {row.residue_number}
    </span>
    <span className="text-[10px] text-text-muted truncate">
      {row.region ?? "—"}
    </span>
    {row.is_bridge && (
      <span className="text-[8px] font-semibold tracking-wide text-warning
                       border border-warning/30 rounded-full px-1.5 py-0.5">
        BRIDGE
      </span>
    )}
  </div>
  <div className="flex items-center gap-2">
    <div className="w-10 h-[3px] rounded-full bg-slate overflow-hidden">
      <div
        className="h-full rounded-full bg-teal"
        style={{ width: `${(row.betweenness / maxBetweenness) * 100}%` }}
      />
    </div>
    <span className="font-mono text-[10px] text-text-primary w-8 text-right tabular-nums">
      {row.betweenness.toFixed(3)}
    </span>
  </div>
</div>
```

---

## Component Interface

```typescript
interface BriefingPanelProps {
  structureId: string | null;
  pdbId: string | null;
  discoveryPhase: DiscoveryPhase;
  hydrationData: HydrationResponse | null;
  onResidueClick: (residueId: string) => void;
  onColorModeChange: (mode: PanelPoincareColorMode) => void;
  onDockTabChange: (tab: string) => void;
}

interface BriefingMetric {
  label: string;
  value: string | number;
  color: 'teal' | 'warning' | 'magenta';
  tooltip: string;
  onClick?: () => void;
}

interface BriefingInsight {
  classification: string;    // 2-3 words from LLM
  interpretation: string;    // 1 sentence from LLM
  loading: boolean;
}

interface SpikeChip {
  residueId: string;
  residueName: string;
  residueNumber: number;
  metricAbbrev: string;
  metricValue: number;
  color: 'teal' | 'warning' | 'magenta';
}
```

---

## LLM Integration Details

### Request Path

`POST /api/agent/chat` with a system-level briefing prompt. The briefing prompt is NOT shown to the user in chat — it's a background request for the instrument readout.

Alternatively, if a dedicated briefing endpoint exists: `POST /api/briefing/insight`

### Caching Strategy

- Cache key: `{structure_id}:{phase}:{hydration_hash}`
- TTL: Until next pipeline run for this structure (invalidated by pipeline_complete event)
- Stale-while-revalidate: Show cached insight immediately, refresh in background if >5 min old

### Failure Handling

If LLM request fails or times out (>5s):
- Show classification as "—" in muted text
- Show interpretation as "Briefing unavailable" in muted text
- Do NOT block the panel — metrics are always available (computed, not LLM-dependent)

---

## Panel Layout Structure (top to bottom)

```
┌─────────────────────────────────┐
│ HEADER: "Briefing" + collapse   │
├─────────────────────────────────┤
│ ┌─────────────────────────────┐ │
│ │ PROTEIN STATE card          │ │
│ │  • 2-3 word classification  │ │
│ │  • 1-sentence interpretation│ │
│ └─────────────────────────────┘ │
│                                 │
│ SECTION: [Phase-specific title] │
│ ┌───────┐ ┌───────┐           │
│ │Metric1│ │Metric2│ (2-col)   │
│ └───────┘ └───────┘           │
│ ┌───────┐                      │
│ │Metric3│ (if 3 metrics)       │
│ └───────┘                      │
│                                 │
│ SPIKE CHIPS                     │
│ [chip] [chip] [chip] [+N]      │
│                                 │
│ › Interpretation paragraph      │
│                                 │
│ SECTION: Allosteric Topology    │
│ (or phase-specific secondary)   │
│ ┌───────┐ ┌───────┐           │
│ │TopoM1 │ │TopoM2 │           │
│ └───────┘ └───────┘           │
│                                 │
│ TOP CENTRALITY TABLE            │
│  GLY 60  Switch II      0.312  │
│  THR 35  P-loop  BRIDGE 0.287  │
│  ...                            │
│                                 │
│ › Topology interpretation       │
└─────────────────────────────────┘
```

---

## Accessibility

- All metric cards have `aria-label` describing the metric name, value, and status (e.g. "Mean epistemic uncertainty: 0.23, amber — noteworthy")
- Spike chips have `role="button"` and `aria-label` with full residue context
- Color is never the sole differentiator — metric values and textual status are always present
- Topology table rows are focusable with keyboard (Enter/Space triggers click)
- Phase transition announcements use `aria-live="polite"` region for screen readers
- Shimmer loading states have `aria-busy="true"` and descriptive `aria-label`
