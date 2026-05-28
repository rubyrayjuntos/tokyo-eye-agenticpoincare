# Feature Brief: Hypothesis Guardrails & Engine

## Goal

Give the agent a structured framework for generating, tracking, testing, and validating scientific hypotheses. Instead of just reporting numbers, the agent should reason about what findings *mean*, propose testable hypotheses, track their status, and flag when evidence contradicts them.

## Why This Matters

Currently the agent reports facts: "residue 12 has high uncertainty." But a researcher wants: "This suggests G12 is a conformational switch point. If true, we'd expect: (1) high betweenness centrality, (2) H-bond network disruption in the G12D mutant, (3) persistence barcode changes in Phase 3. Let me check..."

The hypothesis engine turns the agent from a data retrieval tool into a scientific reasoning partner.

## Core Concepts

### Hypothesis Lifecycle

```
PROPOSED → EVIDENCE_GATHERING → SUPPORTED / CONTRADICTED / INCONCLUSIVE
```

### Hypothesis Structure

```python
@dataclass
class Hypothesis:
    hypothesis_id: str
    structure_id: str
    statement: str              # "G12 is a conformational switch point"
    mechanism: str | None       # "H-bond network disruption upon mutation"
    predictions: list[Prediction]  # Testable predictions
    evidence: list[Evidence]    # Gathered evidence (for/against)
    status: HypothesisStatus    # proposed, gathering, supported, contradicted, inconclusive
    confidence: float           # 0.0–1.0
    created_at: datetime
    updated_at: datetime
```

### Prediction (testable claim)

```python
@dataclass
class Prediction:
    prediction_id: str
    statement: str              # "Betweenness centrality of G12 > 95th percentile"
    test_tool: str              # "get_graph_metrics"
    test_params: dict           # {"structure_id": "4obe", "residue_ids": ["4obe:A:12"]}
    threshold: str              # "betweenness > 0.15"
    result: str | None          # "betweenness = 0.23 (PASS)"
    passed: bool | None
```

### Evidence

```python
@dataclass
class Evidence:
    evidence_id: str
    source_tool: str            # Which tool produced this
    source_run_id: str          # Provenance link
    supports: bool              # True = supports hypothesis, False = contradicts
    strength: float             # 0.0–1.0
    description: str            # "Betweenness of G12 is 0.23, above 95th percentile (0.15)"
```

## Data Model

### New migration: `032_hypothesis_engine.sql`

```sql
CREATE TABLE hypothesis (
    hypothesis_id   TEXT PRIMARY KEY,
    structure_id    TEXT NOT NULL REFERENCES dim_structure(structure_id),
    statement       TEXT NOT NULL,
    mechanism       TEXT,
    status          TEXT NOT NULL DEFAULT 'proposed',  -- proposed, gathering, supported, contradicted, inconclusive
    confidence      DOUBLE PRECISION DEFAULT 0.5,
    created_by      TEXT NOT NULL,  -- 'agent' or user_id
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE hypothesis_prediction (
    prediction_id   TEXT PRIMARY KEY,
    hypothesis_id   TEXT NOT NULL REFERENCES hypothesis(hypothesis_id),
    statement       TEXT NOT NULL,
    test_tool       TEXT,
    test_params     JSONB,
    threshold       TEXT,
    result          TEXT,
    passed          BOOLEAN,
    tested_at       TIMESTAMPTZ
);

CREATE TABLE hypothesis_evidence (
    evidence_id     TEXT PRIMARY KEY,
    hypothesis_id   TEXT NOT NULL REFERENCES hypothesis(hypothesis_id),
    source_tool     TEXT NOT NULL,
    source_run_id   TEXT,
    supports        BOOLEAN NOT NULL,
    strength        DOUBLE PRECISION DEFAULT 0.5,
    description     TEXT NOT NULL,
    gathered_at     TIMESTAMPTZ DEFAULT NOW()
);
```

## Tools to Implement

### `agent/tools/hypothesis_tools.py`

1. **`propose_hypothesis(structure_id, statement, mechanism?, predictions?)`**
   - Agent proposes a hypothesis with testable predictions
   - Stored in governed layer
   - Returns hypothesis_id

2. **`test_hypothesis(hypothesis_id)`**
   - Runs all predictions for a hypothesis using the appropriate tools
   - Updates evidence and confidence score
   - Returns updated status (supported/contradicted/inconclusive)

3. **`get_hypotheses(structure_id?, status?)`**
   - List hypotheses, optionally filtered by structure or status
   - Shows confidence scores and evidence summary

4. **`add_evidence(hypothesis_id, supports, description, source_tool?)`**
   - Manually add evidence (from user observation or external data)
   - Updates confidence score

5. **`evaluate_confidence(hypothesis_id)`**
   - Recalculate confidence based on all evidence
   - Simple Bayesian update: prior × likelihood of evidence

## Guardrails

The hypothesis engine should enforce scientific rigor:

1. **Falsifiability**: Every hypothesis must have at least one prediction that could contradict it
2. **Evidence balance**: Track both supporting and contradicting evidence
3. **Confidence decay**: Hypotheses without new evidence for N days decay toward 0.5 (inconclusive)
4. **Contradiction alerts**: If new pipeline results contradict an active hypothesis, alert the user
5. **No confirmation bias**: When testing, the agent must look for contradicting evidence too

## Agent Behavior

When the agent identifies a pattern (e.g., source leaks clustered in Switch-I), it should:

1. Propose a hypothesis: "Switch-I residues 10-17 form an allosteric communication hub"
2. Generate predictions:
   - "Betweenness centrality of residues 10-17 should be above average"
   - "G12D mutant should show H-bond loss in this region"
   - "Phase 3 persistence should show topological features anchored here"
3. Test predictions using existing tools
4. Report: "Hypothesis SUPPORTED (confidence 0.78): 3/3 predictions passed"

## Integration with Existing Tools

The hypothesis engine *calls* existing tools to test predictions:
- `get_graph_metrics` → test betweenness predictions
- `compare_graphs` → test H-bond change predictions
- `get_source_leaks` → test source-leak clustering predictions
- `compare_wt_mutant` → test displacement predictions
- `generate_plot` → visualize evidence

## Confidence Calculation

Simple weighted evidence model:

```python
def calculate_confidence(evidence_list: list[Evidence]) -> float:
    if not evidence_list:
        return 0.5  # No evidence = maximum uncertainty
    
    supporting = sum(e.strength for e in evidence_list if e.supports)
    contradicting = sum(e.strength for e in evidence_list if not e.supports)
    total = supporting + contradicting
    
    if total == 0:
        return 0.5
    
    return supporting / total  # 0.0 = fully contradicted, 1.0 = fully supported
```

## Testing

- Unit test: hypothesis CRUD, confidence calculation, prediction testing logic
- Integration test: propose → test → evaluate cycle with mock tool results
- Guardrail test: reject hypothesis without falsifiable predictions
- Decay test: verify confidence decays over time without new evidence
