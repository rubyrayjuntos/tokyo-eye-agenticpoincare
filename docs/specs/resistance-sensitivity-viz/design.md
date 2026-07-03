# Design Document: Resistance Sensitivity Visualization

## Overview

Surfaces Phase 4 resistance mapping data (effective resistance, hinge residues, coupling pathways) as a per-residue color mode in the 3D Molecular Viewer. The key insight: Phase 4 stores pairwise effective resistance between residue pairs and identifies hinge residues via the Fiedler vector. We derive a per-residue "resistance sensitivity" score by aggregating each residue's participation in high-coupling pathways, with a hinge bonus for domain-boundary residues.

## Architecture

```mermaid
flowchart LR
    subgraph "DB (existing)"
        FRP[fact_resistance_pathway]
        FRS[fact_resistance_spectral]
    end
    subgraph "Backend"
        HYD[/api/hydrate/{id}/] --> QR[Query resistance data]
        QR --> FRP
        QR --> FRS
        QR --> SCORE[Compute per-residue score]
    end
    subgraph "Frontend"
        HP[HydrationProvider] --> MV[MolecularViewer]
        MV --> CM[Color Mode: Resistance]
        MV --> TT[Tooltip enrichment]
    end
    HYD --> HP
```

## Components and Interfaces

### Per-Residue Score Derivation

Phase 4 gives us pairwise pathway data. To derive per-residue scores:

```python
def compute_per_residue_resistance_score(pathways, spectral, residue_ids):
    """Derive per-residue resistance sensitivity from pathway data.
    
    Score = sum of coupling_strength for all pathways involving this residue,
    normalized to [0, 1]. Hinge residues (from Fiedler vector) get a 1.5× boost
    since they control inter-domain communication.
    """
    scores = {rid: 0.0 for rid in residue_ids}
    
    for pathway in pathways:
        src = pathway["source_residue"]
        tgt = pathway["target_residue"]
        coupling = pathway["coupling_strength"]
        if src in scores:
            scores[src] += coupling
        if tgt in scores:
            scores[tgt] += coupling
    
    # Hinge bonus
    hinge_set = set(spectral.get("hinge_residues", []))
    for rid in scores:
        if rid in hinge_set:
            scores[rid] *= 1.5
    
    # Normalize to [0, 1]
    max_score = max(scores.values()) if scores else 1.0
    return {rid: s / (max_score + 1e-8) for rid, s in scores.items()}
```

### Hydration Endpoint Extension

Add `resistance_data` field to the hydration response:

```python
# In dashboard.py hydration handler
resistance_data = await _fetch_resistance_sensitivity(db, structure_id)

# Response shape:
{
    "resistance_data": {
        "residues": [
            {
                "residue_id": "4obe_A_315",
                "sensitivity_score": 0.82,
                "coupling_count": 7,
                "is_hinge": true,
                "classification": "high_sensitivity"
            }
        ],
        "spectral": {
            "lambda_2": 0.045,
            "hinge_count": 12
        }
    }
}
```

Classification thresholds:
- `high_sensitivity`: score > 0.65
- `moderate`: score > 0.3
- `stable`: score <= 0.3

### Frontend Color Mode

Colormap: white → yellow → orange → deep red (sequential warm, same family as plasticity but distinct hue to avoid confusion).

```typescript
function resistanceToHex(t: number): string {
  // t in [0, 1]: 0=stable (white/gray), 1=high sensitivity (deep red)
  if (t < 0.3) { /* white → light yellow */ }
  else if (t < 0.6) { /* yellow → orange */ }
  else { /* orange → deep red */ }
}
```

## Data Models

### ResistanceData (TypeScript)

```typescript
interface ResistanceResidue {
  residue_id: string;
  sensitivity_score: number;
  coupling_count: number;
  is_hinge: boolean;
  classification: "high_sensitivity" | "moderate" | "stable";
}

interface ResistanceData {
  residues: ResistanceResidue[];
  spectral: {
    lambda_2: number;
    hinge_count: number;
  };
}
```

### HydrationResponse Extension

```typescript
interface HydrationResponse {
  // ... existing fields ...
  resistance_data: ResistanceData | null;
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do.*

### Property 1: Resistance data completeness and shape

*For any* structure with Phase 4 pathway data in the database, the hydration response's `resistance_data.residues` array shall contain entries with valid `residue_id`, numeric `sensitivity_score` in [0, 1], non-negative `coupling_count`, boolean `is_hinge`, and a classification from the allowed set.

**Validates: Requirements 1.1, 1.3**

### Property 2: Resistance score monotonicity with coupling count

*For any* two residues A and B where A participates in strictly more pathways than B (both non-hinge), A's sensitivity_score shall be >= B's sensitivity_score.

**Validates: Requirements 2.1, 3.1**

### Property 3: Tooltip data availability matches color mode data

*For any* residue with a non-null sensitivity_score in the resistance data, the tooltip lookup shall return that score and classification without error.

**Validates: Requirements 3.1, 3.3**

## Error Handling

- No Phase 4 data: `resistance_data` returns null; frontend disables color mode or shows "Run pipeline first" message
- Pathway table empty but spectral exists: return spectral-only data with all residues scored at 0
- Division by zero in normalization: handled by `+ 1e-8` guard

## Testing Strategy

**Property-Based Testing Library:** Hypothesis (Python) for backend, fast-check for frontend if needed.

**Approach:**
- Property tests: Score derivation logic (monotonicity, bounds, hinge bonus)
- Unit tests: Hydration endpoint integration, null handling, classification thresholds
- Each property test runs 100+ iterations

**Tag format:** `Feature: resistance-sensitivity-viz, Property {number}: {property_text}`
