# Requirements Document

## Introduction

The State-Dependent Resistance Profile (SDRP) module replaces the single-consensus resistance classification with a multi-dimensional conformational profile. Instead of aggregating GNN outputs from multiple PDB structures into one "Consensus Class," the SDRP treats each structure as a conformational sensor — capturing how a mutation's resistance mechanism varies across drug-binding states (e.g., imatinib-bound, dasatinib-bound, apo).

The key biological insight: mutations like H396R and Q252H exhibit different resistance mechanisms depending on the protein's conformational state. This is not classification error — it is a high-fidelity readout of the protein's thermodynamic behavior. The SDRP formalizes this as a "Conformational Switch" vs "Static Disruptor" distinction using the State-Sensitivity Score (SSS), computed via Jensen-Shannon Divergence of per-state classification probabilities.

## Glossary

- **SDRP_Engine**: The module that orchestrates multi-structure resistance profiling and computes state-dependent profiles.
- **State_Profile**: A mapping from conformational state (identified by PDB structure and its binding context) to the resistance classification and stability score for that state.
- **Conformational_Sensitivity**: A scalar (0.0–1.0) measuring how much a mutation's classification varies across conformational states. High values indicate "Conformational Switches."
- **State_Sensitivity_Score (SSS)**: The Jensen-Shannon Divergence of the per-state classification probability distributions. Quantifies the "plasticity" of a mutation's resistance mechanism.
- **Stability_Score**: Per-state confidence that the classification is stable (not near a decision boundary). Derived from the distance of metrics to classification thresholds.
- **Conformational_Switch**: A mutation with high SSS (>0.5) whose resistance mechanism changes across binding states (e.g., H396R, Q252H).
- **Static_Disruptor**: A mutation with low SSS (<0.3) whose resistance mechanism is consistent across all conformational states (e.g., T315I, V299L).
- **Binding_Context**: Metadata describing the conformational state of a PDB structure (drug identity, binding mode, apo/holo status).
- **Ensemble_Profile**: The complete SDRP output for a single variant across all provided structures.
- **Mechanism_Shift**: An event where a mutation's classification changes between two conformational states (e.g., Type_II_Allosteric in imatinib-bound → Hybrid in dasatinib-bound).
- **Resistance_Profiler**: The existing single-structure resistance classification module that the SDRP_Engine wraps.

## Requirements

### Requirement 1: Multi-Structure Ensemble Profiling

**User Story:** As a researcher, I want to profile a mutation across multiple PDB structures representing different conformational states, so that I can observe how the resistance mechanism varies with binding context.

#### Acceptance Criteria

1. WHEN a variant and a list of structure entries (each with structure_id and binding_context) are provided, THE SDRP_Engine SHALL run the existing Resistance_Profiler independently against each structure and collect per-structure ResistanceReports
2. WHEN profiling across structures, THE SDRP_Engine SHALL reuse the existing per-structure baseline caching from the Resistance_Profiler (one WT baseline per structure_id)
3. IF any single structure fails during ensemble profiling, THEN THE SDRP_Engine SHALL record the error for that structure and continue processing remaining structures
4. WHEN all structures have been profiled, THE SDRP_Engine SHALL assemble the results into an Ensemble_Profile containing the State_Profile mapping

### Requirement 2: State-Dependent Resistance Profile Schema

**User Story:** As a researcher, I want the output to be a multi-dimensional profile rather than a single consensus class, so that I can see how each conformational state independently classifies the mutation.

#### Acceptance Criteria

1. THE SDRP_Engine SHALL output an Ensemble_Profile containing: variant name, state_profile (mapping from binding_context label to classification and stability_score), conformational_sensitivity, sss_score, clinical_relevance summary, and the category (Conformational_Switch or Static_Disruptor)
2. WHEN a state_profile entry is generated, THE SDRP_Engine SHALL include the mechanism_class, stability_score, site_uncertainty_delta, and max_hub_delta for that state
3. THE SDRP_Engine SHALL NOT produce a single "Consensus Class" field in the output
4. THE SDRP_Engine SHALL serialize all output as JSON with numpy values converted to native Python types

### Requirement 3: State-Sensitivity Score (SSS) Computation

**User Story:** As a researcher, I want a quantitative score measuring how much a mutation's mechanism varies across states, so that I can rank mutations by their conformational plasticity.

#### Acceptance Criteria

1. WHEN an Ensemble_Profile contains classifications from two or more structures, THE SDRP_Engine SHALL compute the SSS as the Jensen-Shannon Divergence of the per-state classification probability vectors
2. THE SDRP_Engine SHALL map each mechanism_class to a probability vector over the class space {Type_I_Steric, Type_II_Allosteric, Hybrid, Neutral} using the confidence_score as the weight for the classified category and distributing (1 - confidence) uniformly across remaining categories
3. WHEN all structures produce the same mechanism_class with high confidence, THE SDRP_Engine SHALL produce an SSS near 0.0
4. WHEN structures produce different mechanism_classes, THE SDRP_Engine SHALL produce an SSS proportional to the divergence between the distributions
5. THE SDRP_Engine SHALL normalize the SSS to the range [0.0, 1.0]

### Requirement 4: Stability Score Computation

**User Story:** As a researcher, I want a per-state stability score indicating how confidently the classification holds for each conformational state, so that I can identify which binding contexts produce borderline classifications.

#### Acceptance Criteria

1. WHEN a per-state classification is produced, THE SDRP_Engine SHALL compute a stability_score between 0.0 and 1.0 based on the margin between the observed metrics and the classification thresholds
2. WHEN the metrics are far from all decision boundaries, THE SDRP_Engine SHALL produce a stability_score near 1.0
3. WHEN the metrics are close to a decision boundary, THE SDRP_Engine SHALL produce a stability_score near 0.0

### Requirement 5: Conformational Switch vs Static Disruptor Categorization

**User Story:** As a researcher, I want mutations automatically categorized as Conformational Switches or Static Disruptors, so that I can quickly identify which variants require state-aware drug design strategies.

#### Acceptance Criteria

1. WHEN the SSS exceeds 0.5, THE SDRP_Engine SHALL categorize the mutation as a Conformational_Switch
2. WHEN the SSS is below 0.3, THE SDRP_Engine SHALL categorize the mutation as a Static_Disruptor
3. WHEN the SSS is between 0.3 and 0.5, THE SDRP_Engine SHALL categorize the mutation as Intermediate with a note indicating borderline plasticity
4. THE SDRP_Engine SHALL generate a clinical_relevance summary string describing the drug-design implications of the categorization

### Requirement 6: Mechanism Shift Detection

**User Story:** As a researcher, I want to know exactly which conformational transitions cause a mechanism shift, so that I can understand the structural basis of state-dependent resistance.

#### Acceptance Criteria

1. WHEN two structures produce different mechanism_classes for the same variant, THE SDRP_Engine SHALL record a Mechanism_Shift entry identifying the source state, target state, source class, and target class
2. WHEN mechanism shifts are detected, THE SDRP_Engine SHALL include them in the Ensemble_Profile output
3. THE SDRP_Engine SHALL annotate each shift with the delta in site_uncertainty and max_hub_delta between the two states

### Requirement 7: Batch Ensemble Profiling

**User Story:** As a researcher, I want to profile multiple variants across the same ensemble of structures in one invocation, so that I can efficiently generate SDRPs for an entire variant library.

#### Acceptance Criteria

1. WHEN a list of variants and a list of structure entries are provided, THE SDRP_Engine SHALL produce an Ensemble_Profile for each variant
2. WHEN processing a batch, THE SDRP_Engine SHALL compute each structure's WT baseline only once (shared across all variants for that structure)
3. THE SDRP_Engine SHALL report progress during batch processing (variants completed / total)
4. THE SDRP_Engine SHALL return results in the same order as the input variant list

### Requirement 8: Persistence of Ensemble Profiles

**User Story:** As a researcher, I want ensemble profiles persisted to the database, so that I can query historical profiles and track how classifications evolve as new structures are added.

#### Acceptance Criteria

1. WHEN an Ensemble_Profile is computed, THE SDRP_Engine SHALL persist it to a fact_ensemble_resistance_profile table with the variant, sss_score, category, state_profile JSON, and mechanism_shifts JSON
2. WHEN persisting, THE SDRP_Engine SHALL record provenance linking the ensemble profile to all individual per-structure resistance profile run_ids
3. IF a profile for the same variant and structure ensemble already exists, THEN THE SDRP_Engine SHALL upsert (update the existing record)

