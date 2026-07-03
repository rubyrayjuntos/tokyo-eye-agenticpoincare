# Requirements Document

## Introduction

The v5 GOSPConeMapper GNN successfully maps protein residues to a Poincaré disc with meaningful radial-angular structure, but its Mixture-of-Experts (MoE) gate is trapped in a high-entropy uniform state — all 4 experts receive near-equal routing weights (~0.25 each) regardless of residue topology. This is caused by insufficient gate input variance, an overpowering balance loss, and a training set of only 11 proteins that lacks the diversity to force expert specialization. V6 resolves this by expanding the gate's discriminative features, implementing asymmetric capacity regularization, and training on a structurally diverse dataset of ~120 proteins spanning all major fold classes.

## Glossary

- **MoE_Gate**: The Mixture-of-Experts routing module that assigns per-residue routing weights to 4 expert MLPs based on topological features
- **Expert_Specialization**: The state where each expert processes a distinct topological role (core packing, interface, functional surface, structural transition) rather than acting as a uniform ensemble
- **Asymmetric_Capacity_Loss**: A loss function that penalizes expert starvation (< 5% usage) but permits legitimate dominance (up to 80%)
- **Expert_Dropout**: Randomly masking one expert during training to build redundancy and prevent catastrophic failure at inference
- **Gate_Feature_Vector**: The expanded input to the MoE gate: [x_tangent, clustering, cone_depth, degree, ρ, ss_onehot]
- **Topological_Role**: One of four spatial categories — Core (buried hub), Interface (domain boundary), Surface (exposed functional), Transition (hinge/bridge)
- **Training_Corpus**: A curated set of ~120 proteins covering all SCOP fold classes with controlled redundancy

## Requirements

### Requirement 1: Expanded MoE Gate Input

**User Story:** As a model architect, I want the MoE gate to receive high-variance topological features directly, so that it has sufficient discriminative signal to break routing symmetry.

#### Acceptance Criteria

1. THE Gate SHALL accept an input vector of [x_tangent, clustering_coefficient, cone_depth, node_degree, rho, ss_onehot_3] (hidden + 7 dimensions)
2. WHEN computing node_degree, THE GraphBuilder SHALL count the number of edges per node from the contact graph and attach it to the PyG Data object
3. WHEN computing ss_onehot, THE GraphBuilder SHALL encode secondary structure as a 3-dimensional one-hot vector [helix, sheet, coil]
4. THE Gate SHALL apply a log(1 + x) transform to node_degree, then normalize both node_degree and rho inputs to zero-mean unit-variance using running statistics computed during training

### Requirement 2: Asymmetric Capacity Loss

**User Story:** As a training engineer, I want the balance loss to penalize expert starvation without forcing uniformity, so that experts can legitimately dominate when the data warrants it.

#### Acceptance Criteria

1. THE Training_Loop SHALL compute asymmetric capacity loss as Σ max(0, min_usage - f_i)² where min_usage = 0.05
2. THE Training_Loop SHALL NOT penalize any expert that receives more than min_usage fraction of tokens

### Requirement 3: Expert Dropout

**User Story:** As a model architect, I want experts to be randomly masked during training, so that remaining experts build redundancy for handling diverse topological inputs.

#### Acceptance Criteria

1. WHILE training, THE Model SHALL randomly mask one expert per forward pass with probability 0.15
2. WHEN an expert is masked, THE System SHALL redistribute its routing weight proportionally among remaining experts
3. WHILE in inference mode, THE Model SHALL use all experts without masking

### Requirement 4: Capacity-Aware Soft Routing

**User Story:** As a training engineer, I want a differentiable pressure valve that discourages expert overload, so that borderline residues naturally flow to underutilized experts.

#### Acceptance Criteria

1. THE Gate SHALL compute an expected load fraction ρ̃_i as the batch-mean of the unpenalized softmax routing probabilities for each expert
2. WHEN ρ̃_i exceeds capacity_threshold (default 0.4), THE Gate SHALL subtract λ · max(0, ρ̃_i - τ)² from the pre-softmax logits of expert i and recompute the softmax
3. THE capacity_threshold SHALL be configurable per training phase

### Requirement 5: Three-Phase Training Schedule

**User Story:** As a training engineer, I want a structured training schedule that first stabilizes representations, then breaks symmetry, then locks routing, so that expert specialization converges reliably.

#### Acceptance Criteria

1. DURING Phase 1 (epochs 1–50), THE System SHALL use balance_coeff=0.1 with no expert dropout
2. DURING Phase 2 (epochs 50–200), THE System SHALL decay balance_coeff to 0.001, enable expert dropout at p=0.15, and switch to asymmetric capacity loss
3. DURING Phase 3 (epochs 200+), THE System SHALL freeze the gate parameters and fine-tune experts independently
4. THE Training_Script SHALL accept phase boundaries as CLI arguments with the above defaults

### Requirement 6: Diverse Training Corpus

**User Story:** As a scientist, I want the model trained on a structurally diverse protein set, so that expert specialization generalizes across fold classes.

#### Acceptance Criteria

1. THE Training_Corpus SHALL contain approximately 120 proteins with resolution ≤ 2.5 Å
2. THE Training_Corpus SHALL include at minimum: 20 all-α, 20 all-β, 20 α/β, 15 α+β, 15 multi-domain with hinge motions, 10 membrane proteins, 10 intrinsically disordered regions, 10 small peptides/miniproteins
3. THE Training_Corpus SHALL enforce maximum 30% sequence identity between any two members (to prevent redundancy)
4. THE Training_Corpus SHALL be defined as a JSON manifest with PDB ID, fold class, chain, and residue count

### Requirement 7: V6 Model Architecture

**User Story:** As a developer, I want the v6 model in a clean directory with no backward-compatibility constraints, so that architectural changes don't pollute the production v5 codebase.

#### Acceptance Criteria

1. THE V6 Model SHALL reside in `science/dtie/v6/gnn/model.py` as a fresh implementation
2. THE V6 Model SHALL preserve the decoupled RadialHead/AngularHead architecture from v5
3. THE V6 Model SHALL use the expanded gate input (hidden+7) and new capacity-aware routing
4. THE V6 Runner SHALL produce outputs compatible with the existing GNNInferenceResult interface
5. THE V6 Model SHALL support loading v5 backbone weights (SE(3) convolutions + RadialHead + AngularHead) for warm-start training

### Requirement 8: Expert Specialization Validation

**User Story:** As a scientist, I want to verify that experts have specialized after training, so that I can confirm the MoE is functioning as a topological parser.

#### Acceptance Criteria

1. WHEN training completes, THE System SHALL compute per-expert routing entropy and report it
2. THE System SHALL flag training as failed if routing entropy exceeds 1.2 (out of max 1.386) after Phase 2
3. THE System SHALL produce a per-expert profile showing mean degree, mean ρ, mean clustering, and SS distribution for residues routed to each expert
4. THE System SHALL persist expert routing statistics to the database alongside embeddings
