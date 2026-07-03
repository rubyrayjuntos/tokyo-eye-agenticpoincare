# Implementation Plan: V6 MoE Specialization

## Overview

Implements the v6 GOSPConeMapper with topologically-routed MoE specialization. Creates a fresh `science/dtie/v6/` directory, expands the gate input vector, implements asymmetric capacity regularization, builds the three-phase training schedule, and curates a ~120 protein training corpus.

## Tasks

- [x] 1. Create v6 directory structure and model skeleton
  - [x] 1.1 Create `science/dtie/v6/__init__.py`, `science/dtie/v6/gnn/__init__.py`, `science/dtie/v6/gnn/model.py`
    - Copy v5 model as starting point, rename class to GOSPConeMapperV6
    - _Requirements: 7.1, 7.2_

  - [x] 1.2 Implement TopologicalMoEGateV6 with expanded input (hidden+7)
    - Gate accepts [x_tangent, clustering, cone_depth, log_degree_norm, rho_norm, ss_onehot]
    - Include running statistics buffers for degree/rho normalization
    - _Requirements: 1.1, 1.4_

  - [x] 1.3 Write property test for gate input dimension correctness
    - **Property 1: Gate input dimension correctness**
    - **Validates: Requirements 1.1, 1.2, 1.3**

- [x] 2. Implement capacity-aware routing and asymmetric loss
  - [x] 2.1 Implement two-pass capacity-aware logit adjustment in the gate
    - First pass: compute expected load from unpenalized softmax
    - Second pass: subtract quadratic penalty for overloaded experts, recompute softmax
    - _Requirements: 4.1, 4.2_

  - [x] 2.2 Implement asymmetric capacity loss function
    - Loss = Σ max(0, 0.05 - f_i)² — only penalizes starvation
    - _Requirements: 2.1, 2.2_

  - [x] 2.3 Write property test for asymmetric capacity loss
    - **Property 2: Asymmetric capacity loss is zero when all experts above minimum**
    - **Validates: Requirements 2.1, 2.2**

  - [x] 2.4 Write property test for capacity penalty monotonicity
    - **Property 4: Capacity penalty monotonically increases with overload**
    - **Validates: Requirements 4.1, 4.2**

- [x] 3. Implement expert dropout
  - [x] 3.1 Add expert dropout to gate forward pass (training only, p=0.15)
    - Mask one random expert by setting logits to -inf
    - Redistribute weight via softmax over remaining experts
    - No-op in eval mode
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 3.2 Write property test for expert dropout probability mass conservation
    - **Property 3: Expert dropout preserves probability mass**
    - **Validates: Requirements 3.1, 3.2**

- [x] 4. Extend GraphBuilder with topological features
  - [x] 4.1 Compute node degree from edge_index and attach to PyG Data
    - Count edges per node, store as `data.degree`
    - _Requirements: 1.2_

  - [x] 4.2 Compute SS one-hot encoding and attach to PyG Data
    - Convert scalar ss_type (H=0, E=1, C=2) to one-hot [3] vector
    - Store as `data.ss_onehot`
    - _Requirements: 1.3_

  - [x] 4.3 Attach raw rho values to PyG Data for gate shortcut
    - Store as `data.rho` tensor
    - _Requirements: 1.1_

- [x] 5. Checkpoint - Ensure model forward pass works end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement v6 training script with three-phase schedule
  - [x] 6.1 Create `experiments/training/v6/train_v6.py` with V6TrainingConfig dataclass
    - Phase 1 (1-50): balance_coeff=0.1, no dropout
    - Phase 2 (50-200): balance_coeff=0.001, dropout=0.15, asymmetric loss
    - Phase 3 (200+): freeze gate, fine-tune experts at 0.5× LR
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 6.2 Implement v5 backbone warm-start loading
    - Load SE(3) convolutions, RadialHead, AngularHead from v5 checkpoint
    - Skip gate and expert weights (fresh initialization)
    - _Requirements: 7.5_

  - [x] 6.3 Add expert specialization validation at end of training
    - Compute per-expert routing entropy, mean degree, mean ρ, SS distribution
    - Flag failure if entropy > 1.2 after Phase 2
    - _Requirements: 8.1, 8.2, 8.3_

- [x] 7. Create v6 GNN runner
  - [x] 7.1 Create `science/dtie/v6/gnn/runner.py` implementing GNNRunner protocol
    - Produce GNNInferenceResult with all required fields
    - Include expert_load and routing_entropy in output metadata
    - _Requirements: 7.4_

  - [x] 7.2 Write property test for v6 output compatibility
    - **Property 5: V6 output compatibility with GNNInferenceResult**
    - **Validates: Requirements 7.4**

- [x] 8. Curate training corpus
  - [x] 8.1 Create `data/training/v6_corpus.json` manifest with fold-class targets
    - Define selection criteria (resolution ≤ 2.5Å, max 30% identity, 20-2000 residues)
    - Populate with ~120 PDB IDs covering all 8 fold classes
    - Include chain, residue count, and fold class for each entry
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 8.2 Write property test for corpus manifest validation
    - **Property 7: Training corpus fold-class coverage**
    - **Validates: Requirements 6.1, 6.2, 6.3**

- [x] 9. Implement v6 loss function (gosp_loss_v6)
  - [x] 9.1 Create gosp_loss_v6 combining all loss terms with phase-aware coefficients
    - Include asymmetric capacity loss, expert dropout awareness
    - Preserve cone_loss, neighborhood_consistency, angular_diversity, domain_sep from v5
    - Add routing entropy monitoring
    - _Requirements: 2.1, 4.1, 5.1, 5.2_

- [x] 10. Checkpoint - Ensure training script runs one epoch without error
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Expert routing persistence
  - [x] 11.1 Extend GNN normalizer to persist expert_load and routing_entropy per run
    - Store in fact_gnn_node_embedding or a new metadata column
    - _Requirements: 8.4_

- [x] 12. Final checkpoint - Full integration test
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required (comprehensive testing from start) — no optional tasks
- The v6 model is a fresh implementation — no backward compatibility constraints with v5 checkpoints for the gate/experts
- V5 backbone weights (SE(3) convolutions + RadialHead + AngularHead) can be warm-started
- The training corpus is curated programmatically using RCSB search + sequence identity filtering
- Property tests validate the mathematical invariants of the routing system
- The three-phase schedule is the most critical piece — incorrect phase transitions will prevent specialization
