# Configure make to use bash instead of sh for better error handling
SHELL := /bin/bash -o pipefail

# Map container UID/GID to host so bind mounts (mlruns, checkpoints) are writable
DOCKER_USER := $(shell id -u):$(shell id -g)
# Forward GNN_INPUT_MODE into the science container. Prefixing
# `GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN)` alone does NOT reach the
# container — compose must receive `-e GNN_INPUT_MODE` (or an environment: entry).
# Without this, targets silently defaulted to legacy_four_vector (node_dim=4).
SCIENCE_RUN := docker compose run --rm -e DB_POOL_MIN_SIZE=2 -e DB_POOL_MAX_SIZE=10 -e GNN_INPUT_MODE --user $(DOCKER_USER)
STAGE_A_MAX_RESIDUES := $(shell PYTHONPATH=. python3 -c "from science.training.corpus_governance import STAGE_A_MAX_RESIDUES; print(STAGE_A_MAX_RESIDUES)" 2>/dev/null || echo 650)
# Host-mapped UID often has no writable $HOME in the container; install test deps to /tmp.
TEST_DEPS_DIR := /tmp/tokyoeye-pytest-deps
TEST_RUN_PREFIX := pip install --quiet --target $(TEST_DEPS_DIR) pytest pytest-asyncio hypothesis httpx && PYTHONPATH=$(TEST_DEPS_DIR):$$PYTHONPATH PYTEST_CACHE_DIR=/tmp/tokyoeye-pytest-cache python -m pytest

# ---------------------------------------------------------------------------
# V6 GNN training lifecycle (science container only)
# ---------------------------------------------------------------------------

train-v6: ## Train v6 GNN (STAGE=1|2|3, RESUME=..., MAX_PROTEINS=N, MAX_RESIDUES=800)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_120.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),$(shell date +%Y%m%d_%H%M%S)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--mlflow-uri http://mlflow:5000 \
		$(if $(STAGE),--phase $(STAGE),) \
		$(if $(MAX_PROTEINS),--max-proteins $(MAX_PROTEINS),) \
		$(if $(MAX_RESIDUES),--max-residues $(MAX_RESIDUES),) \
		$(if $(NO_MLFLOW),--no-mlflow,) \
		$(if $(WARM_START),--warm-start-v5 /app/$(WARM_START),) \
		$(if $(RESUME),--resume /app/$(RESUME),)

# In-repo v3 teacher + benchmark PDBs (science/dtie/v3, science/dtie/assets)
BENCHMARK_PDB_CONTAINER := /app/science/dtie/assets/benchmark_pdbs
V2_TEACHER_CKPT_CONTAINER := /app/science/dtie/v3/checkpoints/v2_bridge_epoch_014.pt

train-v6-benchmark: ## GPU smoke on science/dtie benchmark_pdbs + v3 teacher (EPOCHS=3 MAX_PROTEINS=8)
	@test -d science/dtie/assets/benchmark_pdbs || (echo "Missing science/dtie/assets/benchmark_pdbs" && exit 1)
	@test -f science/dtie/v3/checkpoints/v2_bridge_epoch_014.pt || (echo "Missing v2_bridge teacher checkpoint" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(if $(TRAINING_LOAD_FROM_PDB),docker compose run --rm -e DB_POOL_MIN_SIZE=2 -e DB_POOL_MAX_SIZE=10 -e GNN_INPUT_MODE -e TRAINING_LOAD_FROM_PDB=1 --user $(DOCKER_USER),$(SCIENCE_RUN)) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_benchmark.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),benchmark_$(shell date +%Y%m%d_%H%M%S)) \
		--pdb-dir $(or $(PDB_DIR),$(BENCHMARK_PDB_CONTAINER)) \
		--device $(or $(DEVICE),cuda) \
		--phase $(or $(STAGE),1) \
		--epochs $(or $(EPOCHS),3) \
		--max-proteins $(or $(MAX_PROTEINS),8) \
		--max-residues $(or $(MAX_RESIDUES),600) \
		--no-corpus-cache \
		$(if $(USE_MLFLOW),--mlflow-uri http://mlflow:5000,) \
		$(if $(USE_MLFLOW),,--no-mlflow) \
		--v2-teacher-checkpoint $(V2_TEACHER_CKPT_CONTAINER) \
		$(if $(NO_WARM_START),--no-warm-start,) \
		$(if $(GENTLE_P2),--gentle-phase2,) \
		$(if $(PHASE2_LR),--phase2-lr $(PHASE2_LR),) \
		$(if $(SAVE_EPOCH_SNAPSHOTS),--save-epoch-snapshots,) \
		$(if $(P1C),--p1c,) \
		$(if $(P1C_LR),--p1c-lr $(P1C_LR),) \
		$(if $(P1D),--p1d,) \
		$(if $(P1D_LR),--p1d-lr $(P1D_LR),) \
		$(if $(P1D_EXTEND),--p1d-extend,) \
		$(if $(DISC_DEPTH_SCALE_COEFF),--p1d-disc-depth-scale-coeff $(DISC_DEPTH_SCALE_COEFF),) \
		$(if $(DISC_TARGET_START),--p1d-disc-target-start $(DISC_TARGET_START),) \
		$(if $(DISC_TARGET_END),--p1d-disc-target-end $(DISC_TARGET_END),) \
		$(if $(P1D_FREEZE_RADIAL_EPOCHS),--p1d-freeze-radial-epochs $(P1D_FREEZE_RADIAL_EPOCHS),) \
		$(if $(P1D_MIN_PROBE_R_DEPTH_SASA),--p1d-min-probe-r-depth-sasa $(P1D_MIN_PROBE_R_DEPTH_SASA),) \
		$(if $(P1D_DISC_SPREAD_MIN_STD),--p1d-disc-spread-min-std $(P1D_DISC_SPREAD_MIN_STD),) \
		$(if $(P2_BRIDGE),--p2-bridge,) \
		$(if $(P2_BRIDGE_LR),--p2-bridge-lr $(P2_BRIDGE_LR),) \
		$(if $(P2_BRIDGE_RAMP_EPOCHS),--p2-bridge-ramp-epochs $(P2_BRIDGE_RAMP_EPOCHS),) \
		$(if $(P2_HYPMIX),--p2-hypmix,) \
		$(if $(P2_HYPMIX3),--p2-hypmix3,) \
		$(if $(P2_HYPMIX_FINAL),--p2-hypmix-final,) \
		$(if $(P2_DISC_OCCUPANCY),--p2-disc-occupancy,) \
		$(if $(P2_DISC_OCCUPANCY_V2),--p2-disc-occupancy-v2,) \
		$(if $(P2_DISC_OCCUPANCY_V3),--p2-disc-occupancy-v3,) \
		$(if $(P2_DISC_OCCUPANCY_V4),--p2-disc-occupancy-v4,) \
		$(if $(P2_DISC_OCCUPANCY_V5),--p2-disc-occupancy-v5,) \
		$(if $(P2_DISC_GENTLE_ARCH),--p2-disc-gentle-arch,) \
		$(if $(P2_DISC_PATH_ALIGN),--p2-disc-path-align,) \
		$(if $(P2_DISC_PROJ_RECOVERY),--p2-disc-proj-recovery,) \
		$(if $(P2_DISC_PROJ_RECOVERY_V2),--p2-disc-proj-recovery-v2,) \
		$(if $(P2_DISC_PROJ_RECOVERY_V3),--p2-disc-proj-recovery-v3,) \
		$(if $(P2_DISC_PROJ_RECOVERY_V4),--p2-disc-proj-recovery-v4,) \
		$(if $(P2_DISC_PROJ_RECOVERY_V5),--p2-disc-proj-recovery-v5,) \
		$(if $(P2_REC_ABLATION),--p2-rec-ablation,) \
		$(if $(RADIAL_ANGULAR_RECOMBINE),--radial-angular-recombine $(RADIAL_ANGULAR_RECOMBINE),) \
		$(if $(DISC_RADIAL_SOURCE),--disc-radial-source $(DISC_RADIAL_SOURCE),) \
		$(if $(DISC_OCCUPANCY_COEFF),--p2-disc-occupancy-coeff $(DISC_OCCUPANCY_COEFF),) \
		$(if $(DISC_PATH_ALIGN_COEFF),--p2-disc-path-align-coeff $(DISC_PATH_ALIGN_COEFF),) \
		$(if $(EFF_RANK_COEFF),--p2-disc-eff-rank-coeff $(EFF_RANK_COEFF),) \
		$(if $(BATCH_DIVERSITY_COEFF),--p2-disc-batch-diversity-coeff $(BATCH_DIVERSITY_COEFF),) \
		$(if $(filter 1,$(LEGACY_DISC_PROJECTION)),--legacy-disc-projection,) \
		$(if $(filter 0,$(LEGACY_DISC_PROJECTION)),--no-legacy-disc-projection,) \
		$(if $(RADIAL_FREEZE_EPOCHS),--p2-radial-freeze-epochs $(RADIAL_FREEZE_EPOCHS),) \
		$(if $(DISC_R_STD_FLOOR),--p2-disc-r-std-floor $(DISC_R_STD_FLOOR),) \
		$(if $(DISC_LINE_THICKNESS_FLOOR),--p2-disc-line-thickness-floor $(DISC_LINE_THICKNESS_FLOOR),) \
		$(if $(DISC_SCATTER_INTERVAL),--disc-scatter-interval $(DISC_SCATTER_INTERVAL),) \
		$(if $(FULL_HYP_MOE_TEST),--full-hyp-moe-test,) \
		$(if $(THEORY_TEST_LR),--theory-test-lr $(THEORY_TEST_LR),) \
		$(if $(DEEP_HYPERBOLIC_GATE),--deep-hyperbolic-gate,) \
		$(if $(GATE_DISC_SCALE),--gate-disc-scale $(GATE_DISC_SCALE),) \
		$(if $(GATE_GUMBEL),--gate-gumbel,) \
		$(if $(HYPERBOLIC_EXPERT_MIX),--hyperbolic-expert-mix,) \
		$(if $(P4_EPISTEMIC_DECOUPLING),--p4-epistemic-decoupling,) \
		$(if $(P4_EPISTEMIC_LR),--p4-epistemic-lr $(P4_EPISTEMIC_LR),) \
		$(if $(EPISTEMIC_DECOUPLING_HOLDOUTS),--epistemic-decoupling-holdouts $(EPISTEMIC_DECOUPLING_HOLDOUTS),) \
		$(if $(EPISTEMIC_BF_ALIGN_COEFF),--epistemic-bf-align-coeff $(EPISTEMIC_BF_ALIGN_COEFF),) \
		$(if $(EPISTEMIC_SASA_PEN_COEFF),--epistemic-sasa-pen-coeff $(EPISTEMIC_SASA_PEN_COEFF),) \
		$(if $(SHELL_CORR_EPI_SASA),--shell-corr-epi-sasa-weight $(SHELL_CORR_EPI_SASA),) \
		$(if $(P4_EPISTEMIC_STAGED),--p4-epistemic-staged,) \
		$(if $(P1B),--p1b,) \
		$(if $(P1B_LR),--p1b-lr $(P1B_LR),) \
		$(if $(V2_TEACHER_DEPTH),--v2-teacher-depth-coeff $(V2_TEACHER_DEPTH),) \
		$(if $(V2_TEACHER_EPISTEMIC),--v2-teacher-epistemic-coeff $(V2_TEACHER_EPISTEMIC),) \
		$(if $(DEHYDRON_RIM_RECOVERY),--dehydron-rim-recovery,) \
		$(if $(DEHYDRON_RIM_RECOVERY_LR),--dehydron-rim-recovery-lr $(DEHYDRON_RIM_RECOVERY_LR),) \
		$(if $(LR),--lr $(LR),) \
		$(if $(RESUME),--resume /app/$(RESUME),)

train-v6-p1c-disc: ## P1c disc expansion from shell_p1c best (override RUN_ID, RESUME, EPOCHS)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p1c_disc) EPOCHS=$(or $(EPOCHS),5) \
		P1C=1 NO_WARM_START=1 SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p1c/v6_best.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p1d-disc: ## P1d disc-depth scale from shell_p1c_disc best (MobiusLinear head)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p1d_disc) EPOCHS=$(or $(EPOCHS),5) \
		P1D=1 NO_WARM_START=1 SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p1c_disc/v6_best.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p1d-disc2: ## P1d extension: slower disc ramp + higher scale coeff (8 ep default)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p1d_disc2) EPOCHS=$(or $(EPOCHS),8) \
		P1D=1 P1D_EXTEND=1 NO_WARM_START=1 SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p1d_disc/v6_best.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p2-bridge: ## P2 bridge from disc3 best (relaxed routing save + shell guard)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_bridge) EPOCHS=$(or $(EPOCHS),12) \
		P2_BRIDGE=1 NO_WARM_START=1 SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p1d_disc3/v6_best.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p2-bridge2: ## P2 bridge extension: slower routing ceiling ramp (15 ep default)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_bridge2) EPOCHS=$(or $(EPOCHS),15) \
		P2_BRIDGE=1 P2_BRIDGE_RAMP_EPOCHS=$(or $(P2_BRIDGE_RAMP_EPOCHS),15) \
		NO_WARM_START=1 SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_bridge/v6_best.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p2-hypmix: ## P2 bridge + disc gate inputs + Möbius expert mix (from bridge2 best)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_hypmix) EPOCHS=$(or $(EPOCHS),12) \
		P2_BRIDGE=1 P2_BRIDGE_RAMP_EPOCHS=$(or $(P2_BRIDGE_RAMP_EPOCHS),15) \
		HYPERBOLIC_EXPERT_MIX=1 NO_WARM_START=1 SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_bridge2/v6_best.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p2-hypmix2: ## Hypmix2: stronger MoE pressure, shell≥0.65, disc×2, Gumbel (15 ep)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_hypmix2) EPOCHS=$(or $(EPOCHS),15) \
		P2_BRIDGE=1 P2_HYPMIX=1 P2_BRIDGE_RAMP_EPOCHS=$(or $(P2_BRIDGE_RAMP_EPOCHS),15) \
		HYPERBOLIC_EXPERT_MIX=1 NO_WARM_START=1 SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_bridge2/v6_best.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p2-hypmix3: ## Hypmix3: lock MoE from hypmix2 champion (dropout 0.15, disc×2.5)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_hypmix3) EPOCHS=$(or $(EPOCHS),12) \
		P2_BRIDGE=1 P2_HYPMIX3=1 P2_BRIDGE_RAMP_EPOCHS=$(or $(P2_BRIDGE_RAMP_EPOCHS),15) \
		HYPERBOLIC_EXPERT_MIX=1 NO_WARM_START=1 SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_hypmix2/v6_best.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p2-hypmix-final: ## Final lock-in from ep71 (dropout 0.18, capacity 0.01, disc×2.5)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_hypmix_final) EPOCHS=$(or $(EPOCHS),10) \
		P2_BRIDGE=1 P2_HYPMIX_FINAL=1 P2_BRIDGE_RAMP_EPOCHS=$(or $(P2_BRIDGE_RAMP_EPOCHS),15) \
		HYPERBOLIC_EXPERT_MIX=1 NO_WARM_START=1 SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_hypmix3_extend/epochs/epoch_071.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-theory-test: ## Full hyperbolic MoE theory test from hypmix_final ep78 lock-in
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),theory_test_final) EPOCHS=$(or $(EPOCHS),15) \
		P2_BRIDGE=1 FULL_HYP_MOE_TEST=1 P2_BRIDGE_RAMP_EPOCHS=$(or $(P2_BRIDGE_RAMP_EPOCHS),15) \
		HYPERBOLIC_EXPERT_MIX=1 NO_WARM_START=1 SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_hypmix_final/v6_best.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p2-disc-occupancy: ## Disc occupancy recovery from production (σ₂/σ₁ gate, 15 ep default)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_disc_occupancy) EPOCHS=$(or $(EPOCHS),15) \
		P2_DISC_OCCUPANCY=1 P2_BRIDGE=1 P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),3e-5) \
		RESUME=$(or $(RESUME),checkpoints/v6/tokyo_eyes_v6.pt) \
		$(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p2-disc-occupancy-v2: ## Refined disc recovery (v2 loss + ep90 warm-start, 12 ep default)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_disc_occupancy_v2) EPOCHS=$(or $(EPOCHS),12) \
		P2_DISC_OCCUPANCY_V2=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),2e-5) \
		DISC_OCCUPANCY_COEFF=$(or $(DISC_OCCUPANCY_COEFF),2.8) \
		DISC_R_STD_FLOOR=$(or $(DISC_R_STD_FLOOR),0.02) \
		RADIAL_FREEZE_EPOCHS=$(or $(RADIAL_FREEZE_EPOCHS),4) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_disc_occupancy_snap/epochs/epoch_090.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p2-disc-occupancy-v3: ## Cluster-break push from v2b best_disc (coeff 3.5, LR 1e-5, 12 ep)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_disc_occupancy_v3) EPOCHS=$(or $(EPOCHS),12) \
		P2_DISC_OCCUPANCY_V3=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),1e-5) \
		DISC_OCCUPANCY_COEFF=$(or $(DISC_OCCUPANCY_COEFF),3.5) \
		DISC_R_STD_FLOOR=$(or $(DISC_R_STD_FLOOR),0.045) \
		RADIAL_FREEZE_EPOCHS=$(or $(RADIAL_FREEZE_EPOCHS),5) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_disc_occupancy_v2b/v6_best_disc.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-p2-disc-occupancy-v4: ## Target-corpus disc push (11QE/4OBE/1IVO + eff_rank, 12 ep)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_disc_occupancy_v4) EPOCHS=$(or $(EPOCHS),12) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		P2_DISC_OCCUPANCY_V4=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),1e-5) \
		DISC_OCCUPANCY_COEFF=$(or $(DISC_OCCUPANCY_COEFF),3.5) \
		EFF_RANK_COEFF=$(or $(EFF_RANK_COEFF),1.0) \
		DISC_R_STD_FLOOR=$(or $(DISC_R_STD_FLOOR),0.045) \
		RADIAL_FREEZE_EPOCHS=$(or $(RADIAL_FREEZE_EPOCHS),5) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_disc_occupancy_v2b/v6_best_disc.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-p2-disc-occupancy-v5: ## Batch diversity repulsion on target corpus (v4 + pairwise PC2-residual)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_disc_occupancy_v5) EPOCHS=$(or $(EPOCHS),12) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		P2_DISC_OCCUPANCY_V5=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),1e-5) \
		DISC_OCCUPANCY_COEFF=$(or $(DISC_OCCUPANCY_COEFF),3.5) \
		EFF_RANK_COEFF=$(or $(EFF_RANK_COEFF),1.0) \
		BATCH_DIVERSITY_COEFF=$(or $(BATCH_DIVERSITY_COEFF),1.25) \
		DISC_R_STD_FLOOR=$(or $(DISC_R_STD_FLOOR),0.045) \
		RADIAL_FREEZE_EPOCHS=$(or $(RADIAL_FREEZE_EPOCHS),5) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_disc_occupancy_v4/epochs/epoch_107.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-p2-disc-occupancy-v5b: ## Stronger batch diversity (coeff 2.5) from v5 ep119
	$(MAKE) train-v6-p2-disc-occupancy-v5 RUN_ID=$(or $(RUN_ID),shell_p2_disc_occupancy_v5b) \
		EPOCHS=$(or $(EPOCHS),10) \
		BATCH_DIVERSITY_COEFF=$(or $(BATCH_DIVERSITY_COEFF),2.5) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_disc_occupancy_v5/epochs/epoch_119.pt) \
		$(if $(USE_MLFLOW),USE_MLFLOW=1,) SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda)

train-v6-disc-arch-retrain: ## Retrain pre-routing disc path from production (gentle occupancy, 15 ep)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),disc_arch_retrain_v1) EPOCHS=$(or $(EPOCHS),15) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		LEGACY_DISC_PROJECTION=0 \
		P2_DISC_OCCUPANCY=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),1e-5) \
		DISC_OCCUPANCY_COEFF=$(or $(DISC_OCCUPANCY_COEFF),0.5) \
		DISC_R_STD_FLOOR=$(or $(DISC_R_STD_FLOOR),0.03) \
		RADIAL_FREEZE_EPOCHS=$(or $(RADIAL_FREEZE_EPOCHS),3) \
		RESUME=$(or $(RESUME),checkpoints/v6/tokyo_eyes_v6.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-disc-gentle-retrain: ## Gentle new-path retrain from early recovery (preserve visual spread)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),disc_gentle_arch_v1) EPOCHS=$(or $(EPOCHS),10) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		LEGACY_DISC_PROJECTION=0 \
		P2_DISC_GENTLE_ARCH=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),5e-6) \
		DISC_OCCUPANCY_COEFF=$(or $(DISC_OCCUPANCY_COEFF),1.2) \
		DISC_R_STD_FLOOR=$(or $(DISC_R_STD_FLOOR),0.04) \
		DISC_LINE_THICKNESS_FLOOR=$(or $(DISC_LINE_THICKNESS_FLOOR),0.02) \
		RADIAL_FREEZE_EPOCHS=$(or $(RADIAL_FREEZE_EPOCHS),4) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),3) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_disc_occupancy/phase_2_disc_occupancy_recovery.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-disc-proj-recovery: ## Projection head recovery (thickness/span floors, 20 ep)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),disc_proj_recovery_v1) EPOCHS=$(or $(EPOCHS),20) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		LEGACY_DISC_PROJECTION=0 \
		P2_DISC_PROJ_RECOVERY=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),2e-5) \
		DISC_LINE_THICKNESS_FLOOR=$(or $(DISC_LINE_THICKNESS_FLOOR),0.025) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),2) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_disc_occupancy/phase_2_disc_occupancy_recovery.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-disc-proj-recovery-v5: ## v5: angular_lift (no cone multiply) + lift-path recovery
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),disc_proj_recovery_v5) EPOCHS=$(or $(EPOCHS),20) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		LEGACY_DISC_PROJECTION=0 \
		P2_DISC_PROJ_RECOVERY_V5=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),3e-5) \
		DISC_LINE_THICKNESS_FLOOR=$(or $(DISC_LINE_THICKNESS_FLOOR),0.025) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),2) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_disc_occupancy/phase_2_disc_occupancy_recovery.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-disc-proj-recovery-v4: ## v4: radial+angular+fusion + x_hyp spread loss
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),disc_proj_recovery_v4) EPOCHS=$(or $(EPOCHS),20) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		LEGACY_DISC_PROJECTION=0 \
		P2_DISC_PROJ_RECOVERY_V4=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),2e-5) \
		DISC_LINE_THICKNESS_FLOOR=$(or $(DISC_LINE_THICKNESS_FLOOR),0.025) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),2) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/disc_proj_recovery_v3/phase_2_disc_projection_recovery.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-lever-a-clean-slate: ## Lever A foundation: healthy shell + radial_depth pre-disc (target corpus)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),lever_a_clean_slate) EPOCHS=$(or $(EPOCHS),15) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_hypmix_final/v6_best.pt) \
		LEGACY_DISC_PROJECTION=0 \
		DISC_RADIAL_SOURCE=radial_depth \
		P2_DISC_OCCUPANCY_V4=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),2e-5) \
		DISC_LINE_THICKNESS_FLOOR=$(or $(DISC_LINE_THICKNESS_FLOOR),0.025) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),2) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-lever-a-promote: ## Materialize v6_best.pt from lever_a foundation disc checkpoint (1 ep)
	$(MAKE) train-v6-lever-a-clean-slate RUN_ID=$(or $(RUN_ID),lever_a_clean_slate_promote) EPOCHS=1 \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt)

train-v6-p4-epistemic-decoupling: ## Phase 4: B-factor residual epistemic decoupling (warm-start lever_a)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),p4_epistemic_decoupling) EPOCHS=$(or $(EPOCHS),30) \
		CORPUS=$(or $(CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt) \
		P4_EPISTEMIC_DECOUPLING=1 \
		P4_EPISTEMIC_LR=$(or $(P4_EPISTEMIC_LR),$(LR),1e-4) \
		EPISTEMIC_DECOUPLING_HOLDOUTS=$(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO,4MNE) \
		$(if $(EPISTEMIC_BF_ALIGN_COEFF),EPISTEMIC_BF_ALIGN_COEFF=$(EPISTEMIC_BF_ALIGN_COEFF),) \
		$(if $(EPISTEMIC_SASA_PEN_COEFF),EPISTEMIC_SASA_PEN_COEFF=$(EPISTEMIC_SASA_PEN_COEFF),) \
		$(if $(SHELL_CORR_EPI_SASA),SHELL_CORR_EPI_SASA=$(SHELL_CORR_EPI_SASA),) \
		$(if $(P4_EPISTEMIC_STAGED),P4_EPISTEMIC_STAGED=1,) \
		LEGACY_DISC_PROJECTION=0 \
		DISC_RADIAL_SOURCE=radial_depth \
		V2_TEACHER_DEPTH=$(or $(V2_TEACHER_DEPTH),0.20) \
		V2_TEACHER_EPISTEMIC=0 \
		DISC_LINE_THICKNESS_FLOOR=$(or $(DISC_LINE_THICKNESS_FLOOR),0.18) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),2) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),)

train-v6-p4-epistemic-v3: ## Phase 4 Option B: λ₁-only ep1-10, staged λ₂, shell epi_sasa=0
	$(MAKE) train-v6-p4-epistemic-decoupling RUN_ID=$(or $(RUN_ID),p4_epistemic_v3) EPOCHS=$(or $(EPOCHS),30) \
		P4_EPISTEMIC_STAGED=1 \
		EPISTEMIC_BF_ALIGN_COEFF=$(or $(EPISTEMIC_BF_ALIGN_COEFF),0.22) \
		EPISTEMIC_SASA_PEN_COEFF=$(or $(EPISTEMIC_SASA_PEN_COEFF),0.10) \
		SHELL_CORR_EPI_SASA=$(or $(SHELL_CORR_EPI_SASA),0.0) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt) \
		DEVICE=$(or $(DEVICE),cuda)

probe-encoder-bf-signal: ## Pre-v3 gate: encoder hidden vs B-factor | depth,sasa (CHECKPOINT=...)
	python -m experiments.diagnostics.encoder_bf_signal_probe --run \
		--checkpoint "$(or $(CHECKPOINT),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt)" \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--structures "$(if $(STRUCTURES),$(STRUCTURES),11QE,4OBE,1IVO,4MNE)" \
		--device "$(or $(DEVICE),cpu)" \
		--out "$(or $(JSON_OUT),checkpoints/v6/diagnostics/encoder_bf_signal_probe.json)"

project-crescent-biology: ## Tier 1/2 biology overlays on lever_a crescent (read-only)
	@mkdir -p checkpoints/v6/diagnostics/crescent_projections
	python -m experiments.diagnostics.crescent_biology_projection --run \
		--checkpoint "$(or $(CHECKPOINT),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt)" \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--structures "$(if $(STRUCTURES),$(STRUCTURES),11QE,4OBE,4DSO,1IVO,4MNE)" \
		--device "$(or $(DEVICE),cpu)" \
		--output-dir "$(or $(OUTPUT_DIR),checkpoints/v6/diagnostics/crescent_projections)"

project-pharmacophore-surface: ## 3D ν_epi surface + GOSP pharmacophore labels (Possibility B legend)
	@mkdir -p checkpoints/v6/diagnostics/pharmacophore_surface
	python -m experiments.diagnostics.pharmacophore_surface_render --run \
		--checkpoint "$(or $(CHECKPOINT),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt)" \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--structures "$(if $(STRUCTURES),$(STRUCTURES),11QE,4DSO)" \
		--device "$(or $(DEVICE),cpu)" \
		--output-dir "$(or $(OUTPUT_DIR),checkpoints/v6/diagnostics/pharmacophore_surface)"

prereg-9est-selftest: ## Verify locked 9EST/1FLE benchmark metric code (pre-data)
	python -m experiments.diagnostics.evaluate_9est_prereg --selftest

prereg-9est-phase1: ## Phase 1: lock I_gold from 1FLE→9EST (output-blind)
	python -m experiments.diagnostics.evaluate_9est_prereg --phase1 \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)"

prereg-9est-phase2: ## Phase 2: PeSTo precondition on static 9EST (PESTO_ROOT=/tmp/PeSTo)
	python -m experiments.diagnostics.evaluate_9est_prereg --phase2 \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--pesto-root "$(or $(PESTO_ROOT),/tmp/PeSTo)" \
		--device "$(or $(DEVICE),cpu)"

prereg-9est-run: ## Phase 3+4: blind Tokyo Eye evaluation (requires Phase 2 PASS)
	python -m experiments.diagnostics.evaluate_9est_prereg --run \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--checkpoint "$(or $(CHECKPOINT),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt)" \
		--device "$(or $(DEVICE),cpu)"

demo-9est-cryptic-pipeline: ## Exploratory: offline scan vs I_gold on 9EST (NOT inference prereg)
	python -m experiments.diagnostics.demo_9est_cryptic_pipeline --write-json \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--checkpoint "$(or $(CHECKPOINT),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt)" \
		--device "$(or $(DEVICE),cpu)"

ingest-9est-pipeline: ## Ingest 9EST + queue discovery pathway (requires make up)
	@curl -sf -X POST http://localhost:8000/api/ingest \
		-H "Content-Type: application/json" \
		-d '{"pdb_id":"9EST","force_reingest":true}' | python -m json.tool

ingest-corpus-master-features: ## MASTER features → dim_residue for 12-prot corpus (requires dim rows + DATABASE_URL)
	PYTHONPATH=. python3 -m experiments.training.v6.ingest_corpus_master_features \
		--manifest manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--pdb-dir pdb_cache \
		--verify \
		$(if $(DRY_RUN),--dry-run,)

ingest-corpus-master-features-dry-run: ## Dry-run MASTER feature ingest (compute only)
	$(MAKE) ingest-corpus-master-features DRY_RUN=1

# Active default: v66 feeler-expand corpus. Stage A-12 archive regeneration:
#   make precompute-dehydron-barcodes CORPUS=v6_corpus_stage_a_small_v1.json \
#     OUT_DIR=checkpoints/v65/dehydron_barcode_v1 MAX_PROTEINS=12
precompute-dehydron-barcodes: ## Cache dehydron_barcode_v1_2 sidecars (CORPUS=, OUT_DIR=; default=v66 feeler)
	@test -f manifests/$(or $(CORPUS),v6_corpus_stage_a_feeler_expand_v1.json) || (echo "Missing manifest manifests/$(or $(CORPUS),v6_corpus_stage_a_feeler_expand_v1.json)" && exit 1)
	@mkdir -p $(or $(OUT_DIR),checkpoints/v66/dehydron_barcode_v1) pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v6.precompute_dehydron_barcodes \
		--manifest /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_feeler_expand_v1.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--out-dir /app/$(or $(OUT_DIR),checkpoints/v66/dehydron_barcode_v1) \
		$(if $(BINNED),--binned,) \
		$(if $(MAX_PROTEINS),--max-proteins $(MAX_PROTEINS),)

diagnose-dehydron-bar-length: ## Pure-TDA bar-length + dehydron-count diagnostic (long-lived threshold)
	@test -f manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) || (echo "Missing manifest manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json)" && exit 1)
	@mkdir -p $(or $(OUT_DIR),checkpoints/v65/diagnostics/dehydron_bar_length_v1) pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.dehydron_bar_length_threshold \
		--manifest /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--out-dir /app/$(or $(OUT_DIR),checkpoints/v65/diagnostics/dehydron_bar_length_v1) \
		$(if $(MAX_PROTEINS),--max-proteins $(MAX_PROTEINS),) \
		$(if $(NOISE_FLOOR),--noise-floor $(NOISE_FLOOR),) \
		-v
	@echo "Bar-length diagnostic written under $(or $(OUT_DIR),checkpoints/v65/diagnostics/dehydron_bar_length_v1)"

diagnose-dehydron-scalar-orthogonality: ## Scalar redundancy vs rho/tau/SS + peers (Stage A-12)
	@test -f manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) || (echo "Missing manifest manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json)" && exit 1)
	@test -d $(or $(RAW_BARS_DIR),checkpoints/v65/diagnostics/dehydron_bar_length_v1/bars_raw) || (echo "Missing raw bars — run: make diagnose-dehydron-bar-length" && exit 1)
	@mkdir -p $(or $(OUT_DIR),checkpoints/v65/diagnostics/dehydron_scalar_orthogonality_v1) pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.dehydron_scalar_orthogonality \
		--manifest /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--raw-bars-dir /app/$(or $(RAW_BARS_DIR),checkpoints/v65/diagnostics/dehydron_bar_length_v1/bars_raw) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--out-dir /app/$(or $(OUT_DIR),checkpoints/v65/diagnostics/dehydron_scalar_orthogonality_v1) \
		$(if $(MAX_PROTEINS),--max-proteins $(MAX_PROTEINS),) \
		$(if $(REDUNDANCY_THRESHOLD),--redundancy-threshold $(REDUNDANCY_THRESHOLD),) \
		-v

V66_DBH_BARCODE_DIR := checkpoints/v66/dehydron_barcode_v1
V66_DBH_P2_RESUME := checkpoints/v66/runs/feeler_expand_23_rho_rim_p2_v1/v66_phase2_23prot.pt
V66_FEELER_P3_RESUME := checkpoints/v66/runs/feeler_expand_23_p3_geom_v1/v66_best_disc.pt
V66_FEELER_P4_RESUME := checkpoints/v66/runs/feeler_expand_23_p3_geom_p4_v1/v66_best_disc.pt
V66_FEELER_CHAMPION := $(V66_FEELER_P4_RESUME)
V66_DBH_P3_RESUME := $(V66_FEELER_P3_RESUME)

precompute-dehydron-barcodes-feeler-expand: ## Barcode sidecars for feeler expand 23-prot corpus
	$(MAKE) precompute-dehydron-barcodes \
		CORPUS=v6_corpus_stage_a_feeler_expand_v1.json \
		OUT_DIR=$(V66_DBH_BARCODE_DIR) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),23)

# ---------------------------------------------------------------------------
# V6.5 GNN training (isolated checkpoint namespace + MLflow experiment)
# ---------------------------------------------------------------------------

# Learned-GNN parent for barcode ablation (NOT slim MoE). Override with RESUME=.
# ARCHIVE: current dehydron investigation is v66 feeler + edge barcode
# (docs/specs/dehydron-barcode-input-channel/ablation.md). Do not use these
# train-v65-dbh-* targets for feeler work.
DBH_RESUME_DEFAULT := checkpoints/v65/runs/master_cold_v1/v65_best.pt
DBH_BARCODE_DIR := checkpoints/v65/dehydron_barcode_v1

train-v65-dbh-baseline: ## [ARCHIVE] v65 barcode ablation baseline — use train-v66-feeler-p3-geom-edges
	@echo "WARNING: train-v65-dbh-baseline is archive path; active investigation is v66 feeler edges (ablation.md)"
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f $(or $(RESUME),$(DBH_RESUME_DEFAULT)) || \
		(echo "Missing learned resume checkpoint — run: make train-v65-master-cold RUN_ID=master_cold_v1" && exit 1)
	@mkdir -p mlruns checkpoints/v65/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v65.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v65/runs/$(or $(RUN_ID),dbh_ablation_baseline) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v65 \
		--no-warm-start \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--phase 3 \
		--epochs $(or $(EPOCHS),15) \
		--resume /app/$(or $(RESUME),$(DBH_RESUME_DEFAULT)) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "DBH ablation baseline complete. Run: checkpoints/v65/runs/$(or $(RUN_ID),dbh_ablation_baseline)"

train-v65-dbh-scalars: ## [ARCHIVE] v65 node-global scalars — use train-v66-feeler-p3-geom-edges
	@echo "WARNING: train-v65-dbh-scalars is archive path; node-global scalars deprecated on v66 feeler"
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f $(or $(RESUME),$(DBH_RESUME_DEFAULT)) || \
		(echo "Missing learned resume checkpoint — run: make train-v65-master-cold RUN_ID=master_cold_v1" && exit 1)
	@test -d $(DBH_BARCODE_DIR) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes" && exit 1)
	@mkdir -p mlruns checkpoints/v65/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v65.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v65/runs/$(or $(RUN_ID),dbh_ablation_scalars) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v65 \
		--no-warm-start \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--phase 3 \
		--epochs $(or $(EPOCHS),15) \
		--resume /app/$(or $(RESUME),$(DBH_RESUME_DEFAULT)) \
		--use-dehydron-barcode \
		--dehydron-barcode-dir /app/$(DBH_BARCODE_DIR) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "DBH ablation scalars complete. Run: checkpoints/v65/runs/$(or $(RUN_ID),dbh_ablation_scalars)"

train-v65-dbh-full: ## [ARCHIVE] v65 scalars+binned — deferred; active path is v66 edges
	@echo "WARNING: train-v65-dbh-full is archive path; active investigation is v66 feeler edges (ablation.md)"
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f $(or $(RESUME),$(DBH_RESUME_DEFAULT)) || \
		(echo "Missing learned resume checkpoint — run: make train-v65-master-cold RUN_ID=master_cold_v1" && exit 1)
	@test -d $(DBH_BARCODE_DIR) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes BINNED=1" && exit 1)
	@mkdir -p mlruns checkpoints/v65/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v65.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v65/runs/$(or $(RUN_ID),dbh_ablation_full) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v65 \
		--no-warm-start \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--phase 3 \
		--epochs $(or $(EPOCHS),15) \
		--resume /app/$(or $(RESUME),$(DBH_RESUME_DEFAULT)) \
		--use-dehydron-barcode \
		--use-binned-dehydron \
		--dehydron-barcode-dir /app/$(DBH_BARCODE_DIR) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "DBH ablation full complete. Run: checkpoints/v65/runs/$(or $(RUN_ID),dbh_ablation_full)"

run-9est-pipeline: ## Re-run discovery pathway on ingested 9EST (science container)
	@docker compose exec -T science python -c "import urllib.request,json; print(json.dumps(json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:8001/compute/pipeline', data=json.dumps({'structure_id':'9est'}).encode(), headers={'Content-Type':'application/json'}, method='POST'))), indent=2))"

eval-9est-ingested-sites: ## Compare fact_cryptic_site to I_gold after ingest (STRUCTURE_ID=9est)
	DATABASE_URL=postgresql://tokyoeye:tokyoeye_dev_local@localhost:5432/tokyoeye_dev \
	python -m experiments.diagnostics.demo_9est_cryptic_pipeline --from-db \
		"$(or $(STRUCTURE_ID),9est)" --write-json \
		--output "$(or $(OUTPUT),data/benchmarks/9est_ingest_cryptic_eval.json)"

.PHONY: prereg-9est-record
prereg-9est-record: ## Phase 5 pointer — §12 in PREREG_9EST_1FLE_cryptic_interface.md + 9est_prereg_verdict.md
	@test -f data/benchmarks/9est_phase34_results.json || (echo "Run make prereg-9est-run first" && exit 1)
	@echo "Phase 5 recorded. See:"
	@echo "  experiments/diagnostics/PREREG_9EST_1FLE_cryptic_interface.md (§12)"
	@echo "  data/benchmarks/9est_prereg_verdict.md"

smoke-p4-optimizer: ## Stage 1 gate: AdamW uncertainty_head moves (CHECKPOINT=..., DEVICE=...)
	python -m experiments.training.v6.p4_optimizer_smoke \
		--checkpoint "$(or $(CHECKPOINT),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt)" \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--manifest "$(or $(MANIFEST),manifests/v6_corpus_disc_target.json)" \
		--device "$(or $(DEVICE),cuda)" \
		--lr $(or $(P4_EPISTEMIC_LR),1e-4)

audit-v6-lever-a-foundation: ## Slice audit — lever_a clean slate pre/post control metrics
	$(MAKE) diagnose-radial-angular-slice \
		CHECKPOINT=$(or $(CHECKPOINT),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt) \
		STRUCTURES=$(or $(STRUCTURES),11QE:A,4OBE:A,1IVO:A) \
		LEGACY_DISC_PROJECTION=0 \
		DISC_RADIAL_SOURCE=radial_depth \
		JSON_OUT=$(or $(JSON_OUT),checkpoints/v6/diagnostics/lever_a_foundation_slice.json)

train-v6-lever-c-tangent-mix: ## Lever C: tangent expert mix (hyperbolic_expert_mix off) + radial_depth
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),lever_c_tangent_mix) EPOCHS=$(or $(EPOCHS),12) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt) \
		LEGACY_DISC_PROJECTION=0 \
		DISC_RADIAL_SOURCE=radial_depth \
		P2_BRIDGE=1 GENTLE_P2=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),2e-5) \
		DISC_LINE_THICKNESS_FLOOR=$(or $(DISC_LINE_THICKNESS_FLOOR),0.025) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),2) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-lever-c-post-radial: ## Lever A₂: post-routing depth_routed override (symmetric disc)
	$(MAKE) train-v6-lever-a-promote RUN_ID=$(or $(RUN_ID),lever_c_post_radial) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt) \
		EPOCHS=$(or $(EPOCHS),1)

train-v6-disc-proj-recovery-v3: ## v3: unfreeze angular+fusion+disc head (fix x_hyp wedge)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),disc_proj_recovery_v3) EPOCHS=$(or $(EPOCHS),20) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		LEGACY_DISC_PROJECTION=0 \
		P2_DISC_PROJ_RECOVERY_V3=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),2e-5) \
		DISC_LINE_THICKNESS_FLOOR=$(or $(DISC_LINE_THICKNESS_FLOOR),0.025) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),2) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/disc_proj_recovery_v2/phase_2_disc_projection_recovery.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-disc-proj-recovery-v2: ## v2 recovery: no path_align, mlp_fusion, stronger thickness floor
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),disc_proj_recovery_v2) EPOCHS=$(or $(EPOCHS),20) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		LEGACY_DISC_PROJECTION=0 \
		P2_DISC_PROJ_RECOVERY_V2=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),2e-5) \
		DISC_LINE_THICKNESS_FLOOR=$(or $(DISC_LINE_THICKNESS_FLOOR),0.025) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),2) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_disc_occupancy/phase_2_disc_occupancy_recovery.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-rec-ablation: ## Test residual radial×angular fusion (5 ep, bridge losses only)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),rec_ablation_v1) EPOCHS=$(or $(EPOCHS),5) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		LEGACY_DISC_PROJECTION=0 \
		P2_REC_ABLATION=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),2e-5) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),1) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_disc_occupancy/phase_2_disc_occupancy_recovery.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-disc-path-align: ## Legacy-teacher path alignment (no occupancy; train disc proj + gate readout)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),disc_path_align_v1) EPOCHS=$(or $(EPOCHS),8) \
		CORPUS=$(or $(TARGET_CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		LEGACY_DISC_PROJECTION=0 \
		P2_DISC_PATH_ALIGN=1 P2_BRIDGE=1 \
		P2_BRIDGE_LR=$(or $(P2_BRIDGE_LR),$(LR),1e-5) \
		DISC_PATH_ALIGN_COEFF=$(or $(DISC_PATH_ALIGN_COEFF),10.0) \
		DISC_LINE_THICKNESS_FLOOR=$(or $(DISC_LINE_THICKNESS_FLOOR),0.02) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),2) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/shell_p2_disc_occupancy/phase_2_disc_occupancy_recovery.pt) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-p2-disc-occupancy-snap: ## 5-epoch snapshot pass to materialize epoch_090 from production
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),shell_p2_disc_occupancy_snap) EPOCHS=5 \
		P2_DISC_OCCUPANCY=1 P2_BRIDGE=1 P2_BRIDGE_LR=3e-5 \
		RESUME=checkpoints/v6/tokyo_eyes_v6.pt SAVE_EPOCH_SNAPSHOTS=1 \
		DEVICE=$(or $(DEVICE),cuda) MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

train-v6-mlflow-governance-smoke: ## 1-epoch lever_a resume + full MLflow governance schema (CPU ok)
	$(MAKE) train-v6-lever-a-clean-slate RUN_ID=$(or $(RUN_ID),mlflow_governance_smoke) EPOCHS=1 \
		USE_MLFLOW=1 DEVICE=$(or $(DEVICE),cpu) \
		RESUME=$(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3)

train-v6-stage-a-smoke: ## 1-epoch locked Stage A corpus + MLflow (P_STAGE_A_SMOKE — before full curriculum)
	@test -f manifests/v6_corpus_stage_a.json || (echo "Missing locked Stage A manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt) || \
		(echo "Missing lever_a resume checkpoint for warm-start" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),stage_a_smoke_$(shell date +%Y%m%d_%H%M%S)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cpu) \
		--phase 1 \
		--epochs 1 \
		--max-proteins $(or $(MAX_PROTEINS),8) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt)
	@echo "Smoke complete. Verify with: STAGE_A_SMOKE_RUN_ID=<run_id> make test-stage-a-smoke"

train-v6-stage-a-curriculum: ## Full 3-phase Stage A on locked corpus (25 proteins, epoch snapshots, lever_a warm-start)
	@test -f manifests/v6_corpus_stage_a.json || (echo "Missing locked Stage A manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt) || \
		(echo "Missing lever_a resume checkpoint for warm-start" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),stage_a_$(shell date +%Y%m%d_%H%M%S)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),25) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt) \
		--save-epoch-snapshots
	@echo "Curriculum complete. Read trajectory: checkpoints/v6/runs/$(or $(RUN_ID),stage_a_*)"

gate-p-feature-01: ## P_FEATURE_01 round-trip gate: SSOT recompute == DB ingest == training read
	@test -f manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) || (echo "Missing corpus manifest" && exit 1)
	@mkdir -p data/gates pdb_cache mlruns
	$(SCIENCE_RUN) science python -m science.training.p_feature_01_gate \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--stamp-path /app/data/gates/p_feature_01_passed.json
	@echo "P_FEATURE_01 passed — stamp: data/gates/p_feature_01_passed.json"

train-v6-stage-a-small-corpus: ## 12-protein fold-diverse expansion (requires gate-p-feature-01 stamp)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt) || \
		(echo "Missing lever_a resume checkpoint for warm-start" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),stage_a_small_$(shell date +%Y%m%d_%H%M%S)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt) \
		--save-epoch-snapshots
	@echo "Small corpus curriculum complete. Read trajectory: checkpoints/v6/runs/$(or $(RUN_ID),stage_a_small_*)"

train-v6-stage-a-small-master-cold: ## Cold-start 12-prot MASTER features (P1→P2→P3, lineage root, no resume)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),stage_a_small_master_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--no-warm-start \
		--master-cold-lineage \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--save-epoch-snapshots
	@echo "MASTER cold-start complete. Run: checkpoints/v6/runs/$(or $(RUN_ID),stage_a_small_master_cold_v1)"

train-v6-slim-moe-structural-ssot: ## Cold-start 12-prot slim MoE + frozen structural disc SSOT (inference-aligned)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),slim_moe_structural_ssot_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--no-warm-start \
		--slim-moe-structural-ssot \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--save-epoch-snapshots
	@echo "Slim MoE structural SSOT cold-start complete. Run: checkpoints/v6/runs/$(or $(RUN_ID),slim_moe_structural_ssot_v1)"

# ---------------------------------------------------------------------------
# V6.5 GNN lineage — isolated experiment + checkpoint namespace (see science/dtie/v65/README.md)
# ---------------------------------------------------------------------------

train-v65: ## Train v6.5 fork (STAGE=1|2|3, RUN_ID=..., separate MLflow experiment tokyo-eyes-v65)
	@mkdir -p mlruns checkpoints/v65/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v65.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_120.json) \
		--output-dir /app/checkpoints/v65/runs/$(or $(RUN_ID),$(shell date +%Y%m%d_%H%M%S)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v65 \
		$(if $(STAGE),--phase $(STAGE),) \
		$(if $(MAX_PROTEINS),--max-proteins $(MAX_PROTEINS),) \
		$(if $(MAX_RESIDUES),--max-residues $(MAX_RESIDUES),) \
		$(if $(NO_MLFLOW),--no-mlflow,) \
		$(if $(NO_WARM_START),--no-warm-start,) \
		$(if $(RESUME),--resume /app/$(RESUME),)

train-v65-master-cold: ## v6.5 learned-GNN cold start (MP→geometry→MoE; recommended)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v65/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v65.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v65/runs/$(or $(RUN_ID),master_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v65 \
		--no-warm-start \
		--master-cold-lineage \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.5 master-cold (learned GNN) complete. Run: checkpoints/v65/runs/$(or $(RUN_ID),master_cold_v1)"
	@echo "MLflow experiment: tokyo-eyes-v65 — trainable node_emb/convs/radial→angular→gate"

train-v65-cold-start: train-v65-master-cold ## Alias: recommended v6.5 cold start = learned master-cold

# ---------------------------------------------------------------------------
# V6.6 GNN lineage — feeler cold-start (see science/dtie/v66/README.md)
# ---------------------------------------------------------------------------

# Plain master-cold topology-three-vector under the v6.6 entrypoint (P1 barcode parent).
# Do NOT pass --v66-feeler-lineage or Fix-1 routing-stack flags. Matches the recipe that
# produced checkpoints/v6/runs/master_cold_topology_three_vector_v1 (mislabeled launch;
# see docs/specs/dehydron-barcode-input-channel/ablation.md) but tags lineage/path correctly.
train-v66-master-cold-topology-three-vector: ## P1 parent: plain v6.6 master-cold three-vector (cold, ~200 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),master_cold_topology_three_vector_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--master-cold-lineage \
		--seed $(or $(SEED),1) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--save-epoch-snapshots
	@echo "v66 plain master-cold complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),master_cold_topology_three_vector_v1)"
	@echo "P1 parent artifact: phase_2.pt (expect Training node_dim=3; no feeler/routing-stack flags)"

# P1 matched continues off the locked plain v66 master-cold parent (ablation.md).
# Stage A-12 v1_2 sidecars currently live under checkpoints/v65/dehydron_barcode_v1
# (orthogonality SSOT). Do NOT point these at feeler / Fix-1 resume defaults.
P1_DBH_PARENT := checkpoints/v66/runs/master_cold_topology_three_vector_v1/phase_2.pt
P1_DBH_BARCODE_DIR := checkpoints/v65/dehydron_barcode_v1

train-v66-dbh-baseline: ## P1 baseline continue: [ρ,τ,ss] off locked v66 phase_2 (15 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@test -f $(or $(RESUME),$(P1_DBH_PARENT)) || \
		(echo "Missing P1 parent — run: make train-v66-master-cold-topology-three-vector" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),dbh_ablation_baseline_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--master-cold-lineage \
		--phase 2 \
		--epochs $(or $(EPOCHS),15) \
		--resume /app/$(or $(RESUME),$(P1_DBH_PARENT)) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "P1 baseline complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),dbh_ablation_baseline_v1)"
	@echo "Expect Training node_dim=3; physics_investigation audit path"

train-v66-dbh-scalars: ## P1 scalars continue: + orthogonal dehydron scalars off locked v66 phase_2 (15 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@test -f $(or $(RESUME),$(P1_DBH_PARENT)) || \
		(echo "Missing P1 parent — run: make train-v66-master-cold-topology-three-vector" && exit 1)
	@test -d $(or $(DBH_DIR),$(P1_DBH_BARCODE_DIR)) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes CORPUS=v6_corpus_stage_a_small_v1.json OUT_DIR=$(P1_DBH_BARCODE_DIR) MAX_PROTEINS=12" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),dbh_ablation_scalars_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--master-cold-lineage \
		--phase 2 \
		--epochs $(or $(EPOCHS),15) \
		--resume /app/$(or $(RESUME),$(P1_DBH_PARENT)) \
		--use-dehydron-barcode \
		--dehydron-barcode-dir /app/$(or $(DBH_DIR),$(P1_DBH_BARCODE_DIR)) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "P1 scalars complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),dbh_ablation_scalars_v1)"
	@echo "Expect Training node_dim=7; liveness_barcode_alive=1 before interpreting audits"

train-v66: ## Train v6.6 fork (STAGE=1|2|3, RUN_ID=..., MLflow tokyo-eyes-v66)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_120.json) \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),$(shell date +%Y%m%d_%H%M%S)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		$(if $(STAGE),--phase $(STAGE),) \
		$(if $(MAX_PROTEINS),--max-proteins $(MAX_PROTEINS),) \
		$(if $(MAX_RESIDUES),--max-residues $(MAX_RESIDUES),) \
		$(if $(NO_MLFLOW),--no-mlflow,) \
		$(if $(NO_WARM_START),--no-warm-start,) \
		$(if $(RESUME),--resume /app/$(RESUME),)

train-v66-feeler: ## v6.6 feeler: learned GNN, 20-ep P1, minimal losses, timeout@45% (no weight transfer)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_p1_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--phase 1 \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler P1 complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_p1_v1)"
	@echo "MLflow experiment: tokyo-eyes-v66 — viewers under .../viewers/"

# Chem-MVP (_struct_conn typed edges): PARKED 2026-07-19 — compare-only.
# Active biology/routing SSOT: docs/specs/fix1-s4-restore/README.md
# Historical replay: ALLOW_PARKED=1 make train-v66-chem-mvp-...
# Baseline = role edges only; Chem = role + disulf/covale from fact_covalent_bond.
define _v66_require_allow_parked
	@if [ "$(ALLOW_PARKED)" != "1" ]; then \
		echo "ERROR: $@ is PARKED (chem-MVP successor track; disc fill collapsed vs Fix-1)."; \
		echo "Active SSOT: docs/specs/fix1-s4-restore/README.md"; \
		echo "Recommended train: make train-v66-fix1-healthy-restore"; \
		echo "Historical replay only: ALLOW_PARKED=1 make $@"; \
		exit 1; \
	fi
endef

train-v66-chem-mvp-baseline: ## [PARKED] Chem-MVP matched baseline: role edges, no chem
	$(call _v66_require_allow_parked)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),chem_mvp_baseline_role_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),20) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--phase 1 \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "Chem-MVP baseline complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),chem_mvp_baseline_role_stage_a12_cold_v1)"

train-v66-chem-mvp: ## [PARKED] Chem-MVP cold: role + disulf/covale edges
	$(call _v66_require_allow_parked)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),chem_mvp_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),20) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--chem-edge-mp \
		--phase 1 \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "Chem-MVP cold complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),chem_mvp_stage_a12_cold_v1)"
	@echo "Score per-structure on 1LYZ/1F88/1IVO; refuse 1IVO-dominated pools (ablation.md)"

# Euclidean reach Top-K: matched chem-MVP Stage A-12 cold + euclidean_shortcut_mp.
# SSOT: docs/specs/v66_chem_MVP/ablation_euclidean_reach.md
# Do NOT overwrite locked baseline chem_mvp_stage_a12_cold_v1.
RUN_ID_EUC_REACH_MANIFOLD ?= chem_mvp_euc_reach_manifold_v1

preflight-v66-chem-mvp-euc-reach-manifold: ## Part0 + manifold gate before euc-reach train
	@mkdir -p logs/diagnostics checkpoints/v66/runs/$(RUN_ID_EUC_REACH_MANIFOLD)
	PYTHONPATH=. python experiments/diagnostics/euclidean_reach_preflight.py \
		--run-id $(or $(RUN_ID),$(RUN_ID_EUC_REACH_MANIFOLD)) \
		2>&1 | tee logs/diagnostics/preflight_euc_reach_manifold_$$(date +%Y%m%d_%H%M%S).log

train-v66-chem-mvp-euc-reach-manifold: ## [PARKED] Manifold-sync euc-reach cold (v2)
	$(call _v66_require_allow_parked)
	@$(MAKE) preflight-v66-chem-mvp-euc-reach-manifold RUN_ID=$(or $(RUN_ID),$(RUN_ID_EUC_REACH_MANIFOLD))
	@test -f checkpoints/v66/runs/$(or $(RUN_ID),$(RUN_ID_EUC_REACH_MANIFOLD))/TRAIN_READY.json || \
		(echo "Missing TRAIN_READY.json — preflight failed" && exit 1)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@test -f checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt || \
		(echo "Missing locked chem-MVP baseline checkpoint" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),$(RUN_ID_EUC_REACH_MANIFOLD)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),20) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--chem-edge-mp \
		--euclidean-shortcut-mp \
		--phase 1 \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "Manifold-sync euc-reach cold complete: checkpoints/v66/runs/$(or $(RUN_ID),$(RUN_ID_EUC_REACH_MANIFOLD))"
	@echo "Grade: PYTHONPATH=. python experiments/diagnostics/euclidean_reach_grade_first_ckpt.py --treatment-dir checkpoints/v66/runs/$(or $(RUN_ID),$(RUN_ID_EUC_REACH_MANIFOLD))"

# Prior authorized id (INVALIDATED — construction mismatch): chem_mvp_euc_reach_v1
train-v66-chem-mvp-euc-reach: ## [PARKED] Chem-MVP + Top-K euclidean_shortcut [prior run]
	$(call _v66_require_allow_parked)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@test -f checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt || \
		(echo "Missing locked chem-MVP baseline checkpoint" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),chem_mvp_euc_reach_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),20) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--chem-edge-mp \
		--euclidean-shortcut-mp \
		--phase 1 \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "Euc-reach cold complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),chem_mvp_euc_reach_v1)"
	@echo "Grade vs chem_mvp_stage_a12_cold_v1 on cone_depth + trunk P@K (ablation_euclidean_reach.md §5)"


# ha_edges_v1: heavy-atom packing/dehydron aux on matched chem-MVP Stage A-12 cold.
# Do NOT run until Part 0 passes (docs/specs/graph-communication/ablation.md §2).
train-v66-ha-edges-v1: ## PARKED — ha_edges_v1 stopped (errors on purpose)
	@echo "ERROR: train-v66-ha-edges-v1 is PARKED/STOP (2026-07-19)."
	@echo "D4 Partial + instrument_b par/worse — do not retrain HA packing aux."
	@echo "Active SSOT: docs/specs/fix1-s4-restore/README.md"
	@echo "Checkpoint kept for compare only:"
	@echo "  checkpoints/v66/runs/ha_edges_v1_stage_a12_cold_v1/v66_best.pt"
	@echo "SSOT: docs/specs/graph-communication/ablation.md"
	@exit 1

# |ρ−TAU| swap: matched Stage A-12 cold arms on chem-MVP + T1a z-norm (both arms).
# Scale check locked z-norm ON (std(|ρ−TAU|)/std(τ)≈8.1); do not grade vs chem_mvp without z-norm.
train-v66-chem-mvp-znorm-baseline: ## [PARKED] |ρ−TAU| matched baseline: chem-MVP + z-norm
	$(call _v66_require_allow_parked)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),chem_mvp_znorm_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),20) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--chem-edge-mp \
		--input-feature-zscore \
		--phase 1 \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "Chem-MVP z-norm baseline complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),chem_mvp_znorm_stage_a12_cold_v1)"

train-v66-chem-mvp-tau-abs-dist: ## [PARKED] |ρ−TAU| swap arm: chem-MVP + z-norm + |ρ−TAU|
	$(call _v66_require_allow_parked)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),chem_mvp_tau_abs_dist_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),20) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--chem-edge-mp \
		--input-feature-zscore \
		--replace-tau-abs-dist \
		--phase 1 \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "Chem-MVP |ρ−TAU| swap complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),chem_mvp_tau_abs_dist_stage_a12_cold_v1)"

# Path 2 Move 3: ORPHANED (2026-07-18 cleanup). Do not train.
# See docs/specs/learned-flow-influence/PATH2_DIRECTIONALITY.md
PATH2_SWAP_PARENT := checkpoints/v66/runs/chem_mvp_tau_abs_dist_stage_a12_cold_v1/v66_best.pt
train-v66-path2-dir-diam9: ## ORPHANED — Path 2 agent pilot withdrawn (errors on purpose)
	@echo "ERROR: train-v66-path2-dir-diam9 is ORPHANED (2026-07-18 cleanup)."
	@echo "Do not warm-start Path 2. Resume from locked |ρ−TAU| / matched z-norm trunk."
	@echo "SSOT: docs/specs/learned-flow-influence/PATH2_DIRECTIONALITY.md"
	@exit 1

# Path B hierarchical containment: matched Stage A-12 cold arms on chem-MVP parent stack.
# Baseline = chem on, no containment; Path B = same + contain_up/down (+2 radial MLPs).
train-v66-containment-baseline: ## [PARKED] Containment matched baseline: chem stack, no containment
	$(call _v66_require_allow_parked)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),containment_baseline_chem_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),20) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--chem-edge-mp \
		--phase 1 \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "Containment baseline complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),containment_baseline_chem_stage_a12_cold_v1)"

train-v66-containment-pathb: ## [PARKED] Containment Path B cold: chem + contain_up/down
	$(call _v66_require_allow_parked)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),containment_pathb_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),20) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--chem-edge-mp \
		--containment-edge-mp \
		--phase 1 \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "Containment Path B cold complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),containment_pathb_stage_a12_cold_v1)"
	@echo "Gate order: isolated-seed → liveness → oversmoothing-at-root → flow-influence (ablation.md)"

train-v66-feeler-dbh-scalars: ## v6.6 feeler P3: barcode scalars off P2 parent (15 ep, no disc occupancy) [deprecated — disc collapse]
	@echo "WARNING: node-global barcode caused disc collapse; use train-v66-feeler-p3-geom-edges instead"
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_DBH_P2_RESUME)) || \
		(echo "Missing P2 resume — set RESUME=..." && exit 1)
	@test -d $(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes-feeler-expand" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_dbh_scalars_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--phase 3 \
		--resume /app/$(or $(RESUME),$(V66_DBH_P2_RESUME)) \
		--epochs $(or $(EPOCHS),15) \
		--use-dehydron-barcode \
		--dehydron-barcode-dir /app/$(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler barcode P3 complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_dbh_scalars_v1)"
	@echo "Gate: liveness_barcode_alive=1; compare viewers vs P2 parent"

train-v66-feeler-dbh-edges: ## v6.6 feeler P3: local dehydron edge barcode off P2 parent (15 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_DBH_P2_RESUME)) || \
		(echo "Missing P2 resume — set RESUME=..." && exit 1)
	@test -d $(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes-feeler-expand" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_dbh_edges_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--phase 3 \
		--resume /app/$(or $(RESUME),$(V66_DBH_P2_RESUME)) \
		--epochs $(or $(EPOCHS),15) \
		--dehydron-edge-barcode \
		--dehydron-barcode-dir /app/$(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler edge-barcode P3 complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_dbh_edges_v1)"
	@echo "Gate: liveness_barcode_edge_alive=1; compare viewers vs P2 parent (grid fill, not teardrop)"

train-v66-feeler-p3-geom: ## v6.6 feeler: continue P3 geom from best_disc (20 ep, no barcode)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_P3_RESUME)) || \
		(echo "Missing p3_geom resume — set RESUME=..." && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_p3_geom_v2) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-p3-geom \
		--resume /app/$(or $(RESUME),$(V66_FEELER_P3_RESUME)) \
		--epochs $(or $(EPOCHS),20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler P3 geom continue complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_p3_geom_v2)"
	@echo "Gate: probe_r_proj_depth ≥ 0.30 + 4OBE viewer (ignore corpus σ₂/σ₁ alone)"

train-v66-feeler-p3-geom-half: ## v6.6 feeler: P3 geom continue with 0.5× occupancy stack (20 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_P3_RESUME)) || \
		(echo "Missing p3_geom resume — set RESUME=..." && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_p3_geom_half_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-p3-geom \
		--v66-feeler-p3-geom-half-stack \
		--resume /app/$(or $(RESUME),$(V66_FEELER_P3_RESUME)) \
		--epochs $(or $(EPOCHS),20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler P3 geom half-stack complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_p3_geom_half_v1)"
	@echo "Gate: probe_r_proj_depth ≥ 0.30 + 4OBE viewer (compare vs full-stack v2 drift)"

train-v66-feeler-p3-geom-edges: ## v6.6 feeler: P3 geom + edge barcode off p3_geom champion (15 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_P3_RESUME)) || \
		(echo "Missing p3_geom resume — set RESUME=... or finish feeler_expand_23_p3_geom_v1" && exit 1)
	@test -d $(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes-feeler-expand" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_p3_geom_edges_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-p3-geom-edges \
		--resume /app/$(or $(RESUME),$(V66_FEELER_P3_RESUME)) \
		--epochs $(or $(EPOCHS),15) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--dehydron-edge-barcode \
		--dehydron-barcode-dir /app/$(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler P3 geom + edge barcode complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_p3_geom_edges_v1)"
	@echo "Gate: σ₂/σ₁ ≥ 0.80, disc thick pre ≥ 0.22; compare 4OBE/1R69 viewers vs p3_geom parent"

train-v66-feeler-rim-fanout: ## v6.6 feeler P4: rim fan-out off p3_geom champion (10 ep, no barcode)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_CHAMPION)) || \
		(echo "Missing champion resume — set RESUME=..." && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_p3_geom_p4_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--phase 4 \
		--resume /app/$(or $(RESUME),$(V66_FEELER_CHAMPION)) \
		--epochs $(or $(EPOCHS),10) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler P4 rim fan-out complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_p3_geom_p4_v1)"
	@echo "Gate: probe_r_proj_depth ≥ baseline−0.08 + 4OBE viewer (no disc occupancy stack)"

train-v66-feeler-rim-fanout-model: ## v6.6 warm: P4 ep58 → rim fan-out P12 (15 ep, half geom + rim loss)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_P4_RESUME)) || \
		(echo "Missing P4 ep58 resume — set RESUME=..." && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_p4_v2) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--resume /app/$(or $(RESUME),$(V66_FEELER_P4_RESUME)) \
		--epochs $(or $(EPOCHS),15) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.12) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler rim fan-out WARM complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_p4_v2)"
	@echo "Gate: 4OBE rim_frac + angular span + probe_r_proj_depth ≥ P4−0.08"

V66_FEELER_RIM_FANOUT_POLISH_RESUME := checkpoints/v66/runs/feeler_expand_23_rim_fanout_p4_v2/epochs/epoch_062.pt

train-v66-feeler-rim-fanout-polish: ## v6.6 polish: resume p4_v2 ep62; depth-lock + light rim (10 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_RIM_FANOUT_POLISH_RESUME)) || \
		(echo "Missing polish resume — set RESUME=... or finish feeler_expand_23_rim_fanout_p4_v2" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_polish_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-rim-fanout-polish \
		--resume /app/$(or $(RESUME),$(V66_FEELER_RIM_FANOUT_POLISH_RESUME)) \
		--epochs $(or $(EPOCHS),10) \
		--p2-bridge-lr $(or $(LR),5.0e-5) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.12) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler rim fan-out POLISH complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_polish_v1)"
	@echo "Gate: 4OBE rim_frac ≥ P4, σ₂/σ₁ ≥ 0.60, probe_r_proj_depth ≥ baseline−0.08"

V66_FEELER_ANGULAR_FILL_RESUME := checkpoints/v66/runs/feeler_expand_23_rim_fanout_polish_v1/epochs/epoch_072.pt

train-v66-feeler-rim-fanout-angular: ## v6.6 angular-fill: resume polish ep72; mid-disc rim_* + depth-lock (8 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_ANGULAR_FILL_RESUME)) || \
		(echo "Missing angular-fill resume — set RESUME=... or finish feeler_expand_23_rim_fanout_polish_v1" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_angular_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-rim-fanout-angular \
		--resume /app/$(or $(RESUME),$(V66_FEELER_ANGULAR_FILL_RESUME)) \
		--epochs $(or $(EPOCHS),8) \
		--p2-bridge-lr $(or $(LR),5.0e-5) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.12) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler rim fan-out ANGULAR-FILL complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_angular_v1)"
	@echo "Gate: 1F88/4OBE sparse wedge ↓, R↓, probe ≥ baseline−0.08 (then radius)"

V66_FEELER_ANGULAR_V2_RESUME := checkpoints/v66/runs/feeler_expand_23_rim_fanout_angular_v1/epochs/epoch_080.pt

train-v66-feeler-rim-fanout-angular-v2: ## v6.6 angular-fill v2: resume angular ep80; min_r=0.12 (8 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_ANGULAR_V2_RESUME)) || \
		(echo "Missing angular-v2 resume — set RESUME=... or finish feeler_expand_23_rim_fanout_angular_v1" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_angular_v2) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-rim-fanout-angular-v2 \
		--resume /app/$(or $(RESUME),$(V66_FEELER_ANGULAR_V2_RESUME)) \
		--epochs $(or $(EPOCHS),8) \
		--p2-bridge-lr $(or $(LR),5.0e-5) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.12) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler rim fan-out ANGULAR-FILL v2 complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_angular_v2)"
	@echo "Gate: 1F88 sparse wedge ≤~20° or stall; probe ≥ baseline−0.08; then radius"

V66_FEELER_RADIUS_RESUME := checkpoints/v66/runs/feeler_expand_23_rim_fanout_angular_v2/epochs/epoch_088.pt

train-v66-feeler-rim-fanout-radius: ## v6.6 radius push: resume angular_v2 ep88; depth_tgt=0.55 (8 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_RADIUS_RESUME)) || \
		(echo "Missing radius resume — set RESUME=... or finish feeler_expand_23_rim_fanout_angular_v2" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_radius_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-rim-fanout-radius \
		--resume /app/$(or $(RESUME),$(V66_FEELER_RADIUS_RESUME)) \
		--epochs $(or $(EPOCHS),8) \
		--p2-bridge-lr $(or $(LR),5.0e-5) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler rim fan-out RADIUS complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_radius_v1)"
	@echo "Gate: rim_frac ↑ (~40% 4OBE/1F88); probe ≥ baseline−0.08; wedge must not reopen"

V66_FEELER_COVERAGE_RESUME := checkpoints/v66/runs/feeler_expand_23_rim_fanout_radius_v1/epochs/epoch_096.pt

train-v66-feeler-rim-fanout-coverage: ## v6.6 angular coverage: resume radius ep96; empty-sector bin floor (8 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_COVERAGE_RESUME)) || \
		(echo "Missing coverage resume — set RESUME=... or finish feeler_expand_23_rim_fanout_radius_v1" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_coverage_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-rim-fanout-coverage \
		--resume /app/$(or $(RESUME),$(V66_FEELER_COVERAGE_RESUME)) \
		--epochs $(or $(EPOCHS),8) \
		--p2-bridge-lr $(or $(LR),5.0e-5) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler rim fan-out COVERAGE complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_coverage_v1)"
	@echo "Gate: 1F88 largest angular gap ↓; probe ≥ baseline−0.08; radius must hold"

V66_FEELER_ANTIBARRIER_RESUME := checkpoints/v66/runs/feeler_expand_23_rim_fanout_coverage_v2/epochs/epoch_164.pt
V66_RAF1_PPI_RESUME := checkpoints/v66/runs/feeler_expand_23_rim_fanout_coverage_v2/epochs/epoch_164.pt

train-v66-raf1-ppi-coverage: ## v6.6 RAF1 PPI pathway corpus: coverage transfer from ep164 (40 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json (pathway corpus loads from PDB; stamp is lineage hygiene)" && exit 1)
	@test -f manifests/v6_corpus_raf1_ppi_v1.json || (echo "Missing RAF1 PPI manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_RAF1_PPI_RESUME)) || \
		(echo "Missing RAF1 PPI resume — set RESUME=... or need coverage_v2 epoch_164.pt" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	@echo "NOTE: RAF1 PPI pathway corpus trains via TRAINING_LOAD_FROM_PDB=1 (structures need not be DB-ingested)."
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) -e TRAINING_LOAD_FROM_PDB=1 -e GNN_INPUT_MODE=topology_three_vector science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_raf1_ppi_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),raf1_ppi_coverage_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-rim-fanout-coverage \
		--resume /app/$(or $(RESUME),$(V66_RAF1_PPI_RESUME)) \
		--epochs $(or $(EPOCHS),40) \
		--p2-bridge-lr $(or $(LR),5.0e-5) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 RAF1 PPI coverage transfer complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),raf1_ppi_coverage_v1)"
	@echo "Gate: 4OBE disc vs ep164 baseline; new pathway discs (3OMV/3EQI/2O02) occupancy; probe hold"

train-v66-feeler-rim-fanout-antibarrier: ## v6.6 anti-barrier: resume coverage_v2 ep164; expert θ diversity (25 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_ANTIBARRIER_RESUME)) || \
		(echo "Missing anti-barrier resume — set RESUME=... or finish feeler_expand_23_rim_fanout_coverage_v2" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_antibarrier_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-rim-fanout-antibarrier \
		--resume /app/$(or $(RESUME),$(V66_FEELER_ANTIBARRIER_RESUME)) \
		--epochs $(or $(EPOCHS),25) \
		--p2-bridge-lr $(or $(LR),5.0e-5) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler rim fan-out ANTI-BARRIER complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_antibarrier_v1)"
	@echo "Gate: gap ↓; crest wall_share ↓; r̄ ≳ 0.18; probe ≥ baseline−0.08"

V66_FEELER_EXPERT_ARC_RESUME := checkpoints/v66/runs/feeler_expand_23_rim_fanout_coverage_v2/epochs/epoch_164.pt

train-v66-feeler-rim-fanout-expert-arc: ## v6.6 mild expert-arc: resume coverage_v2 ep164; soft θ sep, 1F88 anchor (20 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_EXPERT_ARC_RESUME)) || \
		(echo "Missing expert-arc resume — set RESUME=... or need coverage_v2 epoch_164.pt" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_expert_arc_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-rim-fanout-expert-arc \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88) \
		--resume /app/$(or $(RESUME),$(V66_FEELER_EXPERT_ARC_RESUME)) \
		--epochs $(or $(EPOCHS),20) \
		--p2-bridge-lr $(or $(LR),5.0e-5) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler mild EXPERT-ARC complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_fanout_expert_arc_v1)"
	@echo "Gates: probe ≥ baseline−0.08; 1F88 gap not worse >~10°; 4OBE circ-R ≲ 0.60"

V66_FEELER_GEOM_PRIOR_RESUME := checkpoints/v66/runs/feeler_expand_23_rim_fanout_coverage_v2/epochs/epoch_164.pt

train-v66-feeler-geom-angular-prior: ## v6.6 geometric angular prior: resume ep164; dehydron/peptide θ (20 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_GEOM_PRIOR_RESUME)) || \
		(echo "Missing geom-prior resume — set RESUME=... or need coverage_v2 epoch_164.pt" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_geom_angular_prior_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88) \
		--resume /app/$(or $(RESUME),$(V66_FEELER_GEOM_PRIOR_RESUME)) \
		--epochs $(or $(EPOCHS),20) \
		--p2-bridge-lr $(or $(LR),5.0e-5) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler GEOMETRIC ANGULAR PRIOR complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_geom_angular_prior_v1)"
	@echo "Gates: probe ≥ baseline−0.08; 1F88 gap; 4OBE circ-R; corr(r,depth) hold"

train-v66-feeler-geom-angular-prior-cold: ## v6.6 geom angular prior COLD: no resume, 10 ep; judge Poincaré HTML viewers
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_geom_angular_prior_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88) \
		--epochs $(or $(EPOCHS),10) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler GEOM ANGULAR PRIOR COLD complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_geom_angular_prior_cold_v1)"
	@echo "True test: open viewers/ HTML Poincaré discs (1F88, 4OBE) — no resume weights"

train-v66-feeler-geom-angular-prior-cold-continue: ## continue cold geom-prior to ≥20 ep; Fix-1 gates+H each epoch
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v66/runs/feeler_expand_23_geom_angular_prior_cold_v1/epochs/epoch_010.pt) || \
		(echo "Missing cold epoch_010 resume — set RESUME=..." && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_geom_angular_prior_cold_to20_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--resume /app/$(or $(RESUME),checkpoints/v66/runs/feeler_expand_23_geom_angular_prior_cold_v1/epochs/epoch_010.pt) \
		--epochs $(or $(EPOCHS),10) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "Continue →20 complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_geom_angular_prior_cold_to20_v1)"
	@echo "Per-epoch Fix-1 + H: .../fix1_gates_per_epoch.jsonl"

train-v66-fix1-s4-stage-a12: ## Fix-1 + S4 hyp-MP, cold, locked Stage A-12; IBU usage gates
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing Stage A-12 manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--hyperbolic-mp-graph \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--epochs $(or $(EPOCHS),20) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots \
		$(if $(SEED),--seed $(SEED),)
	@echo "Fix-1+S4 Stage A-12 complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stage_a12_cold_v1)"
	@echo "Pre-registered usage: USAGE_MOVED / IBU_HOLDS / AMBIGUOUS — see GNNV7_SUCCESS_CRITERIA.md"
	@echo "Log: .../fix1_gates_per_epoch.jsonl (H, max_share, Fix-1 gates)"

train-v66-fix1-s4-t1a-znorm-stage-a12: ## Fix-1+S4 + T1a input z-norm cold Stage A-12 (no |ρ−TAU| yet)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing Stage A-12 manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_t1a_znorm_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--hyperbolic-mp-graph \
		--input-feature-zscore \
		$(if $(REPLACE_TAU_ABS_DIST),--replace-tau-abs-dist,) \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--epochs $(or $(EPOCHS),20) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots \
		$(if $(SEED),--seed $(SEED),)
	@echo "T1a z-norm cold retrain complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_t1a_znorm_stage_a12_cold_v1)"
	@echo "Pre-registered: T1A_RANK_MET / SURPASSED_FWD / ROUTING_RESPONDED / T1A_ROOT_CAUSE_FOR_ROUTING — GNNV7_SUCCESS_CRITERIA.md"
	@echo "Logs: fix1_gates_per_epoch.jsonl + t1a_trunk_rank_per_epoch.jsonl"

train-v66-fix1-s4-gate-sasa-stage-a12: ## Pre-reg: Fix-1+S4 + T1a z-norm + explicit SASA on topology gate (Stage A-12 cold)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing Stage A-12 manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_gate_sasa_stage_a12_cold_v1)
	@printf '%s\n' \
		'# SASA board enrichment — pre-registered 2026-07-14' \
		'lever: explicit normalized SASA on topology gate board' \
		'baseline_lineage: Fix-1 + S4 + T1a input z-norm (no |ρ−TAU|, no scale L3)' \
		'input_feature_zscore: on' \
		'gate_include_sasa: on' \
		'comparators: L2 ge20 soft structure; route_v1 committed tail' \
		'success: GROWS_REAL_NICHE_TWIN_COMMIT | GROWS_REAL_NICHE_CLEAN_EXPOSURE_SPLIT' \
		'see: docs/audit/GNNV7_SUCCESS_CRITERIA.md (HEADLINE / SASA enrichment gate)' \
		> checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_gate_sasa_stage_a12_cold_v1)/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_gate_sasa_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--hyperbolic-mp-graph \
		--input-feature-zscore \
		--gate-include-sasa \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--epochs $(or $(EPOCHS),20) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots \
		$(if $(SEED),--seed $(SEED),)
	@echo "SASA gate enrichment complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_gate_sasa_stage_a12_cold_v1)"
	@echo "Score vs GNNV7_SUCCESS_CRITERIA.md SASA enrichment pre-reg (GROWS_REAL_NICHE_* only)"

train-v66-fix1-s4-proto-repulsion-stage-a12: ## Pre-reg: SASA lineage + nearest-pair prototype repulsion (Stage A-12 cold)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing Stage A-12 manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_proto_repulsion_stage_a12_cold_v1)
	@printf '%s\n' \
		'# Prototype nearest-pair repulsion — pre-registered 2026-07-14 (frozen before train)' \
		'lever: ongoing nearest-pair prototype hyp-distance hinge' \
		'loss: relu(m − min_{i<j} d_H(p_i,p_j))' \
		'prototype_repulsion_coeff: $(or $(PROTO_REPULSION_COEFF),1.0)' \
		'prototype_repulsion_margin: $(or $(PROTO_REPULSION_MARGIN),0.25)' \
		'L0_orthogonal_init: off' \
		'baseline_lineage: Fix-1 + S4 + T1a z-norm + gate_include_sasa' \
		'run_structure: ONE run, ONE coeff; L1/L2/L3 = within-run escalating floors' \
		'no: encoder_h→gate, new board channel, logit_scale L3' \
		'success: PROTO_SEP_L1 | PROTO_SEP_L2 only (see GNNV7_SUCCESS_CRITERIA.md)' \
		'logs: prototype_repulsion_per_epoch.jsonl (all 6 pairs + re-measured degree-swap)' \
		> checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_proto_repulsion_stage_a12_cold_v1)/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_proto_repulsion_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--hyperbolic-mp-graph \
		--input-feature-zscore \
		--gate-include-sasa \
		--prototype-repulsion-coeff $(or $(PROTO_REPULSION_COEFF),1.0) \
		--prototype-repulsion-margin $(or $(PROTO_REPULSION_MARGIN),0.25) \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--epochs $(or $(EPOCHS),20) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots \
		$(if $(SEED),--seed $(SEED),)
	@echo "Prototype repulsion complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_proto_repulsion_stage_a12_cold_v1)"
	@echo "Score vs GNNV7_SUCCESS_CRITERIA.md PROTO_SEP_L1 / PROTO_SEP_L2 (re-measured sensitivity required)"

train-v66-fix1-s4-proto-repulsion-scale-l2-stage-a12: ## Pre-reg: repulsion × elevated scale (~6.61) stack (Stage A-12 cold)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing Stage A-12 manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1)
	@printf '%s\n' \
		'# Repulsion × elevated scale stack — pre-registered 2026-07-14 (frozen before train)' \
		'lever: nearest-pair prototype repulsion + L2 softplus init/floor' \
		'prototype_repulsion_coeff: $(or $(PROTO_REPULSION_COEFF),1.0)' \
		'prototype_repulsion_margin: $(or $(PROTO_REPULSION_MARGIN),0.25)' \
		'gate_logit_softplus_init: $(or $(GATE_SOFTPLUS),6.612216472625732)' \
		'gate_logit_softplus_floor: $(or $(GATE_SOFTPLUS_FLOOR),6.612216472625732)' \
		'baseline_lineage: Fix-1 + S4 + T1a z-norm + gate_include_sasa' \
		'STACK_COMMIT_L1: frac max-p≥0.60 ≥5% (anchored to route_v1 ~6% niche)' \
		'DIST_RANGE_KILLS_SCALE: softplus≥5 AND mean dist_range≤0.15 AND frac_mp≥0.60<2%' \
		'success: STACK_WIN_L1 | STACK_WIN_L2 (see GNNV7_SUCCESS_CRITERIA.md)' \
		'logs: prototype_repulsion_per_epoch.jsonl (pairs + dist_range + softplus + max-p + hard_share + committed_hard_share + per_structure_soft/hard_max)' \
		> checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1)/README.md
	GNN_INPUT_MODE=$(or $(GNN_INPUT_MODE),topology_three_vector) $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--hyperbolic-mp-graph \
		--input-feature-zscore \
		--gate-include-sasa \
		--prototype-repulsion-coeff $(or $(PROTO_REPULSION_COEFF),1.0) \
		--prototype-repulsion-margin $(or $(PROTO_REPULSION_MARGIN),0.25) \
		--gate-logit-softplus-init $(or $(GATE_SOFTPLUS),6.612216472625732) \
		--gate-logit-softplus-floor $(or $(GATE_SOFTPLUS_FLOOR),6.612216472625732) \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--epochs $(or $(EPOCHS),20) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(RESUME),--resume /app/$(RESUME),)
	@echo "Stack complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1)"
	@echo "Score vs GNNV7_SUCCESS_CRITERIA.md STACK_WIN_L1 / STACK_WIN_L2"

train-v66-fix1-s4-proto-repulsion-scale-l2-majority-hinge-stage-a12: ## Pre-reg: stack + majority committed-share hinge λ=0.5 (Stage A-12 cold)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing Stage A-12 manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_majority_hinge_stage_a12_cold_v1)
	@printf '%s\n' \
		'# Majority-conditional committed-share hinge — pre-registered 2026-07-14' \
		'baseline: Fix-1 + S4 + SASA + proto repulsion + softplus floor ≈6.612' \
		'lever: STE hard committed majority share hinge (maj-mask grads)' \
		'majority_committed_share_coeff: $(or $(MAJORITY_SHARE_COEFF),0.5)' \
		'majority_committed_share_tau: $(or $(MAJORITY_SHARE_TAU),0.56)' \
		'coeff_proxy: checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/majority_hinge_coeff_proxy.json' \
		'NO_MOVE at 0.5 → pre-auth rematch 2.5; COMMIT_KILLED → rematch 0.1' \
		'success: MAJORITY_SPLIT_WIN (local cleared + purity + commit L2 + axis)' \
		'logs: prototype_repulsion_per_epoch.jsonl (majority_conditional + dehydron_partition_purity)' \
		> checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_majority_hinge_stage_a12_cold_v1)/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_majority_hinge_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--hyperbolic-mp-graph \
		--input-feature-zscore \
		--gate-include-sasa \
		--prototype-repulsion-coeff $(or $(PROTO_REPULSION_COEFF),1.0) \
		--prototype-repulsion-margin $(or $(PROTO_REPULSION_MARGIN),0.25) \
		--gate-logit-softplus-init $(or $(GATE_SOFTPLUS),6.612216472625732) \
		--gate-logit-softplus-floor $(or $(GATE_SOFTPLUS_FLOOR),6.612216472625732) \
		--majority-committed-share-coeff $(or $(MAJORITY_SHARE_COEFF),0.5) \
		--majority-committed-share-tau $(or $(MAJORITY_SHARE_TAU),0.56) \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--epochs $(or $(EPOCHS),30) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(RESUME),--resume /app/$(RESUME),)
	@echo "Majority hinge complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_majority_hinge_stage_a12_cold_v1)"
	@echo "Score MAJORITY_SPLIT_WIN / COEFF_INCONCLUSIVE / AXIS_SCRAMBLED / COMMIT_KILLED"

train-v66-fix1-s4-proto-repulsion-scale-l2-core-majority-hinge-stage-a12: ## Pre-reg: stack + core-only (dh=0) majority hinge (Stage A-12 cold)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing Stage A-12 manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_core_majority_hinge_stage_a12_cold_v1)
	@printf '%s\n' \
		'# Core-only majority hinge — pre-registered 2026-07-14' \
		'baseline: Fix-1 + S4 + SASA + proto repulsion + softplus floor ≈6.612' \
		'lever: STE hard share among committed∧dehydron=0 only' \
		'core_majority_committed_share_coeff: $(or $(CORE_MAJORITY_SHARE_COEFF),0.25)' \
		'agnostic majority_committed_share_coeff: 0' \
		'direct grad on dh=1: excluded; purity still monitors indirect dilution' \
		'eligible purity: n_committed≥20 and n_minority≥5' \
		'coeff_proxy: core_majority_hinge_coeff_proxy.json (λ=0.25; rematch 1.25 / 0.05)' \
		'success: CORE_MAJORITY_SPLIT_WIN' \
		'logs: prototype_repulsion_per_epoch.jsonl (core_majority_conditional + dehydron_partition_purity)' \
		> checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_core_majority_hinge_stage_a12_cold_v1)/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_core_majority_hinge_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--hyperbolic-mp-graph \
		--input-feature-zscore \
		--gate-include-sasa \
		--prototype-repulsion-coeff $(or $(PROTO_REPULSION_COEFF),1.0) \
		--prototype-repulsion-margin $(or $(PROTO_REPULSION_MARGIN),0.25) \
		--gate-logit-softplus-init $(or $(GATE_SOFTPLUS),6.612216472625732) \
		--gate-logit-softplus-floor $(or $(GATE_SOFTPLUS_FLOOR),6.612216472625732) \
		--majority-committed-share-coeff 0 \
		--core-majority-committed-share-coeff $(or $(CORE_MAJORITY_SHARE_COEFF),0.25) \
		--core-majority-committed-share-tau $(or $(CORE_MAJORITY_SHARE_TAU),0.56) \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--epochs $(or $(EPOCHS),30) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(RESUME),--resume /app/$(RESUME),)
	@echo "Core majority hinge complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_core_majority_hinge_stage_a12_cold_v1)"
	@echo "Score CORE_MAJORITY_SPLIT_WIN / COEFF_INCONCLUSIVE / AXIS_SCRAMBLED / COMMIT_KILLED"

train-v66-fix1-s4-proto-repulsion-scale-l2-core-quota-stage-a12: ## Pre-reg: stack + core capacity quotas τ=0.40 (Stage A-12 cold)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing Stage A-12 manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_core_quota_stage_a12_cold_v1)
	@printf '%s\n' \
		'# Core capacity quotas — pre-registered 2026-07-14' \
		'baseline: Fix-1 + S4 + SASA + proto repulsion + softplus floor ≈6.612' \
		'lever: hard core (dh=0) capacity τ_cap=0.40 + frozen dehydron-dominant second-choice ban' \
		'tie-break: unplaced wins over force cross-axis (QUOTA_STARVE if mean unplaced>5%)' \
		'snapshot: ge0 freeze of dehydron-dominant mask; never recomputed' \
		'agnostic/core majority share coeffs: 0' \
		'success: CORE_QUOTA_WIN' \
		'logs: prototype_repulsion_per_epoch.jsonl (core_quota_conditional + dehydron_partition_purity)' \
		'artifact: core_quota_dominant_ge0.json' \
		> checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_core_quota_stage_a12_cold_v1)/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_core_quota_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--hyperbolic-mp-graph \
		--input-feature-zscore \
		--gate-include-sasa \
		--prototype-repulsion-coeff $(or $(PROTO_REPULSION_COEFF),1.0) \
		--prototype-repulsion-margin $(or $(PROTO_REPULSION_MARGIN),0.25) \
		--gate-logit-softplus-init $(or $(GATE_SOFTPLUS),6.612216472625732) \
		--gate-logit-softplus-floor $(or $(GATE_SOFTPLUS_FLOOR),6.612216472625732) \
		--majority-committed-share-coeff 0 \
		--core-majority-committed-share-coeff 0 \
		--core-capacity-quota-tau $(or $(CORE_QUOTA_TAU),0.40) \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--epochs $(or $(EPOCHS),30) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(RESUME),--resume /app/$(RESUME),)
	@echo "Core quota complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_core_quota_stage_a12_cold_v1)"
	@echo "Score CORE_QUOTA_WIN / CORE_QUOTA_NO_MOVE / AXIS_SCRAMBLED_BY_QUOTA / COMMIT_KILLED_BY_QUOTA / QUOTA_STARVE"

train-v66-fix1-s4-proto-repulsion-scale-l2-stage-a12-seed2: ## Seed-2 cold continuous ge1→30 (no resume); distribution floors pre-reg
	@if [ -n "$(RESUME)" ]; then \
		echo "Seed-2 forbids RESUME (seed-1 ge21 seam confound). Unset RESUME and use cold EPOCHS=30."; \
		exit 1; \
	fi
	$(MAKE) train-v66-fix1-s4-proto-repulsion-scale-l2-stage-a12 \
		RUN_ID=$(or $(RUN_ID),fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1) \
		SEED=$(or $(SEED),2) \
		EPOCHS=$(or $(EPOCHS),30)
	@echo "Seed-2 cold complete. Score STACK_WIN_* + committed_distribution (COMMITTED_DISTRIBUTION_PASS)."
	@echo "Floors: ≥8/12 committed proteins, protein share≤35%, committed_hard_max<0.56, per_structure_committed_hard_max<0.56"

# Recommended v6.6 biology/routing cold train (2026-07-19 restore).
# Never overwrite the locked sealed run dir; default RUN_ID is a new restore sibling.
# Sealed SSOT: checkpoints/v66/runs/fix1_s4_stack_initseed_controlled_3d_seed2_v1/v66_healthy_sealed.pt
# Docs: docs/specs/fix1-s4-restore/README.md
train-v66-fix1-healthy-restore: ## Recommended: Fix-1+S4 healthy recipe (seed2, 30 ep, new RUN_ID)
	@if [ -n "$(RESUME)" ]; then \
		echo "Healthy restore forbids RESUME (cold Fix-1 stack only). Unset RESUME."; \
		exit 1; \
	fi
	@case "$(or $(RUN_ID),fix1_s4_stack_initseed_controlled_3d_seed2_restore_v1)" in \
		fix1_s4_stack_initseed_controlled_3d_seed2_v1) \
			echo "ERROR: refuse overwrite of locked healthy run dir."; \
			echo "Omit RUN_ID or use a new id (default: ..._restore_v1)."; \
			exit 1 ;; \
	esac
	$(MAKE) train-v66-fix1-s4-proto-repulsion-scale-l2-stage-a12 \
		RUN_ID=$(or $(RUN_ID),fix1_s4_stack_initseed_controlled_3d_seed2_restore_v1) \
		SEED=$(or $(SEED),2) \
		EPOCHS=$(or $(EPOCHS),30) \
		GNN_INPUT_MODE=topology_three_vector
	@echo "Healthy restore train complete."
	@echo "Locked SSOT (do not overwrite): checkpoints/v66/runs/fix1_s4_stack_initseed_controlled_3d_seed2_v1/v66_healthy_sealed.pt"
	@echo "Docs: docs/specs/fix1-s4-restore/README.md"

# ---------------------------------------------------------------------------
# Fix-1 expand lineage (post-restore) — MLflow experiment tokyo-eyes-v66-fix1-expand
# Docs: docs/specs/fix1-s4-restore/corpus-expansion.md
# ---------------------------------------------------------------------------
FIX1_SEALED_CKPT := checkpoints/v66/runs/fix1_s4_stack_initseed_controlled_3d_seed2_v1/v66_healthy_sealed.pt
FIX1_EXPAND_EXPERIMENT := tokyo-eyes-v66-fix1-expand
FIX1_EXPAND23_RUN ?= fix1_s4_expand23_continue_v1
FIX1_STAGE_A25_RUN ?= fix1_s4_stage_a25_continue_v2
FIX1_RAF1_RUN ?= fix1_s4_raf1_mix_continue_v1
FIX1_SPARSITY_RUN ?= fix1_s4_sparsity_sealed_continue_v1
# Default corpus for sparsity bet (Stage A-12 preferred for hub grade; override CORPUS=…_feeler_expand_v1.json)
FIX1_SPARSITY_CORPUS ?= v6_corpus_stage_a_small_v1.json
FIX1_SPARSITY_MAX_PROTEINS ?= 12
FIX1_SPARSITY_COEFF ?= 0.0075
FIX1_SPARSITY_WARMUP ?= 8

register-v66-fix1-expand-lineage: ## P0: register MLflow lineage root for Fix-1 expand
	@test -f $(FIX1_SEALED_CKPT) || (echo "Missing sealed ckpt $(FIX1_SEALED_CKPT)" && exit 1)
	$(SCIENCE_RUN) science python -m experiments.training.v66.register_fix1_expand_lineage \
		--tracking-uri http://mlflow:5000 \
		--experiment $(FIX1_EXPAND_EXPERIMENT)
	@echo "Lineage stamp: data/gates/fix1_expand_mlflow_lineage_root.json"
	@echo "UI: http://localhost:5000 → experiment $(FIX1_EXPAND_EXPERIMENT)"

# Shared Fix-1 stack flags for expand continues (hyp-MP + repulsion + z-norm + SASA + geom prior).
V66_FIX1_EXPAND_STACK := --v66-feeler-lineage --v66-feeler-rim-fanout-model --v66-feeler-geom-angular-prior --hyperbolic-mp-graph --input-feature-zscore --gate-include-sasa --prototype-repulsion-coeff $(or $(PROTO_REPULSION_COEFF),1.0) --prototype-repulsion-margin $(or $(PROTO_REPULSION_MARGIN),0.25) --gate-logit-softplus-init $(or $(GATE_SOFTPLUS),6.612216472625732) --gate-logit-softplus-floor $(or $(GATE_SOFTPLUS_FLOOR),6.612216472625732) --geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) --geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) --rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) --rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) --num-experts $(or $(NUM_EXPERTS),4) --feature-liveness-probe --save-epoch-snapshots

train-v66-fix1-expand23-continue: ## P1: Fix-1 stack continue 12→23 (resume sealed; new MLflow lineage)
	@test -f data/gates/fix1_expand_mlflow_lineage_root.json || \
		(echo "Missing lineage root — run: make register-v66-fix1-expand-lineage" && exit 1)
	@test -f $(FIX1_SEALED_CKPT) || (echo "Missing sealed ckpt" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	$(MAKE) gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_EXPAND23_RUN))
	@printf '%s\n' \
		'# Fix-1 expand P1 — feeler_expand 23 continue from sealed SSOT' \
		'mlflow_experiment: $(FIX1_EXPAND_EXPERIMENT)' \
		'resume: $(FIX1_SEALED_CKPT)' \
		'docs: docs/specs/fix1-s4-restore/corpus-expansion.md' \
		> checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_EXPAND23_RUN))/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_EXPAND23_RUN)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),15) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment $(FIX1_EXPAND_EXPERIMENT) \
		--resume /app/$(FIX1_SEALED_CKPT) \
		--no-warm-start \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		$(V66_FIX1_EXPAND_STACK)
	PYTHONPATH=. python -m experiments.training.v66.grade_fix1_expand_resilience \
		--run-dir checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_EXPAND23_RUN)) \
		--continue-epochs $(or $(EPOCHS),15)
	@echo "P1 complete. Grade: checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_EXPAND23_RUN))/resilience_gate.json"

train-v66-fix1-stage-a25-continue: ## P2: Fix-1 Stage A hold continue (MASTER-safe; 3CON/4GQB off)
	@test -f data/gates/fix1_expand_mlflow_lineage_root.json || \
		(echo "Missing lineage root — run: make register-v66-fix1-expand-lineage" && exit 1)
	@test -f checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_EXPAND23_RUN))/resilience_gate.json || \
		(echo "Missing P1 Pass stamp — run: make train-v66-fix1-expand23-continue" && exit 1)
	@python -c "import json,sys; r=json.load(open('checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_EXPAND23_RUN))/resilience_gate.json')); sys.exit(0 if r.get('passed') else 1)" || \
		(echo "P1 resilience gate FAIL — do not start P2" && exit 1)
	@test -f manifests/v6_corpus_stage_a_fix1_hold_v1.json || (echo "Missing fix1 hold manifest" && exit 1)
	$(MAKE) gate-p-feature-01 CORPUS=v6_corpus_stage_a_fix1_hold_v1.json
	@RESUME_CKPT=$$(ls checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_EXPAND23_RUN))/phase_12.pt checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_EXPAND23_RUN))/v66_phase12_*.pt 2>/dev/null | head -1); \
	test -n "$$RESUME_CKPT" || RESUME_CKPT=checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_EXPAND23_RUN))/epochs/$$(ls checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_EXPAND23_RUN))/epochs/ 2>/dev/null | sort | tail -1); \
	test -f "$$RESUME_CKPT" || (echo "Missing P1 resume ckpt" && exit 1); \
	mkdir -p mlruns checkpoints/v66/runs pdb_cache checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_STAGE_A25_RUN)); \
	printf '%s\n' \
		'# Fix-1 expand P2 — Stage A hold continue (3CON/4GQB MASTER-blocked; true 25 deferred)' \
		"resume: $$RESUME_CKPT" \
		'mlflow_experiment: $(FIX1_EXPAND_EXPERIMENT)' \
		'corpus: v6_corpus_stage_a_fix1_hold_v1.json' \
		> checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_STAGE_A25_RUN))/README.md; \
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_fix1_hold_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_STAGE_A25_RUN)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),15) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment $(FIX1_EXPAND_EXPERIMENT) \
		--resume /app/$$RESUME_CKPT \
		--no-warm-start \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		$(V66_FIX1_EXPAND_STACK)
	PYTHONPATH=. python -m experiments.training.v66.grade_fix1_expand_resilience \
		--run-dir checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_STAGE_A25_RUN)) \
		--continue-epochs $(or $(EPOCHS),15)
	@echo "P2 complete. Grade: checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_STAGE_A25_RUN))/resilience_gate.json"

train-v66-fix1-raf1-continue: ## P3: Fix-1 RAF1 mix transfer (RAF1 hold + Stage A anchors)
	@test -f checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_STAGE_A25_RUN))/resilience_gate.json || \
		(echo "Missing P2 Pass stamp — run: make train-v66-fix1-stage-a25-continue" && exit 1)
	@python -c "import json,sys; r=json.load(open('checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_STAGE_A25_RUN))/resilience_gate.json')); sys.exit(0 if r.get('passed') else 1)" || \
		(echo "P2 resilience gate FAIL — do not start P3" && exit 1)
	@test -f manifests/v6_corpus_raf1_ppi_fix1_mix_v1.json || (echo "Missing RAF1 fix1 mix manifest" && exit 1)
	$(MAKE) gate-p-feature-01 CORPUS=v6_corpus_raf1_ppi_fix1_mix_v1.json
	@RESUME_CKPT=$$(ls checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_STAGE_A25_RUN))/phase_12.pt checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_STAGE_A25_RUN))/v66_phase12_*.pt 2>/dev/null | head -1); \
	test -n "$$RESUME_CKPT" || RESUME_CKPT=checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_STAGE_A25_RUN))/epochs/$$(ls checkpoints/v66/runs/$(or $(RESUME_RUN),$(FIX1_STAGE_A25_RUN))/epochs/ 2>/dev/null | sort | tail -1); \
	test -f "$$RESUME_CKPT" || (echo "Missing P2 resume ckpt" && exit 1); \
	mkdir -p mlruns checkpoints/v66/runs pdb_cache checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_RAF1_RUN)); \
	printf '%s\n' \
		'# Fix-1 expand P3 — RAF1 mix continue (pathway hold + Stage A anchors; not PPI edges)' \
		"resume: $$RESUME_CKPT" \
		'mlflow_experiment: $(FIX1_EXPAND_EXPERIMENT)' \
		'corpus: v6_corpus_raf1_ppi_fix1_mix_v1.json' \
		> checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_RAF1_RUN))/README.md; \
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_raf1_ppi_fix1_mix_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_RAF1_RUN)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),13) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),15) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment $(FIX1_EXPAND_EXPERIMENT) \
		--resume /app/$$RESUME_CKPT \
		--no-warm-start \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--p2-bridge-lr $(or $(LR),3.0e-5) \
		$(V66_FIX1_EXPAND_STACK)
	PYTHONPATH=. python -m experiments.training.v66.grade_fix1_expand_resilience \
		--run-dir checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_RAF1_RUN)) \
		--continue-epochs $(or $(EPOCHS),15) \
		--profile transfer
	@echo "P3 complete. Grade: checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_RAF1_RUN))/resilience_gate.json"

grade-v66-fix1-biology: ## P4: biology grade pack vs sealed + optional promoted champion
	@test -f $(FIX1_SEALED_CKPT) || (echo "Missing sealed ckpt" && exit 1)
	$(SCIENCE_RUN) science python -m experiments.training.v66.grade_fix1_biology_pack \
		--sealed-checkpoint /app/$(FIX1_SEALED_CKPT) \
		--champion-dir /app/checkpoints/v66/runs/$(or $(CHAMPION_RUN),$(FIX1_RAF1_RUN)) \
		--output-dir /app/checkpoints/v66/diagnostics/fix1_expand_biology \
		--device $(or $(DEVICE),cuda)
	@echo "P4 biology pack → checkpoints/v66/diagnostics/fix1_expand_biology/"

# Fix-1 sealed continue + mean-residue routing entropy sparsity (hub-rehab bet).
# Resume ONLY v66_healthy_sealed.pt — never P2/P3 expand champions.
# Default RUN_ID: fix1_s4_sparsity_sealed_continue_v1
# Docs: docs/specs/routing-entropy-sparsity/design.md
train-v66-fix1-sparsity-sealed-continue: ## Fix-1 sealed continue + mean-residue entropy sparsity
	# resume HEALTHY_FIX1_CKPT / v66_healthy_sealed.pt
	# --routing-entropy-sparsity-coeff 0.0075
	# --routing-entropy-sparsity-warmup-epochs 8
	# corpus: Stage A-12 small OR feeler_expand
	# mlflow experiment tokyo-eyes-v66-fix1-expand
	# Fix-1 stack flags unchanged
	@test -f data/gates/fix1_expand_mlflow_lineage_root.json || \
		(echo "Missing lineage root — run: make register-v66-fix1-expand-lineage" && exit 1)
	@test -f $(FIX1_SEALED_CKPT) || (echo "Missing sealed ckpt $(FIX1_SEALED_CKPT)" && exit 1)
	@case "$(or $(RUN_ID),$(FIX1_SPARSITY_RUN))" in \
		fix1_s4_stack_initseed_controlled_3d_seed2_v1) \
			echo "ERROR: refuse overwrite of locked healthy sealed run dir."; \
			exit 1 ;; \
	esac
	@test -f manifests/$(or $(CORPUS),$(FIX1_SPARSITY_CORPUS)) || \
		(echo "Missing corpus manifests/$(or $(CORPUS),$(FIX1_SPARSITY_CORPUS))" && exit 1)
	$(MAKE) gate-p-feature-01 CORPUS=$(or $(CORPUS),$(FIX1_SPARSITY_CORPUS))
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache \
		checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_SPARSITY_RUN))
	@printf '%s\n' \
		'# Fix-1 sparsity sealed continue — mean-residue routing entropy' \
		'run_id: $(or $(RUN_ID),$(FIX1_SPARSITY_RUN))' \
		'resume: $(FIX1_SEALED_CKPT)' \
		'mlflow_experiment: $(FIX1_EXPAND_EXPERIMENT)' \
		'corpus: $(or $(CORPUS),$(FIX1_SPARSITY_CORPUS))' \
		'routing_entropy_sparsity_coeff: $(or $(SPARSITY_COEFF),$(FIX1_SPARSITY_COEFF))' \
		'routing_entropy_sparsity_warmup_epochs: $(or $(SPARSITY_WARMUP),$(FIX1_SPARSITY_WARMUP))' \
		'telemetry: routing_sparsity_per_epoch.jsonl' \
		'docs: docs/specs/routing-entropy-sparsity/design.md' \
		> checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_SPARSITY_RUN))/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),$(FIX1_SPARSITY_CORPUS)) \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_SPARSITY_RUN)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),$(FIX1_SPARSITY_MAX_PROTEINS)) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),15) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment $(FIX1_EXPAND_EXPERIMENT) \
		--resume /app/$(FIX1_SEALED_CKPT) \
		--no-warm-start \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--routing-entropy-sparsity-coeff $(or $(SPARSITY_COEFF),$(FIX1_SPARSITY_COEFF)) \
		--routing-entropy-sparsity-warmup-epochs $(or $(SPARSITY_WARMUP),$(FIX1_SPARSITY_WARMUP)) \
		$(V66_FIX1_EXPAND_STACK)
	@echo "Sparsity sealed continue complete."
	@echo "Run: checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_SPARSITY_RUN))"
	@echo "Collision log: .../routing_sparsity_per_epoch.jsonl"
	@echo "Next: grade hub knockout on 4OBE; record sparsity_gate.json (Task 6)"

# Confirm-continue after sparsity gate rewrite: resume epoch_045, λ=0.0075 locked.
# Save eligibility uses mean_residue_H ∈ [0.50, 0.90] + max_share < 0.45 (no H(f̄)≤1.21).
FIX1_SPARSITY_CONFIRM_RUN ?= fix1_s4_sparsity_confirm_continue_v1
FIX1_SPARSITY_CONFIRM_RESUME ?= checkpoints/v66/runs/fix1_s4_sparsity_sealed_continue_v1/epochs/epoch_045.pt

train-v66-fix1-sparsity-confirm-continue: ## Confirm mean-H band + hub hold from sparsity ep045
	@test -f $(FIX1_SPARSITY_CONFIRM_RESUME) || \
		(echo "Missing resume $(FIX1_SPARSITY_CONFIRM_RESUME) — run sparsity sealed continue first" && exit 1)
	@test -f data/gates/fix1_expand_mlflow_lineage_root.json || \
		(echo "Missing lineage root — run: make register-v66-fix1-expand-lineage" && exit 1)
	@case "$(or $(RUN_ID),$(FIX1_SPARSITY_CONFIRM_RUN))" in \
		fix1_s4_stack_initseed_controlled_3d_seed2_v1|fix1_s4_sparsity_sealed_continue_v1) \
			echo "ERROR: refuse overwrite of sealed/parent sparsity run dir."; \
			exit 1 ;; \
	esac
	@test -f manifests/$(or $(CORPUS),$(FIX1_SPARSITY_CORPUS)) || \
		(echo "Missing corpus manifests/$(or $(CORPUS),$(FIX1_SPARSITY_CORPUS))" && exit 1)
	$(MAKE) gate-p-feature-01 CORPUS=$(or $(CORPUS),$(FIX1_SPARSITY_CORPUS))
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache \
		checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_SPARSITY_CONFIRM_RUN))
	@printf '%s\n' \
		'# Fix-1 sparsity confirm continue — mean-residue save gates' \
		'run_id: $(or $(RUN_ID),$(FIX1_SPARSITY_CONFIRM_RUN))' \
		'resume: $(FIX1_SPARSITY_CONFIRM_RESUME)' \
		'lambda_sparse: $(or $(SPARSITY_COEFF),$(FIX1_SPARSITY_COEFF))' \
		'save: mean_residue_H in [0.50, 0.90]; max_share < 0.45' \
		'monitor: H(fbar) warn < 1.00; abort <= 0.80' \
		'docs: docs/specs/routing-entropy-sparsity/design.md' \
		> checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_SPARSITY_CONFIRM_RUN))/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),$(FIX1_SPARSITY_CORPUS)) \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_SPARSITY_CONFIRM_RUN)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),$(FIX1_SPARSITY_MAX_PROTEINS)) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),5) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment $(FIX1_EXPAND_EXPERIMENT) \
		--resume /app/$(FIX1_SPARSITY_CONFIRM_RESUME) \
		--no-warm-start \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--routing-entropy-sparsity-coeff $(or $(SPARSITY_COEFF),$(FIX1_SPARSITY_COEFF)) \
		--routing-entropy-sparsity-warmup-epochs $(or $(SPARSITY_WARMUP),0) \
		$(V66_FIX1_EXPAND_STACK)
	@echo "Sparsity confirm continue complete."
	@echo "Run: checkpoints/v66/runs/$(or $(RUN_ID),$(FIX1_SPARSITY_CONFIRM_RUN))"
	@echo "Grade: hub knockout 4OBE on final epoch; expect mean_H band + ρ≥0.45"

# Banked sparsity phase champion (confirm_v2 epoch 48). Next biology probe: G12D hub migration.
FIX1_SPARSITY_CHAMPION_RUN ?= fix1_s4_sparsity_confirm_continue_v2
FIX1_SPARSITY_CHAMPION_CKPT ?= checkpoints/v66/runs/$(FIX1_SPARSITY_CHAMPION_RUN)/v66_sparsity_champion.pt
LEDGER_B_SENSITIVITY_EPOCHS ?= 46,48,50
FIX1_SPARSITY_CHAMPION_GATE ?= data/gates/fix1_sparsity_champion.json
HEALTHY_FIX1_CKPT ?= checkpoints/v66/runs/fix1_s4_stack_initseed_controlled_3d_seed2_v1/v66_healthy_sealed.pt

# Structural monopoly closeout: full-chain Gini(out_effect) baseline vs sparsity champion.
# Panel: 2SHP / 3PP0 / 4OBE / 4DSO / 5VQ2. Never hub-slice H. ΔG = G_base − G_champion.
grade-v66-fix1-gini-reduction-analysis: ## Full-chain Gini ΔG sealed baseline vs sparsity champion
	@test -f $(HEALTHY_FIX1_CKPT) || \
		(echo "Missing baseline $(HEALTHY_FIX1_CKPT)" && exit 1)
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.gini_reduction_analysis \
		--baseline /app/$(HEALTHY_FIX1_CKPT) \
		--champion /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/gini_reduction_analysis.json
	@echo "Gini reduction → checkpoints/v66/diagnostics/routing_sparsity/gini_reduction_analysis.json"
	@echo "ΔG = G(HEALTHY_FIX1_CKPT) − G(FIX1_SPARSITY_CHAMPION_CKPT); positive ⇒ monopoly reduced"

# SHP2 inactive→active OOD: 2SHP → 6CRF on sparsity champion (three locked arms).
# Prereg: data/gates/shp2_2shp_6mcf_ood_prereg.json (active PDB corrected 6MCF→6CRF)
# Spec: docs/specs/shp2-2shp-6mcf-ood/design.md
grade-v66-fix1-shp2-2shp-6crf-ood: ## SHP2 2SHP→6CRF OOD migration on sparsity champion
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@test -f data/gates/shp2_2shp_6mcf_ood_prereg.json || \
		(echo "Missing prereg data/gates/shp2_2shp_6mcf_ood_prereg.json" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	@echo "NOTE: active open SHP2 is 6CRF (E76K). 6MCF is NOT SHP2 (7SK/Tat) — historical label corrected."
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.shp2_2shp_6mcf_ood \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--prereg /app/data/gates/shp2_2shp_6mcf_ood_prereg.json \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/shp2_2shp_6crf_ood.json
	@echo "SHP2 OOD → checkpoints/v66/diagnostics/routing_sparsity/shp2_2shp_6crf_ood.json"
	@echo "Pass: Spearman≥0.50 (shared resseq); Jaccard(H10%)≥0.50 on all 6CRF perturbations; G(6CRF)∈[0.12,0.25]"
	@echo "Docs: docs/specs/shp2-2shp-6mcf-ood/design.md"

# Alias kept for the original make name after 6MCF→6CRF identity correction.
grade-v66-fix1-shp2-2shp-6mcf-ood: grade-v66-fix1-shp2-2shp-6crf-ood ## Alias → 6CRF gate (6MCF was wrong PDB)

grade-v66-fix1-sparsity-g12d-hub-migration: ## Next: KRAS G12D hub migration on sparsity champion
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@test -f $(FIX1_SPARSITY_CHAMPION_GATE) || \
		(echo "Missing gate stamp $(FIX1_SPARSITY_CHAMPION_GATE)" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.kras_hub_migration_ood \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/kras_hub_migration_4obe_4dso.json
	@echo "G12D hub migration → checkpoints/v66/diagnostics/routing_sparsity/kras_hub_migration_4obe_4dso.json"
	@echo "Pass: R_4DSO > R_4OBE (trunk in-flow into 163). Docs: docs/specs/routing-entropy-sparsity/next-phase.md"

# Phase A triangulation: 4OBE / 4DSO / 5VQ2 on sparsity champion (forward knockout + classical ΔE).
# Spec: docs/specs/kras-topo-structural-inference/design.md
# Note: historical triad edge ΔE is report-only; rewiring closeout → grade-v66-kras-topo-edge-four-quadrant
grade-v66-fix1-sparsity-kras-topo-matrix: ## Primary topo matrix 4OBE/4DSO/5VQ2 on sparsity champion
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@test -f $(FIX1_SPARSITY_CHAMPION_GATE) || \
		(echo "Missing gate stamp $(FIX1_SPARSITY_CHAMPION_GATE)" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.kras_topo_structural_matrix \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/kras_topo_matrix_4obe_4dso_5vq2.json
	@echo "Topo matrix → checkpoints/v66/diagnostics/routing_sparsity/kras_topo_matrix_4obe_4dso_5vq2.json"
	@echo "Blocking: ρ(4DSO,5VQ2)>ρ(4OBE,5VQ2); switch-lock 12-32@11Å / 12-61@10Å. Edge ΔE on triad = report-only."
	@echo "Docs: docs/specs/kras-topo-structural-inference/design.md"

# Four-quadrant classical edge ΔE rematch (structure-only; no checkpoint).
# Roster: 4LPK / 6GOD / 5US4 / 6GOF. Spec: edge-delta-four-quadrant.md
grade-v66-kras-topo-edge-four-quadrant: ## KRAS G12D four-quadrant Cα edge-ΔE rematch
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	$(SCIENCE_RUN) science python -m experiments.diagnostics.kras_topo_edge_delta_four_quadrant \
		--pdb-dir /tmp/dtie_pdb_cache \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/kras_topo_edge_delta_four_quadrant.json
	@echo "Four-quadrant → checkpoints/v66/diagnostics/routing_sparsity/kras_topo_edge_delta_four_quadrant.json"
	@echo "Pass: |ΔE_CA(5US4,6GOD)|<|ΔE_CA(4LPK,6GOD)| AND |ΔE_CA(5US4,6GOF)|<|ΔE_CA(4LPK,6GOD)|"
	@echo "Docs: docs/specs/kras-topo-structural-inference/edge-delta-four-quadrant.md"

# Matched-state residue-12 graft: WT scaffold ← mut site 12 (features+Cα); scramble control.
# Spec: docs/specs/kras-topo-structural-inference/kras-g12-residue-graft-prereg.md
grade-v66-kras-g12-residue-graft: ## KRAS G12 residue-12 graft validation (4LPK/5US4 + 6GOD/6GOF)
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@test -f data/gates/kras_g12_residue_graft_prereg.json || \
		(echo "Missing pre-reg stamp data/gates/kras_g12_residue_graft_prereg.json" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.kras_g12_residue_graft_validation \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/kras_g12_residue_graft.json
	@echo "Graft validation → checkpoints/v66/diagnostics/routing_sparsity/kras_g12_residue_graft.json"
	@echo "Pass: ρ(graft,mut)>ρ(WT,mut) AND ρ(graft,mut)>ρ(scramble,mut) on both OFF and ON arms"
	@echo "Docs: docs/specs/kras-topo-structural-inference/kras-g12-residue-graft-prereg.md"

# Neighborhood / conduit graft (OFF arm only): 4LPK ← N12 from 5US4.
# Spec: docs/specs/kras-topo-structural-inference/kras-g12-neighborhood-graft-prereg.md
grade-v66-kras-g12-neighborhood-graft: ## KRAS G12 neighborhood conduit graft (4LPK←5US4)
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@test -f data/gates/kras_g12_neighborhood_graft_prereg.json || \
		(echo "Missing pre-reg stamp data/gates/kras_g12_neighborhood_graft_prereg.json" && exit 1)
	@test -f data/gates/kras_g12_residue_graft_closeout.json || \
		(echo "Missing single-site closeout data/gates/kras_g12_residue_graft_closeout.json" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.kras_g12_neighborhood_graft_validation \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/kras_g12_neighborhood_graft.json
	@echo "Neighborhood graft → checkpoints/v66/diagnostics/routing_sparsity/kras_g12_neighborhood_graft.json"
	@echo "Pass: Δρ_neigh > Δρ_single AND ρ(neigh,mut) > ρ(scramble,mut)"
	@echo "Docs: docs/specs/kras-topo-structural-inference/kras-g12-neighborhood-graft-prereg.md"

# Child 4: post-lift hyperbolic latent graft (OFF 4LPK←5US4 x_hyp at N12).
# Spec: docs/specs/kras-topo-structural-inference/kras-g12-hyperbolic-latent-graft-prereg.md
grade-v66-kras-g12-hyperbolic-latent-graft: ## KRAS G12 hyp latent graft (post-lift x_hyp)
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@test -f data/gates/kras_g12_hyperbolic_latent_graft_prereg.json || \
		(echo "Missing pre-reg stamp data/gates/kras_g12_hyperbolic_latent_graft_prereg.json" && exit 1)
	@test -f data/gates/kras_g12_neighborhood_graft_closeout.json || \
		(echo "Missing OFF neighborhood closeout" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.kras_g12_hyperbolic_latent_graft_validation \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/kras_g12_hyperbolic_latent_graft.json
	@echo "Hyp latent graft → checkpoints/v66/diagnostics/routing_sparsity/kras_g12_hyperbolic_latent_graft.json"
	@echo "Pass: Δρ_hyp_depth > Δρ_euc_neigh_on_hyp_depth AND ρ(hyp_graft,mut) > ρ(hyp_scramble,mut)"
	@echo "Docs: docs/specs/kras-topo-structural-inference/kras-g12-hyperbolic-latent-graft-prereg.md"

# Basin routing automaton Child 1: four-quadrant observation classify smoke.
# Spec: docs/specs/kras-topo-structural-inference/kras-basin-observation-classify-prereg.md
grade-v66-kras-basin-observation-classify: ## KRAS basin observation classify (4LPK/5US4/6GOD/6GOF)
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@test -f data/gates/kras_basin_observation_classify_prereg.json || \
		(echo "Missing pre-reg stamp data/gates/kras_basin_observation_classify_prereg.json" && exit 1)
	@test -f data/gates/kras_g12_hyperbolic_latent_graft_closeout.json || \
		(echo "Missing Child 4 closeout data/gates/kras_g12_hyperbolic_latent_graft_closeout.json" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.kras_basin_observation_classify_validation \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/kras_basin_observation_classify.json
	@echo "Basin classify → checkpoints/v66/diagnostics/routing_sparsity/kras_basin_observation_classify.json"
	@echo "Pass: guards + same_nuc > cross_nuc + N12 |Δ| > scramble on OFF and ON"
	@echo "Docs: docs/specs/kras-topo-structural-inference/kras-basin-observation-classify-prereg.md"

# Tokyo Eye v7 — Hyp MP forward smoke (isolated from v66).
# Spec: docs/specs/tokyo-eye-v7/forward-smoke-prereg.md
grade-v7-forward-smoke: ## Tokyo Eye v7 Hyp MP forward smoke
	@test -f data/gates/tokyo_eye_v7_lineage_opened.json || \
		(echo "Missing v7 lineage stamp data/gates/tokyo_eye_v7_lineage_opened.json" && exit 1)
	@test -f data/gates/tokyo_eye_v7_forward_smoke_prereg.json || \
		(echo "Missing forward smoke pre-reg" && exit 1)
	@mkdir -p checkpoints/v7/diagnostics
	$(SCIENCE_RUN) science python -m experiments.training.v7.forward_smoke
	@echo "v7 smoke → checkpoints/v7/diagnostics/tokyo_eye_v7_forward_smoke.json"
	@echo "Docs: docs/specs/tokyo-eye-v7/README.md"

# Register MLflow experiment tokyo-eyes-v7 + lineage-root run (isolated from v66).
mlflow-register-v7-lineage: ## Create tokyo-eyes-v7 MLflow experiment + lineage root
	@mkdir -p data/gates checkpoints/v7/diagnostics
	$(SCIENCE_RUN) -e MLFLOW_TRACKING_URI=http://mlflow:5000 science python -m experiments.training.v7.register_mlflow_lineage \
		--tracking-uri http://mlflow:5000 \
		--experiment tokyo-eyes-v7 \
		--stamp /app/data/gates/tokyo_eye_v7_mlflow_lineage_root.json
	@echo "MLflow → experiment tokyo-eyes-v7; stamp data/gates/tokyo_eye_v7_mlflow_lineage_root.json"
	@echo "Registered model name (on promote): TokyoEye-v7"

# Seal cutover-scaffold checkpoint (init weights; not biology champion).
seal-v7-cutover-scaffold: ## Seal TokyoEye cutover scaffold .pt for contract production
	@mkdir -p checkpoints/v7 data/gates
	$(SCIENCE_RUN) science python -m experiments.training.v7.seal_cutover_scaffold \
		--out /app/checkpoints/v7/tokyo_eye_v7_cutover_scaffold.pt
	@echo "Scaffold → checkpoints/v7/tokyo_eye_v7_cutover_scaffold.pt"

# B′ surgical warmstart from Fix-1 sparsity champion (deny convs/norms; cold hyp_mp).
FIX1_SPARSITY_CHAMPION_CKPT ?= checkpoints/v66/runs/fix1_s4_sparsity_confirm_continue_v2/v66_sparsity_champion.pt
V7_BPRIME_WARMSTART_CKPT ?= checkpoints/v7/tokyo_eye_v7_bprime_warmstart.pt
V7_BPRIME_HEALTH_RUN ?= tokyo_eye_v7_bprime_health_v1
# Fix-1 mitigations without S4 --hyperbolic-mp-graph (v7 Hyp MP is primary).
V7_BPRIME_STACK := --v66-feeler-lineage --v66-feeler-rim-fanout-model --v66-feeler-geom-angular-prior --input-feature-zscore --gate-include-sasa --prototype-repulsion-coeff $(or $(PROTO_REPULSION_COEFF),1.0) --prototype-repulsion-margin $(or $(PROTO_REPULSION_MARGIN),0.25) --gate-logit-softplus-init $(or $(GATE_SOFTPLUS),6.612216472625732) --gate-logit-softplus-floor $(or $(GATE_SOFTPLUS_FLOOR),6.612216472625732) --geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) --geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) --rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) --rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) --num-experts $(or $(NUM_EXPERTS),4) --feature-liveness-probe --save-epoch-snapshots

seal-v7-bprime-warmstart: ## Surgical B′: Fix-1 champion → TokyoEye (deny Euc trunk)
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || (echo "Missing donor $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@mkdir -p checkpoints/v7 data/gates
	$(SCIENCE_RUN) science python -m experiments.training.v7.seal_bprime_warmstart \
		--donor /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--out /app/$(V7_BPRIME_WARMSTART_CKPT)
	@echo "B′ warmstart → $(V7_BPRIME_WARMSTART_CKPT)"

train-v7-bprime-health: ## Stage-A health micro-run from B′ warmstart (Hyp MP primary)
	@test -f data/gates/tokyo_eye_v7_bprime_health_prereg.json || (echo "Missing prereg stamp" && exit 1)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@$(MAKE) seal-v7-bprime-warmstart
	@mkdir -p checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_HEALTH_RUN)) pdb_cache
	@printf '%s\n' \
		'# Tokyo Eye v7 B′ health — surgical warmstart from Fix-1 champion' \
		'donor: $(FIX1_SPARSITY_CHAMPION_CKPT)' \
		'warmstart: $(V7_BPRIME_WARMSTART_CKPT)' \
		'spec: docs/specs/tokyo-eye-v7/bprime-health-warmstart-prereg.md' \
		> checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_HEALTH_RUN))/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v7.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_HEALTH_RUN)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--epochs $(or $(EPOCHS),8) \
		--gnn-lineage v7 \
		--mlflow-experiment tokyo-eyes-v7 \
		--resume /app/$(V7_BPRIME_WARMSTART_CKPT) \
		$(V7_BPRIME_STACK) \
		$(if $(NO_MLFLOW),--no-mlflow,)
	@echo "Health run → checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_HEALTH_RUN))"

V7_BPRIME_HEALTH_DISC ?= checkpoints/v7/runs/tokyo_eye_v7_bprime_health_v1/v7_best_disc.pt
V7_BPRIME_HEALTH_CONTINUE_RUN ?= tokyo_eye_v7_bprime_health_continue_v1

# Longer continue from health disc saver — aim for disc_r≥0.25 + eligible v7_best (not promote).
train-v7-bprime-health-continue: ## Continue B′ health from v7_best_disc (longer; no promote)
	@test -f $(or $(RESUME),$(V7_BPRIME_HEALTH_DISC)) || \
		(echo "Missing resume ckpt $(or $(RESUME),$(V7_BPRIME_HEALTH_DISC))" && exit 1)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_HEALTH_CONTINUE_RUN)) pdb_cache
	@printf '%s\n' \
		'# Tokyo Eye v7 B′ health continue — from disc ckpt (not promote)' \
		'resume: $(or $(RESUME),$(V7_BPRIME_HEALTH_DISC))' \
		'prior: tokyo_eye_v7_bprime_health_v1 (health FAIL disc_r≈0.203)' \
		'target: disc_r_mean≥0.25 + eligible v7_best if routing allows' \
		'spec: docs/specs/tokyo-eye-v7/bprime-health-warmstart-prereg.md' \
		> checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_HEALTH_CONTINUE_RUN))/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v7.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_HEALTH_CONTINUE_RUN)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--epochs $(or $(EPOCHS),24) \
		--gnn-lineage v7 \
		--mlflow-experiment tokyo-eyes-v7 \
		--resume /app/$(or $(RESUME),$(V7_BPRIME_HEALTH_DISC)) \
		$(V7_BPRIME_STACK) \
		$(if $(NO_MLFLOW),--no-mlflow,)
	@echo "Continue → checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_HEALTH_CONTINUE_RUN))"

# Option B: disc occupancy / depth-scale bump + sparsity-style save (no legacy H≤1.21).
V7_BPRIME_DISC_RESUME ?= checkpoints/v7/runs/tokyo_eye_v7_bprime_health_continue_v1/v7_best_disc.pt
V7_BPRIME_DISC_CONTINUE_RUN ?= tokyo_eye_v7_bprime_disc_continue_v1
V7_BPRIME_DISC_OCC ?= 0.70
V7_BPRIME_DISC_DEPTH_SCALE ?= 1.60
V7_BPRIME_DISC_DEPTH_TARGET ?= 0.60
V7_BPRIME_MEAN_H_MIN ?= 0.25
V7_BPRIME_MEAN_H_MAX ?= 0.90

train-v7-bprime-disc-continue: ## B′ disc-health continue (prereg Option B; no promote)
	@test -f data/gates/tokyo_eye_v7_bprime_disc_health_continue_prereg.json || \
		(echo "Missing prereg stamp data/gates/tokyo_eye_v7_bprime_disc_health_continue_prereg.json" && exit 1)
	@test -f $(or $(RESUME),$(V7_BPRIME_DISC_RESUME)) || \
		(echo "Missing resume ckpt $(or $(RESUME),$(V7_BPRIME_DISC_RESUME))" && exit 1)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_DISC_CONTINUE_RUN)) pdb_cache
	@printf '%s\n' \
		'# Tokyo Eye v7 B′ disc-health continue — Option B (not promote)' \
		'resume: $(or $(RESUME),$(V7_BPRIME_DISC_RESUME))' \
		'disc_occupancy_coeff: $(or $(DISC_OCC),$(V7_BPRIME_DISC_OCC))' \
		'disc_depth_scale_coeff: $(or $(DISC_DEPTH_SCALE),$(V7_BPRIME_DISC_DEPTH_SCALE))' \
		'disc_depth_scale_target: $(or $(DISC_DEPTH_TARGET),$(V7_BPRIME_DISC_DEPTH_TARGET))' \
		'sparsity_style_save: mean_residue ∈ [$(or $(MEAN_H_MIN),$(V7_BPRIME_MEAN_H_MIN)), $(or $(MEAN_H_MAX),$(V7_BPRIME_MEAN_H_MAX))]' \
		'spec: docs/specs/tokyo-eye-v7/bprime-disc-health-continue-prereg.md' \
		> checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_DISC_CONTINUE_RUN))/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v7.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_DISC_CONTINUE_RUN)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--epochs $(or $(EPOCHS),24) \
		--gnn-lineage v7 \
		--mlflow-experiment tokyo-eyes-v7 \
		--resume /app/$(or $(RESUME),$(V7_BPRIME_DISC_RESUME)) \
		$(V7_BPRIME_STACK) \
		--sparsity-style-save \
		--routing-entropy-mean-residue-min-save $(or $(MEAN_H_MIN),$(V7_BPRIME_MEAN_H_MIN)) \
		--routing-entropy-mean-residue-max-save $(or $(MEAN_H_MAX),$(V7_BPRIME_MEAN_H_MAX)) \
		--disc-occupancy-coeff $(or $(DISC_OCC),$(V7_BPRIME_DISC_OCC)) \
		--disc-depth-scale-coeff $(or $(DISC_DEPTH_SCALE),$(V7_BPRIME_DISC_DEPTH_SCALE)) \
		--disc-depth-scale-target $(or $(DISC_DEPTH_TARGET),$(V7_BPRIME_DISC_DEPTH_TARGET)) \
		$(if $(NO_MLFLOW),--no-mlflow,)
	@echo "Disc continue → checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_DISC_CONTINUE_RUN))"

# Core radial floor: lift e1 (high-ρ low-τ) off origin without rim push.
V7_BPRIME_CORE_RESUME ?= checkpoints/v7/runs/tokyo_eye_v7_bprime_disc_continue_v1/v7_best_disc.pt
V7_BPRIME_CORE_FLOOR_RUN ?= tokyo_eye_v7_bprime_core_floor_continue_v1
V7_BPRIME_CORE_FLOOR_COEFF ?= 1.0
V7_BPRIME_CORE_FLOOR_MIN_R ?= 0.15

train-v7-bprime-core-floor-continue: ## B′ e1 core radial floor continue (prereg; no promote)
	@test -f data/gates/tokyo_eye_v7_bprime_core_radial_floor_prereg.json || \
		(echo "Missing prereg stamp data/gates/tokyo_eye_v7_bprime_core_radial_floor_prereg.json" && exit 1)
	@test -f $(or $(RESUME),$(V7_BPRIME_CORE_RESUME)) || \
		(echo "Missing resume ckpt $(or $(RESUME),$(V7_BPRIME_CORE_RESUME))" && exit 1)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_CORE_FLOOR_RUN)) pdb_cache
	@printf '%s\n' \
		'# Tokyo Eye v7 B′ core radial floor continue — e1 origin lift (not promote)' \
		'resume: $(or $(RESUME),$(V7_BPRIME_CORE_RESUME))' \
		'core_radial_floor_coeff: $(or $(CORE_FLOOR_COEFF),$(V7_BPRIME_CORE_FLOOR_COEFF))' \
		'core_radial_floor_min_r: $(or $(CORE_FLOOR_MIN_R),$(V7_BPRIME_CORE_FLOOR_MIN_R))' \
		'spec: docs/specs/tokyo-eye-v7/bprime-core-radial-floor-prereg.md' \
		> checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_CORE_FLOOR_RUN))/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v7.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_CORE_FLOOR_RUN)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--epochs $(or $(EPOCHS),24) \
		--gnn-lineage v7 \
		--mlflow-experiment tokyo-eyes-v7 \
		--resume /app/$(or $(RESUME),$(V7_BPRIME_CORE_RESUME)) \
		$(V7_BPRIME_STACK) \
		--sparsity-style-save \
		--routing-entropy-mean-residue-min-save $(or $(MEAN_H_MIN),$(V7_BPRIME_MEAN_H_MIN)) \
		--routing-entropy-mean-residue-max-save $(or $(MEAN_H_MAX),$(V7_BPRIME_MEAN_H_MAX)) \
		--disc-occupancy-coeff $(or $(DISC_OCC),$(V7_BPRIME_DISC_OCC)) \
		--disc-depth-scale-coeff $(or $(DISC_DEPTH_SCALE),$(V7_BPRIME_DISC_DEPTH_SCALE)) \
		--disc-depth-scale-target $(or $(DISC_DEPTH_TARGET),$(V7_BPRIME_DISC_DEPTH_TARGET)) \
		--core-radial-floor-coeff $(or $(CORE_FLOOR_COEFF),$(V7_BPRIME_CORE_FLOOR_COEFF)) \
		--core-radial-floor-min-r $(or $(CORE_FLOOR_MIN_R),$(V7_BPRIME_CORE_FLOOR_MIN_R)) \
		$(if $(NO_MLFLOW),--no-mlflow,)
	@echo "Core-floor continue → checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_CORE_FLOOR_RUN))"

# Seal disc-health bank (epoch_041) as restore SSOT for v7 B′.
ARCHAEOLOGY_HYPMP_CKPT ?= checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1/v7_healthy_sealed.pt
V7_BPRIME_UNC_RESUME ?= $(ARCHAEOLOGY_HYPMP_CKPT)
V7_BPRIME_UNC_RUN ?= tokyo_eye_v7_bprime_uncertainty_heads_v1

seal-v7-bprime-healthy: ## Seal v7 B′ disc-health bank (epoch_041 → v7_healthy_sealed.pt)
	@test -f checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1/epochs/epoch_041.pt || \
		(echo "Missing epoch_041 health bank" && exit 1)
	@mkdir -p checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1 data/gates
	$(SCIENCE_RUN) science python -m experiments.training.v7.seal_healthy_bprime \
		--run-dir /app/checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1 \
		--epoch-ckpt /app/checkpoints/v7/runs/tokyo_eye_v7_bprime_core_floor_continue_v1/epochs/epoch_041.pt \
		--epoch 41 \
		--out /app/$(ARCHAEOLOGY_HYPMP_CKPT)
	@echo "Healthy sealed → $(ARCHAEOLOGY_HYPMP_CKPT)"
	@echo "Gate → data/gates/tokyo_eye_v7_bprime_healthy_sealed.json"

# B1 teleconnections: AlleleSens conduit vs scramble on sealed v7 Θ (4LPK/5US4, 6GOD/6GOF).
grade-v7-b1-teleconnections: ## B1 AlleleSens teleconnections grade (prereg frozen)
	@test -f data/gates/tokyo_eye_v7_b1_teleconnections_prereg.json || \
		(echo "Missing prereg data/gates/tokyo_eye_v7_b1_teleconnections_prereg.json" && exit 1)
	@test -f $(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT)) || \
		(echo "Missing sealed Θ $(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT)) — run make seal-v7-bprime-healthy" && exit 1)
	@mkdir -p checkpoints/v7/diagnostics/b1_teleconnections data/gates pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.b1_teleconnections_grade \
		--checkpoint /app/$(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/$(or $(OUT),checkpoints/v7/diagnostics/b1_teleconnections/b1_teleconnections.json) \
		--closeout /app/$(or $(CLOSEOUT),data/gates/tokyo_eye_v7_b1_teleconnections_closeout.json); \
	status=$$?; \
	echo "B1 artifact → $(or $(OUT),checkpoints/v7/diagnostics/b1_teleconnections/b1_teleconnections.json)"; \
	echo "Closeout → $(or $(CLOSEOUT),data/gates/tokyo_eye_v7_b1_teleconnections_closeout.json)"; \
	echo "Docs: docs/specs/tokyo-eye-v7/b1-teleconnections-prereg.md"; \
	exit $$status

# Compare-only: G12D hub migration on sealed v7 (knockout on x_hyp; Jacobian forbidden)
grade-v7-kras-g12d-hub-migration: ## v7 G12D hub migration (4OBE→4DSO knockout x_hyp)
	@test -f $(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT)) || \
		(echo "Missing sealed Θ $(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT))" && exit 1)
	@mkdir -p checkpoints/v7/diagnostics/hub_migration logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.v7_kras_g12d_hub_migration \
		--checkpoint /app/$(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v7/diagnostics/hub_migration/kras_hub_migration_4obe_4dso.json
	@echo "v7 hub migration → checkpoints/v7/diagnostics/hub_migration/kras_hub_migration_4obe_4dso.json"
	@echo "Pass: R_out_163(4DSO) > R_out_163(4OBE) on x_hyp geodesics"
	@echo "Docs: docs/specs/tokyo-eye-v7/kras-g12d-hub-migration-prereg.md"

# Cheap Hyp-MP telemetry (one forward; no knockout)
grade-v7-hyp-mp-telemetry: ## Cheap Hyp-MP hub proxy telemetry
	@test -f $(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT)) || \
		(echo "Missing ckpt $(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT))" && exit 1)
	@mkdir -p checkpoints/v7/diagnostics/hyp_mp_telemetry
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.v7_hyp_mp_telemetry_grade \
		--checkpoint /app/$(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--targets "$(or $(TARGETS),4OBE:A 4LPK:A 6GOD:A)" \
		--output /app/$(or $(OUT),checkpoints/v7/diagnostics/hyp_mp_telemetry/hyp_mp_telemetry_baseline.json)
	@echo "Hyp-MP telemetry → $(or $(OUT),checkpoints/v7/diagnostics/hyp_mp_telemetry/hyp_mp_telemetry_baseline.json)"

# Hyp biology MP child lineage — fail-closed Cα audit smoke (Option A degree-0).
grade-v7-hyp-biology-mp-smoke: ## Biology-only Hyp MP ontology smoke (no Cα fallback)
	@test -f data/gates/tokyo_eye_v7_hyp_biology_mp_prereg.json || \
		(echo "Missing prereg data/gates/tokyo_eye_v7_hyp_biology_mp_prereg.json" && exit 1)
	@test -f $(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT)) || \
		(echo "Missing ckpt $(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT))" && exit 1)
	@mkdir -p checkpoints/v7/runs/tokyo_eye_v7_hyp_biology_mp_v1
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.v7_hyp_biology_mp_smoke \
		--checkpoint /app/$(or $(CHECKPOINT),$(ARCHAEOLOGY_HYPMP_CKPT)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--targets "$(or $(TARGETS),4OBE:A)" \
		--out /app/$(or $(OUT),checkpoints/v7/runs/tokyo_eye_v7_hyp_biology_mp_v1/hyp_biology_mp_smoke.json)
	@echo "Hyp biology MP smoke → $(or $(OUT),checkpoints/v7/runs/tokyo_eye_v7_hyp_biology_mp_v1/hyp_biology_mp_smoke.json)"
	@echo "Docs: docs/specs/tokyo-eye-v7/hyp-biology-mp-prereg.md"

# Hyp biology MP continue-train (child lineage; never overwrites sealed healthy).
V7_HYP_BIOLOGY_MP_RUN ?= tokyo_eye_v7_hyp_biology_mp_v1
train-v7-hyp-biology-mp: ## Continue from HEALTHY_V7 with biology-only Hyp MP
	@test -f data/gates/tokyo_eye_v7_hyp_biology_mp_prereg.json || \
		(echo "Missing prereg data/gates/tokyo_eye_v7_hyp_biology_mp_prereg.json" && exit 1)
	@test -f $(or $(RESUME),$(ARCHAEOLOGY_HYPMP_CKPT)) || \
		(echo "Missing resume $(or $(RESUME),$(ARCHAEOLOGY_HYPMP_CKPT))" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(or $(RUN_ID),$(V7_HYP_BIOLOGY_MP_RUN)) logs/training
	GNN_INPUT_MODE=topology_three_vector TRAINING_LOAD_FROM_PDB=1 $(SCIENCE_RUN) science python -m experiments.training.v7.hyp_biology_mp_train \
		--resume /app/$(or $(RESUME),$(ARCHAEOLOGY_HYPMP_CKPT)) \
		--manifest /app/$(or $(CORPUS),manifests/v6_corpus_stage_a_small_v1.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--output-dir /app/checkpoints/v7/runs/$(or $(RUN_ID),$(V7_HYP_BIOLOGY_MP_RUN)) \
		--device $(or $(DEVICE),cuda) \
		--epochs $(or $(EPOCHS),8) \
		--lr $(or $(LR),5e-5) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--min-disc-r-mean-hold $(or $(MIN_DISC_R),0.25)
	@echo "Hyp biology MP run → checkpoints/v7/runs/$(or $(RUN_ID),$(V7_HYP_BIOLOGY_MP_RUN))"
	@echo "Best: checkpoints/v7/runs/$(or $(RUN_ID),$(V7_HYP_BIOLOGY_MP_RUN))/v7_hyp_biology_mp_best.pt"

# Cold Hyp MP + simple Cα graph + funnel/disc curriculum (no Fix-1 / HEALTHY_V7 warmstart).
V7_COLD_HYP_MP_FUNNEL_RUN ?= tokyo_eye_v7_cold_hyp_mp_funnel_v1
train-v7-cold-hyp-mp-funnel: ## Cold Hyp MP + Cα graph + funnel curriculum
	@test -f data/gates/tokyo_eye_v7_cold_hyp_mp_funnel_prereg.json || \
		(echo "Missing prereg data/gates/tokyo_eye_v7_cold_hyp_mp_funnel_prereg.json" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(or $(RUN_ID),$(V7_COLD_HYP_MP_FUNNEL_RUN)) logs/training
	GNN_INPUT_MODE=topology_three_vector TRAINING_LOAD_FROM_PDB=1 $(SCIENCE_RUN) science python -m experiments.training.v7.cold_hyp_mp_funnel_train \
		--manifest /app/$(or $(CORPUS),manifests/v6_corpus_stage_a_small_v1.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--output-dir /app/checkpoints/v7/runs/$(or $(RUN_ID),$(V7_COLD_HYP_MP_FUNNEL_RUN)) \
		--device $(or $(DEVICE),cuda) \
		--epochs $(or $(EPOCHS),24) \
		--lr $(or $(LR),1e-4) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--min-disc-r-mean-hold $(or $(MIN_DISC_R),0.25) \
		--disc-hold-warmup-epochs $(or $(DISC_WARMUP),6) \
		$(if $(RESUME),--resume /app/$(RESUME),) \
		$(if $(USE_CUTOVER_SCAFFOLD),--use-cutover-scaffold,)
	@echo "Cold Hyp MP funnel → checkpoints/v7/runs/$(or $(RUN_ID),$(V7_COLD_HYP_MP_FUNNEL_RUN))"
	@echo "Best: checkpoints/v7/runs/$(or $(RUN_ID),$(V7_COLD_HYP_MP_FUNNEL_RUN))/v7_cold_hyp_mp_funnel_best.pt"
	@echo "Docs: docs/specs/tokyo-eye-v7/cold-hyp-mp-funnel-prereg.md"

# Cold funnel continue: sealed feelers + light MoE anti-monopoly + ER > 1.5 (MLflow on).
V7_COLD_HYP_MP_FUNNEL_ANGFILL_RUN ?= tokyo_eye_v7_cold_hyp_mp_funnel_angfill_v1
V7_COLD_HYP_MP_FUNNEL_BEST ?= checkpoints/v7/runs/$(V7_COLD_HYP_MP_FUNNEL_RUN)/v7_cold_hyp_mp_funnel_best.pt
train-v7-cold-hyp-mp-funnel-angfill: ## Continue cold best: rim/geom + specialization + ER
	@test -f data/gates/tokyo_eye_v7_cold_hyp_mp_funnel_angfill_prereg.json || \
		(echo "Missing prereg data/gates/tokyo_eye_v7_cold_hyp_mp_funnel_angfill_prereg.json" && exit 1)
	@test -f $(or $(RESUME),$(V7_COLD_HYP_MP_FUNNEL_BEST)) || \
		(echo "Missing cold best — run make train-v7-cold-hyp-mp-funnel" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(or $(RUN_ID),$(V7_COLD_HYP_MP_FUNNEL_ANGFILL_RUN)) logs/training
	GNN_INPUT_MODE=topology_three_vector TRAINING_LOAD_FROM_PDB=1 \
	MLFLOW_TRACKING_URI=http://mlflow:5000 $(SCIENCE_RUN) -e MLFLOW_TRACKING_URI science python -m experiments.training.v7.cold_hyp_mp_funnel_angfill_train \
		--resume /app/$(or $(RESUME),$(V7_COLD_HYP_MP_FUNNEL_BEST)) \
		--manifest /app/$(or $(CORPUS),manifests/v6_corpus_stage_a_small_v1.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--output-dir /app/checkpoints/v7/runs/$(or $(RUN_ID),$(V7_COLD_HYP_MP_FUNNEL_ANGFILL_RUN)) \
		--device $(or $(DEVICE),cuda) \
		--epochs $(or $(EPOCHS),16) \
		--lr $(or $(LR),5e-5) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--min-disc-r-mean-hold $(or $(MIN_DISC_R),0.25) \
		--min-disc-effective-rank $(or $(MIN_ER),1.5) \
		$(if $(NO_MLFLOW),--no-mlflow,)
	@echo "Angfill → checkpoints/v7/runs/$(or $(RUN_ID),$(V7_COLD_HYP_MP_FUNNEL_ANGFILL_RUN))"
	@echo "Best: checkpoints/v7/runs/$(or $(RUN_ID),$(V7_COLD_HYP_MP_FUNNEL_ANGFILL_RUN))/v7_cold_hyp_mp_funnel_angfill_best.pt"
	@echo "MLflow: http://localhost:5000  experiment tokyo-eyes-v7  run $(or $(RUN_ID),$(V7_COLD_HYP_MP_FUNNEL_ANGFILL_RUN))"
	@echo "Docs: docs/specs/tokyo-eye-v7/cold-hyp-mp-funnel-angfill-prereg.md"

grade-v7-cold-hyp-mp-funnel-vs-sealed: ## Sealed Θ vs cold-funnel best (report-only basin+migration)
	@test -f $(or $(SEALED),$(ARCHAEOLOGY_HYPMP_CKPT)) || (echo "Missing sealed" && exit 1)
	@test -f $(or $(COLD),checkpoints/v7/runs/$(V7_COLD_HYP_MP_FUNNEL_RUN)/v7_cold_hyp_mp_funnel_best.pt) || \
		(echo "Missing cold best — run make train-v7-cold-hyp-mp-funnel" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(V7_COLD_HYP_MP_FUNNEL_RUN)
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.v7_cold_hyp_mp_funnel_vs_sealed \
		--sealed /app/$(or $(SEALED),$(ARCHAEOLOGY_HYPMP_CKPT)) \
		--cold /app/$(or $(COLD),checkpoints/v7/runs/$(V7_COLD_HYP_MP_FUNNEL_RUN)/v7_cold_hyp_mp_funnel_best.pt) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/$(or $(OUT),checkpoints/v7/runs/$(V7_COLD_HYP_MP_FUNNEL_RUN)/sealed_vs_cold_scorecard.json) \
		$(if $(SKIP_MIGRATION),--skip-migration,)
	@echo "Scorecard → $(or $(OUT),checkpoints/v7/runs/$(V7_COLD_HYP_MP_FUNNEL_RUN)/sealed_vs_cold_scorecard.json)"

grade-v7-hyp-biology-mp-vs-sealed: ## Sealed Θ vs biology-best scorecard (basin + migration)
	@test -f $(or $(SEALED),$(ARCHAEOLOGY_HYPMP_CKPT)) || (echo "Missing sealed" && exit 1)
	@test -f $(or $(BIOLOGY),checkpoints/v7/runs/$(V7_HYP_BIOLOGY_MP_RUN)/v7_hyp_biology_mp_best.pt) || \
		(echo "Missing biology best — run make train-v7-hyp-biology-mp" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(V7_HYP_BIOLOGY_MP_RUN)
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.v7_hyp_biology_mp_sealed_vs_biology \
		--sealed /app/$(or $(SEALED),$(ARCHAEOLOGY_HYPMP_CKPT)) \
		--biology /app/$(or $(BIOLOGY),checkpoints/v7/runs/$(V7_HYP_BIOLOGY_MP_RUN)/v7_hyp_biology_mp_best.pt) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/$(or $(OUT),checkpoints/v7/runs/$(V7_HYP_BIOLOGY_MP_RUN)/sealed_vs_biology_scorecard.json) \
		$(if $(SKIP_MIGRATION),--skip-migration,)
	@echo "Scorecard → $(or $(OUT),checkpoints/v7/runs/$(V7_HYP_BIOLOGY_MP_RUN)/sealed_vs_biology_scorecard.json)"

grade-v7-hyp-biology-mp-three-arm: ## Classical vs sealed+biology vs trained Hyp-MP
	@test -f data/gates/tokyo_eye_v7_hyp_biology_mp_three_arm_prereg.json || \
		(echo "Missing three-arm prereg" && exit 1)
	@test -f $(or $(SEALED),$(ARCHAEOLOGY_HYPMP_CKPT)) || (echo "Missing sealed" && exit 1)
	@test -f $(or $(TRAINED),checkpoints/v7/runs/$(V7_HYP_BIOLOGY_MP_RUN)/v7_hyp_biology_mp_best.pt) || \
		(echo "Missing trained best" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(V7_HYP_BIOLOGY_MP_RUN)
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.v7_hyp_biology_mp_three_arm \
		--sealed /app/$(or $(SEALED),$(ARCHAEOLOGY_HYPMP_CKPT)) \
		--trained /app/$(or $(TRAINED),checkpoints/v7/runs/$(V7_HYP_BIOLOGY_MP_RUN)/v7_hyp_biology_mp_best.pt) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/$(or $(OUT),checkpoints/v7/runs/$(V7_HYP_BIOLOGY_MP_RUN)/three_arm_biology_graph_scorecard.json) \
		$(if $(SKIP_MIGRATION),--skip-migration,)
	@echo "Three-arm → $(or $(OUT),checkpoints/v7/runs/$(V7_HYP_BIOLOGY_MP_RUN)/three_arm_biology_graph_scorecard.json)"

# Phase A nucleotide basin continue from sealed healthy
V7_PHASE_A_RUN ?= tokyo_eye_v7_phase_a_nucleotide_basin_v1
train-v7-phase-a-nucleotide-basin: ## Phase A OFF↔ON basin contrastive continue
	@test -f data/gates/tokyo_eye_v7_phase_a_basin_train_prereg.json || \
		(echo "Missing Phase A prereg stamp" && exit 1)
	@test -f $(or $(RESUME),$(ARCHAEOLOGY_HYPMP_CKPT)) || \
		(echo "Missing resume $(or $(RESUME),$(ARCHAEOLOGY_HYPMP_CKPT))" && exit 1)
	@test -f manifests/v7_phase_a_nucleotide_basin_v1.json || (echo "Missing Phase A manifest" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(or $(RUN_ID),$(V7_PHASE_A_RUN)) logs/training
	GNN_INPUT_MODE=topology_three_vector TRAINING_LOAD_FROM_PDB=1 $(SCIENCE_RUN) science python -m experiments.training.v7.phase_a_basin_train \
		--resume /app/$(or $(RESUME),$(ARCHAEOLOGY_HYPMP_CKPT)) \
		--manifest /app/manifests/v7_phase_a_nucleotide_basin_v1.json \
		--pdb-dir /tmp/dtie_pdb_cache \
		--output-dir /app/checkpoints/v7/runs/$(or $(RUN_ID),$(V7_PHASE_A_RUN)) \
		--device $(or $(DEVICE),cuda) \
		--epochs $(or $(EPOCHS),12) \
		--basin-coeff $(or $(BASIN_COEFF),0.10) \
		--basin-margin $(or $(BASIN_MARGIN),0.50) \
		--min-disc-r-mean-hold $(or $(MIN_DISC_R),0.25)
	@echo "Phase A run → checkpoints/v7/runs/$(or $(RUN_ID),$(V7_PHASE_A_RUN))"

grade-v7-phase-a-nucleotide-basin: ## Grade Phase A basin continue (cheap + migration spot-check)
	@test -f data/gates/tokyo_eye_v7_phase_a_basin_train_prereg.json || \
		(echo "Missing Phase A prereg" && exit 1)
	@test -f $(or $(CHECKPOINT),checkpoints/v7/runs/$(V7_PHASE_A_RUN)/v7_phase_a_best.pt) || \
		(echo "Missing Phase A ckpt — run make train-v7-phase-a-nucleotide-basin" && exit 1)
	@mkdir -p checkpoints/v7/diagnostics/phase_a_basin data/gates
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.v7_phase_a_basin_grade \
		--checkpoint /app/$(or $(CHECKPOINT),checkpoints/v7/runs/$(V7_PHASE_A_RUN)/v7_phase_a_best.pt) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v7/diagnostics/phase_a_basin/phase_a_basin_grade.json \
		--closeout /app/data/gates/tokyo_eye_v7_phase_a_basin_train_closeout.json
	@echo "Phase A grade → checkpoints/v7/diagnostics/phase_a_basin/phase_a_basin_grade.json"

train-v7-bprime-uncertainty-heads: ## Heads-only ale/epi recovery from sealed health (prereg)
	@test -f data/gates/tokyo_eye_v7_bprime_uncertainty_heads_prereg.json || \
		(echo "Missing prereg stamp data/gates/tokyo_eye_v7_bprime_uncertainty_heads_prereg.json" && exit 1)
	@$(MAKE) seal-v7-bprime-healthy
	@test -f $(or $(RESUME),$(V7_BPRIME_UNC_RESUME)) || \
		(echo "Missing sealed resume $(or $(RESUME),$(V7_BPRIME_UNC_RESUME))" && exit 1)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_UNC_RUN)) pdb_cache
	@printf '%s\n' \
		'# Tokyo Eye v7 B′ uncertainty heads — sealed health, trunk frozen' \
		'resume: $(or $(RESUME),$(V7_BPRIME_UNC_RESUME))' \
		'phase: v7_bprime_uncertainty_heads (epistemic_uncertainty_only_train)' \
		'disc_hold: min_disc_r_mean_hold=0.25' \
		'spec: docs/specs/tokyo-eye-v7/bprime-uncertainty-heads-prereg.md' \
		> checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_UNC_RUN))/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v7.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_UNC_RUN)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--epochs $(or $(EPOCHS),24) \
		--gnn-lineage v7 \
		--mlflow-experiment tokyo-eyes-v7 \
		--resume /app/$(or $(RESUME),$(V7_BPRIME_UNC_RESUME)) \
		$(V7_BPRIME_STACK) \
		--v7-bprime-uncertainty-heads \
		--p4-epistemic-lr $(or $(UNC_LR),5e-5) \
		--min-disc-r-mean-hold $(or $(DISC_HOLD),0.25) \
		--sparsity-style-save \
		--routing-entropy-mean-residue-min-save $(or $(MEAN_H_MIN),$(V7_BPRIME_MEAN_H_MIN)) \
		--routing-entropy-mean-residue-max-save $(or $(MEAN_H_MAX),$(V7_BPRIME_MEAN_H_MAX)) \
		$(if $(NO_MLFLOW),--no-mlflow,)
	@echo "Uncertainty heads → checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_UNC_RUN))"

# Pre-authorized rematch-1 after rematch-0 flat ale/epi + disc hold.
V7_BPRIME_UNC_REMATCH_RESUME ?= checkpoints/v7/runs/tokyo_eye_v7_bprime_uncertainty_heads_v1/v7_phase4_12prot.pt
V7_BPRIME_UNC_REMATCH_RUN ?= tokyo_eye_v7_bprime_uncertainty_heads_rematch_v1

train-v7-bprime-uncertainty-heads-rematch: ## Rematch-1: higher anticollapse/decorrelation (heads-only)
	@test -f data/gates/tokyo_eye_v7_bprime_uncertainty_heads_prereg.json || \
		(echo "Missing prereg stamp" && exit 1)
	@test -f data/gates/tokyo_eye_v7_bprime_uncertainty_heads_closeout.json || \
		(echo "Grade rematch-0 first: make grade-v7-bprime-uncertainty-heads" && exit 1)
	@test -f $(or $(RESUME),$(V7_BPRIME_UNC_REMATCH_RESUME)) || \
		(echo "Missing rematch resume $(or $(RESUME),$(V7_BPRIME_UNC_REMATCH_RESUME))" && exit 1)
	@mkdir -p checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_UNC_REMATCH_RUN)) pdb_cache
	@printf '%s\n' \
		'# Tokyo Eye v7 B′ uncertainty heads rematch-1 — higher anticollapse/decorr' \
		'resume: $(or $(RESUME),$(V7_BPRIME_UNC_REMATCH_RESUME))' \
		'prior: tokyo_eye_v7_bprime_uncertainty_heads_v1 (rematch-0 FAIL flat ale/epi, disc held)' \
		'spec: docs/specs/tokyo-eye-v7/bprime-uncertainty-heads-prereg.md' \
		> checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_UNC_REMATCH_RUN))/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v7.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_UNC_REMATCH_RUN)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--epochs $(or $(EPOCHS),24) \
		--gnn-lineage v7 \
		--mlflow-experiment tokyo-eyes-v7 \
		--resume /app/$(or $(RESUME),$(V7_BPRIME_UNC_REMATCH_RESUME)) \
		$(V7_BPRIME_STACK) \
		--v7-bprime-uncertainty-heads-rematch \
		--p4-epistemic-lr $(or $(UNC_LR),5e-5) \
		--min-disc-r-mean-hold $(or $(DISC_HOLD),0.25) \
		--sparsity-style-save \
		--routing-entropy-mean-residue-min-save $(or $(MEAN_H_MIN),$(V7_BPRIME_MEAN_H_MIN)) \
		--routing-entropy-mean-residue-max-save $(or $(MEAN_H_MAX),$(V7_BPRIME_MEAN_H_MAX)) \
		$(if $(NO_MLFLOW),--no-mlflow,)
	@echo "Uncertainty rematch → checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_UNC_REMATCH_RUN))"

grade-v7-bprime-uncertainty-heads: ## Grade rematch-0/1 ale/epi + disc hold closeout
	@test -f $(or $(METRICS),checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_UNC_RUN))/metrics.json) || \
		(echo "Missing metrics.json" && exit 1)
	$(SCIENCE_RUN) science python -m experiments.training.v7.grade_uncertainty_heads \
		--metrics /app/$(or $(METRICS),checkpoints/v7/runs/$(or $(RUN_ID),$(V7_BPRIME_UNC_RUN))/metrics.json) \
		--out /app/$(or $(OUT),data/gates/tokyo_eye_v7_bprime_uncertainty_heads_closeout.json)
	@echo "Closeout → $(or $(OUT),data/gates/tokyo_eye_v7_bprime_uncertainty_heads_closeout.json)"

# 4-expert overlay charts (MLflow UI cannot put expert_0..3 on one axes).
plot-v7-expert-overlay: ## HTML overlay of expert_* metrics from a run metrics.json
	@test -f $(or $(METRICS),checkpoints/v7/runs/tokyo_eye_v7_bprime_health_continue_v1/metrics.json) || \
		(echo "Missing METRICS=…/metrics.json" && exit 1)
	@mkdir -p checkpoints/v7/diagnostics
	$(SCIENCE_RUN) science python -m experiments.diagnostics.plot_expert_metrics_overlay \
		--metrics /app/$(or $(METRICS),checkpoints/v7/runs/tokyo_eye_v7_bprime_health_continue_v1/metrics.json) \
		--out /app/$(or $(OUT),checkpoints/v7/diagnostics/expert_overlay.html) \
		--title "$(or $(TITLE),Per-expert metric overlay)"
	@echo "Open $(or $(OUT),checkpoints/v7/diagnostics/expert_overlay.html) in a browser"

# Cold / continue train under checkpoints/v7 + tokyo-eyes-v7.
train-v7: ## Tokyo Eye v7 train (gnn_lineage=v7; Hyp MP primary)
	@mkdir -p checkpoints/v7/runs/$(or $(RUN_ID),tokyo_eye_v7_cold_v1) pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v7.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v7/runs/$(or $(RUN_ID),tokyo_eye_v7_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),2) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--epochs $(or $(EPOCHS),5) \
		--gnn-lineage v7 \
		--mlflow-experiment tokyo-eyes-v7 \
		$(if $(RESUME),--resume /app/$(RESUME),) \
		$(if $(NO_MLFLOW),--no-mlflow,)
	@echo "Train complete → checkpoints/v7/runs/$(or $(RUN_ID),tokyo_eye_v7_cold_v1)"

export-corpus-viewers-v7: ## Export HTML viewers from a v7 checkpoint
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v7.export_corpus_viewers \
		--checkpoint /app/$(or $(CHECKPOINT),checkpoints/v7/tokyo_eye_v7_cutover_scaffold.pt) \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v7/viewers/$(or $(RUN_ID),cutover_scaffold) \
		--device $(or $(DEVICE),cpu)

promote-production-v7: ## Promote a v7 .pt into contract as TokyoEye production
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=checkpoints/v7/runs/.../v7_best.pt" && exit 1)
	$(SCIENCE_RUN) science python -m science.training.promote \
		--checkpoint-path /app/$(CHECKPOINT) \
		--checkpoint-id $(or $(CHECKPOINT_ID),tokyo_eye_v7_champion) \
		--model-id tokyo_eye_v7 \
		--status production
	@echo "Promoted → onboard_contract tokyo_eye_v7 production"

verify-v7-production: ## Health-style verify of production TokyoEye checkpoint
	$(SCIENCE_RUN) science python -c "from science.tokyo_eye.TokyoEye import verify_tokyo_eye_checkpoint; import json; print(json.dumps(verify_tokyo_eye_checkpoint(), indent=2))"

# ON-arm neighborhood conduit graft (Child 3): 6GOD ← 6GOF (N12 recomputed).
# Spec: docs/specs/kras-topo-structural-inference/kras-g12-neighborhood-graft-on-prereg.md
grade-v66-kras-g12-neighborhood-graft-on: ## KRAS G12 neighborhood conduit graft ON (6GOD←6GOF)
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@test -f data/gates/kras_g12_neighborhood_graft_on_prereg.json || \
		(echo "Missing pre-reg stamp data/gates/kras_g12_neighborhood_graft_on_prereg.json" && exit 1)
	@test -f data/gates/kras_g12_neighborhood_graft_closeout.json || \
		(echo "Missing OFF neighborhood closeout" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.kras_g12_neighborhood_graft_validation \
		--arm on \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/kras_g12_neighborhood_graft_on.json
	@echo "ON neighborhood graft → checkpoints/v66/diagnostics/routing_sparsity/kras_g12_neighborhood_graft_on.json"
	@echo "Pass: Δρ_neigh > Δρ_single AND ρ(neigh,mut) > ρ(scramble,mut)"
	@echo "Docs: docs/specs/kras-topo-structural-inference/kras-g12-neighborhood-graft-on-prereg.md"

# Platform smoke: generic flow↔betweenness concordance on non-KRAS TRAINING_TARGETS.
# No residue-ID gates. Default panel: 3PP0 (kinase) / 2SHP (phosphatase) / 2HHB (blind fold).
grade-v66-fix1-general-hub-alignment: ## Generic hub concordance smoke (non-KRAS TRAINING_TARGETS)
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.topo_structural_engine \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--pass-rho $(or $(PASS_RHO),0.50) \
		--k-frac $(or $(K_FRAC),0.10) \
		--targets $(or $(TARGETS),3PP0:A 2SHP:A 2HHB:B) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/general_hub_alignment_smoke.json \
		$(if $(WITH_KRAS_AUDIT),--with-kras-audit,)
	@echo "General hub alignment → checkpoints/v66/diagnostics/routing_sparsity/general_hub_alignment_smoke.json"
	@echo "Pass: ρ(out_effect, CB) > 0.50 + cutoff stability on each panel structure"
	@echo "Docs: docs/specs/kras-topo-structural-inference/platform-concordance.md"

# Ledger B: pre-registered SRC/SHP2 interface recall@top-10% knockout flow.
# Residue sets: data/gates/ledger_b_interface_prereg_src_shp2.json ONLY.
grade-v66-fix1-ledger-b-interface-alignment: ## Ledger B interface alignment (SRC/SHP2 pre-reg)
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@test -f data/gates/ledger_b_interface_prereg_src_shp2.json || \
		(echo "Missing Ledger B pre-reg stamp" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.ledger_b_interface_alignment \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--prereg /app/data/gates/ledger_b_interface_prereg_src_shp2.json \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/ledger_b_interface_alignment.json
	@echo "Ledger B → checkpoints/v66/diagnostics/routing_sparsity/ledger_b_interface_alignment.json"
	@echo "Pass: recall@top-10% flow on pre-reg I ≥ 0.25 for BOTH 3PP0 and 2SHP"
	@echo "Docs: docs/specs/kras-topo-structural-inference/ledger-b-interface-prereg.md"

# Phase 4b: full hub lists + literature map (interpretative; no threshold change)
grade-v66-fix1-ledger-b-phase4b-hub-map: ## Phase 4b hub inventory + literature map
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@test -f data/gates/ledger_b_interface_prereg_src_shp2.json || \
		(echo "Missing Ledger B pre-reg stamp" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.ledger_b_phase4b_hub_map \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--prereg /app/data/gates/ledger_b_interface_prereg_src_shp2.json \
		--grade-artifact /app/checkpoints/v66/diagnostics/routing_sparsity/ledger_b_interface_alignment.json \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/ledger_b_phase4b_hub_map.json \
		--markdown /app/checkpoints/v66/diagnostics/routing_sparsity/ledger_b_phase4b_hub_literature_map.md
	@cp -f checkpoints/v66/diagnostics/routing_sparsity/ledger_b_phase4b_hub_literature_map.md \
		docs/specs/kras-topo-structural-inference/ledger-b-phase4b-hub-literature-map.md
	@echo "Phase 4b → checkpoints/v66/diagnostics/routing_sparsity/ledger_b_phase4b_hub_map.json"
	@echo "Literature map → docs/specs/kras-topo-structural-inference/ledger-b-phase4b-hub-literature-map.md"

# Phase 4b′: 2SHP focal-hub sensitivity (jitter ρ>0.85, sparsity trajectory, wrapping)
grade-v66-fix1-ledger-b-sensitivity-check: ## 2SHP R32/I310/N308/V457 sensitivity
	@test -f $(FIX1_SPARSITY_CHAMPION_CKPT) || \
		(echo "Missing champion $(FIX1_SPARSITY_CHAMPION_CKPT)" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/routing_sparsity logs/training
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.diagnostics.ledger_b_sensitivity_check \
		--checkpoint /app/$(FIX1_SPARSITY_CHAMPION_CKPT) \
		--run-dir /app/checkpoints/v66/runs/$(FIX1_SPARSITY_CHAMPION_RUN) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--jitter-seeds $(or $(JITTER_SEEDS),3) \
		--epochs $(or $(EPOCHS),$(LEDGER_B_SENSITIVITY_EPOCHS)) \
		--output /app/checkpoints/v66/diagnostics/routing_sparsity/ledger_b_sensitivity_check.json
	@echo "Sensitivity → checkpoints/v66/diagnostics/routing_sparsity/ledger_b_sensitivity_check.json"
	@echo "Bars: jitter Spearman > 0.85; focal hubs stay top-10% across sparsity epochs"

train-v66-fix1-s4-proto-repulsion-scale-l2-gram-cond-stage-a12: ## Pre-reg: stack + saturating Gram logdet hinge λ=0.001 (Stage A-12 cold)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing Stage A-12 manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_gram_cond_stage_a12_cold_v1)
	@printf '%s\n' \
		'# Full-bank Gram logdet hinge — frozen 2026-07-15 (before train)' \
		'baseline: Fix-1 + S4 + SASA + proto repulsion + softplus floor ≈6.612' \
		'lever: saturating ReLU(τ − logdet(G+εI))² on unit-normalized prototype tangents' \
		'prototype_gram_logdet_coeff: $(or $(GRAM_LOGDET_COEFF),0.001)' \
		'prototype_gram_logdet_tau: $(or $(GRAM_LOGDET_TAU),-1.15)' \
		'keep: nearest-pair repulsion coeff=1.0 margin=0.25' \
		'monopole: OUT OF SCOPE (log only)' \
		'coeff_proxy: checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/gram_logdet_hinge_coeff_proxy.json' \
		'NO_MOVE → rematch 0.005; COMMIT_KILLED → rematch 0.0002' \
		'success: GRAM_COND_WIN on BOTH seeds (bank + relative purity + commit + axis)' \
		'logs: prototype_repulsion_per_epoch.jsonl (gram_conditional + gram_eig_* + gram_logdet)' \
		> checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_gram_cond_stage_a12_cold_v1)/README.md
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_gram_cond_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--hyperbolic-mp-graph \
		--input-feature-zscore \
		--gate-include-sasa \
		--prototype-repulsion-coeff $(or $(PROTO_REPULSION_COEFF),1.0) \
		--prototype-repulsion-margin $(or $(PROTO_REPULSION_MARGIN),0.25) \
		--prototype-gram-logdet-coeff $(or $(GRAM_LOGDET_COEFF),0.001) \
		--prototype-gram-logdet-tau $(or $(GRAM_LOGDET_TAU),-1.15) \
		--gate-logit-softplus-init $(or $(GATE_SOFTPLUS),6.612216472625732) \
		--gate-logit-softplus-floor $(or $(GATE_SOFTPLUS_FLOOR),6.612216472625732) \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--epochs $(or $(EPOCHS),30) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(RESUME),--resume /app/$(RESUME),)
	@echo "Gram cond complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_stack_gram_cond_stage_a12_cold_v1)"
	@echo "Score GRAM_COND_WIN / GRAM_COND_NO_MOVE / AXIS_SCRAMBLED_BY_GRAM / COMMIT_KILLED_BY_GRAM"

train-v66-fix1-s4-proto-repulsion-scale-l2-gram-cond-stage-a12-seed2: ## Gram cond seed-2 cold (same λ; both seeds required)
	$(MAKE) train-v66-fix1-s4-proto-repulsion-scale-l2-gram-cond-stage-a12 \
		RUN_ID=$(or $(RUN_ID),fix1_s4_stack_gram_cond_stage_a12_cold_seed2_v1) \
		SEED=$(or $(SEED),2) \
		EPOCHS=$(or $(EPOCHS),30)

train-v66-fix1-s4-scale-l1-stage-a12: ## L1: Fix-1+S4 + gate softplus≈2.65 (×2); Stage A-12 cold
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_small_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing Stage A-12 manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_scale_l1_stage_a12_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-geom-angular-prior \
		--hyperbolic-mp-graph \
		--input-feature-zscore \
		--gate-logit-softplus-init $(or $(GATE_SOFTPLUS),2.6451666355133057) \
		--gate-logit-softplus-floor $(or $(GATE_SOFTPLUS_FLOOR),2.6451666355133057) \
		--geometric-angular-kappa $(or $(GEOM_KAPPA),1.0) \
		--geometric-angular-alpha $(or $(GEOM_ALPHA),0.7853981633974483) \
		--epoch-anchor-pdb-ids $(or $(EPOCH_ANCHORS),1F88,4OBE) \
		--epochs $(or $(EPOCHS),20) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.14) \
		--rim-fanout-min-r $(or $(RIM_FANOUT_MIN_R),0.20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots \
		$(if $(SEED),--seed $(SEED),)
	@echo "L1 scale train complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),fix1_s4_scale_l1_stage_a12_cold_v1)"
	@echo "Pre-registered: SCALE_TRAIN_FIX / SHARPENED_NOISE / IBU / AMBIGUOUS — GNNV7_SUCCESS_CRITERIA.md"
	@echo "Logs: fix1_gates_per_epoch.jsonl + scale_train_structure_per_epoch.jsonl"

audit-hyperbolic-gate-logit-spread: ## Pre-softmax hyp gate logit spread (IBU commitment / saturation)
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=..." && exit 1)
	$(SCIENCE_RUN) science python -m experiments.diagnostics.hyperbolic_gate_logit_spread \
		--checkpoint /app/$(CHECKPOINT) \
		--corpus /app/$(or $(CORPUS),manifests/v6_corpus_stage_a_small_v1.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output-dir /app/$(or $(OUT),checkpoints/v66/diagnostics/hyperbolic_gate_logit_spread)

audit-trunk-hidden-occupancy: ## T1 trunk: HD encoder/pre-gate rank (not 2D disc)
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=..." && exit 1)
	$(SCIENCE_RUN) science python -m experiments.diagnostics.trunk_hidden_occupancy \
		--checkpoint /app/$(CHECKPOINT) \
		--corpus /app/$(or $(CORPUS),manifests/v6_corpus_stage_a_small_v1.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output-dir /app/$(or $(OUT),checkpoints/v66/diagnostics/trunk_hidden_occupancy)

audit-t1-trunk-origin: ## T1a/T1b: raw x vs embed; init vs early vs late rank
	@test -n "$(CHECKPOINT_LATE)" || (echo "Set CHECKPOINT_LATE=... [CHECKPOINT_EARLY=...]" && exit 1)
	$(SCIENCE_RUN) science python -m experiments.diagnostics.t1_trunk_origin_audit \
		--checkpoint-late /app/$(CHECKPOINT_LATE) \
		$(if $(CHECKPOINT_EARLY),--checkpoint-early /app/$(CHECKPOINT_EARLY),) \
		--corpus /app/$(or $(CORPUS),manifests/v6_corpus_stage_a_small_v1.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output-dir /app/$(or $(OUT),checkpoints/v66/diagnostics/t1_trunk_origin)

audit-t1a-znorm-forward: ## T1a fwd-only z-norm: pre_mp vs encoder_h ceiling check
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=..." && exit 1)
	$(SCIENCE_RUN) science python -m experiments.diagnostics.t1a_znorm_forward_probe \
		--checkpoint /app/$(CHECKPOINT) \
		--corpus /app/$(or $(CORPUS),manifests/v6_corpus_stage_a_small_v1.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--output-dir /app/$(or $(OUT),checkpoints/v66/diagnostics/t1a_znorm_forward_probe)

train-v66-feeler-rim-fanout-cold: ## v6.6 cold: P1+P2 angular fill → rim fan-out P12 (37 ep default; EPOCHS=P12 only)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_rim_fanout_cold_23_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-fanout-model \
		--v66-feeler-rim-fanout-cold-curriculum \
		--epochs $(or $(EPOCHS),10) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--spoke-edge-scale $(or $(SPOKE_SCALE),1.5) \
		--ribbon-edge-scale $(or $(RIBBON_SCALE),1.35) \
		--rim-fanout-strength $(or $(RIM_FANOUT_STRENGTH),0.12) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler rim fan-out COLD complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_rim_fanout_cold_23_v1)"
	@echo "Gate: 4OBE σ₂/σ₁ + rim_frac + viewer vs feeler_expand_23_v1 (P12-only cold → rank-1 filament)"

train-v66-feeler-angular-lift: ## v6.6 feeler: angular_lift + lift-path recovery off P3 (15 ep, no disc occupancy)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_P3_RESUME)) || \
		(echo "Missing P3 resume — set RESUME=... or finish feeler_expand_23_p3_geom_v1" && exit 1)
	@test -d $(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes-feeler-expand" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_angular_lift_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-angular-lift \
		--resume /app/$(or $(RESUME),$(V66_FEELER_P3_RESUME)) \
		--epochs $(or $(EPOCHS),15) \
		--p2-bridge-lr $(or $(LR),1.25e-4) \
		--dehydron-edge-barcode \
		--dehydron-barcode-dir /app/$(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler angular_lift complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_angular_lift_v1)"

train-v66-feeler-coupling: ## v6.6 feeler: cross-subgraph coupling edges off P3 (12 ep, no disc occupancy)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_P3_RESUME)) || \
		(echo "Missing P3 resume — set RESUME=... or finish feeler_expand_23_p3_geom_v1" && exit 1)
	@test -d $(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes-feeler-expand" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_coupling_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-coupling \
		--resume /app/$(or $(RESUME),$(V66_FEELER_P3_RESUME)) \
		--epochs $(or $(EPOCHS),12) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--dehydron-edge-barcode \
		--dehydron-barcode-dir /app/$(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler coupling complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_coupling_v1)"
	@echo "Compare 4OBE/1F88 viewers: spike should pull body cloud vs P3 parent"

train-v66-feeler-no-exclusivity: ## v6.6 feeler Exp1: dehydron pairs keep packing/ribbon/spoke (12 ep, P3 parent)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_P3_RESUME)) || \
		(echo "Missing P3 resume — set RESUME=... or finish feeler_expand_23_p3_geom_v1" && exit 1)
	@test -d $(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes-feeler-expand" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_no_excl_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-no-exclusivity \
		--no-dehydron-exclusivity \
		--resume /app/$(or $(RESUME),$(V66_FEELER_P3_RESUME)) \
		--epochs $(or $(EPOCHS),12) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--dehydron-edge-barcode \
		--dehydron-barcode-dir /app/$(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler no-exclusivity complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_no_excl_v1)"
	@echo "Gate: make audit-rim-dehydron-angular CHECKPOINT=... STRUCTURES='4OBE 1R69' LEARNED=1"

train-v66-feeler-dehydron-angular: ## v6.6 feeler Exp2: dehydron angular scale 0.3 off P3 (12 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_P3_RESUME)) || \
		(echo "Missing P3 resume — set RESUME=... or finish feeler_expand_23_p3_geom_v1" && exit 1)
	@test -d $(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes-feeler-expand" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_dbh_ang_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-dehydron-angular \
		--dehydron-angular-scale $(or $(DEHYDRON_ANGULAR_SCALE),0.3) \
		--resume /app/$(or $(RESUME),$(V66_FEELER_P3_RESUME)) \
		--epochs $(or $(EPOCHS),12) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--dehydron-edge-barcode \
		--dehydron-barcode-dir /app/$(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler dehydron-angular complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_dbh_ang_v1)"

train-v66-feeler-rim-decouple: ## v6.6 feeler Exp1+2: no exclusivity + dehydron angular 0.3 off P3 (12 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_P3_RESUME)) || \
		(echo "Missing P3 resume — set RESUME=... or finish feeler_expand_23_p3_geom_v1" && exit 1)
	@test -d $(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes-feeler-expand" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_decouple_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-decouple \
		--dehydron-angular-scale $(or $(DEHYDRON_ANGULAR_SCALE),0.3) \
		--resume /app/$(or $(RESUME),$(V66_FEELER_P3_RESUME)) \
		--epochs $(or $(EPOCHS),12) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--dehydron-edge-barcode \
		--dehydron-barcode-dir /app/$(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler rim-decouple complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_decouple_v1)"
	@echo "Gate: make audit-rim-dehydron-angular CHECKPOINT=checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_decouple_v1)/v66_phase9_23prot.pt STRUCTURES='4OBE 1R69' LEARNED=1"

train-v66-feeler-rim-detach: ## v6.6 feeler: full dehydron θ-detach (scale=0) + no exclusivity, 200 ep off P3
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(V66_FEELER_P3_RESUME)) || \
		(echo "Missing P3 resume — set RESUME=... (do not resume failed no-excl/ang/coupling runs)" && exit 1)
	@test -d $(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes-feeler-expand" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_detach_200ep_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--v66-feeler-rim-decouple \
		--dehydron-angular-scale $(or $(DEHYDRON_ANGULAR_SCALE),0.0) \
		--resume /app/$(or $(RESUME),$(V66_FEELER_P3_RESUME)) \
		--epochs $(or $(EPOCHS),200) \
		--p2-bridge-lr $(or $(LR),1.0e-4) \
		--dehydron-edge-barcode \
		--dehydron-barcode-dir /app/$(or $(DBH_DIR),$(V66_DBH_BARCODE_DIR)) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler rim-detach complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_rim_detach_200ep_v1)"

audit-rim-dehydron-angular: ## Exp3: within-rim dehydron angular spread (SSOT vs learned)
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=path/to/model.pt" && exit 1)
	@mkdir -p checkpoints/v66/diagnostics/rim_dehydron_angular
	$(SCIENCE_RUN) science python -m experiments.diagnostics.rim_dehydron_angular_audit \
		--checkpoint /app/$(CHECKPOINT) \
		--structures $(or $(STRUCTURES),4OBE 1R69) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cpu) \
		--output-dir /app/checkpoints/v66/diagnostics/rim_dehydron_angular \
		$(if $(LEARNED),--learned-disc,--structural-ssot)
	@echo "Compare 4OBE/1F88 viewers vs P3 parent — wedge should open without disc occupancy pressure"

train-v66-feeler-p2: ## v6.6 feeler P2: resume P1, unfreeze angular, soft routing (20 ep, timeout@45%)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v66/runs/feeler_p1_v1/v66_best.pt) || \
		(echo "Missing P1 resume checkpoint — run: make train-v66-feeler" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_p2_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--phase 2 \
		--resume /app/$(or $(RESUME),checkpoints/v66/runs/feeler_p1_v1/v66_best.pt) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler P2 complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_p2_v1)"
	@echo "Compare viewers vs P1; watch routing_H / max expert share / disc_r_std"

train-v66-feeler-expand: ## v6.6 feeler expand: resume P2 → Stage A 23-prot (ex-3CON/4GQB), same light recipe (50 ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01 CORPUS=v6_corpus_stage_a_feeler_expand_v1.json" && exit 1)
	@test -f manifests/v6_corpus_stage_a_feeler_expand_v1.json || (echo "Missing feeler expand manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v66/runs/feeler_p2_v4_50ep/v66_phase2_12prot.pt) || \
		(echo "Missing expand resume checkpoint — set RESUME=... or finish feeler_p2_v4_50ep" && exit 1)
	@mkdir -p mlruns checkpoints/v66/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v66.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_feeler_expand_v1.json \
		--output-dir /app/checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),23) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--epochs $(or $(EPOCHS),50) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v66 \
		--no-warm-start \
		--v66-feeler-lineage \
		--phase 2 \
		--resume /app/$(or $(RESUME),checkpoints/v66/runs/feeler_p2_v4_50ep/v66_phase2_12prot.pt) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "v6.6 feeler expand complete. Run: checkpoints/v66/runs/$(or $(RUN_ID),feeler_expand_23_v1)"
	@echo "Watch σ2/σ1, e1/e3 geometry split, routing_H drift, e0 core niche — no new losses"

train-v65-slim-cold-start: ## LEGACY: slim MoE + frozen structural disc (not representation learning)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v65/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v65.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v65/runs/$(or $(RUN_ID),cold_start_v8) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v65 \
		--no-warm-start \
		--slim-moe-structural-ssot \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--phase $(or $(STAGE),1) \
		--save-epoch-snapshots
	@echo "LEGACY slim MoE cold-start complete. Run: checkpoints/v65/runs/$(or $(RUN_ID),cold_start_v8)"
	@echo "Prefer: make train-v65-master-cold — slim freezes node_emb/convs and bypasses learned geometry"

train-v65-slim-p2: ## Continue slim MoE P2 (timeout@50%/1ep, no floor/ceiling; RESUME=... EPOCHS=...)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v65/runs/cold_start_v8_p2/v65_best.pt) || \
		(echo "Missing resume checkpoint — set RESUME=..." && exit 1)
	@mkdir -p mlruns checkpoints/v65/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v65.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v65/runs/$(or $(RUN_ID),cold_start_v8_p2b) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v65 \
		--no-warm-start \
		--slim-moe-structural-ssot \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--phase 2 \
		--epochs $(or $(EPOCHS),30) \
		--resume /app/$(or $(RESUME),checkpoints/v65/runs/cold_start_v8_p2/v65_best.pt) \
		--save-epoch-snapshots
	@echo "Slim MoE P2 complete. Run: checkpoints/v65/runs/$(or $(RUN_ID),cold_start_v8_p2b)"

train-v65-slim-p3: ## P3 routing consolidate: freeze e2 + e2-only timeout@45%/1ep
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v65/runs/cold_start_v8_p3/v65_phase3_12prot.pt) || \
		(echo "Missing resume checkpoint — set RESUME=..." && exit 1)
	@mkdir -p mlruns checkpoints/v65/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v65.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v65/runs/$(or $(RUN_ID),cold_start_v8_p3b) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v65 \
		--no-warm-start \
		--slim-moe-structural-ssot \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--phase 3 \
		--epochs $(or $(EPOCHS),15) \
		--resume /app/$(or $(RESUME),checkpoints/v65/runs/cold_start_v8_p3/v65_phase3_12prot.pt) \
		--save-epoch-snapshots
	@echo "Slim MoE P3 complete. Run: checkpoints/v65/runs/$(or $(RUN_ID),cold_start_v8_p3b)"
	@echo "P3: freeze e2, e2-only timeout@45%/1ep, eval_min≥0.05 eval_max≤0.55 H≤1.30"

validate-learned-ssot-gate: ## Print SSOT policy for CHECKPOINT=... (pre-production flip gate)
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=path/to/v65_best.pt" && exit 1)
	PYTHONPATH=. python -m experiments.diagnostics.validate_learned_ssot_gate \
		--checkpoint $(CHECKPOINT) \
		$(if $(OUT),--out $(OUT),)

train-v65-dbh-scalars-cold: ## [ARCHIVE] v65 cold scalars — wrong lineage for feeler investigation
	@echo "WARNING: train-v65-dbh-scalars-cold is archive; active path is make train-v66-feeler-p3-geom-edges"
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -d $(DBH_BARCODE_DIR) || \
		(echo "Missing barcode sidecars — run: make precompute-dehydron-barcodes" && exit 1)
	@mkdir -p mlruns checkpoints/v65/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v65.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v65/runs/$(or $(RUN_ID),dbh_scalars_master_cold_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--seed $(or $(SEED),1) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--mlflow-experiment tokyo-eyes-v65 \
		--no-warm-start \
		--master-cold-lineage \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--use-dehydron-barcode \
		--dehydron-barcode-dir /app/$(DBH_BARCODE_DIR) \
		--feature-liveness-probe \
		--save-epoch-snapshots
	@echo "DBH scalars master-cold complete. Run: checkpoints/v65/runs/$(or $(RUN_ID),dbh_scalars_master_cold_v1)"
	@echo "Liveness probe must stay green before claiming barcode inference influence."

train-v6-slim-moe-routing-recovery: ## MoE routing recovery off slim SSOT checkpoint (structural disc frozen)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v6/runs/slim_moe_structural_ssot_cold_v1/v6_phase2_12prot.pt) || \
		(echo "Missing resume checkpoint — set RESUME=..." && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),slim_moe_route_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/slim_moe_structural_ssot_cold_v1/v6_phase2_12prot.pt) \
		--topology-routing-recovery \
		--structural-disc-frozen \
		--topology-routing-recovery-lr $(or $(LR),3e-5) \
		--epochs $(or $(EPOCHS),40) \
		--p2-bridge-ramp-epochs $(or $(P2_BRIDGE_RAMP_EPOCHS),$(EPOCHS),40) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--save-epoch-snapshots
	@echo "Slim MoE routing recovery complete. Viewers: checkpoints/v6/runs/$(or $(RUN_ID),slim_moe_route_v1)/viewers/"

train-v6-topology-routing-recovery: ## P2 MoE routing: ep169, 4 experts, structure gate, depth decouple
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v6/runs/master_cold_topology_v1/epochs/epoch_169.pt) || \
		(echo "Missing resume checkpoint — set RESUME=..." && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),master_cold_topology_route_v3) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/master_cold_topology_v1/epochs/epoch_169.pt) \
		--topology-routing-recovery \
		--topology-routing-recovery-lr $(or $(LR),3e-5) \
		--epochs $(or $(EPOCHS),40) \
		--p2-bridge-ramp-epochs $(or $(P2_BRIDGE_RAMP_EPOCHS),$(EPOCHS),40) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--save-epoch-snapshots
	@echo "Topology routing recovery complete. Run: checkpoints/v6/runs/$(or $(RUN_ID),master_cold_topology_route_v3)"

train-v6-topology-gate-disc-recovery: ## Gate-only + light disc recovery off route_v3 (freeze depth bias, 20ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v6/runs/master_cold_topology_route_v3/epochs/epoch_209.pt) || \
		(echo "Missing resume checkpoint — set RESUME=..." && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),master_cold_topology_gate_disc_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/master_cold_topology_route_v3/epochs/epoch_209.pt) \
		--topology-gate-disc-recovery \
		--topology-gate-disc-recovery-lr $(or $(LR),2e-5) \
		--epochs $(or $(EPOCHS),20) \
		--p2-bridge-ramp-epochs $(or $(P2_BRIDGE_RAMP_EPOCHS),$(EPOCHS),20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--save-epoch-snapshots
	@echo "Topology gate+disc recovery complete. Run: checkpoints/v6/runs/$(or $(RUN_ID),master_cold_topology_gate_disc_v1)"

train-v6-topology-crescent-recovery: ## Open 1D crescent: angular+disc wedge off route_v3 (25ep)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v6/runs/master_cold_topology_route_v3/epochs/epoch_209.pt) || \
		(echo "Missing resume checkpoint — set RESUME=..." && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),master_cold_topology_crescent_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/master_cold_topology_route_v3/epochs/epoch_209.pt) \
		--topology-crescent-recovery \
		--topology-crescent-recovery-lr $(or $(LR),2e-5) \
		--epochs $(or $(EPOCHS),25) \
		--p2-bridge-ramp-epochs $(or $(P2_BRIDGE_RAMP_EPOCHS),$(EPOCHS),25) \
		--p2-disc-line-thickness-floor $(or $(DISC_THICKNESS_FLOOR),0.03) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--save-epoch-snapshots
	@echo "Topology crescent recovery complete. Run: checkpoints/v6/runs/$(or $(RUN_ID),master_cold_topology_crescent_v1)"

train-v6-stage-a-small-master-cold-smoke: ## 1-epoch MASTER cold-start smoke — all 12 structures (P_MASTER_COLD_SMOKE)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f manifests/v6_corpus_stage_a_small_v1.json || (echo "Missing small Stage A manifest" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),stage_a_small_master_cold_smoke_12_$(shell date +%Y%m%d_%H%M%S)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cpu) \
		--phase 1 \
		--epochs 1 \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--no-warm-start \
		--master-cold-lineage \
		--no-corpus-cache \
		--save-epoch-snapshots
	@echo "Smoke complete. Verify with: MASTER_COLD_SMOKE_RUN_ID=<run_id> make test-stage-a-small-master-cold-smoke"

test-stage-a-small-master-cold-smoke: ## Assert P_MASTER_COLD_SMOKE on MASTER_COLD_SMOKE_RUN_ID MLflow run
	@test -n "$$MASTER_COLD_SMOKE_RUN_ID" || (echo "Set MASTER_COLD_SMOKE_RUN_ID to the smoke run id" && exit 1)
	MLFLOW_TRACKING_URI=$(or $(MLFLOW_TRACKING_URI),file:./mlruns) \
	MLFLOW_ALLOW_FILE_STORE=true \
	python3 -m pytest tests/test_stage_a_small_master_cold_smoke.py::test_master_cold_smoke_mlflow_run_from_env -v

train-v6-residue-stage1: ## ResidueStage1 BCE on 12-protein corpus (warm-start stage_a_small_v1)
	@test -f manifests/v6_corpus_residue_stage1.json || (echo "Missing ResidueStage1 manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v6/runs/stage_a_small_v1/v6_best_disc.pt) || \
		(echo "Missing stage_a_small_v1 checkpoint for warm-start" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_residue_stage1.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),residue_stage1_$(shell date +%Y%m%d_%H%M%S)) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/stage_a_small_v1/v6_best.pt) \
		--residue-stage1 \
		--residue-stage1-lr $(or $(RESIDUE_STAGE1_LR),1e-4) \
		--residue-stage1-epochs $(or $(EPOCHS),30) \
		--no-corpus-cache
	@echo "ResidueStage1 complete. Compare pocket_bce / interface_bce in MLflow."

train-v6-residue-stage2: ## ResidueStage2 pipeline cryptic + source-leak BCE (warm-start residue_stage1)
	@test -f manifests/v6_corpus_residue_stage2_pipeline.json || (echo "Missing ResidueStage2 manifest" && exit 1)
	@test -f $(or $(RESUME),checkpoints/v6/runs/residue_stage1_v1/v6_best.pt) || \
		(echo "Missing resume checkpoint: $(or $(RESUME),checkpoints/v6/runs/residue_stage1_v1/v6_best.pt)" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_residue_stage2_pipeline.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),residue_stage2_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/residue_stage1_v1/v6_best.pt) \
		--residue-stage2 \
		--residue-stage2-lr $(or $(RESIDUE_STAGE2_LR),1e-4) \
		--residue-stage2-epochs $(or $(EPOCHS),30) \
		--no-corpus-cache
	@echo "ResidueStage2 complete. Compare leak_bce / pocket_bce in MLflow."

train-v6-p4-from-residue-stage2: ## P4 staged epistemic on 12-protein corpus (warm-start residue_stage2)
	@test -f $(or $(RESUME),checkpoints/v6/runs/residue_stage2_v1/v6_phase1_12prot.pt) || \
		(echo "Missing resume checkpoint: $(or $(RESUME),checkpoints/v6/runs/residue_stage2_v1/v6_phase1_12prot.pt)" && exit 1)
	@test -f science/dtie/v3/checkpoints/v2_bridge_epoch_014.pt || \
		(echo "Missing v2_bridge teacher checkpoint" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),p4_epi_rs2_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/residue_stage2_v1/v6_phase1_12prot.pt) \
		--p4-epistemic-decoupling \
		--p4-epistemic-staged \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),1e-4) \
		--epistemic-bf-align-coeff $(or $(EPISTEMIC_BF_ALIGN_COEFF),0.22) \
		--epistemic-sasa-pen-coeff $(or $(EPISTEMIC_SASA_PEN_COEFF),0.10) \
		--shell-corr-epi-sasa-weight $(or $(SHELL_CORR_EPI_SASA),0.0) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),30) \
		--save-epoch-snapshots \
		--v2-teacher-checkpoint $(V2_TEACHER_CKPT_CONTAINER) \
		--v2-teacher-epistemic-coeff 0 \
		--no-corpus-cache
	@echo "P4-from-residue-stage2 complete. Inspect std(epi), r(epi,ale), probe_r_epi_sasa in MLflow."

train-v6-exhaustive-curriculum: ## Chain P4→P4-tight→RS2 refresh (~70 epochs, sequential)
	@chmod +x experiments/training/v6/run_exhaustive_curriculum.sh
	bash experiments/training/v6/run_exhaustive_curriculum.sh

probe-v6-biology: ## KRAS biology probes (Probe 1/4/5) on CHECKPOINT; optional COMPARE=
	@mkdir -p checkpoints/v6/diagnostics pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.biology_probes \
		--checkpoint /app/$(or $(CHECKPOINT),checkpoints/v6/runs/rs2_post_p4_v1/v6_best.pt) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		$(if $(COMPARE),--compare /app/$(COMPARE),) \
		--output /app/$(or $(OUTPUT),checkpoints/v6/diagnostics/biology_probes_$(or $(RUN_ID),latest).json)

train-v6-p4-uncertainty-calibration: ## P4 uncertainty decoupling warm-start rs2_post_p4_v1 (25 ep)
	@test -f $(or $(RESUME),checkpoints/v6/runs/rs2_post_p4_v1/v6_best.pt) || \
		(echo "Missing resume checkpoint: $(or $(RESUME),checkpoints/v6/runs/rs2_post_p4_v1/v6_best.pt)" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),p4_uncertainty_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/rs2_post_p4_v1/v6_best.pt) \
		--p4-uncertainty-calibration \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),25) \
		--save-epoch-snapshots \
		--v2-teacher-checkpoint $(V2_TEACHER_CKPT_CONTAINER) \
		--v2-teacher-epistemic-coeff 0 \
		--no-corpus-cache
	@echo "P4 uncertainty calibration complete. Check probe_r_epi_ale, epistemic_std_mean in MLflow."

train-v6-p4-head-decouple: ## P4 split epi/ale trunks + decorrelation loss (20 ep, rs2 warm-start)
	@test -f $(or $(RESUME),checkpoints/v6/runs/rs2_post_p4_v1/v6_best.pt) || \
		(echo "Missing resume checkpoint: $(or $(RESUME),checkpoints/v6/runs/rs2_post_p4_v1/v6_best.pt)" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),p4_head_decouple_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/rs2_post_p4_v1/v6_best.pt) \
		--p4-head-decouple \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),20) \
		--save-epoch-snapshots \
		--v2-teacher-checkpoint $(V2_TEACHER_CKPT_CONTAINER) \
		--v2-teacher-epistemic-coeff 0 \
		--no-corpus-cache
	@echo "P4 head decouple complete. Target: probe_r_epi_ale < 0.70 for v6_best save."

G3_ROUTE_RESUME ?= checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt
G3_EPOCHS ?= 12
G4_WVP_WEIGHTS ?= 2.8,1.0,0.3
G4_WVP_EPOCHS ?= 4
G4_ISO_EPOCHS ?= 12
G4_ALE_ONLY_EPOCHS ?= 12

train-v6-g3-p4-decorr-only: ## G3 ablation A: decorr-only Phase 4 from route_v1 (no B-factor/SASA supervision)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f $(or $(RESUME),$(G3_ROUTE_RESUME)) || \
		(echo "Missing resume: $(or $(RESUME),$(G3_ROUTE_RESUME))" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),g3_p4_decorr_only_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),$(G3_ROUTE_RESUME)) \
		--p4-head-decouple-decorr-only \
		--structural-disc-frozen \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		--epochs $(or $(EPOCHS),$(G3_EPOCHS)) \
		--save-epoch-snapshots \
		--no-corpus-cache
	@echo "G3 ablation A complete: checkpoints/v6/runs/$(or $(RUN_ID),g3_p4_decorr_only_v1)"

train-v6-g3-p4-full: ## G3 ablation B: full Phase 4 head decouple from route_v1 (B-factor/SASA supervision on)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f $(or $(RESUME),$(G3_ROUTE_RESUME)) || \
		(echo "Missing resume: $(or $(RESUME),$(G3_ROUTE_RESUME))" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),g3_p4_full_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),$(G3_ROUTE_RESUME)) \
		--p4-head-decouple \
		--structural-disc-frozen \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),$(G3_EPOCHS)) \
		--save-epoch-snapshots \
		--no-corpus-cache
	@echo "G3 ablation B complete: checkpoints/v6/runs/$(or $(RUN_ID),g3_p4_full_v1)"

eval-g3-ablation: ## Compare G3 A/B checkpoints (G4a P8 + G5b + P11 circularity gate)
	@test -f $(or $(G3_DECORR_CKPT),checkpoints/v6/runs/g3_p4_decorr_only_v1/v6_best.pt) \
		-o -f $(or $(G3_DECORR_CKPT),checkpoints/v6/runs/g3_p4_decorr_only_v1/v6_phase4_12prot.pt) || \
		(echo "Missing decorr-only checkpoint (v6_best.pt or v6_phase4_12prot.pt)" && exit 1)
	@test -f $(or $(G3_FULL_CKPT),checkpoints/v6/runs/g3_p4_full_v1/v6_best.pt) \
		-o -f $(or $(G3_FULL_CKPT),checkpoints/v6/runs/g3_p4_full_v1/v6_phase4_12prot.pt) || \
		(echo "Missing full-supervision checkpoint" && exit 1)
	@DECORR=$$(test -f $(or $(G3_DECORR_CKPT),checkpoints/v6/runs/g3_p4_decorr_only_v1/v6_best.pt) && echo $(or $(G3_DECORR_CKPT),checkpoints/v6/runs/g3_p4_decorr_only_v1/v6_best.pt) || echo $(or $(G3_DECORR_CKPT),checkpoints/v6/runs/g3_p4_decorr_only_v1/v6_phase4_12prot.pt)); \
	FULL=$$(test -f $(or $(G3_FULL_CKPT),checkpoints/v6/runs/g3_p4_full_v1/v6_best.pt) && echo $(or $(G3_FULL_CKPT),checkpoints/v6/runs/g3_p4_full_v1/v6_best.pt) || echo $(or $(G3_FULL_CKPT),checkpoints/v6/runs/g3_p4_full_v1/v6_phase4_12prot.pt)); \
	mkdir -p checkpoints/v6/diagnostics; \
	TRAINING_LOAD_FROM_PDB=1 $(SCIENCE_RUN) science python experiments/diagnostics/g3_ablation_eval.py \
		--decor-only /app/$$DECORR \
		--full-supervision /app/$$FULL \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cpu) \
		--json-out /app/checkpoints/v6/diagnostics/g3_ablation_report.json \
		--pdb-local

eval-g5-provenance: ## G5 + G5b epistemic provenance (SASA/ρ/teacher bootstrap + OOD 1PGB)
	@mkdir -p checkpoints/v6/diagnostics
	TRAINING_LOAD_FROM_PDB=1 $(SCIENCE_RUN) science python experiments/diagnostics/g5_epistemic_provenance_audit.py \
		--checkpoints route_v1:/app/checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt \
		--checkpoints g3_decorr:/app/checkpoints/v6/runs/g3_p4_decorr_only_v1/v6_phase4_12prot.pt \
		--checkpoints g3_full:/app/checkpoints/v6/runs/g3_p4_full_v1/v6_phase4_12prot.pt \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cpu) \
		--json-out /app/checkpoints/v6/diagnostics/g5_provenance_report.json \
		--pdb-local

eval-g4-holdout: ## G4 holdout P8 eval (aleatoric shaping circularity baseline)
	@mkdir -p checkpoints/v6/diagnostics
	TRAINING_LOAD_FROM_PDB=1 $(SCIENCE_RUN) science python experiments/diagnostics/g4_aleatoric_holdout_eval.py \
		--checkpoint /app/$(or $(G4_CKPT),checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cpu) \
		--holdout-seeds $(or $(G4_HOLDOUT_SEEDS),42) \
		--json-out /app/checkpoints/v6/diagnostics/g4_holdout_report.json \
		--pdb-local

eval-g4-holdout-multi: ## G4 eval with holdout seeds 42,7 (eval-only rotation)
	$(MAKE) eval-g4-holdout G4_HOLDOUT_SEEDS=42,7 G4_CKPT=$(or $(G4_CKPT),checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt)

train-v6-p4-v3-aleatoric-shaping: ## Phase 4 + v3 aleatoric shaping (G4 holdout train mask)
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f $(or $(RESUME),$(G3_ROUTE_RESUME)) || \
		(echo "Missing resume: $(or $(RESUME),$(G3_ROUTE_RESUME))" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),p4_v3_aleatoric_shaping_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),$(G3_ROUTE_RESUME)) \
		--p4-v3-aleatoric-shaping \
		--structural-disc-frozen \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		$(if $(W_VAR_PENALTY),--w-var-penalty $(W_VAR_PENALTY),) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),$(G3_EPOCHS)) \
		--save-epoch-snapshots \
		--no-corpus-cache
	@echo "G4 v3 aleatoric shaping complete: checkpoints/v6/runs/$(or $(RUN_ID),p4_v3_aleatoric_shaping_v1)"

train-v6-p4-g4-shaping-only-isolation: ## G4 isolation: uncertainty-head-only, only v3 shaping loss active
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f $(or $(RESUME),$(G3_ROUTE_RESUME)) || \
		(echo "Missing resume: $(or $(RESUME),$(G3_ROUTE_RESUME))" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),g4_shaping_only_iso_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),$(G3_ROUTE_RESUME)) \
		--p4-g4-shaping-only-isolation \
		--structural-disc-frozen \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		$(if $(W_VAR_PENALTY),--w-var-penalty $(W_VAR_PENALTY),) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),$(G4_ISO_EPOCHS)) \
		--save-epoch-snapshots \
		--no-corpus-cache
	@echo "G4 shaping-only isolation complete: checkpoints/v6/runs/$(or $(RUN_ID),g4_shaping_only_iso_v1)"

train-v6-p4-g4-ale-only-unshaped: ## G4 isolation: ale-only branch trainable, no shaping losses
	@test -f data/gates/p_feature_01_passed.json || \
		(echo "Missing P_FEATURE_01 gate stamp — run: make gate-p-feature-01" && exit 1)
	@test -f $(or $(RESUME),$(G3_ROUTE_RESUME)) || \
		(echo "Missing resume: $(or $(RESUME),$(G3_ROUTE_RESUME))" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),g4_ale_only_unshaped_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),$(G3_ROUTE_RESUME)) \
		--p4-g4-ale-only-unshaped \
		--structural-disc-frozen \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),$(G4_ALE_ONLY_EPOCHS)) \
		--save-epoch-snapshots \
		--no-corpus-cache
	@echo "G4 ale-only unshaped isolation complete: checkpoints/v6/runs/$(or $(RUN_ID),g4_ale_only_unshaped_v1)"

g4-var-penalty-sweep: ## Short w_var_penalty probes {2.8,1.0,0.3} + frozen separation eval
	@test -f $(or $(RESUME),$(G3_ROUTE_RESUME)) || \
		(echo "Missing resume: $(or $(RESUME),$(G3_ROUTE_RESUME))" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs checkpoints/v6/diagnostics pdb_cache
	TRAINING_LOAD_FROM_PDB=1 $(SCIENCE_RUN) science python experiments/diagnostics/g4_var_penalty_sweep.py \
		--resume /app/$(or $(RESUME),$(G3_ROUTE_RESUME)) \
		--epochs $(or $(G4_WVP_EPOCHS),4) \
		--device $(or $(DEVICE),cuda) \
		--weights $(G4_WVP_WEIGHTS) \
		--json-out /app/checkpoints/v6/diagnostics/g4_wvp_sweep_report.json

g4-var-penalty-sweep-eval: ## Eval-only for existing g4_wvp_* probe runs
	TRAINING_LOAD_FROM_PDB=1 $(SCIENCE_RUN) science python experiments/diagnostics/g4_var_penalty_sweep.py \
		--eval-only \
		--device $(or $(DEVICE),cuda) \
		--weights $(G4_WVP_WEIGHTS) \
		--json-out /app/checkpoints/v6/diagnostics/g4_wvp_sweep_report.json

gnnv7-retrain-g3: ## Run G3 A/B Phase 4 ablations then eval circularity gate
	$(MAKE) train-v6-g3-p4-decorr-only RUN_ID=$(or $(G3_DECORR_RUN),g3_p4_decorr_only_v1) EPOCHS=$(or $(EPOCHS),$(G3_EPOCHS))
	$(MAKE) train-v6-g3-p4-full RUN_ID=$(or $(G3_FULL_RUN),g3_p4_full_v1) EPOCHS=$(or $(EPOCHS),$(G3_EPOCHS))
	$(MAKE) eval-g3-ablation G3_DECORR_CKPT=checkpoints/v6/runs/$(or $(G3_DECORR_RUN),g3_p4_decorr_only_v1)/v6_best.pt \
		G3_FULL_CKPT=checkpoints/v6/runs/$(or $(G3_FULL_RUN),g3_p4_full_v1)/v6_best.pt

train-v6-gnnv7-routing-recovery: ## GNNv7: MoE routing touch-up from route_v1 (hyperbolic MP graph during training)
	@test -f $(or $(RESUME),$(G3_ROUTE_RESUME)) || \
		(echo "Missing resume: $(or $(RESUME),$(G3_ROUTE_RESUME))" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	GNN_INPUT_MODE=topology_three_vector $(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),gnnv7_route_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),$(G3_ROUTE_RESUME)) \
		--topology-routing-recovery \
		--structural-disc-frozen \
		--topology-routing-recovery-lr $(or $(LR),3e-5) \
		--epochs $(or $(EPOCHS),20) \
		--p2-bridge-ramp-epochs $(or $(P2_BRIDGE_RAMP_EPOCHS),$(EPOCHS),20) \
		--num-experts $(or $(NUM_EXPERTS),4) \
		--save-epoch-snapshots
	@echo "GNNv7 routing recovery complete: checkpoints/v6/runs/$(or $(RUN_ID),gnnv7_route_v1)"

train-v6-gnnv7-p4-full: ## GNNv7: full Phase 4 after routing recovery (requires G3 pass)
	@test -f $(or $(RESUME),checkpoints/v6/runs/gnnv7_route_v1/v6_best.pt) || \
		(echo "Missing resume — run train-v6-gnnv7-routing-recovery first" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/v6_corpus_stage_a_small_v1.json \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),gnnv7_p4_full_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/gnnv7_route_v1/v6_best.pt) \
		--p4-head-decouple \
		--structural-disc-frozen \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),20) \
		--save-epoch-snapshots \
		--no-corpus-cache
	@echo "GNNv7 Phase 4 complete: checkpoints/v6/runs/$(or $(RUN_ID),gnnv7_p4_full_v1)"

gnnv7-retrain-full: ## Full GNNv7 retrain after G3 pass (routing recovery + Phase 4 full)
	$(MAKE) train-v6-gnnv7-routing-recovery RUN_ID=$(or $(GNNV7_ROUTE_RUN),gnnv7_route_v1) EPOCHS=$(or $(ROUTING_EPOCHS),20)
	$(MAKE) train-v6-gnnv7-p4-full RUN_ID=$(or $(GNNV7_P4_RUN),gnnv7_p4_full_v1) \
		RESUME=checkpoints/v6/runs/$(or $(GNNV7_ROUTE_RUN),gnnv7_route_v1)/v6_best.pt EPOCHS=$(or $(P4_EPOCHS),20)

gnnv7-retrain: ## Recommended sequence: G3 A/B gate, then full retrain if pass
	@echo "=== GNNv7 retrain step 1/2: G3 A/B ablations ==="
	$(MAKE) gnnv7-retrain-g3
	@echo "=== GNNv7 retrain step 2/2: full routing + P4 (only if G3 passed) ==="
	@if [ -f checkpoints/v6/diagnostics/g3_ablation_report.json ] && \
		python3 -c "import json; r=json.load(open('checkpoints/v6/diagnostics/g3_ablation_report.json')); exit(0 if r['g3_report']['g3_pass'] else 1)"; then \
		$(MAKE) gnnv7-retrain-full; \
	else \
		echo "G3 gate failed or report missing — skipping full retrain. Inspect checkpoints/v6/diagnostics/g3_ablation_report.json"; \
		exit 1; \
	fi

train-v6-p4-head-decouple-continue: ## Continue head decouple from phase checkpoint (12 ep, sasa gate 0.79)
	@test -f $(or $(RESUME),checkpoints/v6/runs/p4_head_decouple_v1/v6_phase4_12prot.pt) || \
		(echo "Missing resume: $(or $(RESUME),checkpoints/v6/runs/p4_head_decouple_v1/v6_phase4_12prot.pt)" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),p4_head_decouple_v2) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/p4_head_decouple_v1/v6_phase4_12prot.pt) \
		--p4-head-decouple \
		--max-probe-r-epi-sasa-save $(or $(MAX_PROBE_R_EPI_SASA_SAVE),0.79) \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),12) \
		--save-epoch-snapshots \
		--v2-teacher-checkpoint $(V2_TEACHER_CKPT_CONTAINER) \
		--v2-teacher-epistemic-coeff 0 \
		--no-corpus-cache
	@echo "P4 head decouple continue complete."

train-v6-p4-gate-promotion: ## Gate-only routing pass from head-decouple best (20 ep, backbone uncertainty probes)
	@test -f $(or $(RESUME),checkpoints/v6/runs/p4_head_decouple_v2/v6_best.pt) || \
		(echo "Missing resume: $(or $(RESUME),checkpoints/v6/runs/p4_head_decouple_v2/v6_best.pt)" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),p4_gate_promotion_v3) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/p4_head_decouple_v2/v6_best.pt) \
		--p4-gate-promotion \
		--max-probe-r-epi-sasa-save $(or $(MAX_PROBE_R_EPI_SASA_SAVE),0.79) \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),3e-5) \
		--epochs $(or $(EPOCHS),20) \
		$(if $(P4_GATE_BALANCE_COEFF),--p4-gate-balance-coeff $(P4_GATE_BALANCE_COEFF),) \
		$(if $(P4_GATE_LOAD_FLOOR_COEFF),--p4-gate-load-floor-coeff $(P4_GATE_LOAD_FLOOR_COEFF),) \
		$(if $(P4_GATE_LOAD_FLOOR_MIN),--p4-gate-load-floor-min $(P4_GATE_LOAD_FLOOR_MIN),) \
		$(if $(GATE_GUMBEL),--gate-gumbel,) \
		--save-epoch-snapshots \
		--v2-teacher-epistemic-coeff 0 \
		--no-corpus-cache
	@echo "P4 gate promotion complete. Uncertainty save gates use backbone tangent during gate training."

train-v6-p4-gate-touchup: ## Routed-path uncertainty recalibration after gate pass (15 ep)
	@test -f $(or $(RESUME),checkpoints/v6/runs/$(or $(GATE_PROMOTION_RUN),p4_gate_promotion_v3)/v6_phase2_12prot.pt) || \
		(echo "Missing resume: $(or $(RESUME),checkpoints/v6/runs/$(or $(GATE_PROMOTION_RUN),p4_gate_promotion_v3)/v6_phase2_12prot.pt)" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_stage_a_small_v1.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),p4_gate_touchup_v3) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/$(or $(GATE_PROMOTION_RUN),p4_gate_promotion_v3)/v6_phase2_12prot.pt) \
		--p4-gate-uncertainty-touchup \
		--max-probe-r-epi-sasa-save $(or $(MAX_PROBE_R_EPI_SASA_SAVE),0.79) \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),15) \
		--save-epoch-snapshots \
		--v2-teacher-epistemic-coeff 0 \
		--no-corpus-cache
	@echo "P4 gate touchup complete. Production uncertainty path is routed tangent."

train-v6-p4-gate-full-promotion: ## Gate pass + routed touchup (v3 defaults)
	$(MAKE) train-v6-p4-gate-promotion RUN_ID=$(or $(GATE_PROMOTION_RUN),p4_gate_promotion_v3)
	$(MAKE) train-v6-p4-gate-touchup \
		GATE_PROMOTION_RUN=$(or $(GATE_PROMOTION_RUN),p4_gate_promotion_v3) \
		RUN_ID=$(or $(GATE_TOUCHUP_RUN),p4_gate_touchup_v3)

CORPUS25_MANIFEST := v6_corpus_stage_a_expand_v1.json
CORPUS25_PROTEINS := 23
CORPUS25_RESUME := checkpoints/v6/runs/p4_corpus25_touchup_v4c/v6_best.pt
CORPUS25_GATE_TOUCHUP_RESUME := checkpoints/v6/runs/$(or $(GATE_PROMOTION_RUN),p4_corpus25_gate_v1)/v6_best.pt

train-v6-p4-corpus25-gate: ## Gate on Stage A expand corpus (23 prot) from touchup v5 (30 ep, moderate MoE + Gumbel)
	@test -f manifests/$(CORPUS25_MANIFEST) || (echo "Missing expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(CORPUS25_RESUME)) || \
		(echo "Missing resume: $(or $(RESUME),$(CORPUS25_RESUME))" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(CORPUS25_MANIFEST) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),p4_corpus25_gate_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),$(CORPUS25_PROTEINS)) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),$(CORPUS25_RESUME)) \
		--p4-corpus25-gate-promotion \
		--gate-gumbel \
		--max-probe-r-epi-sasa-save $(or $(MAX_PROBE_R_EPI_SASA_SAVE),0.79) \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),3e-5) \
		--epochs $(or $(EPOCHS),30) \
		$(if $(P4_GATE_BALANCE_COEFF),--p4-gate-balance-coeff $(P4_GATE_BALANCE_COEFF),) \
		$(if $(P4_GATE_LOAD_FLOOR_COEFF),--p4-gate-load-floor-coeff $(P4_GATE_LOAD_FLOOR_COEFF),) \
		$(if $(P4_GATE_LOAD_FLOOR_MIN),--p4-gate-load-floor-min $(P4_GATE_LOAD_FLOOR_MIN),) \
		--save-epoch-snapshots \
		--v2-teacher-epistemic-coeff 0 \
		--no-corpus-cache
	@echo "Corpus-25 gate pass complete."

train-v6-p4-corpus25-touchup: ## Routed uncertainty touchup after corpus expand gate (default: from gate v6_best)
	@test -f manifests/$(CORPUS25_MANIFEST) || (echo "Missing expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(CORPUS25_GATE_TOUCHUP_RESUME)) || \
		(echo "Missing resume: $(or $(RESUME),$(CORPUS25_GATE_TOUCHUP_RESUME))" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(CORPUS25_MANIFEST) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),p4_corpus25_touchup_v1) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),$(CORPUS25_PROTEINS)) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),$(CORPUS25_GATE_TOUCHUP_RESUME)) \
		--p4-gate-uncertainty-touchup \
		--max-probe-r-epi-sasa-save $(or $(MAX_PROBE_R_EPI_SASA_SAVE),0.79) \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),15) \
		--save-epoch-snapshots \
		--v2-teacher-epistemic-coeff 0 \
		--no-corpus-cache
	@echo "Corpus-25 touchup complete."

train-v6-p4-corpus25-touchup-extended: ## Extended SASA recal touchup (25 ep, gate frozen)
	@test -f manifests/$(CORPUS25_MANIFEST) || (echo "Missing expand manifest" && exit 1)
	@test -f $(or $(RESUME),$(CORPUS25_RESUME)) || \
		(echo "Missing resume: $(or $(RESUME),$(CORPUS25_RESUME))" && exit 1)
	@mkdir -p mlruns checkpoints/v6/runs pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(CORPUS25_MANIFEST) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),p4_corpus25_touchup_v4) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),$(CORPUS25_PROTEINS)) \
		--max-residues $(or $(MAX_RESIDUES),$(STAGE_A_MAX_RESIDUES)) \
		--mlflow-uri http://mlflow:5000 \
		--resume /app/$(or $(RESUME),$(CORPUS25_RESUME)) \
		--p4-corpus25-touchup-extended \
		--max-probe-r-epi-sasa-save $(or $(MAX_PROBE_R_EPI_SASA_SAVE),0.79) \
		--p4-epistemic-lr $(or $(P4_EPISTEMIC_LR),5e-5) \
		--epistemic-decoupling-holdouts $(or $(EPISTEMIC_DECOUPLING_HOLDOUTS),1IVO\,4MNE) \
		--epochs $(or $(EPOCHS),25) \
		--save-epoch-snapshots \
		--v2-teacher-epistemic-coeff 0 \
		--no-corpus-cache
	@echo "Corpus-25 extended touchup complete."

train-v6-p4-corpus25-full: ## Corpus expand gate + touchup from CORPUS25_RESUME champion
	$(MAKE) train-v6-p4-corpus25-gate \
		RUN_ID=$(or $(GATE_PROMOTION_RUN),p4_corpus25_gate_v1) \
		RESUME=$(or $(RESUME),$(CORPUS25_RESUME))
	$(MAKE) train-v6-p4-corpus25-touchup \
		GATE_PROMOTION_RUN=$(or $(GATE_PROMOTION_RUN),p4_corpus25_gate_v1) \
		RUN_ID=$(or $(GATE_TOUCHUP_RUN),p4_corpus25_touchup_v1)

train-v6-p4-corpus25-push: ## Extended touchup → gate → touchup from gate best (v3b champion)
	$(MAKE) train-v6-p4-corpus25-touchup-extended \
		RUN_ID=$(or $(TOUCHUP_EXTENDED_RUN),p4_corpus25_touchup_v4) \
		RESUME=$(or $(RESUME),$(CORPUS25_RESUME))
	@ext_run="$(or $(TOUCHUP_EXTENDED_RUN),p4_corpus25_touchup_v4)"; \
	ext_dir="checkpoints/v6/runs/$$ext_run"; \
	if [ -f "$$ext_dir/v6_best.pt" ]; then \
	  ext_ckpt="$$ext_dir/v6_best.pt"; \
	elif [ -f "$$ext_dir/v6_phase4_$(CORPUS25_PROTEINS)prot.pt" ]; then \
	  ext_ckpt="$$ext_dir/v6_phase4_$(CORPUS25_PROTEINS)prot.pt"; \
	  echo "Touchup-extended did not beat prior v6_best score — gate resumes from $$ext_ckpt"; \
	else \
	  ext_ckpt="$$(ls -1 $$ext_dir/v6_phase*_*prot.pt 2>/dev/null | tail -1)"; \
	  if [ -z "$$ext_ckpt" ]; then \
	    echo "No checkpoint in $$ext_dir after touchup-extended (expected v6_best.pt or v6_phase*_*prot.pt)"; \
	    exit 1; \
	  fi; \
	  echo "Gate resumes from phase checkpoint $$ext_ckpt"; \
	fi; \
	$(MAKE) train-v6-p4-corpus25-gate \
		RUN_ID=$(or $(GATE_PROMOTION_RUN),p4_corpus25_gate_v4) \
		RESUME="$$ext_ckpt"
	$(MAKE) train-v6-p4-corpus25-touchup \
		GATE_PROMOTION_RUN=$(or $(GATE_PROMOTION_RUN),p4_corpus25_gate_v4) \
		RUN_ID=$(or $(GATE_TOUCHUP_RUN),p4_corpus25_touchup_v4b)

auto-train-v6-corpus25: ## Playbook-driven autonomous corpus-25 loop (DRY_RUN=1, ONCE=1, DIAGNOSE=1, MULTI=N, APPROVE=1, PLAYBOOK=...)
	@mkdir -p checkpoints/v6/runs/auto_trainer
	PYTHONPATH=. python3 -m experiments.training.v6.auto_trainer \
		--playbook manifests/auto_trainer/$(or $(PLAYBOOK),corpus12_recovery_playbook.yaml) \
		$(if $(DRY_RUN),--dry-run,) \
		$(if $(ONCE),--once,) \
		$(if $(DIAGNOSE),--diagnose,) \
		$(if $(MULTI),--multi $(MULTI),) \
		$(if $(APPROVE),--approve,) \
		$(if $(MAX_ITERATIONS),--max-iterations $(MAX_ITERATIONS),)

auto-train-v6-status: ## JSON status for champion, program goals, and next planned action
	@mkdir -p checkpoints/v6/runs/auto_trainer
	PYTHONPATH=. python3 -m experiments.training.v6.auto_trainer --status \
		--playbook manifests/auto_trainer/$(or $(PLAYBOOK),corpus12_recovery_playbook.yaml)

assess-v6-12prot-baseline: ## Side-by-side assess on 12-prot manifest (CHAMPION=... BASELINE=...)
	@mkdir -p checkpoints/v6/runs/baseline_compare_12prot
	$(MAKE) assess-v6 \
		CHECKPOINT=/app/checkpoints/v6/runs/$(or $(CHAMPION),auto_touchup_20260704_135750)/v6_best.pt \
		CORPUS=v6_corpus_stage_a_small_v1.json MAX_PROTEINS=12 MAX_RESIDUES=650 DEVICE=$(or $(DEVICE),cuda) \
		OUTPUT=/app/checkpoints/v6/runs/baseline_compare_12prot/champion_on_12prot.json
	$(MAKE) assess-v6 \
		CHECKPOINT=/app/checkpoints/v6/runs/$(or $(BASELINE),residue_stage2_v1/v6_phase1_12prot.pt) \
		CORPUS=v6_corpus_stage_a_small_v1.json MAX_PROTEINS=12 MAX_RESIDUES=650 DEVICE=$(or $(DEVICE),cuda) \
		OUTPUT=/app/checkpoints/v6/runs/baseline_compare_12prot/baseline_on_12prot.json

export-corpus-viewers: ## Export NGL + Poincaré disc + split HTML for each corpus structure
	@mkdir -p data/local_objects/gnn_viewer
	$(SCIENCE_RUN) science python -m experiments.training.v6.export_corpus_viewers \
		--checkpoint /app/$(or $(CHECKPOINT),checkpoints/v6/runs/residue_stage2_v1/v6_phase1_12prot.pt) \
		--corpus /app/$(or $(CORPUS),manifests/v6_corpus_stage_a_small_v1.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),12) \
		$(if $(STRUCTURAL_DISC_FROZEN),--structural-disc-frozen,) \
		$(if $(OUTPUT_ROOT),--output-root /app/$(OUTPUT_ROOT),)
	@echo "Viewers under $(or $(OUTPUT_ROOT),data/local_objects/gnn_viewer)/<pdb>/"

test-stage-a-smoke: ## Assert P_STAGE_A_SMOKE on STAGE_A_SMOKE_RUN_ID MLflow run
	@test -n "$$STAGE_A_SMOKE_RUN_ID" || (echo "Set STAGE_A_SMOKE_RUN_ID to the smoke run id" && exit 1)
	MLFLOW_TRACKING_URI=$(or $(MLFLOW_TRACKING_URI),file:./mlruns) \
	MLFLOW_ALLOW_FILE_STORE=true \
	STAGE_A_SMOKE_FULL=$${STAGE_A_SMOKE_FULL:-0} \
	python3 -m pytest tests/test_stage_a_integration_smoke.py::test_stage_a_smoke_mlflow_run_from_env -v

sync-corpus-pins: ## Print SHA256 constants for corpus_governance.py (same commit as JSON)
	uv run python experiments/training/v6/sync_corpus_pins.py

sync-p-curv-fixture: ## Refresh P_CURV_01 CI checkpoint from lever_a production path
	@test -f checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt || \
		(echo "Missing lever_a checkpoint — cannot refresh fixture" && exit 1)
	@mkdir -p tests/fixtures/checkpoints
	cp checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt \
		tests/fixtures/checkpoints/lever_a_v6_best_disc.pt
	@echo "Updated tests/fixtures/checkpoints/lever_a_v6_best_disc.pt"

seed-p-curv-fixture: ## Apply curvature SSOT seed to DATABASE_URL (after migrations)
	@test -n "$$DATABASE_URL" || (echo "Set DATABASE_URL first" && exit 1)
	uv run python -c "import os, psycopg; \
		conn = psycopg.connect(os.environ['DATABASE_URL']); \
		cur = conn.cursor(); \
		cur.execute(open('tests/fixtures/seed_curvature_ssot.sql').read()); \
		conn.commit(); conn.close()"
	@echo "Applied tests/fixtures/seed_curvature_ssot.sql"

test-p-curv-01: ## Run P_CURV_01 SSOT probe (needs DATABASE_URL + seed-p-curv-fixture)
	@test -n "$$DATABASE_URL" || (echo "Set DATABASE_URL first" && exit 1)
	P_CURV_01_CHECKPOINT_PATH=tests/fixtures/checkpoints/lever_a_v6_best_disc.pt \
		uv run pytest tests/test_curvature_ssot_gate.py::test_p_curv_01_probe_sources_match -v

train-v6-theory-test-mlflow: ## Theory test + MLflow ON — seed baseline for promote-v6-from-run
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),mlflow_theory_baseline) EPOCHS=$(or $(EPOCHS),15) \
		USE_MLFLOW=1 P2_BRIDGE=1 FULL_HYP_MOE_TEST=1 P2_BRIDGE_RAMP_EPOCHS=$(or $(P2_BRIDGE_RAMP_EPOCHS),15) \
		HYPERBOLIC_EXPERT_MIX=1 NO_WARM_START=1 SAVE_EPOCH_SNAPSHOTS=1 DEVICE=$(or $(DEVICE),cuda) \
		RESUME=$(or $(RESUME),checkpoints/v6/tokyo_eyes_v6.pt) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),8)

assess-v6-benchmark: ## Assess v6 checkpoint on science/dtie benchmark_pdbs (use v6_best.pt)
	@test -n "$(CHECKPOINT)" || (echo "Usage: make assess-v6-benchmark CHECKPOINT=/app/checkpoints/v6/runs/RUN/v6_best.pt" && exit 1)
	@test -d science/dtie/assets/benchmark_pdbs || (echo "Missing science/dtie/assets/benchmark_pdbs" && exit 1)
	@mkdir -p pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.assess_checkpoint \
		--checkpoint "$(CHECKPOINT)" \
		--corpus /app/manifests/v6_corpus_benchmark.json \
		--pdb-dir $(BENCHMARK_PDB_CONTAINER) \
		--device $(or $(DEVICE),cuda) \
		--no-promotion-exit \
		$(if $(MAX_PROTEINS),--max-proteins $(MAX_PROTEINS),) \
		$(if $(MAX_RESIDUES),--max-residues $(MAX_RESIDUES),) \
		$(if $(OUTPUT),--output "$(OUTPUT)",)

viz-v6-shell: ## Matplotlib shell dashboard (CHECKPOINT=..., optional COMPARE=..., OUTPUT=...)
	@test -n "$(CHECKPOINT)" || (echo "Usage: make viz-v6-shell CHECKPOINT=/app/checkpoints/v6/runs/RUN/v6_best.pt" && exit 1)
	@test -d science/dtie/assets/benchmark_pdbs || (echo "Missing science/dtie/assets/benchmark_pdbs" && exit 1)
	@mkdir -p checkpoints/v6/runs
	$(SCIENCE_RUN) science python -m experiments.training.v6.visualize_shell \
		--checkpoint "$(CHECKPOINT)" \
		$(if $(COMPARE),--compare "$(COMPARE)",) \
		--corpus /app/manifests/v6_corpus_benchmark.json \
		--pdb-dir $(BENCHMARK_PDB_CONTAINER) \
		--device $(or $(DEVICE),cuda) \
		--max-proteins $(or $(MAX_PROTEINS),8) \
		$(if $(OUTPUT),--output "$(OUTPUT)",) \
		$(if $(LABEL),--label "$(LABEL)",) \
		$(if $(COMPARE_LABEL),--compare-label "$(COMPARE_LABEL)",) \
		$(if $(PER_PROTEIN_PANELS),--per-protein-panels $(PER_PROTEIN_PANELS),)

viz-v6-training: ## Metrics curves + filmstrip from RUN_DIR (metrics.json + epoch snapshots)
	@test -n "$(RUN_DIR)" || (echo "Usage: make viz-v6-training RUN_DIR=checkpoints/v6/runs/shell_no_warm_p2" && exit 1)
	$(SCIENCE_RUN) science python -m experiments.training.v6.visualize_training \
		--run-dir /app/$(RUN_DIR) \
		--pdb-dir $(BENCHMARK_PDB_CONTAINER) \
		--corpus /app/manifests/v6_corpus_benchmark.json \
		--device $(or $(DEVICE),cuda) \
		$(if $(FILMSTRIP_PROTEIN),--filmstrip-protein $(FILMSTRIP_PROTEIN),) \
		$(if $(NO_FILMSTRIP),--no-filmstrip,)

train-v6-curriculum: ## Full P1→P2 curriculum (stops if P1 fails; MAX_PROTEINS, MAX_RESIDUES=600)
	@RUN_ID=$(or $(RUN_ID),$(shell date +%Y%m%d_%H%M%S)); \
	echo "=== V6 curriculum run $$RUN_ID ==="; \
	$(MAKE) train-v6 STAGE=1 RUN_ID=$$RUN_ID MAX_PROTEINS=$(MAX_PROTEINS) CORPUS=$(or $(CORPUS),v6_corpus_200.json) \
		MAX_RESIDUES=$(or $(MAX_RESIDUES),600) DEVICE=$(or $(DEVICE),cuda) NO_MLFLOW=$(NO_MLFLOW) && \
	$(MAKE) train-v6 STAGE=2 RUN_ID=$$RUN_ID \
		MAX_PROTEINS=$(MAX_PROTEINS) CORPUS=$(or $(CORPUS),v6_corpus_200.json) \
		MAX_RESIDUES=$(or $(MAX_RESIDUES),600) DEVICE=$(or $(DEVICE),cuda) NO_MLFLOW=$(NO_MLFLOW) && \
	echo "Done. Assess: make assess-v6 CHECKPOINT=/app/checkpoints/v6/runs/$$RUN_ID/v6_best.pt MAX_PROTEINS=$(MAX_PROTEINS) CORPUS=$(or $(CORPUS),v6_corpus_200.json)"

assess-v6: ## Assess v6 checkpoint (CHECKPOINT=/app/..., OUTPUT=/app/... optional)
	@test -n "$(CHECKPOINT)" || (echo "Usage: make assess-v6 CHECKPOINT=/app/checkpoints/v6/runs/.../v6_best.pt" && exit 1)
	@mkdir -p pdb_cache
	$(SCIENCE_RUN) science python -m experiments.training.v6.assess_checkpoint \
		--checkpoint "$(CHECKPOINT)" \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_120.json) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cuda) \
        $(if $(MAX_PROTEINS),--max-proteins $(MAX_PROTEINS),) \
        $(if $(MAX_RESIDUES),--max-residues $(MAX_RESIDUES),) \
        $(if $(OUTPUT),--output "$(OUTPUT)",)

eval-v6: assess-v6 ## Alias for assess-v6

diagnose-radial-angular-slice: ## Rank-collapse audit — pre/post routing disc (CHECKPOINT=...)
	@test -n "$(CHECKPOINT)" || (echo "Usage: make diagnose-radial-angular-slice CHECKPOINT=checkpoints/v6/runs/.../phase_2_disc_occupancy_recovery.pt" && exit 1)
	@mkdir -p checkpoints/v6/diagnostics
	python -m experiments.diagnostics.radial_angular_slice_audit \
		--checkpoint "$(CHECKPOINT)" \
		--structures "$(or $(STRUCTURES),11QE:A,4OBE:A,1IVO:A)" \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cpu) \
		$(if $(filter 1,$(LEGACY_DISC_PROJECTION)),--legacy-disc-projection,) \
		$(if $(filter 0,$(LEGACY_DISC_PROJECTION)),--no-legacy-disc-projection,) \
		$(if $(RADIAL_ANGULAR_RECOMBINE),--radial-angular-recombine $(RADIAL_ANGULAR_RECOMBINE),) \
		$(if $(DISC_RADIAL_SOURCE),--disc-radial-source $(DISC_RADIAL_SOURCE),) \
		--json-out "$(or $(JSON_OUT),checkpoints/v6/diagnostics/slice_audit_$(shell basename $(CHECKPOINT) .pt).json)"

diagnose-embedding: ## Disc occupancy audit — full tables (CHECKPOINT=... or COMPARE_CHECKPOINTS='a b c'; PDB_LOCAL=0 needs DB)
	@test -n "$(or $(CHECKPOINT),$(COMPARE_CHECKPOINTS))" || (echo "Usage: make diagnose-embedding CHECKPOINT=checkpoints/v6/tokyo_eyes_v6.pt STRUCTURES=11QE:A" && exit 1)
	@mkdir -p pdb_cache checkpoints/v6/diagnostics
	python -m experiments.diagnostics.embedding_occupancy_audit \
		$(if $(COMPARE_CHECKPOINTS),--compare-checkpoints $(COMPARE_CHECKPOINTS),--checkpoint "$(CHECKPOINT)") \
		--structures "$(or $(STRUCTURES),11QE:A,4OBE:A,1IVO:A,4MNE:A)" \
		--pdb-dir /tmp/dtie_pdb_cache \
		$(if $(filter 0,$(PDB_LOCAL)),,--pdb-local) \
		--device $(or $(DEVICE),cpu) \
		$(if $(JSON_OUT),--json-out "$(JSON_OUT)",) \
		$(if $(EXPORT_SCATTER),--export-scatter "$(EXPORT_SCATTER)",)

audit-dehydron-topology: ## Dehydron cone alignment + disc panels + P_DEHYDRON_CONE_01 pass/fail (CHECKPOINT=...)
	@test -f "$(or $(CHECKPOINT),checkpoints/v6/tokyo_eyes_v6.pt)" || \
		(echo "Missing checkpoint: set CHECKPOINT=checkpoints/v6/tokyo_eyes_v6.pt" && exit 1)
	@mkdir -p checkpoints/v6/diagnostics/dehydron_topology
	@CKPT="$(or $(CHECKPOINT),checkpoints/v6/tokyo_eyes_v6.pt)"; \
	STEM=$$(basename "$$CKPT" .pt); \
	python3 -m experiments.training.v6.dehydron_cone_alignment_audit \
		--checkpoint "$$CKPT" \
		--corpus "$(or $(CORPUS),manifests/v6_corpus_disc_target.json)" \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--pdb-local \
		--device $(or $(DEVICE),cpu) \
		--plot-dir "$(or $(PLOT_DIR),checkpoints/v6/diagnostics/dehydron_topology)" \
		--output "$(or $(JSON_OUT),checkpoints/v6/diagnostics/dehydron_topology/$${STEM}_alignment.json)" \
		--gate-exit

edge-telemetry-baseline: ## Edge telemetry MVP — telemetry-alive + collapsed-routing baseline (track only, not gate)
	@mkdir -p checkpoints/v6/runs
	python3 -m experiments.diagnostics.edge_telemetry_baseline \
		--corpus "$(or $(CORPUS),manifests/v6_corpus_stage_a_small_v1.json)" \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--pdb-local \
		--device $(or $(DEVICE),cpu) \
		--compare-checkpoints \
			checkpoints/v6/runs/slim_moe_structural_ssot_cold_v1/v6_best.pt \
			checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt \
		--json-out "$(or $(JSON_OUT),checkpoints/v6/runs/edge_telemetry_mvp_baseline.json)"

residue-uncertainty-audit: ## Per-residue ν_epi / ν_ale CSV+JSON (CHECKPOINT=... STRUCTURES=1MBN:A)
	@test -f "$(or $(CHECKPOINT),checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt)" || \
		(echo "Missing CHECKPOINT" && exit 1)
	@mkdir -p checkpoints/v6/diagnostics
	python3 -m experiments.diagnostics.residue_uncertainty_audit \
		--checkpoint "$(or $(CHECKPOINT),checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt)" \
		--structures "$(or $(STRUCTURES),1MBN:A)" \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--pdb-local \
		--device $(or $(DEVICE),cpu) \
		--json-out "$(or $(JSON_OUT),checkpoints/v6/diagnostics/residue_uncertainty_audit.json)" \
		--csv-out "$(or $(CSV_OUT),checkpoints/v6/diagnostics/residue_uncertainty_audit.csv)"

aleatoric-corpus-diagnostics: ## Residue-first aleatoric health + active-learning triage (CHECKPOINT=... MANIFEST=...)
	@test -f "$(or $(CHECKPOINT),checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt)" || \
		(echo "Missing CHECKPOINT" && exit 1)
	@mkdir -p checkpoints/v6/diagnostics
	python3 -m experiments.diagnostics.aleatoric_corpus_diagnostics \
		--checkpoint "$(or $(CHECKPOINT),checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt)" \
		--manifest "$(or $(MANIFEST),manifests/v6_corpus_stage_a_small_v1.json)" \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--pdb-local \
		--device $(or $(DEVICE),cpu) \
		--t-ale-percentile $(or $(T_ALE_PERCENTILE),90) \
		$(if $(T_ALE),--t-ale $(T_ALE),) \
		--json-out "$(or $(JSON_OUT),checkpoints/v6/diagnostics/aleatoric_corpus_diagnostics.json)"

aleatoric-independence-probe: ## Does ν_ale vary beyond ρ + expert? (CHECKPOINT=... MANIFEST=...)
	@test -f "$(or $(CHECKPOINT),checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt)" || \
		(echo "Missing CHECKPOINT" && exit 1)
	@mkdir -p checkpoints/v6/diagnostics
	python3 -m experiments.diagnostics.aleatoric_independence_probe \
		--checkpoint "$(or $(CHECKPOINT),checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt)" \
		--manifest "$(or $(MANIFEST),manifests/v6_corpus_stage_a_small_v1.json)" \
		--pdb-dir "$(or $(PDB_DIR),/tmp/dtie_pdb_cache)" \
		--pdb-local \
		--device $(or $(DEVICE),cpu) \
		$(if $(INCLUDE_GEOMETRY),--include-geometry,) \
		--json-out "$(or $(JSON_OUT),checkpoints/v6/diagnostics/aleatoric_independence_probe.json)"

train-v6-dehydron-rim-recovery: ## τ→rim cone recovery warm-start off production v6 (preserve disc occupancy)
	$(MAKE) train-v6-benchmark RUN_ID=$(or $(RUN_ID),dehydron_rim_recovery_v1) EPOCHS=$(or $(EPOCHS),12) \
		CORPUS=$(or $(CORPUS),v6_corpus_disc_target.json) \
		PDB_DIR=/tmp/dtie_pdb_cache \
		RESUME=$(or $(RESUME),checkpoints/v6/tokyo_eyes_v6.pt) \
		DEHYDRON_RIM_RECOVERY=1 \
		DEHYDRON_RIM_RECOVERY_LR=$(or $(LR),$(DEHYDRON_RIM_RECOVERY_LR),2e-5) \
		LEGACY_DISC_PROJECTION=0 \
		DISC_RADIAL_SOURCE=radial_depth \
		V2_TEACHER_DEPTH=0 \
		V2_TEACHER_EPISTEMIC=0 \
		DISC_LINE_THICKNESS_FLOOR=$(or $(DISC_LINE_THICKNESS_FLOOR),0.025) \
		DISC_SCATTER_INTERVAL=$(or $(DISC_SCATTER_INTERVAL),2) \
		SAVE_EPOCH_SNAPSHOTS=1 $(if $(USE_MLFLOW),USE_MLFLOW=1,) DEVICE=$(or $(DEVICE),cuda) \
		MAX_PROTEINS=$(or $(MAX_PROTEINS),3) \
		TRAINING_LOAD_FROM_PDB=1
	@echo "Dehydron rim recovery complete. Audit: make audit-dehydron-topology CHECKPOINT=checkpoints/v6/runs/$(or $(RUN_ID),dehydron_rim_recovery_v1)/v6_best.pt"

diagnose-angular-shell: ## Step-2 angular geometry test — radius shells + perm null (CHECKPOINT=...)
	@test -n "$(CHECKPOINT)" || (echo "Usage: make diagnose-angular-shell CHECKPOINT=checkpoints/v6/runs/shell_p2_hypmix_final/v6_best.pt" && exit 1)
	@mkdir -p checkpoints/v6/diagnostics
	python -m experiments.diagnostics.angular_shell_geometry_test \
		--checkpoint "$(CHECKPOINT)" \
		--structures "$(or $(STRUCTURES),11QE:A,4OBE:A,1IVO:A,4MNE:A)" \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cpu) \
		--n-perm $(or $(N_PERM),1000) \
		--output-json "$(or $(JSON_OUT),checkpoints/v6/diagnostics/angular_shell_$(shell basename $(CHECKPOINT) .pt).json)"

diagnose-epistemic-baseline: ## Step-4 go/no-go + B-factor residual baseline (CHECKPOINT=...)
	@mkdir -p checkpoints/v6/diagnostics
	python -m experiments.diagnostics.epistemic_decoupling_baseline \
		--checkpoint "$(or $(CHECKPOINT),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt)" \
		--structures "$(if $(STRUCTURES),$(STRUCTURES),11QE,4OBE,1IVO,4MNE)" \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cpu) \
		--out "$(or $(JSON_OUT),checkpoints/v6/diagnostics/epistemic_decoupling_baseline.json)"

promote-v6: ## Promote checkpoint to contract candidate (CHECKPOINT=host path under checkpoints/)
	@test -n "$(CHECKPOINT)" || (echo "Usage: make promote-v6 CHECKPOINT=checkpoints/v6/runs/.../v6_best.pt" && exit 1)
	$(SCIENCE_RUN) science python -m science.training.promote \
		--checkpoint-path "/app/$(CHECKPOINT)" \
		--checkpoint-id $(or $(CHECKPOINT_ID),tokyo_eyes_v6_candidate) \
		--status $(or $(STATUS),candidate) \
		$(if $(RUN_ID),--run-id "$(RUN_ID)",)

promote-v6-from-run: ## Promote v6_best from MLflow run (MLFLOW_RUN_ID=... from mlflow-ui)
	@test -n "$(MLFLOW_RUN_ID)" || (echo "Usage: make promote-v6-from-run MLFLOW_RUN_ID=<run_id>" && exit 1)
	$(SCIENCE_RUN) science python -m science.training.promote \
		--mlflow-run-id "$(MLFLOW_RUN_ID)" \
		--checkpoint-id $(or $(CHECKPOINT_ID),tokyo_eyes_v6_candidate) \
		--status $(or $(STATUS),candidate) \
		--mlflow-tracking-uri $(or $(MLFLOW_TRACKING_URI),http://localhost:5000)

promote-production-v6: ## Copy v6_best.pt → tokyo_eyes_v6.pt (SOURCE=checkpoints/v6/runs/.../v6_best.pt)
	@test -f "$(or $(SOURCE),checkpoints/v6/runs/full_hyp_moe_test/v6_best.pt)" || \
		(echo "Missing SOURCE checkpoint on host; set SOURCE=checkpoints/v6/runs/.../v6_best.pt" && exit 1)
	$(SCIENCE_RUN) science cp \
		"/app/$(or $(SOURCE),checkpoints/v6/runs/full_hyp_moe_test/v6_best.pt)" \
		/app/checkpoints/v6/tokyo_eyes_v6.pt
	@echo "✓ Promoted to checkpoints/v6/tokyo_eyes_v6.pt (restart science to reload)"

verify-v6-gnn: ## Verify hyperbolic_moe import + production checkpoint loads (science container)
	$(SCIENCE_RUN) science python -c "\
from science.dtie.v6.gnn.hyperbolic_moe import HyperbolicPrototypeGate; \
from science.contracts.model_registry import checkpoint_status, get_production_checkpoint_path; \
from science.dtie.v6.gnn.model import verify_v6_checkpoint; \
print('hyperbolic_moe OK'); \
print(checkpoint_status(get_production_checkpoint_path())); \
print(verify_v6_checkpoint())"

test-v6-gnn-integration: ## Run tests/test_v6_gnn_integration.py in science container
	$(SCIENCE_RUN) -v $(PWD)/tests:/app/tests science \
		sh -c "$(TEST_RUN_PREFIX) tests/test_v6_gnn_integration.py -v --tb=short"

mlflow-ui: ## Start MLflow 3 server (http://localhost:5000); also started by make up
	@mkdir -p mlflow-artifacts mlruns
	docker compose up -d mlflow
	@echo "✓ MLflow server at http://localhost:5000 (Postgres-backed registry)"

mlflow-ui-logs: ## Tail MLflow server container logs
	docker compose logs -f mlflow

mlflow-repair-store: ## Fix legacy local mlruns/ metadata (archive only; prefer Postgres server)
	@python3 experiments/training/v6/repair_mlflow_store.py --store mlruns

lifecycle-status: ## Print GNN lifecycle 360° status (host; needs PYTHONPATH)
	PYTHONPATH=. MLFLOW_TRACKING_URI=$(or $(MLFLOW_TRACKING_URI),http://localhost:5000) \
		python3 -c "from science.training.lifecycle import lifecycle_status; import json; print(json.dumps(lifecycle_status(), indent=2, default=str))"

promote-champion: ## Register checkpoint as MLflow champion + sync contract (LINEAGE=v6.5 CHECKPOINT=...)
	@test -n "$(CHECKPOINT)" || (echo "Usage: make promote-champion LINEAGE=v6.5 CHECKPOINT=checkpoints/..." && exit 1)
	PYTHONPATH=. MLFLOW_TRACKING_URI=$(or $(MLFLOW_TRACKING_URI),http://localhost:5000) \
		python3 -c "from science.training.lifecycle import register_and_alias; import json; \
print(json.dumps(register_and_alias(lineage_id='$(or $(LINEAGE),v6.5)', checkpoint_path='$(CHECKPOINT)', alias='champion', sync_contract=True), indent=2, default=str))"

promote-challenger: ## Register checkpoint as MLflow challenger (LINEAGE=v6.5 CHECKPOINT=...)
	@test -n "$(CHECKPOINT)" || (echo "Usage: make promote-challenger LINEAGE=v6.5 CHECKPOINT=checkpoints/..." && exit 1)
	PYTHONPATH=. MLFLOW_TRACKING_URI=$(or $(MLFLOW_TRACKING_URI),http://localhost:5000) \
		python3 -c "from science.training.lifecycle import register_and_alias; import json; \
print(json.dumps(register_and_alias(lineage_id='$(or $(LINEAGE),v6.5)', checkpoint_path='$(CHECKPOINT)', alias='challenger', sync_contract=True), indent=2, default=str))"

.PHONY: help up down kill build rebuild logs ps migrate psql dev dev-frontend test test-host test-docker test-integration test-integration-docker test-all lint format typecheck clean train-v6 train-v6-curriculum assess-v6 eval-v6 diagnose-embedding audit-dehydron-topology train-v6-dehydron-rim-recovery promote-v6 promote-v6-from-run promote-production-v6 verify-v6-gnn test-v6-gnn-integration mlflow-ui mlflow-ui-logs train-v6-theory-test-mlflow train-v6-mlflow-governance-smoke train-v6-stage-a-smoke train-v6-stage-a-curriculum test-stage-a-smoke auto-train-v6-corpus25 auto-train-v6-status sync-corpus-pins sync-p-curv-fixture seed-p-curv-fixture test-p-curv-01

# ---------------------------------------------------------------------------
# Docker Compose shortcuts
# ---------------------------------------------------------------------------

up: ## Start all services (detached)
	docker compose up -d
	@echo "✓ All services started. Check status with: make ps"

down: ## Stop all services gracefully
	docker compose down

kill: ## Force-stop and remove all containers and volumes
	docker compose down -v --remove-orphans
	docker rm -f tokyoeye_db tokyoeye_agent tokyoeye_science tokyoeye_mlflow_ui mlflow_ui 2>/dev/null || true
	@echo "✓ All containers and volumes removed"

build: ## Rebuild all images from scratch (no cache)
	docker compose build --no-cache

rebuild: ## Rebuild and restart all services
	docker compose up -d --build
	@echo "✓ Services rebuilt and restarted"

build-agent: ## Rebuild agent image only
	docker compose build --no-cache agent

build-science: ## Rebuild science image only (shared by science + mlflow)
	docker compose build --no-cache science mlflow

build-db: ## Rebuild database (normally not needed)
	docker compose build --no-cache db

# ---------------------------------------------------------------------------
# Individual services
# ---------------------------------------------------------------------------

up-db: ## Start only the database
	docker compose up -d db
	@docker compose logs -f db | grep -q "ready to accept connections" && echo "✓ Database ready"

up-agent: ## Start DB + agent (for API-only development)
	docker compose up -d db agent
	@echo "✓ Database and agent started. API available at http://localhost:8000"

up-science: ## Start DB + science container (for compute jobs)
	docker compose up -d db science
	@echo "✓ Database and science container ready"

# ---------------------------------------------------------------------------
# Logs & status
# ---------------------------------------------------------------------------

logs: ## Tail logs from all services
	docker compose logs -f

logs-agent: ## Tail agent logs only
	docker compose logs -f agent

logs-science: ## Tail science container logs only
	docker compose logs -f science

logs-mlflow: ## Tail MLflow UI logs only
	docker compose logs -f mlflow

logs-db: ## Tail database logs only
	docker compose logs -f db

ps: ## Show running containers and status
	docker compose ps

status: ## Show detailed service status
	@echo "=== Container Status ===" && \
	docker compose ps && \
	echo "" && \
	echo "=== Database Health ===" && \
	docker compose exec -T db pg_isready -U tokyoeye -d tokyoeye_dev || echo "Database not ready" && \
	echo "" && \
	echo "=== Agent Health ===" && \
	curl -s http://localhost:8000/health | jq . || echo "Agent not responding"

# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

migrate: ## Run database migrations
	docker compose exec agent python -c "import asyncio; from data.db import run_migrations; asyncio.run(run_migrations())"
	@echo "✓ Migrations complete"

psql: ## Open interactive psql shell
	docker compose exec db psql -U tokyoeye -d tokyoeye_dev

db-reset: ## Reset database (destructive - use with caution)
	@read -p "Are you sure? This will delete all data. Type 'yes' to confirm: " confirm && \
	[ "$$confirm" = "yes" ] && \
	docker compose down -v && \
	docker compose up -d db && \
	echo "✓ Database reset complete" || echo "Aborted"

# ---------------------------------------------------------------------------
# Development (local, no Docker)
# ---------------------------------------------------------------------------

dev: ## Run agent locally with uvicorn (hot-reload, requires local setup)
	uv run uvicorn agent.coordinator.app:app --reload --port 8000

dev-frontend: ## Run Discovery Cockpit frontend (Vite on :3000, proxies API to :8000)
	@if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then \
		echo "⚠ Agent not reachable on :8000 — run 'make up' or 'make dev' first"; \
	fi
	cd visualizer/frontend && npm run dev

dev-science: ## Run science scripts locally (requires CUDA/GPU setup)
	uv run python -m science.dtie.v5.orchestrator.pipeline

# ---------------------------------------------------------------------------
# Onboard contract codegen
# ---------------------------------------------------------------------------

contract-sync: ## Regenerate job_schema.json and frontend onboard types from contract
	uv run python -c "from science.compute.job_schema import write_job_schema; write_job_schema()"
	uv run python -m science.contracts.generate_typescript

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: ## Run unit tests (no integration tests)
	uv run pytest tests/ -v --tb=short -m "not integration"

test-host: ## Run unit tests with host python (no uv/docker; needs local deps)
	python3 -m pytest tests/ -v --tb=short -m "not integration"

test-integration: ## Run integration tests (requires running DB + TEST_DATABASE_URL)
	uv run pytest tests/ -v --tb=short -m "integration"

test-integration-docker: ## Run integration tests in science container (docker compose up)
	$(SCIENCE_RUN) -e TEST_DATABASE_URL=postgresql://tokyoeye:tokyoeye_dev_local@db:5432/tokyoeye_dev \
		-v $(PWD)/tests:/app/tests science \
		sh -c "$(TEST_RUN_PREFIX) tests/test_science_container_integration.py -m integration -v --tb=short"

test-docker: ## Run unit tests in science container (avoids host torch-scatter build)
	$(SCIENCE_RUN) -v $(PWD)/tests:/app/tests science \
		sh -c "$(TEST_RUN_PREFIX) tests/ -m 'not integration' -q --tb=short"

test-all: ## Run all tests (unit + integration)
	uv run pytest tests/ -v --tb=short

test-coverage: ## Run tests with coverage report
	uv run pytest tests/ -v --cov=agent --cov=science --cov=data --cov-report=html --tb=short

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint: ## Check code style and imports
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format code and fix issues
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Run mypy type checking
	uv run mypy science/ data/ agent/ --ignore-missing-imports

quality: lint typecheck ## Run all quality checks

# ---------------------------------------------------------------------------
# Cleanup and maintenance
# ---------------------------------------------------------------------------

clean: ## Remove local build artifacts, caches, and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -delete
	@echo "✓ Cleaned up artifacts"

docker-clean: ## Remove dangling Docker images and build cache
	docker image prune -f
	docker builder prune -f
	@echo "✓ Docker cleanup complete"

prune: clean docker-clean ## Deep cleanup of all build artifacts

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

pip-compile: ## Regenerate pinned requirements files from pyproject.toml
	uv pip compile pyproject.toml --extra agent -o requirements-agent.txt
	uv pip compile pyproject.toml --extra science -o requirements-science.txt
	@echo "✓ Requirements files regenerated"

# ---------------------------------------------------------------------------
# Info and help
# ---------------------------------------------------------------------------

project-hub: ## Print live Fix-1 biology roadmap from phase status JSON
	@python scripts/project_hub.py
	@echo ""
	@echo "Full hub: docs/PROJECT_HUB.md"

help: ## Show this help message
	@echo "Tokyo Eye Agenticpoincare — Make Commands" && \
	echo "" && \
	grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

version: ## Show version and dependency info
	@echo "Tokyo Eye Agenticpoincare v0.1.0" && \
	echo "" && \
	echo "Python:" && python --version && \
	echo "" && \
	echo "Docker:" && docker --version && \
	echo "Docker Compose:" && docker compose --version

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Pipeline audit CLI
# ---------------------------------------------------------------------------

audit-structure: ## Query audit events for a structure (STRUCTURE_ID=4obe)
	@test -n "$(STRUCTURE_ID)" || (echo "Usage: make audit-structure STRUCTURE_ID=4obe [SINCE=7d] [SEVERITY=warning]" && exit 1)
	python scripts/audit_pipeline.py --structure-id "$(STRUCTURE_ID)" \
		$(if $(SINCE),--since "$(SINCE)",) \
		$(if $(SEVERITY),--severity "$(SEVERITY)",)

shell-signal-gate: ## τ-rim shell + uncertainty gate (STRUCTURE_ID=4obe; STRICT=1 to fail)
	@test -n "$(STRUCTURE_ID)" || (echo "Usage: make shell-signal-gate STRUCTURE_ID=4obe [STRICT=1]" && exit 1)
	python -m experiments.diagnostics.shell_signal_ssot_gate \
		--structure-id "$(STRUCTURE_ID)" \
		$(if $(STRICT),--strict,)

audit-summary: ## Summarize pipeline audit events (optional SINCE=7d JOB_NAME=gnn_inference)
	python scripts/audit_pipeline.py --summary --limit 500 \
		$(if $(SINCE),--since "$(SINCE)",) \
		$(if $(JOB_NAME),--job-name "$(JOB_NAME)",) \
		$(if $(SEVERITY),--severity "$(SEVERITY)",)

audit-retention: ## Prune audit events older than RETENTION_DAYS (default 90)
	python scripts/audit_retention.py --days $(or $(RETENTION_DAYS),90) $(if $(DRY_RUN),--dry-run,)

# ---------------------------------------------------------------------------
# TokyoEye active trunk (implementation package: science/tokyo_eye/v8)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# TokyoEye active trunk (implementation package: science/tokyo_eye/v8)
# ---------------------------------------------------------------------------
.PHONY: train-tokyoeye-smoke train-tokyoeye-experiment export-tokyoeye-viewers
train-tokyoeye-smoke: ## Synthetic 1-epoch TokyoEye harness via science container (CUDA; no MLflow)
	@mkdir -p checkpoints/tokyoeye/runs
	MLFLOW_TRACKING_URI=http://mlflow:5000 $(SCIENCE_RUN) -e MLFLOW_TRACKING_URI science \
		python -m experiments.training.v8.run_v8_experiment \
		--smoke --no-mlflow \
		--device $(or $(DEVICE),cuda) \
		--run-name $(or $(RUN_NAME),tokyoeye_smoke) \
		--out-dir /app/checkpoints/tokyoeye/runs

train-tokyoeye-experiment: ## TokyoEye MLflow run (default Mode A: 4OBE:A; set MANIFEST= for Mode B)
	@mkdir -p checkpoints/tokyoeye/runs checkpoints/tokyoeye/pretrained
	@echo "Equiformer ckpt: $(or $(EQUIFORMER_CKPT),checkpoints/tokyoeye/pretrained/equiformer_v3_baseline.pt)"
	@echo "Data: PDB=$(or $(PDB),4OBE) CHAIN=$(or $(CHAIN),A) MANIFEST=$(or $(MANIFEST),)"
	MLFLOW_TRACKING_URI=$(or $(MLFLOW_URI),http://mlflow:5000) $(SCIENCE_RUN) -e MLFLOW_TRACKING_URI science \
		python -m experiments.training.v8.run_v8_experiment \
		--epochs $(or $(EPOCHS),10) \
		--device $(or $(DEVICE),cuda) \
		--run-name $(or $(RUN_NAME),tokyoeye_live) \
		--out-dir /app/checkpoints/tokyoeye/runs \
		--mlflow-uri $(or $(MLFLOW_URI),http://mlflow:5000) \
		--mlflow-experiment tokyoeye/equiformer-v3-moe/geometric/full-stack \
		--pdb $(or $(PDB),4OBE) \
		--chain $(or $(CHAIN),A) \
		--pdb-dir /tmp/dtie_pdb_cache \
		$(if $(MANIFEST),--manifest /app/$(MANIFEST),) \
		$(if $(EQUIFORMER_CKPT),--equiformer-ckpt /app/$(EQUIFORMER_CKPT),--equiformer-ckpt /app/checkpoints/tokyoeye/pretrained/equiformer_v3_baseline.pt) \
		$(if $(NO_MLFLOW),--no-mlflow,) \
		$(if $(EXPORT_VIEWERS),--export-viewers,) \
		$(if $(FREEZE_BACKBONE),--freeze-backbone,) \
		$(if $(NO_GRAPH_CACHE),--no-graph-cache,) \
		$(if $(GUMBEL_SCHEDULE),--gumbel-schedule $(GUMBEL_SCHEDULE),) \
		$(if $(CV_COEFF),--cv-coeff $(CV_COEFF),) \
		$(if $(MOE_QUOTA_COEFF),--moe-quota-coeff $(MOE_QUOTA_COEFF),) \
		$(if $(INIT_CKPT),--init-ckpt /app/$(INIT_CKPT),)
	@echo "MLflow → experiment tokyoeye/equiformer-v3-moe/geometric/full-stack  run $(or $(RUN_NAME),tokyoeye_live)"
	@echo "Ckpts → checkpoints/tokyoeye/runs/$(or $(RUN_NAME),tokyoeye_live)/"

export-tokyoeye-viewers: ## Export Poincaré disc HTML (PDB= or MANIFEST=; default 4OBE)
	@mkdir -p data/local_objects/gnn_viewer/tokyoeye
	$(SCIENCE_RUN) science \
		python -m experiments.training.v8.export_viewers \
		--ckpt /app/$(or $(CKPT),checkpoints/tokyoeye/runs/tokyoeye_4obe_a/tokyoeye_best.pt) \
		--pdb $(or $(PDB),4OBE) \
		--chain $(or $(CHAIN),A) \
		--pdb-dir /tmp/dtie_pdb_cache \
		--out-root /app/data/local_objects/gnn_viewer/tokyoeye \
		--device $(or $(DEVICE),cuda) \
		$(if $(MANIFEST),--manifest /app/$(MANIFEST),) \
		$(if $(DEHYDRON_WRAP_MAX),--dehydron-wrap-max $(DEHYDRON_WRAP_MAX),) \
		$(if $(FREEZE_BACKBONE),--freeze-backbone,)
	@echo "Viewers → data/local_objects/gnn_viewer/tokyoeye/"
