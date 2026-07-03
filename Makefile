# Configure make to use bash instead of sh for better error handling
SHELL := /bin/bash -o pipefail

# Map container UID/GID to host so bind mounts (mlruns, checkpoints) are writable
DOCKER_USER := $(shell id -u):$(shell id -g)
SCIENCE_RUN := docker compose run --rm --user $(DOCKER_USER)
STAGE_A_MAX_RESIDUES := $(shell uv run python -c "from science.training.corpus_governance import STAGE_A_MAX_RESIDUES; print(STAGE_A_MAX_RESIDUES)")
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
		--mlflow-uri file:/app/mlruns \
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
	$(SCIENCE_RUN) science python -m experiments.training.v6.launch_training \
		--corpus /app/manifests/$(or $(CORPUS),v6_corpus_benchmark.json) \
		--output-dir /app/checkpoints/v6/runs/$(or $(RUN_ID),benchmark_$(shell date +%Y%m%d_%H%M%S)) \
		--pdb-dir $(or $(PDB_DIR),$(BENCHMARK_PDB_CONTAINER)) \
		--device $(or $(DEVICE),cuda) \
		--phase $(or $(STAGE),1) \
		--epochs $(or $(EPOCHS),3) \
		--max-proteins $(or $(MAX_PROTEINS),8) \
		--max-residues $(or $(MAX_RESIDUES),600) \
		--no-corpus-cache \
		$(if $(USE_MLFLOW),--mlflow-uri file:/app/mlruns,) \
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
		--mlflow-uri file:/app/mlruns \
		--resume /app/$(or $(RESUME),checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt)
	@echo "Smoke complete. Verify with: STAGE_A_SMOKE_RUN_ID=<run_id> make test-stage-a-smoke"

test-stage-a-smoke: ## Assert P_STAGE_A_SMOKE on STAGE_A_SMOKE_RUN_ID MLflow run
	@test -n "$$STAGE_A_SMOKE_RUN_ID" || (echo "Set STAGE_A_SMOKE_RUN_ID to the smoke run id" && exit 1)
	MLFLOW_TRACKING_URI=$(or $(MLFLOW_TRACKING_URI),file:./mlruns) \
	MLFLOW_ALLOW_FILE_STORE=true \
	STAGE_A_SMOKE_FULL=$${STAGE_A_SMOKE_FULL:-0} \
	uv run pytest tests/test_stage_a_integration_smoke.py::test_stage_a_smoke_mlflow_run_from_env -v

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

diagnose-embedding: ## Disc occupancy audit — full tables (CHECKPOINT=... or COMPARE_CHECKPOINTS='a b c')
	@test -n "$(or $(CHECKPOINT),$(COMPARE_CHECKPOINTS))" || (echo "Usage: make diagnose-embedding CHECKPOINT=checkpoints/v6/tokyo_eyes_v6.pt STRUCTURES=11QE:A" && exit 1)
	@mkdir -p pdb_cache checkpoints/v6/diagnostics
	python -m experiments.diagnostics.embedding_occupancy_audit \
		$(if $(COMPARE_CHECKPOINTS),--compare-checkpoints $(COMPARE_CHECKPOINTS),--checkpoint "$(CHECKPOINT)") \
		--structures "$(or $(STRUCTURES),11QE:A,4OBE:A,1IVO:A,4MNE:A)" \
		--pdb-dir /tmp/dtie_pdb_cache \
		--device $(or $(DEVICE),cpu) \
		$(if $(JSON_OUT),--json-out "$(JSON_OUT)",) \
		$(if $(EXPORT_SCATTER),--export-scatter "$(EXPORT_SCATTER)",)

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
		--mlflow-tracking-uri file:/app/mlruns

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

mlflow-ui: ## Open MLflow UI for training runs (http://localhost:5000)
	@mkdir -p mlruns
	$(SCIENCE_RUN) -p 5000:5000 science \
		mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:/app/mlruns

.PHONY: help up down kill build rebuild logs ps migrate psql dev dev-frontend test test-host test-docker test-integration test-integration-docker test-all lint format typecheck clean train-v6 train-v6-curriculum assess-v6 eval-v6 diagnose-embedding promote-v6 promote-v6-from-run promote-production-v6 verify-v6-gnn test-v6-gnn-integration mlflow-ui train-v6-theory-test-mlflow train-v6-mlflow-governance-smoke train-v6-stage-a-smoke test-stage-a-smoke sync-corpus-pins sync-p-curv-fixture seed-p-curv-fixture test-p-curv-01

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
	docker rm -f tokyoeye_db tokyoeye_agent tokyoeye_science 2>/dev/null || true
	@echo "✓ All containers and volumes removed"

build: ## Rebuild all images from scratch (no cache)
	docker compose build --no-cache

rebuild: ## Rebuild and restart all services
	docker compose up -d --build
	@echo "✓ Services rebuilt and restarted"

build-agent: ## Rebuild agent image only
	docker compose build --no-cache agent

build-science: ## Rebuild science image only
	docker compose build --no-cache science

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

audit-summary: ## Summarize pipeline audit events (optional SINCE=7d JOB_NAME=gnn_inference)
	python scripts/audit_pipeline.py --summary --limit 500 \
		$(if $(SINCE),--since "$(SINCE)",) \
		$(if $(JOB_NAME),--job-name "$(JOB_NAME)",) \
		$(if $(SEVERITY),--severity "$(SEVERITY)",)

audit-retention: ## Prune audit events older than RETENTION_DAYS (default 90)
	python scripts/audit_retention.py --days $(or $(RETENTION_DAYS),90) $(if $(DRY_RUN),--dry-run,)
