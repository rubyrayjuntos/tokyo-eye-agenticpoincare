#!/usr/bin/env bash
# Sequential v6 training rounds: P4 epistemic → P4 tighten → ResidueStage2 refresh.
# Warm-start chain: residue_stage2_v1 → p4_epi_rs2_v1 → p4_epi_rs2_v2 → rs2_post_p4_v1
#
# Resume mid-chain (e.g. after round 1 finished without v6_best.pt):
#   START_ROUND=2 bash experiments/training/v6/run_exhaustive_curriculum.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="checkpoints/v6/runs/exhaustive_curriculum_${STAMP}.log"
mkdir -p checkpoints/v6/runs mlruns pdb_cache

exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "Exhaustive v6 curriculum — started $(date -Iseconds)"
echo "Log: $LOG"
echo "START_ROUND=${START_ROUND:-1}"
echo "================================================================"

BASE_RS2="checkpoints/v6/runs/residue_stage2_v1/v6_phase1_12prot.pt"
CORPUS="v6_corpus_stage_a_small_v1.json"
DEVICE="${DEVICE:-cuda}"
START_ROUND="${START_ROUND:-1}"

# Pick best available checkpoint for a run dir (P4 may finish with phase weights only).
resolve_run_checkpoint() {
  local run_id="$1"
  local d="checkpoints/v6/runs/${run_id}"
  local candidates=(
    "${d}/v6_best.pt"
    "${d}/v6_best_disc.pt"
    "${d}/v6_phase4_12prot.pt"
    "${d}/v6_phase1_12prot.pt"
    "${d}/phase_4_epistemic-depth_decoupling_(b-factor_residual,_staged_λ).pt"
    "${d}/phase_4_epistemic-depth_decoupling_(b-factor_residual).pt"
    "${d}/residuestage2.pt"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  # Last resort: newest phase_*.pt in run dir
  local latest
  latest="$(ls -t "${d}"/phase_*.pt 2>/dev/null | head -1 || true)"
  if [[ -n "$latest" && -f "$latest" ]]; then
    echo "$latest"
    return 0
  fi
  return 1
}

_assess() {
  local run_id="$1"
  local host_ckpt
  if ! host_ckpt="$(resolve_run_checkpoint "$run_id")"; then
    echo "WARN: no checkpoint to assess for ${run_id} (skipping)"
    return 0
  fi
  echo "Assessing ${run_id} from ${host_ckpt}"
  make assess-v6 \
    CHECKPOINT="/app/${host_ckpt}" \
    CORPUS="$CORPUS" \
    MAX_PROTEINS=12 \
    DEVICE="$DEVICE" \
    OUTPUT="/app/checkpoints/v6/runs/${run_id}/assess.json" \
    || echo "WARN: assess failed for ${run_id} (continuing)"
}

if [[ "$START_ROUND" -le 1 ]]; then
  if [[ ! -f "$BASE_RS2" ]]; then
    echo "Missing warm-start: $BASE_RS2" >&2
    exit 1
  fi
  echo ""
  echo "=== Round 1/3: P4 staged epistemic (30 ep) from residue_stage2_v1 ==="
  make train-v6-p4-from-residue-stage2 \
    RUN_ID=p4_epi_rs2_v1 \
    RESUME="$BASE_RS2" \
    EPOCHS=30 \
    DEVICE="$DEVICE"
  _assess p4_epi_rs2_v1
fi

if [[ "$START_ROUND" -le 2 ]]; then
  P4_R1_CKPT="$(resolve_run_checkpoint p4_epi_rs2_v1)" || {
    echo "Missing p4_epi_rs2_v1 checkpoint for round 2" >&2
    exit 1
  }
  echo ""
  echo "=== Round 2/3: P4 uncertainty tighten (20 ep) from ${P4_R1_CKPT} ==="
  make train-v6-p4-from-residue-stage2 \
    RUN_ID=p4_epi_rs2_v2 \
    RESUME="$P4_R1_CKPT" \
    EPOCHS=20 \
    P4_EPISTEMIC_LR=5e-5 \
    EPISTEMIC_BF_ALIGN_COEFF=0.28 \
    EPISTEMIC_SASA_PEN_COEFF=0.08 \
    DEVICE="$DEVICE"
  _assess p4_epi_rs2_v2
fi

if [[ "$START_ROUND" -le 3 ]]; then
  P4_R2_CKPT="$(resolve_run_checkpoint p4_epi_rs2_v2)" || {
    echo "Missing p4_epi_rs2_v2 checkpoint for round 3" >&2
    exit 1
  }
  echo ""
  echo "=== Round 3/3: ResidueStage2 pipeline refresh (20 ep) from ${P4_R2_CKPT} ==="
  make train-v6-residue-stage2 \
    RUN_ID=rs2_post_p4_v1 \
    RESUME="$P4_R2_CKPT" \
    EPOCHS=20 \
    RESIDUE_STAGE2_LR=5e-5 \
    DEVICE="$DEVICE"
  _assess rs2_post_p4_v1
fi

FINAL_CKPT="$(resolve_run_checkpoint rs2_post_p4_v1)" || FINAL_CKPT=""
echo ""
echo "=== Export dual HTML viewers for final checkpoint ==="
if [[ -n "$FINAL_CKPT" ]]; then
  make export-corpus-viewers \
    CHECKPOINT="$FINAL_CKPT" \
    CORPUS="manifests/${CORPUS}" \
    DEVICE="$DEVICE" \
    || echo "WARN: viewer export failed (non-fatal)"
else
  echo "WARN: no final checkpoint for viewer export"
fi

echo ""
echo "================================================================"
echo "Exhaustive curriculum complete — $(date -Iseconds)"
echo "Final run: checkpoints/v6/runs/rs2_post_p4_v1"
echo "================================================================"
