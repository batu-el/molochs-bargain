#!/bin/bash
# ---------------------------------------------------------------------------
# new_experiments/submit_experiment.sh
#
# End-to-end SLURM submission for the current experiment configuration:
#
#   - 5 models x 3 tasks                         (15 train shards in parallel)
#   - 4 trained methods: rft, tfb, dpo, kto      (config.TRAINED_METHODS)
#   - 1 epoch, batch=16                          (config.NUM_EPOCHS, PER_DEVICE_BATCH)
#   - Single fixed audience: 20 train people     (config.AUDIENCES, NUM_VOTERS_TRAIN)
#   - 4 base-vs-trained compete pairs            (compete.METHOD_PAIRS)
#   - q1/q2 misalignment probes per task         (probes.PROBES)
#
# Submits three SLURM steps with afterok dependencies, so cancelling any
# upstream step automatically cancels the dependents:
#
#   1. TRAIN  (15-shard array) -> run_train_shard.sh
#                                  -- prep + generate1 + build_train + train
#                                     (rft, tfb, dpo, kto) + generate22 + generate2
#   2. EVAL   (15-shard array, afterok TRAIN) -> run_eval_shard_per_model.sh
#                                  -- compete (per-model part) + probes (per-model part)
#   3. MERGE  (1 job, afterok EVAL) -> run_eval_merge.sh
#                                  -- consolidates parts -> competition.json + probes/*.csv
#
# Before this runs, we make sure the 20-person audience files exist. The
# materialization is idempotent and runs inline (~1s).
#
# Usage:
#   bash new_experiments/submit_experiment.sh                # full pipeline
#   bash new_experiments/submit_experiment.sh smoke          # --limit 64 everywhere
#   LIMIT=128 bash new_experiments/submit_experiment.sh      # custom prompt cap
#   MAX_CONC=16 BATCH_SIZE=16 bash new_experiments/submit_experiment.sh
#   THROTTLE="%5" bash new_experiments/submit_experiment.sh  # cap concurrent shards
#   TRAIN_TIME=24:00:00 EVAL_TIME=12:00:00 bash new_experiments/submit_experiment.sh
#
# Inspect logs:
#   tail -F slurm-outputs/ne-shard-<TRAIN_JOBID>_*.out      # train shards
#   tail -F slurm-outputs/ne-eval-pm-<EVAL_JOBID>_*.out     # eval shards
#   tail -F slurm-outputs/ne-eval-merge-<MERGE_JOBID>.out   # merge
#
# Cancel everything (cascade via dependencies):
#   scancel <TRAIN_JOBID>
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-full}"

# --- knobs (env-overridable) ---
LIMIT="${LIMIT:-}"
MAX_CONC="${MAX_CONC:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"          # matches config.PER_DEVICE_BATCH default
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"
EVAL_TIME="${EVAL_TIME:-12:00:00}"
ARRAY_RANGE="${ARRAY_RANGE:-0-14}"      # all 5 models x 3 tasks
THROTTLE="${THROTTLE:-}"                # e.g. "%5" to cap concurrent shards

case "$mode" in
    full)
        echo "[submit-experiment] FULL pipeline (20-person train audience, 4 methods, 1 epoch)"
        ;;
    smoke)
        LIMIT="${LIMIT:-64}"
        TRAIN_TIME="${TRAIN_TIME:-04:00:00}"
        EVAL_TIME="${EVAL_TIME:-02:00:00}"
        echo "[submit-experiment] SMOKE pipeline (limit=$LIMIT)"
        ;;
    *)
        echo "Unknown mode '$mode'. Use 'full' or 'smoke'." >&2
        exit 1
        ;;
esac

ARRAY_SPEC="${ARRAY_RANGE}${THROTTLE}"

# --- secrets sanity check ---
if [[ -f .env ]]; then
    grep -q '^TINKER_API_KEY=' .env || { echo "ERROR: TINKER_API_KEY missing from .env" >&2; exit 1; }
    grep -q '^OPENAI_API_KEY=' .env || { echo "ERROR: OPENAI_API_KEY missing from .env" >&2; exit 1; }
else
    echo "ERROR: no .env file found in repo root." >&2
    exit 1
fi

mkdir -p slurm-outputs slurm-errors

# --- 0. Materialize the fixed 20-person audience files (idempotent) ---
echo
echo "[submit-experiment] (0) materializing 20-person train audience ..."
PYTHON="${PYTHON:-python}"
"$PYTHON" -m new_experiments.scripts.build_audiences

# --- 1. TRAIN array (5 models x 3 tasks) ---
TRAIN_EXPORTS="ALL"
[[ -n "$LIMIT"      ]] && TRAIN_EXPORTS="${TRAIN_EXPORTS},LIMIT=${LIMIT}"
[[ -n "$MAX_CONC"   ]] && TRAIN_EXPORTS="${TRAIN_EXPORTS},MAX_CONC=${MAX_CONC}"
[[ -n "$BATCH_SIZE" ]] && TRAIN_EXPORTS="${TRAIN_EXPORTS},BATCH_SIZE=${BATCH_SIZE}"

echo
echo "[submit-experiment] (1) submitting TRAIN array (${ARRAY_SPEC}, --time=${TRAIN_TIME}) ..."
TRAIN_JOB_ID=$(sbatch \
    --time="$TRAIN_TIME" \
    --array="$ARRAY_SPEC" \
    --export="${TRAIN_EXPORTS}" \
    --parsable \
    new_experiments/run_train_shard.sh)
echo "    TRAIN_JOB_ID=${TRAIN_JOB_ID}"

# --- 2. EVAL array, depends on TRAIN array completing ---
EVAL_EXPORTS="ALL"
[[ -n "$LIMIT" ]] && EVAL_EXPORTS="${EVAL_EXPORTS},LIMIT=${LIMIT}"

echo
echo "[submit-experiment] (2) submitting EVAL array (afterok:${TRAIN_JOB_ID}, --time=${EVAL_TIME}) ..."
EVAL_JOB_ID=$(sbatch \
    --time="$EVAL_TIME" \
    --array="$ARRAY_SPEC" \
    --dependency="afterok:${TRAIN_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    --export="${EVAL_EXPORTS}" \
    --parsable \
    new_experiments/run_eval_shard_per_model.sh)
echo "    EVAL_JOB_ID=${EVAL_JOB_ID}"

# --- 3. MERGE, depends on EVAL array completing ---
echo
echo "[submit-experiment] (3) submitting MERGE (afterok:${EVAL_JOB_ID}) ..."
MERGE_JOB_ID=$(sbatch \
    --dependency="afterok:${EVAL_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    --parsable \
    new_experiments/run_eval_merge.sh)
echo "    MERGE_JOB_ID=${MERGE_JOB_ID}"

# --- summary ---
echo
echo "============================================================"
echo "  submitted experiment pipeline"
echo "    TRAIN array : ${TRAIN_JOB_ID}    (${ARRAY_SPEC} = up to 15 shards)"
echo "    EVAL  array : ${EVAL_JOB_ID}    (depends on afterok:${TRAIN_JOB_ID})"
echo "    MERGE       : ${MERGE_JOB_ID}    (depends on afterok:${EVAL_JOB_ID})"
echo "============================================================"
echo
echo "Tail logs:"
echo "  tail -F slurm-outputs/ne-shard-${TRAIN_JOB_ID}_*.out"
echo "  tail -F slurm-outputs/ne-eval-pm-${EVAL_JOB_ID}_*.out"
echo "  tail -F slurm-outputs/ne-eval-merge-${MERGE_JOB_ID}.out"
echo
echo "Inspect job state:"
echo "  squeue -u \$USER"
echo "  sacct -j ${TRAIN_JOB_ID},${EVAL_JOB_ID},${MERGE_JOB_ID} --format=JobID,State,ExitCode,Elapsed"
echo
echo "Cancel a single train shard (0..14):"
echo "  scancel ${TRAIN_JOB_ID}_<task_id>"
echo "Cancel everything (eval + merge auto-cancel via dependency):"
echo "  scancel ${TRAIN_JOB_ID} ${EVAL_JOB_ID} ${MERGE_JOB_ID}"
