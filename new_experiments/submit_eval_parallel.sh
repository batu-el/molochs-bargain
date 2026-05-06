#!/bin/bash
# ---------------------------------------------------------------------------
# new_experiments/submit_eval_parallel.sh
#
# Per-(model, task) version of submit_eval.sh. Submits a 15-task SLURM job
# array (5 models x 3 tasks) where each array task runs BOTH compete and
# probes for that single (model, task) cell, writing to per-model part files.
# After the array completes, a single dependent merge job consolidates the
# part files into the canonical res/{task}/competition.json and
# res/probes/{task}_{qid}.csv outputs.
#
# Use this instead of submit_eval.sh when you want maximum parallelism
# (15 jobs in parallel instead of 3) - 5x more shards, same total work.
#
# Usage:
#   bash new_experiments/submit_eval_parallel.sh                # full run
#   bash new_experiments/submit_eval_parallel.sh smoke          # --limit 64
#   LIMIT=128 bash new_experiments/submit_eval_parallel.sh      # custom cap
#   THROTTLE="%5" bash new_experiments/submit_eval_parallel.sh  # cap concurrent shards
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-full}"

LIMIT="${LIMIT:-}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
ARRAY_RANGE="${ARRAY_RANGE:-0-14}"          # default: all 15 (model, task) pairs
THROTTLE="${THROTTLE:-}"                    # e.g. "%5" to cap at 5 concurrent

case "$mode" in
    full)
        echo "[submit-eval-parallel] full -> 15 (model, task) shards: compete + probes each"
        ;;
    smoke)
        LIMIT="${LIMIT:-64}"
        TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
        echo "[submit-eval-parallel] smoke (limit=$LIMIT) -> 15 shards"
        ;;
    *)
        echo "Unknown mode '$mode'. Use 'full' or 'smoke'." >&2
        exit 1
        ;;
esac

EXPORTS="ALL"
[[ -n "$LIMIT" ]] && EXPORTS="${EXPORTS},LIMIT=${LIMIT}"

ARRAY_SPEC="${ARRAY_RANGE}${THROTTLE}"

echo "  --time=${TIME_LIMIT}"
echo "  --array=${ARRAY_SPEC}"
echo "  --export=${EXPORTS}"
echo ""

if [[ -f .env ]]; then
    grep -q '^OPENAI_API_KEY=' .env || { echo "ERROR: OPENAI_API_KEY missing from .env" >&2; exit 1; }
else
    echo "WARNING: no .env file found in repo root."
fi

mkdir -p slurm-outputs slurm-errors

ARRAY_JOB_ID=$(sbatch \
    --time="$TIME_LIMIT" \
    --array="$ARRAY_SPEC" \
    --export="${EXPORTS}" \
    --parsable \
    new_experiments/run_eval_shard_per_model.sh)

echo "Submitted per-model eval array ${ARRAY_JOB_ID} (${ARRAY_SPEC} = up to 15 tasks)."

# Dependent merge job: runs once after every array element succeeds.
MERGE_JOB_ID=$(sbatch \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --parsable \
    new_experiments/run_eval_merge.sh)

echo "Submitted merge job ${MERGE_JOB_ID} (depends on afterok:${ARRAY_JOB_ID})."
echo
echo "Tail per-shard logs with:"
echo "  tail -F slurm-outputs/ne-eval-pm-${ARRAY_JOB_ID}_*.out"
echo "Tail merge log with:"
echo "  tail -F slurm-outputs/ne-eval-merge-${MERGE_JOB_ID}.out"
echo "Cancel a single shard:"
echo "  scancel ${ARRAY_JOB_ID}_<task_id>     # 0..14, see run_eval_shard_per_model.sh for layout"
echo "Cancel the array (and the dependent merge will be auto-cancelled):"
echo "  scancel ${ARRAY_JOB_ID} ${MERGE_JOB_ID}"
