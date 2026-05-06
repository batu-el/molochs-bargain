#!/bin/bash
# ---------------------------------------------------------------------------
# new_experiments/submit_eval.sh
#
# Submits a 3-task SLURM job array (one per high-level task) where each
# array task runs BOTH compete and probes for that task in a single shard.
# Run this AFTER submit_train_parallel.sh has produced every (model, task)
# step2 output under new_experiments/res/.
#
# Usage:
#   bash new_experiments/submit_eval.sh                 # full
#   bash new_experiments/submit_eval.sh smoke           # --limit 64
#   LIMIT=128 bash new_experiments/submit_eval.sh       # custom cap
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-full}"

LIMIT="${LIMIT:-}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
ARRAY_RANGE="${ARRAY_RANGE:-0-2}"          # default: all 3 tasks
THROTTLE="${THROTTLE:-}"                   # e.g. "%2" to cap concurrent shards

case "$mode" in
    full)
        echo "[submit-eval] full -> 3 (task) shards: compete + probes each"
        ;;
    smoke)
        LIMIT="${LIMIT:-64}"
        TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
        echo "[submit-eval] smoke (limit=$LIMIT) -> 3 shards"
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

JOB_ID=$(sbatch \
    --time="$TIME_LIMIT" \
    --array="$ARRAY_SPEC" \
    --export="${EXPORTS}" \
    --parsable \
    new_experiments/run_eval_shard.sh)

echo "Submitted eval array ${JOB_ID} (${ARRAY_SPEC} = up to 3 tasks)."
echo
echo "Tail per-task logs with:"
echo "  tail -F slurm-outputs/ne-eval-${JOB_ID}_*.out"
echo "Cancel a single task:"
echo "  scancel ${JOB_ID}_<task_id>     # 0=elections, 1=sales, 2=sm"
echo "Cancel the whole array:"
echo "  scancel ${JOB_ID}"
