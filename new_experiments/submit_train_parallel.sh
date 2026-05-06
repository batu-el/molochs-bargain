#!/bin/bash
# ---------------------------------------------------------------------------
# new_experiments/submit_train_parallel.sh
#
# Submits a 15-task SLURM job array (5 models x 3 tasks) where each array
# task runs the full per-(model, task) training pipeline (prep, generate1,
# build_train, train, generate22, generate2) in parallel.
#
# After the array finishes, run new_experiments/submit_eval.sh to launch
# the per-task compete + probes shards.
#
# Usage:
#   bash new_experiments/submit_train_parallel.sh                # full run
#   bash new_experiments/submit_train_parallel.sh smoke          # --limit 64
#   LIMIT=128 bash new_experiments/submit_train_parallel.sh      # custom cap
#   MAX_CONC=16 bash new_experiments/submit_train_parallel.sh    # throttle
#
# Inspect logs:
#   tail -F slurm-outputs/ne-shard-<JOBID>_*.out
#   sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-full}"

LIMIT="${LIMIT:-}"
MAX_CONC="${MAX_CONC:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
ARRAY_RANGE="${ARRAY_RANGE:-0-14}"          # default: all 15 (model, task) pairs
THROTTLE="${THROTTLE:-}"                    # e.g. "%5" to cap at 5 concurrent

case "$mode" in
    full)
        echo "[submit-parallel] full pipeline -> 15 (model, task) shards"
        ;;
    smoke)
        LIMIT="${LIMIT:-64}"
        TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
        echo "[submit-parallel] smoke pipeline (limit=$LIMIT) -> 15 shards"
        ;;
    *)
        echo "Unknown mode '$mode'. Use 'full' or 'smoke'." >&2
        exit 1
        ;;
esac

EXPORTS="ALL"
[[ -n "$LIMIT"      ]] && EXPORTS="${EXPORTS},LIMIT=${LIMIT}"
[[ -n "$MAX_CONC"   ]] && EXPORTS="${EXPORTS},MAX_CONC=${MAX_CONC}"
[[ -n "$BATCH_SIZE" ]] && EXPORTS="${EXPORTS},BATCH_SIZE=${BATCH_SIZE}"

ARRAY_SPEC="${ARRAY_RANGE}${THROTTLE}"

echo "  --time=${TIME_LIMIT}"
echo "  --array=${ARRAY_SPEC}"
echo "  --export=${EXPORTS}"
echo ""

# Sanity check that .env has the required keys before submission.
if [[ -f .env ]]; then
    grep -q '^TINKER_API_KEY=' .env || { echo "ERROR: TINKER_API_KEY missing from .env" >&2; exit 1; }
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
    new_experiments/run_train_shard.sh)

echo "Submitted job array ${JOB_ID} (${ARRAY_SPEC} = up to 15 tasks)."
echo
echo "Tail per-shard logs with:"
echo "  tail -F slurm-outputs/ne-shard-${JOB_ID}_*.out"
echo "Cancel a single shard:"
echo "  scancel ${JOB_ID}_<task_id>     # e.g. ${JOB_ID}_0 .. ${JOB_ID}_14"
echo "Cancel the whole array:"
echo "  scancel ${JOB_ID}"
echo
echo "After ALL 15 shards complete, launch evaluation with:"
echo "  bash new_experiments/submit_eval.sh ${mode}"
