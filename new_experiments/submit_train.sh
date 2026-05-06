#!/bin/bash
# ---------------------------------------------------------------------------
# new_experiments/submit_train.sh
#
# Submits the full new_experiments pipeline (5 models x 3 tasks x {base, rft,
# tfb}) as a SINGLE sbatch job that runs every stage sequentially. Mirrors
# the convention used by ../submit_train.sh.
#
# Usage:
#   bash new_experiments/submit_train.sh                       # full run (~$1100, ~30-50 h)
#   bash new_experiments/submit_train.sh smoke                 # --limit 64 (~$70, ~3-6 h)
#   LIMIT=128 bash new_experiments/submit_train.sh             # custom cap
#   ONLY=probes bash new_experiments/submit_train.sh           # only one stage
#   SKIP=generate1,generate22 bash new_experiments/submit_train.sh
#
# Resume after a partial run:
#   SKIP=prep,generate1,build_train bash new_experiments/submit_train.sh
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:-full}"

# Default knobs
LIMIT="${LIMIT:-}"
ONLY="${ONLY:-}"
SKIP="${SKIP:-}"
MAX_CONC="${MAX_CONC:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"

case "$mode" in
    full)
        echo "[submit] full pipeline -> sbatch new_experiments/run_train.sh"
        ;;
    smoke)
        LIMIT="${LIMIT:-64}"
        TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
        echo "[submit] smoke pipeline (limit=$LIMIT) -> sbatch new_experiments/run_train.sh"
        ;;
    *)
        echo "Unknown mode '$mode'. Use 'full' or 'smoke'."
        exit 1
        ;;
esac

# ----- Build --export string for sbatch -----
EXPORTS="ALL"
[[ -n "$LIMIT"      ]] && EXPORTS="${EXPORTS},LIMIT=${LIMIT}"
[[ -n "$ONLY"       ]] && EXPORTS="${EXPORTS},ONLY=${ONLY}"
[[ -n "$SKIP"       ]] && EXPORTS="${EXPORTS},SKIP=${SKIP}"
[[ -n "$MAX_CONC"   ]] && EXPORTS="${EXPORTS},MAX_CONC=${MAX_CONC}"
[[ -n "$BATCH_SIZE" ]] && EXPORTS="${EXPORTS},BATCH_SIZE=${BATCH_SIZE}"

echo "  --time=${TIME_LIMIT}"
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
    --export="${EXPORTS}" \
    --parsable \
    new_experiments/run_train.sh)

echo "Submitted as job ${JOB_ID}."
echo "Tail logs with:  tail -F slurm-outputs/ne-${JOB_ID}.out slurm-errors/ne-${JOB_ID}.err"
echo "Cancel with:     scancel ${JOB_ID}"
