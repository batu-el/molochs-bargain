#!/bin/bash
#SBATCH --job-name=ne-eval
#SBATCH --partition=owners
#SBATCH --account=jamesz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# No GPU - both stages are pure OpenAI API calls (gpt-4o-mini voters and probes).
#SBATCH --time=12:00:00
#SBATCH --output=slurm-outputs/ne-eval-%A_%a.out
#SBATCH --error=slurm-errors/ne-eval-%A_%a.err
#SBATCH --array=0-2

# ---------------------------------------------------------------------------
# new_experiments/run_eval_shard.sh  (SLURM job array, 3 tasks)
#
# One array task per high-level task (task_elections / task_sales / task_sm).
# Each shard runs BOTH compete and probes for its task in a single job:
#
#   1. compete  - pairwise voter competition over all 5 models x 4 base-vs-
#                 trained method pairs (base vs rft / tfb / dpo / kto) using
#                 gpt-4o-mini, evaluated under the single fixed train
#                 audience (the same N people that scored the training
#                 rollouts in generate1).
#   2. probes   - q1/q2 misalignment probes over all 5 models x {base, rft,
#                 tfb, dpo, kto} using gpt-4o-mini (override via
#                 config.PROBE_MODEL_NAME).
#
# Both stages iterate internally over every model in config.MODELS, so a
# per-task split gives a clean ~3x speedup without further coordination.
# Submit via new_experiments/submit_eval.sh.
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p slurm-outputs slurm-errors

# ----- Conda / env setup (mirrors run_train_shard.sh) -----
CONDA_HOME="${CONDA_HOME:-/scratch/users/batuel/miniconda3}"
CONDA_ENV="${CONDA_ENV:-tinker311}"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="${HF_HOME:-/scratch/users/batuel/}"
export HF_HUB_DISABLE_XET=1

export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-30}"
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-3}"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

REPO_ROOT="$(pwd -P)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set in .env}"

LIMIT="${LIMIT:-}"
PYTHON="${PYTHON:-python}"

LIMIT_ARG=()
[[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")

TASKS_LIST=("task_elections" "task_sales" "task_sm")
NTASKS=${#TASKS_LIST[@]}

TID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TID < 0 || TID >= NTASKS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TID} out of range [0, ${NTASKS})" >&2
    exit 1
fi

TASK="${TASKS_LIST[$TID]}"

echo "============================================================"
echo "[eval ${TID}/${NTASKS}]  task=${TASK}"
echo "  LIMIT=${LIMIT:-<full>}"
echo "============================================================"

run_step() {
    local name="$1"; shift
    echo "------------------------------------------------------------"
    echo "[eval ${TID}] step=${name}  cmd=$*"
    echo "------------------------------------------------------------"
    "$@"
}

# ----- Per-task evaluation pipeline (compete + probes in one job) -----
run_step compete \
    "$PYTHON" -m new_experiments.src.compete --task "$TASK" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}

run_step probes \
    "$PYTHON" -m new_experiments.src.probes --task "$TASK" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}

echo "============================================================"
echo "[eval ${TID}] DONE  task=${TASK}"
echo "============================================================"
