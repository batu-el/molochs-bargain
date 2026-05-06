#!/bin/bash
#SBATCH --job-name=ne-eval-pm
#SBATCH --partition=owners
#SBATCH --account=jamesz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# No GPU - both stages are pure OpenAI API calls (gpt-4o-mini voters and probes).
#SBATCH --time=12:00:00
#SBATCH --output=slurm-outputs/ne-eval-pm-%A_%a.out
#SBATCH --error=slurm-errors/ne-eval-pm-%A_%a.err
#SBATCH --array=0-14

# ---------------------------------------------------------------------------
# new_experiments/run_eval_shard_per_model.sh  (SLURM job array, 5 models x 3 tasks)
#
# Per-(model, task) version of run_eval_shard.sh: 15 array tasks instead of 3,
# so each (model, task) cell is its own job. Each shard runs:
#
#   1. compete  - pairwise voter competition for THIS model on THIS task
#                 over the 3 method pairs (base/rft, base/tfb, rft/tfb), using
#                 gpt-4o-mini voters. Writes a per-model part file:
#                   res/{task}/competition_parts/{model}.json
#   2. probes   - q1/q2 misalignment probes for THIS model on THIS task using
#                 gpt-4o-mini (override via config.PROBE_MODEL_NAME). Writes:
#                   res/probes/{task}_{qid}_parts/{model}.csv
#
# After ALL 15 shards complete, run new_experiments/run_eval_merge.sh once
# to consolidate the per-model part files into the canonical paths
# (res/{task}/competition.json and res/probes/{task}_{qid}.csv) that the
# downstream analysis notebooks read. Use submit_eval_parallel.sh to
# submit the array AND the dependent merge job in one go.
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p slurm-outputs slurm-errors

# ----- Conda / env setup (mirrors run_eval_shard.sh) -----
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

# ----- Map array task ID -> (model, task) -----
# Order MUST match config.MODELS so logs are easy to cross-reference and the
# 15-cell layout is identical to run_train_shard.sh.
MODELS_LIST=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-32B"
    "openai/gpt-oss-20b"
)
TASKS_LIST=("task_elections" "task_sales" "task_sm")

NMODELS=${#MODELS_LIST[@]}
NTASKS=${#TASKS_LIST[@]}
TOTAL=$(( NMODELS * NTASKS ))

TID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TID < 0 || TID >= TOTAL )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TID} out of range [0, ${TOTAL})" >&2
    exit 1
fi

MODEL_IDX=$(( TID / NTASKS ))
TASK_IDX=$((  TID % NTASKS ))
MODEL="${MODELS_LIST[$MODEL_IDX]}"
TASK="${TASKS_LIST[$TASK_IDX]}"

echo "============================================================"
echo "[eval-pm ${TID}/${TOTAL}]  task=${TASK}  model=${MODEL}"
echo "  LIMIT=${LIMIT:-<full>}"
echo "============================================================"

run_step() {
    local name="$1"; shift
    echo "------------------------------------------------------------"
    echo "[eval-pm ${TID}] step=${name}  cmd=$*"
    echo "------------------------------------------------------------"
    "$@"
}

# ----- Per-(task, model) evaluation pipeline -----
run_step compete \
    "$PYTHON" -m new_experiments.src.compete \
        --task "$TASK" --model "$MODEL" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}

run_step probes \
    "$PYTHON" -m new_experiments.src.probes \
        --task "$TASK" --model "$MODEL" ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}

echo "============================================================"
echo "[eval-pm ${TID}] DONE  task=${TASK}  model=${MODEL}"
echo "============================================================"
