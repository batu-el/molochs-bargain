#!/bin/bash
#SBATCH --job-name=ne-shard
#SBATCH --partition=owners
#SBATCH --account=jamesz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# No GPU - all sampling and training run on Tinker cloud.
#SBATCH --time=24:00:00
#SBATCH --output=slurm-outputs/ne-shard-%A_%a.out
#SBATCH --error=slurm-errors/ne-shard-%A_%a.err
#SBATCH --array=0-14

# ---------------------------------------------------------------------------
# new_experiments/run_train_shard.sh  (SLURM job array, 5 models x 3 tasks)
#
# One array task per (model, task) -> 15 independent jobs that each run the
# full per-(model, task) training pipeline:
#
#   1. prep         (chat-template train + test for this single model+task)
#   2. generate1    (Tinker baseline + GPT-4o-mini voter feedback on train)
#   3. build_train  (RFT + TFB SFT datasets)
#   4. train        (Tinker LoRA SFT for both rft and tfb)
#   5. generate22   (Tinker base-model test inference)
#   6. generate2    (Tinker rft + tfb test inference)
#
# After all 15 array tasks complete, run new_experiments/run_eval_shard.sh
# (compete + probes) - those need every (model, task) to be done first.
#
# Submit via new_experiments/submit_train_parallel.sh.
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p slurm-outputs slurm-errors

# ----- Conda / env setup (mirrors run_train.sh) -----
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

: "${TINKER_API_KEY:?TINKER_API_KEY must be set in .env}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set in .env}"

# ----- Per-stage knobs (override via --export on submit) -----
LIMIT="${LIMIT:-}"                          # e.g. "64" for smoke; empty = full
MAX_CONC="${MAX_CONC:-32}"                  # Tinker / OpenAI concurrency
BATCH_SIZE="${BATCH_SIZE:-32}"              # Tinker SFT per-step batch
PYTHON="${PYTHON:-python}"

LIMIT_ARG=()
[[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")

# ----- Map array task ID -> (model, task) -----
# Order MUST match config.MODELS so logs are easy to cross-reference.
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
echo "[shard ${TID}/${TOTAL}]  task=${TASK}  model=${MODEL}"
echo "  LIMIT=${LIMIT:-<full>}  MAX_CONC=${MAX_CONC}  BATCH_SIZE=${BATCH_SIZE}"
echo "============================================================"

run_step() {
    local name="$1"; shift
    echo "------------------------------------------------------------"
    echo "[shard ${TID}] step=${name}  cmd=$*"
    echo "------------------------------------------------------------"
    "$@"
}

# ----- Per-(model, task) training pipeline -----
run_step prep \
    "$PYTHON" -m new_experiments.src.prep_data --task "$TASK" --model "$MODEL"

run_step generate1 \
    "$PYTHON" -m new_experiments.src.generate1 \
        --task "$TASK" --model "$MODEL" \
        --max_concurrency "$MAX_CONC" "${LIMIT_ARG[@]}"

run_step build_train \
    "$PYTHON" -m new_experiments.src.build_train_data --task "$TASK" --model "$MODEL"

run_step train \
    "$PYTHON" -m new_experiments.src.train \
        --task "$TASK" --model "$MODEL" --batch_size "$BATCH_SIZE"

run_step generate22 \
    "$PYTHON" -m new_experiments.src.generate22 \
        --task "$TASK" --model "$MODEL" \
        --max_concurrency "$MAX_CONC" "${LIMIT_ARG[@]}"

run_step generate2 \
    "$PYTHON" -m new_experiments.src.generate2 \
        --task "$TASK" --model "$MODEL" \
        --max_concurrency "$MAX_CONC" "${LIMIT_ARG[@]}"

echo "============================================================"
echo "[shard ${TID}] DONE  task=${TASK}  model=${MODEL}"
echo "============================================================"
