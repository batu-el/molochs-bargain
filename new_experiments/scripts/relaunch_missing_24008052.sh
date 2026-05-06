#!/usr/bin/env bash
# Relaunch the missing pieces from Slurm array 24008052.
#
# - Rebuild/train/regenerate pairs whose shard failed before model states existed.
# - Regenerate only final test outputs for Llama-3.3 pairs that already have states.
#
# Usage:
#   bash new_experiments/scripts/relaunch_missing_24008052.sh
#
# Optional overrides:
#   MAX_CONC=16 BATCH_SIZE=32 TIME_LIMIT_TRAIN=12:00:00 TIME_LIMIT_GEN=06:00:00 \
#     bash new_experiments/scripts/relaunch_missing_24008052.sh

set -euo pipefail

cd "$(dirname "$0")/../.."

MAX_CONC="${MAX_CONC:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TIME_LIMIT_TRAIN="${TIME_LIMIT_TRAIN:-12:00:00}"
TIME_LIMIT_GEN="${TIME_LIMIT_GEN:-06:00:00}"

if [[ -f .env ]]; then
    grep -q '^TINKER_API_KEY=' .env || { echo "ERROR: TINKER_API_KEY missing from .env" >&2; exit 1; }
    grep -q '^OPENAI_API_KEY=' .env || { echo "ERROR: OPENAI_API_KEY missing from .env" >&2; exit 1; }
else
    echo "WARNING: no .env file found in repo root."
fi

mkdir -p slurm-outputs slurm-errors

TRAIN_JOB_ID=$(sbatch \
    --parsable \
    --job-name=ne-repair-train \
    --partition=owners \
    --account=jamesz \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time="$TIME_LIMIT_TRAIN" \
    --array=0-4 \
    --output=slurm-outputs/ne-repair-train-%A_%a.out \
    --error=slurm-errors/ne-repair-train-%A_%a.err \
    --export=ALL,MAX_CONC="$MAX_CONC",BATCH_SIZE="$BATCH_SIZE" <<'SBATCH'
#!/bin/bash
set -euo pipefail

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
    source .env
    set +a
fi

REPO_ROOT="$(pwd -P)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

: "${TINKER_API_KEY:?TINKER_API_KEY must be set in .env}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set in .env}"

TASKS=(
    "task_sales"
    "task_elections"
    "task_elections"
    "task_sales"
    "task_sm"
)
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-32B"
    "openai/gpt-oss-20b"
    "openai/gpt-oss-20b"
    "openai/gpt-oss-20b"
)
OLD_SHARDS=(1 9 12 13 14)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
TASK="${TASKS[$IDX]}"
MODEL="${MODELS[$IDX]}"
OLD_SHARD="${OLD_SHARDS[$IDX]}"

echo "============================================================"
echo "[repair-train ${IDX}/5] old_shard=${OLD_SHARD} task=${TASK} model=${MODEL}"
echo "  MAX_CONC=${MAX_CONC:-32} BATCH_SIZE=${BATCH_SIZE:-32}"
echo "============================================================"

python -m new_experiments.src.build_train_data --task "$TASK" --model "$MODEL"
python -m new_experiments.src.train --task "$TASK" --model "$MODEL" --batch_size "${BATCH_SIZE:-32}"
python -m new_experiments.src.generate22 --task "$TASK" --model "$MODEL" --max_concurrency "${MAX_CONC:-32}"
python -m new_experiments.src.generate2 --task "$TASK" --model "$MODEL" --max_concurrency "${MAX_CONC:-32}"
SBATCH
)

GEN_JOB_ID=$(sbatch \
    --parsable \
    --job-name=ne-repair-gen \
    --partition=owners \
    --account=jamesz \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time="$TIME_LIMIT_GEN" \
    --array=0-2 \
    --output=slurm-outputs/ne-repair-gen-%A_%a.out \
    --error=slurm-errors/ne-repair-gen-%A_%a.err \
    --export=ALL,MAX_CONC="$MAX_CONC" <<'SBATCH'
#!/bin/bash
set -euo pipefail

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
    source .env
    set +a
fi

REPO_ROOT="$(pwd -P)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

: "${TINKER_API_KEY:?TINKER_API_KEY must be set in .env}"

TASKS=("task_elections" "task_sales" "task_sm")
MODEL="meta-llama/Llama-3.3-70B-Instruct"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
TASK="${TASKS[$IDX]}"

echo "============================================================"
echo "[repair-gen ${IDX}/3] task=${TASK} model=${MODEL}"
echo "  MAX_CONC=${MAX_CONC:-32}"
echo "============================================================"

python -m new_experiments.src.generate22 --task "$TASK" --model "$MODEL" --max_concurrency "${MAX_CONC:-32}"
python -m new_experiments.src.generate2 --task "$TASK" --model "$MODEL" --max_concurrency "${MAX_CONC:-32}"
SBATCH
)

echo "Submitted repair train array: ${TRAIN_JOB_ID} (old shards 1,9,12,13,14)"
echo "Submitted repair generation array: ${GEN_JOB_ID} (Llama-3.3 tasks 3,4,5 outputs)"
echo
echo "Tail logs with:"
echo "  tail -F slurm-outputs/ne-repair-train-${TRAIN_JOB_ID}_*.out slurm-outputs/ne-repair-gen-${GEN_JOB_ID}_*.out"
