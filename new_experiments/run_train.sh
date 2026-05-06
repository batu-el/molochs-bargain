#!/bin/bash
#SBATCH --job-name=ne-train
#SBATCH --partition=owners
#SBATCH --account=jamesz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# No GPU — all sampling and training run on Tinker cloud.
#SBATCH --time=48:00:00
#SBATCH --output=slurm-outputs/ne-%j.out
#SBATCH --error=slurm-errors/ne-%j.err

# ---------------------------------------------------------------------------
# new_experiments/run_train.sh  (SLURM batch)
#
# Single SLURM job that runs the *entire* new_experiments pipeline
# sequentially across all 5 models x 3 tasks x {base, rft, tfb}:
#
#   1. prep         — chat-template every (task, model, split)
#   2. generate1    — Tinker baseline sampling + GPT-4o-mini voter feedback (train)
#   3. build_train  — RFT + TFB SFT datasets
#   4. train        — Tinker LoRA SFT (5 models x 3 tasks x 2 methods = 30 runs)
#   5. generate22   — Tinker base-model test inference
#   6. generate2    — Tinker trained-model test inference
#   7. compete      — pairwise GPT-4o-mini voter competition
#   8. probes       — gpt-4o-mini misalignment probes (override via config.PROBE_MODEL_NAME)
#
# Use submit_train.sh to enqueue. Override settings via --export, e.g.
#   sbatch --export=LIMIT=64,SKIP=probes new_experiments/run_train.sh
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p slurm-outputs slurm-errors

# ----- Conda / env setup (matches existing run_train.sh on Sherlock) -----
CONDA_HOME="${CONDA_HOME:-/scratch/users/batuel/miniconda3}"
CONDA_ENV="${CONDA_ENV:-tinker311}"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="${HF_HOME:-/scratch/users/batuel/}"
export HF_HUB_DISABLE_XET=1

# Tighter OpenAI defaults so a single hung request doesn't stall a stage
# (see new_experiments/src/openai_patch.py for details).
export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-30}"
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-3}"

# Load secrets (.env with TINKER_API_KEY, OPENAI_API_KEY, HF_TOKEN, ...).
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

REPO_ROOT="$(pwd -P)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ----- Per-stage knobs (override via --export) -----
LIMIT="${LIMIT:-}"                          # e.g. "64" for smoke; empty = full
ONLY="${ONLY:-}"                            # comma-list of stages to run alone
SKIP="${SKIP:-}"                            # comma-list of stages to skip
MAX_CONC="${MAX_CONC:-32}"                  # Tinker / OpenAI concurrency
BATCH_SIZE="${BATCH_SIZE:-16}"              # Tinker SFT per-step batch

# Build the run_experiments.sh argv from the env vars above.
RUN_ARGS=()
[[ -n "$LIMIT"     ]] && RUN_ARGS+=("--limit" "$LIMIT")
[[ -n "$ONLY"      ]] && RUN_ARGS+=("--only" "$ONLY")
[[ -n "$SKIP"      ]] && RUN_ARGS+=("--skip" "$SKIP")
RUN_ARGS+=("--max_concurrency" "$MAX_CONC")
RUN_ARGS+=("--batch_size" "$BATCH_SIZE")

set -x

# ----- Sanity check: required secrets are set -----
: "${TINKER_API_KEY:?TINKER_API_KEY must be set in .env}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set in .env}"

# ----- Run the entire pipeline sequentially -----
bash new_experiments/run_experiments.sh "${RUN_ARGS[@]}"

set +x
echo "[DONE] new_experiments full pipeline. Results under new_experiments/res/"
