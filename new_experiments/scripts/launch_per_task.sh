#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch_per_task.sh — fan out the heavy stages across tasks in parallel.
#
# Generation and probe stages are I/O bound (Tinker / OpenAI), so running
# the three tasks in parallel cuts wall time roughly 3x. Training stays
# sequential per (model, task, method) to avoid Tinker queue contention.
#
# Usage:
#   bash new_experiments/scripts/launch_per_task.sh                # all stages
#   bash new_experiments/scripts/launch_per_task.sh --skip train   # skip a stage
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

SKIP=""
LIMIT_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip)  SKIP="$2"; shift 2 ;;
        --limit) LIMIT_ARG="--limit $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

: "${TINKER_API_KEY:?TINKER_API_KEY must be set}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"

TASKS=(task_elections task_sales task_sm)
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_LOG="new_experiments/logs/per_task_${STAMP}"
mkdir -p "$ROOT_LOG"

skipped() { [[ ",$SKIP," == *",$1,"* ]]; }

# --- Stage 1: prep (cheap, all tasks at once) ---
if ! skipped prep; then
    echo "[per_task] prep (all tasks, both models)"
    python -m new_experiments.src.prep_data 2>&1 | tee "$ROOT_LOG/prep.log"
fi

# --- Stage 2: generate1 (parallel across tasks) ---
if ! skipped generate1; then
    echo "[per_task] generate1 (parallel across tasks)"
    pids=()
    for task in "${TASKS[@]}"; do
        (
            python -m new_experiments.src.generate1 --task "$task" --max_concurrency 32 $LIMIT_ARG \
                2>&1 | tee "$ROOT_LOG/generate1_${task}.log"
        ) &
        pids+=($!)
    done
    wait "${pids[@]}"
fi

# --- Stage 3: build_train_data (cheap, sequential) ---
if ! skipped build_train; then
    echo "[per_task] build_train_data"
    python -m new_experiments.src.build_train_data 2>&1 | tee "$ROOT_LOG/build_train.log"
fi

# --- Stage 4: train (parallel across tasks; methods serial inside each task) ---
if ! skipped train; then
    echo "[per_task] train (parallel across tasks)"
    pids=()
    for task in "${TASKS[@]}"; do
        (
            python -m new_experiments.src.train --task "$task" --batch_size 32 \
                2>&1 | tee "$ROOT_LOG/train_${task}.log"
        ) &
        pids+=($!)
    done
    wait "${pids[@]}"
fi

# --- Stage 5: generate22 (parallel across tasks) ---
if ! skipped generate22; then
    echo "[per_task] generate22 (parallel across tasks)"
    pids=()
    for task in "${TASKS[@]}"; do
        (
            python -m new_experiments.src.generate22 --task "$task" --max_concurrency 32 $LIMIT_ARG \
                2>&1 | tee "$ROOT_LOG/generate22_${task}.log"
        ) &
        pids+=($!)
    done
    wait "${pids[@]}"
fi

# --- Stage 6: generate2 (parallel across tasks) ---
if ! skipped generate2; then
    echo "[per_task] generate2 (parallel across tasks)"
    pids=()
    for task in "${TASKS[@]}"; do
        (
            python -m new_experiments.src.generate2 --task "$task" --max_concurrency 32 $LIMIT_ARG \
                2>&1 | tee "$ROOT_LOG/generate2_${task}.log"
        ) &
        pids+=($!)
    done
    wait "${pids[@]}"
fi

# --- Stage 7: compete (parallel across tasks) ---
if ! skipped compete; then
    echo "[per_task] compete (parallel across tasks)"
    pids=()
    for task in "${TASKS[@]}"; do
        (
            python -m new_experiments.src.compete --task "$task" $LIMIT_ARG \
                2>&1 | tee "$ROOT_LOG/compete_${task}.log"
        ) &
        pids+=($!)
    done
    wait "${pids[@]}"
fi

# --- Stage 8: probes (parallel across tasks) ---
if ! skipped probes; then
    echo "[per_task] probes (parallel across tasks)"
    pids=()
    for task in "${TASKS[@]}"; do
        (
            python -m new_experiments.src.probes --task "$task" $LIMIT_ARG \
                2>&1 | tee "$ROOT_LOG/probes_${task}.log"
        ) &
        pids+=($!)
    done
    wait "${pids[@]}"
fi

echo "[per_task] DONE.  Logs -> $ROOT_LOG"
