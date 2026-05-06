#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch_smoke.sh — end-to-end smoke test (~$70, ~30-90 min)
#
# Runs the full pipeline against just 16 prompts per task. Goal is to verify
# Tinker auth, OpenAI auth, training succeeds, and all stage outputs land
# in the right paths before committing to a full ($450+) run.
#
# Usage:
#   bash new_experiments/scripts/launch_smoke.sh
#   bash new_experiments/scripts/launch_smoke.sh --model Qwen/Qwen3-30B-A3B-Instruct-2507
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

MODEL_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_FILTER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

LIMIT_ARG="--limit 16"
LOG_DIR="new_experiments/logs/smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "[smoke] logs -> $LOG_DIR"
echo "[smoke] limit=16, models=${MODEL_FILTER:-ALL}"
echo "[smoke] expected cost ~ \$70 (mostly Tinker training; sampling and OpenAI scale with --limit)"

if [[ -n "$MODEL_FILTER" ]]; then
    # Re-run each stage with the model filter
    for stage in prep generate1 build_train train generate22 generate2 compete probes; do
        echo "[smoke] stage=$stage model=$MODEL_FILTER"
        case "$stage" in
            prep)         python -m new_experiments.src.prep_data --model "$MODEL_FILTER" 2>&1 | tee "$LOG_DIR/${stage}.log" ;;
            generate1)    python -m new_experiments.src.generate1 --model "$MODEL_FILTER" $LIMIT_ARG 2>&1 | tee "$LOG_DIR/${stage}.log" ;;
            build_train)  python -m new_experiments.src.build_train_data --model "$MODEL_FILTER" 2>&1 | tee "$LOG_DIR/${stage}.log" ;;
            train)        python -m new_experiments.src.train --model "$MODEL_FILTER" 2>&1 | tee "$LOG_DIR/${stage}.log" ;;
            generate22)   python -m new_experiments.src.generate22 --model "$MODEL_FILTER" $LIMIT_ARG 2>&1 | tee "$LOG_DIR/${stage}.log" ;;
            generate2)    python -m new_experiments.src.generate2 --model "$MODEL_FILTER" $LIMIT_ARG 2>&1 | tee "$LOG_DIR/${stage}.log" ;;
            compete)      python -m new_experiments.src.compete $LIMIT_ARG 2>&1 | tee "$LOG_DIR/${stage}.log" ;;
            probes)       python -m new_experiments.src.probes $LIMIT_ARG 2>&1 | tee "$LOG_DIR/${stage}.log" ;;
        esac
    done
else
    bash new_experiments/run_experiments.sh --limit 16 2>&1 | tee "$LOG_DIR/full.log"
fi

echo "[smoke] DONE. Inspect $LOG_DIR and new_experiments/res/"
