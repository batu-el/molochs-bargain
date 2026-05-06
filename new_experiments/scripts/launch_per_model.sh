#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch_per_model.sh — run the pipeline one model at a time, sequentially.
#
# Useful when you want to validate Qwen-30B (~$30 total) before paying for
# Llama-3.3-70B (~$130 in Tinker plus shared OpenAI cost).
#
# Stages run per-model (everything except compete + probes which are
# multi-model by design and run only after the second model finishes):
#   prep -> generate1 -> build_train_data -> train -> generate22 -> generate2
#
# Usage:
#   bash new_experiments/scripts/launch_per_model.sh                 # both models
#   bash new_experiments/scripts/launch_per_model.sh --qwen-only     # cheap model
#   bash new_experiments/scripts/launch_per_model.sh --llama-only    # large model
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

QWEN="Qwen/Qwen3-30B-A3B-Instruct-2507"
LLAMA="meta-llama/Llama-3.3-70B-Instruct"

MODELS=("$QWEN" "$LLAMA")
SKIP_COMPETE_PROBES=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --qwen-only)  MODELS=("$QWEN");  SKIP_COMPETE_PROBES=1; shift ;;
        --llama-only) MODELS=("$LLAMA"); SKIP_COMPETE_PROBES=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

: "${TINKER_API_KEY:?TINKER_API_KEY must be set}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_LOG="new_experiments/logs/per_model_${STAMP}"
mkdir -p "$ROOT_LOG"

run_per_model() {
    local model="$1"
    local model_safe="${model//\//__}"
    local log="$ROOT_LOG/${model_safe}.log"
    echo "[per_model] $model -> $log"
    {
        echo "=== prep ==="
        python -m new_experiments.src.prep_data --model "$model"
        echo "=== generate1 ==="
        python -m new_experiments.src.generate1 --model "$model" --max_concurrency 32
        echo "=== build_train_data ==="
        python -m new_experiments.src.build_train_data --model "$model"
        echo "=== train ==="
        python -m new_experiments.src.train --model "$model" --batch_size 16
        echo "=== generate22 ==="
        python -m new_experiments.src.generate22 --model "$model" --max_concurrency 32
        echo "=== generate2 ==="
        python -m new_experiments.src.generate2 --model "$model" --max_concurrency 32
    } 2>&1 | tee "$log"
}

for m in "${MODELS[@]}"; do
    run_per_model "$m"
done

if [[ "$SKIP_COMPETE_PROBES" == "0" ]]; then
    echo "=== compete (all models) ==="
    python -m new_experiments.src.compete 2>&1 | tee "$ROOT_LOG/compete.log"
    echo "=== probes (all models) ==="
    python -m new_experiments.src.probes 2>&1 | tee "$ROOT_LOG/probes.log"
else
    echo "[per_model] Skipped compete/probes (single-model run). Run them manually after both models finish:"
    echo "  python -m new_experiments.src.compete"
    echo "  python -m new_experiments.src.probes"
fi

echo "[per_model] DONE.  Logs -> $ROOT_LOG"
