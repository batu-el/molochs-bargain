#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# new_experiments/run_experiments.sh
#
# End-to-end driver for the Llama-3.3-70B / Qwen3-30B-A3B replication of
# Moloch's Bargain through the Tinker API. Mirrors the original artsco
# pipeline (data prep -> baseline + audience -> RFT/TFB datasets -> SFT ->
# test inference -> pairwise competition -> misalignment probes).
#
# Usage:
#   bash new_experiments/run_experiments.sh                  # full pipeline
#   bash new_experiments/run_experiments.sh --skip-train     # skip steps
#   bash new_experiments/run_experiments.sh --only generate2 # one stage
#   bash new_experiments/run_experiments.sh --limit 16       # debug limit
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

# Auto-load secrets from .env if present (TINKER_API_KEY, OPENAI_API_KEY, HF_TOKEN, ...)
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# ---------- defaults ----------
LIMIT=""
ONLY=""
SKIP=""
MAX_CONC="${MAX_CONC:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PYTHON="${PYTHON:-python}"

# ---------- argparse ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)  LIMIT="--limit $2"; shift 2 ;;
        --only)   ONLY="$2"; shift 2 ;;
        --skip)   SKIP="$2"; shift 2 ;;
        --max_concurrency) MAX_CONC="$2"; shift 2 ;;
        --batch_size)      BATCH_SIZE="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed -e 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------- env sanity ----------
: "${TINKER_API_KEY:?TINKER_API_KEY must be set (https://tinker-console.thinkingmachines.ai/)}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (used by voters and probes)}"
: "${HF_TOKEN:?HF_TOKEN must be set (for tokenizer access)}"

mkdir -p new_experiments/data new_experiments/models new_experiments/res new_experiments/logs

run_stage() {
    local name="$1"; shift
    if [[ -n "$ONLY" && "$ONLY" != "$name" ]]; then return 0; fi
    if [[ -n "$SKIP" && ",$SKIP," == *",$name,"* ]]; then
        echo "=========================================="
        echo "==  SKIP $name"
        echo "=========================================="
        return 0
    fi
    echo "=========================================="
    echo "==  STAGE: $name"
    echo "==  $@"
    echo "=========================================="
    "$@"
}

# ---------- 1. per-model chat templates (train + test) ----------
run_stage prep \
    $PYTHON -m new_experiments.src.prep_data

# ---------- 2. baseline + voter feedback (train split) ----------
run_stage generate1 \
    $PYTHON -m new_experiments.src.generate1 \
        --max_concurrency "$MAX_CONC" $LIMIT

# ---------- 3. build RFT and TFB SFT datasets ----------
run_stage build_train \
    $PYTHON -m new_experiments.src.build_train_data

# ---------- 4. SFT through Tinker (one run per task x model x method) ----------
run_stage train \
    $PYTHON -m new_experiments.src.train \
        --batch_size "$BATCH_SIZE"

# ---------- 5. baseline test inference ----------
run_stage generate22 \
    $PYTHON -m new_experiments.src.generate22 \
        --max_concurrency "$MAX_CONC" $LIMIT

# ---------- 6. trained-model test inference ----------
run_stage generate2 \
    $PYTHON -m new_experiments.src.generate2 \
        --max_concurrency "$MAX_CONC" $LIMIT

# ---------- 7. pairwise voter competition ----------
run_stage compete \
    $PYTHON -m new_experiments.src.compete $LIMIT

# ---------- 8. misalignment probes (q1, q2 per task) ----------
run_stage probes \
    $PYTHON -m new_experiments.src.probes $LIMIT

echo "=========================================="
echo "==  DONE.  Results under new_experiments/res/"
echo "=========================================="
