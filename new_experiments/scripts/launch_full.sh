#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch_full.sh — full pipeline as a detached background job.
#
# Estimated cost: ~$460 total (see scripts/estimate_costs.py).
#   - generate1 (Tinker sample + GPT-4o-mini voters):  ~$102
#   - train     (Tinker LoRA SFT):                     ~$ 44
#   - generate2 + generate22 (Tinker sample on test):  ~$ 22
#   - compete   (GPT-4o-mini pairwise voters):         ~$114
#   - probes    (GPT-4o q1/q2):                        ~$180
#
# Wall time depends on Tinker queue and OpenAI throughput; budget 6-12 hours.
#
# Usage:
#   bash new_experiments/scripts/launch_full.sh                   # detach + log
#   bash new_experiments/scripts/launch_full.sh --foreground      # block
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

FOREGROUND=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground) FOREGROUND=1; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

: "${TINKER_API_KEY:?TINKER_API_KEY must be set}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="new_experiments/logs/full_${STAMP}"
mkdir -p "$LOG_DIR"

echo "[full] cost estimate:"
python -m new_experiments.scripts.estimate_costs | tail -20
echo

if [[ "$FOREGROUND" == "1" ]]; then
    bash new_experiments/run_experiments.sh "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_DIR/run.log"
else
    nohup bash new_experiments/run_experiments.sh "${EXTRA_ARGS[@]}" \
        >"$LOG_DIR/run.log" 2>&1 &
    PID=$!
    echo "$PID" > "$LOG_DIR/pid"
    echo "[full] started PID=$PID, logs -> $LOG_DIR/run.log"
    echo "[full] tail -f $LOG_DIR/run.log"
    echo "[full] kill \$(cat $LOG_DIR/pid)   # to abort"
fi
