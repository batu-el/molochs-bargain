#!/bin/bash
#SBATCH --job-name=ne-eval-merge
#SBATCH --partition=owners
#SBATCH --account=jamesz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# No GPU, no API calls - pure file I/O over per-model part files.
#SBATCH --time=00:15:00
#SBATCH --output=slurm-outputs/ne-eval-merge-%j.out
#SBATCH --error=slurm-errors/ne-eval-merge-%j.err

# ---------------------------------------------------------------------------
# new_experiments/run_eval_merge.sh
#
# One-shot consolidation step that runs AFTER run_eval_shard_per_model.sh
# finishes. Walks the per-model part directories produced by the 15-shard
# array and writes the canonical consolidated outputs:
#
#   res/{task}/competition_parts/{model}.json   ->  res/{task}/competition.json
#   res/probes/{task}_{qid}_parts/{model}.csv   ->  res/probes/{task}_{qid}.csv
#
# Submit this with --dependency=afterok:<arrayJobId> after the per-model
# eval array, or run it standalone via submit_eval_parallel.sh which wires
# the dependency for you.
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p slurm-outputs slurm-errors

CONDA_HOME="${CONDA_HOME:-/scratch/users/batuel/miniconda3}"
CONDA_ENV="${CONDA_ENV:-tinker311}"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

REPO_ROOT="$(pwd -P)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python}"

echo "============================================================"
echo "[eval-merge] consolidating per-model parts -> canonical outputs"
echo "============================================================"

"$PYTHON" -m new_experiments.src.compete --merge
"$PYTHON" -m new_experiments.src.probes  --merge

echo "============================================================"
echo "[eval-merge] DONE"
echo "============================================================"
