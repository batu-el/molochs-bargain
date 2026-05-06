#!/bin/bash
#SBATCH --job-name=lsp-train
#SBATCH --partition=owners
#SBATCH --account=jamesz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# No GPU requested (training happens on Tinker cloud)
#SBATCH --time=48:00:00
#SBATCH --output=slurm-outputs/slurm-%j.out
#SBATCH --error=slurm-errors/slurm-%j.err

# Single trainer launcher. The 3 supported methods are encoded as
# (seller_prompt_mode, personalization_probe, personalization_reward):
#   simple            -> simple,       probe=off, reward=False
#   pers_no_reward    -> personalized, probe=on,  reward=False
#   pers_with_reward  -> personalized, probe=on,  reward=True
# Use submit_train.sh to sweep all 3 methods x both settings.

set -euo pipefail

mkdir -p slurm-outputs slurm-errors

# Setup conda environment
CONDA_HOME="/scratch/users/batuel/miniconda3"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate tinker311

# Set environment variables
export HF_HOME="/scratch/users/batuel/"
export VLLM_DISABLE_COMPILE_CACHE=1
export HF_HUB_DISABLE_XET=1
export EXPECT_TOTAL_GPUS=0

# Load all API keys from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

REPO_ROOT="$(pwd -P)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

set -x

# Required (passed via --export from submit script)
setting="${setting}"                                # "likelihood" or "preference"
dataset_category="${dataset_category}"              # "groceries" or "movies"
random_initial_vote="${random_initial_vote}"        # "True" or "False"
seller_prompt_mode="${seller_prompt_mode}"          # "simple" or "personalized"
personalization_probe="${personalization_probe}"    # "True" or "False"
personalization_reward="${personalization_reward}"  # "True" or "False"

# Optional seller (trained) base model. Empty -> LSPConfig default
# (`meta-llama/Llama-3.1-8B-Instruct`). Currently supported by tinker for
# LoRA RL: `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`.
# Both use the `llama3` renderer so renderer_name doesn't need to change.
model_name="${model_name:-}"

# Optional model overrides
buyer_model="${buyer_model:-gpt-5-mini}"
buyer_reasoning_effort="${buyer_reasoning_effort:-low}"
buyer_prompt_mode="${buyer_prompt_mode:-persona}"  # "persona" or "uniform"
# Optional second training buyer (Together open-weight). When non-empty, GRPO
# groups split 50/50 even/odd between primary openai buyer and this one.
secondary_buyer_model="${secondary_buyer_model:-}"
secondary_buyer_reasoning_effort="${secondary_buyer_reasoning_effort:-}"
# Optional eval buyer overrides. When eval_secondary_buyer_model is set, every
# eval scenario is rolled out with BOTH eval_buyer_model (openai) and
# eval_secondary_buyer_model (together) — that's how we compare buyer backends
# on the same held-out test demographics x test products.
eval_buyer_model="${eval_buyer_model:-}"
eval_buyer_reasoning_effort="${eval_buyer_reasoning_effort:-}"
eval_secondary_buyer_model="${eval_secondary_buyer_model:-}"
eval_secondary_buyer_reasoning_effort="${eval_secondary_buyer_reasoning_effort:-}"
violation_probe_model="${violation_probe_model:-gpt-5-mini}"
# Dynamic violation probe (4th method): when "True", the violation probe's
# rubric is refreshed every 10 GRADIENT STEPS by a two-LLM meta pipeline
# (extractor -> incorporator) that operates on ~20 conversations from the
# most recent step. Total rubric length stays capped to the static rubric's
# length so per-call probe cost never grows.
dynamic_violation_probe="${dynamic_violation_probe:-False}"
dynamic_violation_probe_meta_model="${dynamic_violation_probe_meta_model:-}"
personalization_probe_model="${personalization_probe_model:-gpt-5-mini}"
vote_probe_model="${vote_probe_model:-gpt-4o-mini}"
output_dir="${output_dir:-outputs}"
# Optional explicit run id. When set, becomes BOTH the on-disk leaf folder
# (`output_dir/method/setting/category/riv/<run_id>`) AND the W&B run name —
# so distinct experiments are easy to tell apart in the UI / on disk.
# Leave empty to fall back to the auto-generated timestamped id.
run_id="${run_id:-}"

# Optional training-loop / RL overrides. Empty means "use LSPConfig default".
#   num_steps              -> total gradient steps (default 200)
#   eval_every             -> eval every N steps (default 50)
#   save_every             -> checkpoint cadence in steps (default 10)
#   kl_penalty_coef        -> KL anchor against base model (default 0.0; see
#                             LSPConfig docstring for tradeoffs). When > 0,
#                             tinker also requires a reference base model —
#                             we let LSPConfig synthesize that from
#                             `model_name` unless `kl_reference_base_model`
#                             is explicitly passed.
#   kl_reference_base_model -> override the KL reference (e.g. point at a
#                              different snapshot / size).
num_steps="${num_steps:-}"
eval_every="${eval_every:-}"
save_every="${save_every:-}"
kl_penalty_coef="${kl_penalty_coef:-}"
kl_reference_base_model="${kl_reference_base_model:-}"
# Optional optimizer override. LSPConfig default (4e-5) is tuned for the 8B
# seller. For 70B you typically want a smaller LR (e.g. 2e-5).
learning_rate="${learning_rate:-}"

# Build the chz CLI args. Optional buyer overrides are only forwarded when
# non-empty so the LSPConfig defaults (None) keep the old single-buyer
# behavior for callers that don't care about the dual-buyer eval.
optional_args=()
if [ -n "$secondary_buyer_model" ]; then
    optional_args+=("cfg.secondary_buyer_model=${secondary_buyer_model}")
fi
if [ -n "$secondary_buyer_reasoning_effort" ]; then
    optional_args+=("cfg.secondary_buyer_reasoning_effort=${secondary_buyer_reasoning_effort}")
fi
if [ -n "$eval_buyer_model" ]; then
    optional_args+=("cfg.eval_buyer_model=${eval_buyer_model}")
fi
if [ -n "$eval_buyer_reasoning_effort" ]; then
    optional_args+=("cfg.eval_buyer_reasoning_effort=${eval_buyer_reasoning_effort}")
fi
if [ -n "$eval_secondary_buyer_model" ]; then
    optional_args+=("cfg.eval_secondary_buyer_model=${eval_secondary_buyer_model}")
fi
if [ -n "$eval_secondary_buyer_reasoning_effort" ]; then
    optional_args+=("cfg.eval_secondary_buyer_reasoning_effort=${eval_secondary_buyer_reasoning_effort}")
fi
if [ -n "$run_id" ]; then
    optional_args+=("cfg.run_id=${run_id}")
fi
if [ -n "$dynamic_violation_probe_meta_model" ]; then
    optional_args+=("cfg.dynamic_violation_probe_meta_model=${dynamic_violation_probe_meta_model}")
fi
if [ -n "$num_steps" ]; then
    optional_args+=("cfg.num_steps=${num_steps}")
fi
if [ -n "$eval_every" ]; then
    optional_args+=("cfg.eval_every=${eval_every}")
fi
if [ -n "$save_every" ]; then
    optional_args+=("cfg.save_every=${save_every}")
fi
if [ -n "$kl_penalty_coef" ]; then
    optional_args+=("cfg.kl_penalty_coef=${kl_penalty_coef}")
fi
if [ -n "$kl_reference_base_model" ]; then
    optional_args+=("cfg.kl_reference_base_model=${kl_reference_base_model}")
fi
if [ -n "$model_name" ]; then
    optional_args+=("cfg.model_name=${model_name}")
fi
if [ -n "$learning_rate" ]; then
    optional_args+=("cfg.learning_rate=${learning_rate}")
fi

# Violation probe is always ON (the simple prompt's anti-fabrication rules
# are reinforced by the -6 penalty floor when the judge flags a violation).
python -m src.trainer \
    cfg.setting="${setting}" \
    cfg.dataset_category="${dataset_category}" \
    cfg.random_initial_vote="${random_initial_vote}" \
    cfg.seller_prompt_mode="${seller_prompt_mode}" \
    cfg.buyer_prompt_mode="${buyer_prompt_mode}" \
    cfg.buyer_model="${buyer_model}" \
    cfg.buyer_reasoning_effort="${buyer_reasoning_effort}" \
    cfg.violation_probe=True \
    cfg.violation_probe_model="${violation_probe_model}" \
    cfg.dynamic_violation_probe="${dynamic_violation_probe}" \
    cfg.personalization_probe="${personalization_probe}" \
    cfg.personalization_probe_model="${personalization_probe_model}" \
    cfg.personalization_reward="${personalization_reward}" \
    cfg.vote_probe_model="${vote_probe_model}" \
    cfg.output_dir="${output_dir}" \
    "${optional_args[@]}"

set +x
echo "[DONE] setting=${setting}, category=${dataset_category}, random_initial_vote=${random_initial_vote}, seller_prompt_mode=${seller_prompt_mode}, buyer_prompt_mode=${buyer_prompt_mode}, personalization_probe=${personalization_probe}, personalization_reward=${personalization_reward}, dynamic_violation_probe=${dynamic_violation_probe}, secondary_buyer_model=${secondary_buyer_model:-<none>}, eval_secondary_buyer_model=${eval_secondary_buyer_model:-<none>}, output_dir=${output_dir}"
