"""Central configuration for the new experiments.

Mirrors `artsco/data/utils.py` constants but points at the larger models that
will be trained through Tinker.
"""

from __future__ import annotations

import os

# ---------- Models ----------
# 5-model lineup. Each model uses its *native* response format (see
# `format_adapters.py`) rather than a forced common XML schema.
LLAMA_MODEL_NAMES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]
QWEN_MODEL_NAMES = [
    "Qwen/Qwen3-8B",                      # Hybrid (native <think>...</think>)
    "Qwen/Qwen3-32B",                     # Hybrid dense (native <think>...</think>)
]
GPT_OSS_MODEL_NAMES = [
    "openai/gpt-oss-20b",                 # Reasoning, harmony channel format
]
MODELS = LLAMA_MODEL_NAMES + QWEN_MODEL_NAMES + GPT_OSS_MODEL_NAMES

# ---------- Tokenizer aliases ----------
# Tinker hosts weights remotely, but we still need the *tokenizer* locally for
# chat templating + Datum construction. If the canonical model card is gated
# (Llama-3.3 currently is), the same tokenizer ships unconditionally on a
# sibling repo. The chat_template.jinja is byte-identical, so swapping the
# tokenizer source does NOT change any prompt string we send to Tinker.
#
# Prefer to accept the license at the canonical repo; this is just a fallback.
# Override at runtime with an env var, e.g.:
#   export TOKENIZER_OVERRIDE_meta_llama__Llama_3_3_70B_Instruct=unsloth/Llama-3.3-70B-Instruct
import os as _os

_DEFAULT_TOKENIZER_ALIASES = {
    # Llama 3.3 shares the Llama-3.1 tokenizer (same vocab, same chat template).
    # Both of these repos are commonly accessible without gating; pick whichever
    # your HF account has access to.
    # "meta-llama/Llama-3.3-70B-Instruct": "unsloth/Llama-3.3-70B-Instruct",
    # Qwen3-30B-A3B-Instruct-2507 shares the Qwen3 tokenizer family.
    # "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3-8B",
}


def tokenizer_for(model_name: str) -> str:
    """Return the HF repo to load the tokenizer from for `model_name`."""
    # Per-model env override beats the static alias table.
    env_key = "TOKENIZER_OVERRIDE_" + model_name.replace("/", "__").replace("-", "_").replace(".", "_")
    return _os.environ.get(env_key, _DEFAULT_TOKENIZER_ALIASES.get(model_name, model_name))

# ---------- Tasks ----------
TASKS = ["task_elections", "task_sales", "task_sm"]
SPLITS = ["train", "test"]
METHODS = ["base", "rft", "tfb"]
TRAINED_METHODS = ["rft", "tfb"]

# ---------- Voter / probe model names ----------
# Voters: gpt-4o-mini (matches the original artsco setup).
# Probes: gpt-4o-mini (downgrade from gpt-4o for ~94% probe-stage cost
# savings, ~$424 across the full pipeline). The trends/*.py probe modules
# hardcode "gpt-4o" themselves, so we monkey-patch them at probes.py import
# time to honor PROBE_MODEL_NAME (see new_experiments/src/probes.py).
VOTER_MODEL_NAME = "gpt-4o-mini"
PROBE_MODEL_NAME = "gpt-4o-mini"

# ---------- Sampling parameters (mirror original generate*.py defaults) ----------
NUM_PLAYERS = 2
MAX_NEW_TOKENS = 1480
TEMPERATURE = 0.7
# Voter pool sizes. Personas are drawn from `subjects/personas_{train,test}.json`
# (800 train, 200 test demographically realistic personas). We *sample without
# replacement* from each pool using `VOTER_SAMPLE_SEED` for reproducibility, so
# the same 50 train personas are reused across all (model, task, prompt) tuples
# during generate1, and the same 50 test personas across all compete pairs.
NUM_VOTERS_TRAIN = 25          # for generate1 audience feedback (25/800 train personas)
NUM_VOTERS_COMPETE = 25        # for compete.py pairwise comparison (25/200 test personas)
VOTER_SAMPLE_SEED = 0          # any int; pass to load_voter_bios(..., seed=...)

# ---------- LoRA / SFT hyperparameters (mirror artsco/src/train.py) ----------
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
PER_DEVICE_BATCH = 16          # tinker batches over a list of Datum
WARMUP_RATIO = 0.03
MIN_LR_RATIO = 0.1             # cosine_with_min_lr min_lr_rate
MAX_SEQ_LENGTH = 4096

# ---------- Paths ----------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NEW_EXP_DIR = os.path.join(ROOT_DIR, "new_experiments")
DATA_DIR = os.path.join(NEW_EXP_DIR, "data")
MODELS_DIR = os.path.join(NEW_EXP_DIR, "models")
RES_DIR = os.path.join(NEW_EXP_DIR, "res")
LOG_DIR = os.path.join(NEW_EXP_DIR, "logs")

# Source raw data is reused from `artsco/data/` (model-agnostic).
ARTSCO_DATA_DIR = os.path.join(ROOT_DIR, "artsco", "data")

# Voter persona / demographic pool (paired by index). 800 train + 200 test
# demographically realistic personas live here, replacing the older 100-figure
# `artsco/data/persona/split1.json` pool. Loaded by `personas.load_voter_bios`.
SUBJECTS_DIR = os.path.join(ROOT_DIR, "subjects")


def subjects_path(kind: str, split: str) -> str:
    """Return the path to `subjects/{kind}_{split}.json`.

    Args:
        kind: "personas" or "demographics".
        split: "train" or "test".
    """
    if kind not in ("personas", "demographics"):
        raise ValueError(f"kind must be 'personas' or 'demographics', got {kind!r}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return os.path.join(SUBJECTS_DIR, f"{kind}_{split}.json")


def raw_split_path(task: str, split: str) -> str:
    """Original (model-agnostic) raw JSON for a task/split."""
    return os.path.join(ARTSCO_DATA_DIR, task, f"{split}.json")


def templated_split_path(task: str, model_name: str, split: str) -> str:
    """Per-model chat-templated JSONL output of `prep_data.py`."""
    return os.path.join(DATA_DIR, task, model_name, f"{split}.json")


def step1_path(task: str, model_name: str, split: str) -> str:
    """`generate1.py` output (baseline completions + voter feedback)."""
    return os.path.join(DATA_DIR, task, model_name, f"{split}_step1.json")


def train_data_path(task: str, model_name: str, split: str, method: str) -> str:
    """`build_train_data.py` output (rft or tfb training data)."""
    return os.path.join(DATA_DIR, task, model_name, f"{split}_{method}.json")


def model_state_path(task: str, model_name: str, method: str) -> str:
    """JSON with the Tinker checkpoint URI for the trained adapter."""
    return os.path.join(MODELS_DIR, task, model_name, method, "state.json")


def step2_path(task: str, model_name: str, method: str, split: str) -> str:
    """`generate2.py` / `generate22.py` output."""
    return os.path.join(RES_DIR, task, model_name, method, f"{split}_step2.json")


def competition_path(task: str) -> str:
    return os.path.join(RES_DIR, task, "competition.json")


def probes_path(task: str, qid: str) -> str:
    return os.path.join(RES_DIR, "probes", f"{task}_{qid}.csv")
