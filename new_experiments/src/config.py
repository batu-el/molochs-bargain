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
# Five training methods plus the untrained `base` control. The contrastive
# preference methods (dpo, kto) need a frozen *reference* model -- we always
# use the base model as the reference; see `tinker_utils.compute_ref_logprob_sums`.
#   rft : SFT NLL on the (prompt, winner) only.
#   tfb : SFT NLL on (prompt, voter_think) warm-up + RFT data appended.
#   dpo : pairwise contrastive on (prompt, winner, loser) vs reference.
#   kto : per-example KTO loss on (prompt, completion, desirable=True/False)
#         with batch-mean KL anchor against reference.
METHODS = ["base", "rft", "tfb", "dpo", "kto"]
TRAINED_METHODS = ["rft", "tfb", "dpo", "kto"]
# Methods that need a frozen base-model reference for the loss.
PREF_METHODS = ["dpo", "kto"]

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
# ---------- Fixed audience (train only) ----------
# We commit to a *single* fixed audience drawn from the train pool, used both
# during training (generate1 audience feedback) AND during evaluation
# (compete pairwise voter competition). Materialized once into
# `subjects/train_{persona,demographic}_{N}.json` by
# `new_experiments/scripts/build_audiences.py` (seed=0 sample of the 800 train
# demographic-realistic persona pool). The same N people see every
# (model, task, prompt, method).
#
# A separate `..._test_*.json` audience file *can* be materialized via the
# same script (and `audience_path("...", "test")` will resolve to it), but the
# default pipeline below does NOT use it -- compete only iterates over
# `AUDIENCES`.
#
# N is baked into the file name suffix so multiple sizes can coexist on disk
# (e.g. `..._20.json` and `..._50.json`); the active size is selected by
# `NUM_VOTERS_TRAIN`.
NUM_VOTERS_TRAIN = 20          # generate1 + compete (the only audience used)
NUM_VOTERS_TEST = 20           # only used by build_audiences.py if you also
                               # want to materialize a held-out test audience
                               # for ad-hoc analyses; not wired to compete.
VOTER_SAMPLE_SEED = 0          # only used by build_audiences.py at materialization time
# Audience identifiers iterated over by compete.py + estimate_costs.py.
# Currently single-audience (train only); add "test" here to also re-evaluate
# with the held-out audience.
AUDIENCES = ("train",)

# Voter bio surface form. The 800/200 subject pool ships with a *paired*
# free-form persona text and a structured demographics dict per person:
#   - "persona"      : free-form persona paragraph only           (DEFAULT)
#   - "demographics" : demographic list only                      (ABLATION)
#   - "both"         : demographics header + persona paragraph
# The same (seed, n) sample selects the same underlying 25 people regardless
# of bio_mode, so persona vs demographics runs are directly comparable.
VOTER_BIO_MODE = "persona"

# ---------- LoRA / SFT hyperparameters (mirror artsco/src/train.py) ----------
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1                 # single pass over the training set for every method
PER_DEVICE_BATCH = 16          # tinker batches over a list of Datum
WARMUP_RATIO = 0.03
MIN_LR_RATIO = 0.1             # cosine_with_min_lr min_lr_rate
MAX_SEQ_LENGTH = 4096

# ---------- Preference-method hyperparameters (DPO + KTO) ----------
# DPO sigmoid logistic loss with reference-anchored log-ratios.
DPO_BETA = 0.1
# KTO per-example loss vs. reference, with batch-mean KL anchor (z_ref).
KTO_BETA = 0.1
KTO_DESIRABLE_WEIGHT = 1.0     # paper default lambda_D
KTO_UNDESIRABLE_WEIGHT = 1.0   # paper default lambda_U
# DPO/KTO need pairs/labels in a deterministic order inside the SFT batch:
#   - DPO: (chosen, rejected) interleaved every 2 entries -> batch must be even.
#   - KTO: (desirable, undesirable) interleaved every 2 entries (each prompt
#          contributes one of each, so the batch is naturally balanced).
PREF_BATCH_PAIR_STRIDE = 2

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
    """Return the path to `subjects/{kind}_{split}.json` (the full 800/200 pool).

    Args:
        kind: "personas" or "demographics".
        split: "train" or "test".
    """
    if kind not in ("personas", "demographics"):
        raise ValueError(f"kind must be 'personas' or 'demographics', got {kind!r}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return os.path.join(SUBJECTS_DIR, f"{kind}_{split}.json")


def _audience_size_for(split: str) -> int:
    return NUM_VOTERS_TRAIN if split == "train" else NUM_VOTERS_TEST


def audience_path(kind: str, split: str, n: int | None = None) -> str:
    """Return the path to the fixed audience file for `(kind, split, n)`.

    Layout:
        subjects/train_persona_{n}.json
        subjects/test_persona_{n}.json
        subjects/train_demographic_{n}.json
        subjects/test_demographic_{n}.json

    `n` defaults to `NUM_VOTERS_TRAIN` for split="train" and `NUM_VOTERS_TEST`
    for split="test", so most callers don't need to pass it. Multiple sizes can
    coexist on disk side by side (e.g. `..._20.json` and `..._50.json`) -- the
    config constants pick which one is wired into generate1 / compete.

    Materialized once by `new_experiments/scripts/build_audiences.py` so every
    (model, task, method) sees the *same* people.

    Args:
        kind: "persona" (singular) or "demographic" (singular).
        split: "train" or "test".
        n: optional override of the audience size; defaults to the matching
            NUM_VOTERS_{TRAIN,TEST} constant.
    """
    if kind not in ("persona", "demographic"):
        raise ValueError(f"kind must be 'persona' or 'demographic', got {kind!r}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    size = n if n is not None else _audience_size_for(split)
    return os.path.join(SUBJECTS_DIR, f"{split}_{kind}_{size}.json")


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


def ref_logprob_cache_path(task: str, model_name: str, method: str) -> str:
    """Sidecar cache of base-model reference logprob sums for DPO/KTO.

    Computed once per (task, model, method) by `tinker_utils.compute_ref_logprob_sums`
    and reused across re-runs of `train.py` so we only pay the base-model
    forward pass once per training entry.
    """
    return os.path.join(MODELS_DIR, task, model_name, method, "ref_logprobs.json")


def step2_path(task: str, model_name: str, method: str, split: str) -> str:
    """`generate2.py` / `generate22.py` output."""
    return os.path.join(RES_DIR, task, model_name, method, f"{split}_step2.json")


def competition_path(task: str) -> str:
    return os.path.join(RES_DIR, task, "competition.json")


def probes_path(task: str, qid: str) -> str:
    return os.path.join(RES_DIR, "probes", f"{task}_{qid}.csv")
