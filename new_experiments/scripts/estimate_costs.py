"""Cost estimator for the new_experiments pipeline.

Walks the same pipeline as `run_experiments.sh` and prints a per-stage,
per-model cost breakdown using:

- Tinker pricing (snapshot below) for sampling and training.
- OpenAI pricing for VOTER_MODEL_NAME (voters) and PROBE_MODEL_NAME (probes).

Token counts come from real tokenization when raw data files exist,
otherwise from heuristics derived from the original Qwen3-8B step1 outputs.

Usage:
    python -m new_experiments.scripts.estimate_costs                  # default
    python -m new_experiments.scripts.estimate_costs --limit 64       # smoke test
    python -m new_experiments.scripts.estimate_costs --json           # machine readable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List

# Make repo root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from new_experiments.src.config import (  # noqa: E402
    LORA_RANK,
    MODELS,
    NUM_EPOCHS,
    NUM_PLAYERS,
    NUM_VOTERS_COMPETE,
    NUM_VOTERS_TRAIN,
    PROBE_MODEL_NAME,
    TASKS,
    TRAINED_METHODS,
    VOTER_BIO_MODE,
    VOTER_MODEL_NAME,
    raw_split_path,
)


# Mean voter bio length per VOTER_BIO_MODE, measured on the full
# subjects/personas_{train,test}.json + demographics_{train,test}.json pool
# (1000 personas) using chars/4 + ~15% safety margin for tokenizer overhead.
BIO_TOKENS_PER_MODE = {
    "persona":      360,    # measured ~306 -> +safety
    "demographics": 120,    # measured  ~98 -> +safety
    "both":         500,    # measured ~414 -> +safety
}
BIO_TOKENS = BIO_TOKENS_PER_MODE.get(VOTER_BIO_MODE, 360)


# ---------- Pricing (USD per million tokens) ----------
# Tinker prices snapshotted from https://tinker-docs.thinkingmachines.ai/tinker/models/
# (May 2026 lineup; verified against the Tinker Console for the 5-model lineup).
# Llama-3.3-70B billed at the Llama-3.1-70B tier. Qwen3-8B billed at the
# Llama-3.1-8B tier (same dense 8B class).
TINKER_PRICES: Dict[str, Dict[str, float]] = {
    "meta-llama/Llama-3.1-8B-Instruct":  {"prefill": 0.13, "sample": 0.40, "train": 0.40},
    "meta-llama/Llama-3.3-70B-Instruct": {"prefill": 1.05, "sample": 3.16, "train": 3.16},
    "Qwen/Qwen3-8B":                     {"prefill": 0.13, "sample": 0.40, "train": 0.40},
    "Qwen/Qwen3-32B":                    {"prefill": 0.49, "sample": 1.47, "train": 1.47},
    "openai/gpt-oss-20b":                {"prefill": 0.12, "sample": 0.30, "train": 0.36},
}

# OpenAI prices (gpt-4o-mini for voters and probes by default; gpt-4o kept
# as a fallback price for runs that flip PROBE_MODEL_NAME back. Jan 2026 list.)
OPENAI_PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}

# Heuristic token counts per task (measured from artsco/.../train_step1.json on Qwen3-8B,
# inflated 1.3x to be conservative for 70B-class outputs and a bit of chat-template overhead).
TASK_TOKENS = {
    "task_elections": {"prompt": 720, "completion": 350, "tfb_prompt": 1100, "voter_think": 130},
    "task_sales":     {"prompt": 470, "completion": 220, "tfb_prompt":  900, "voter_think": 130},
    "task_sm":        {"prompt": 490, "completion": 190, "tfb_prompt":  870, "voter_think": 130},
}

DEFAULT_N_TRAIN = 1024
DEFAULT_N_TEST = 1024


@dataclass
class StageCost:
    name: str
    detail: str = ""
    tinker_prefill_tokens: int = 0
    tinker_sample_tokens: int = 0
    tinker_train_tokens: int = 0
    openai_input_tokens: Dict[str, int] = field(default_factory=dict)  # model -> tokens
    openai_output_tokens: Dict[str, int] = field(default_factory=dict)
    by_model_cost: Dict[str, float] = field(default_factory=dict)
    openai_cost: float = 0.0

    @property
    def tinker_cost(self) -> float:
        return sum(self.by_model_cost.values())

    @property
    def total(self) -> float:
        return self.tinker_cost + self.openai_cost


def _row_count(task: str, split: str) -> int:
    """Count rows in raw JSON; fall back to defaults if file is missing."""
    path = raw_split_path(task, split)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return len(json.load(f))
        except Exception:
            pass
    return DEFAULT_N_TRAIN if split == "train" else DEFAULT_N_TEST


def _apply_limit(n: int, limit: int | None) -> int:
    return min(n, limit) if limit else n


def _tinker_cost(model: str, prefill: int, sample: int, train: int) -> float:
    p = TINKER_PRICES[model]
    return (
        prefill / 1_000_000 * p["prefill"]
        + sample / 1_000_000 * p["sample"]
        + train / 1_000_000 * p["train"]
    )


def _openai_cost(model: str, in_tok: int, out_tok: int) -> float:
    p = OPENAI_PRICES[model]
    return in_tok / 1_000_000 * p["input"] + out_tok / 1_000_000 * p["output"]


# ---------- Per-stage estimators ----------
def estimate_generate1(limit: int | None) -> StageCost:
    """Baseline + voter feedback on train split (Tinker sample + GPT-4o-mini voters)."""
    sc = StageCost(name="generate1", detail="baseline sampling + GPT-4o-mini voter feedback (train)")

    for task in TASKS:
        n = _apply_limit(_row_count(task, "train"), limit)
        tok = TASK_TOKENS[task]
        # 1 prefill per prompt, NUM_PLAYERS samples per prompt
        sc.tinker_prefill_tokens += n * tok["prompt"]
        sc.tinker_sample_tokens += n * NUM_PLAYERS * tok["completion"]

        # Voter calls (gpt-4o-mini): NUM_VOTERS_TRAIN voters x n prompts.
        # Each voter sees both candidates.
        voter_calls = NUM_VOTERS_TRAIN * n
        # voter prompt = bio (BIO_TOKENS, depends on VOTER_BIO_MODE) +
        # instructions (~250) + 2 candidates (~2 * completion).
        in_per_call = BIO_TOKENS + 250 + 2 * tok["completion"]
        out_per_call = 200  # short think + <vote>X</vote>
        sc.openai_input_tokens["gpt-4o-mini"] = sc.openai_input_tokens.get("gpt-4o-mini", 0) + voter_calls * in_per_call
        sc.openai_output_tokens["gpt-4o-mini"] = sc.openai_output_tokens.get("gpt-4o-mini", 0) + voter_calls * out_per_call

    for model in MODELS:
        sc.by_model_cost[model] = _tinker_cost(
            model, sc.tinker_prefill_tokens, sc.tinker_sample_tokens, 0
        )
    sc.openai_cost = _openai_cost(
        "gpt-4o-mini",
        sc.openai_input_tokens.get("gpt-4o-mini", 0),
        sc.openai_output_tokens.get("gpt-4o-mini", 0),
    ) * len(MODELS)  # voters run once per (model, task, prompt)
    return sc


def estimate_train() -> StageCost:
    """LoRA SFT on RFT and TFB datasets (Tinker train tokens)."""
    sc = StageCost(
        name="train",
        detail=f"Tinker LoRA SFT (rank={LORA_RANK}, epochs={NUM_EPOCHS}, methods=rft+tfb)",
    )

    # Approximate dataset sizes after build_train_data (mirrors step2.1 logic).
    # ~50% of train prompts have a non-tied winner, x3 replication = 1.5 * n entries.
    # TFB = rft + min(rft_size, voter_thinks_total) -> ~2 * rft_size.
    for task in TASKS:
        n = _row_count(task, "train")
        tok = TASK_TOKENS[task]
        n_rft = int(n * 0.5 * 3)               # ~1536
        n_tfb = int(n_rft * 2)                  # ~3072 (rft + voter-think samples)
        # RFT entries: prompt + completion
        rft_tokens = n_rft * (tok["prompt"] + tok["completion"])
        # TFB voter-think entries: tfb_prompt (~candidates+context) + voter_think (~130)
        # Half of TFB is rft_list, half is voter-think entries (approximation).
        tfb_voter = (n_tfb - n_rft) * (tok["tfb_prompt"] + tok["voter_think"])
        tfb_rft = n_rft * (tok["prompt"] + tok["completion"])
        train_tokens = (rft_tokens + tfb_voter + tfb_rft) * NUM_EPOCHS
        sc.tinker_train_tokens += train_tokens

    for model in MODELS:
        sc.by_model_cost[model] = _tinker_cost(model, 0, 0, sc.tinker_train_tokens)
    return sc


def estimate_generate2(limit: int | None) -> StageCost:
    """Test inference for base + rft + tfb (Tinker sample only)."""
    sc = StageCost(name="generate2/22", detail="Tinker sampling on test split (base + rft + tfb)")
    n_methods = 1 + len(TRAINED_METHODS)  # base + rft + tfb
    for task in TASKS:
        n = _apply_limit(_row_count(task, "test"), limit)
        tok = TASK_TOKENS[task]
        sc.tinker_prefill_tokens += n * tok["prompt"] * n_methods
        sc.tinker_sample_tokens += n * NUM_PLAYERS * tok["completion"] * n_methods
    for model in MODELS:
        sc.by_model_cost[model] = _tinker_cost(
            model, sc.tinker_prefill_tokens, sc.tinker_sample_tokens, 0
        )
    return sc


def estimate_compete(limit: int | None) -> StageCost:
    """Pairwise voter competition between methods (gpt-4o-mini)."""
    sc = StageCost(
        name="compete",
        detail=f"{NUM_VOTERS_COMPETE} voters, 3 method pairs (base/rft/tfb), gpt-4o-mini",
    )
    pairs = 3  # (base, rft), (base, tfb), (rft, tfb)
    for task in TASKS:
        n = _apply_limit(_row_count(task, "test"), limit)
        tok = TASK_TOKENS[task]
        calls = NUM_VOTERS_COMPETE * n * pairs * len(MODELS)
        # bio (BIO_TOKENS, see generate1) + instructions (~250) + 2 candidates.
        in_per_call = BIO_TOKENS + 250 + 2 * tok["completion"]
        out_per_call = 200
        sc.openai_input_tokens["gpt-4o-mini"] = sc.openai_input_tokens.get("gpt-4o-mini", 0) + calls * in_per_call
        sc.openai_output_tokens["gpt-4o-mini"] = sc.openai_output_tokens.get("gpt-4o-mini", 0) + calls * out_per_call
    sc.openai_cost = _openai_cost(
        "gpt-4o-mini",
        sc.openai_input_tokens.get("gpt-4o-mini", 0),
        sc.openai_output_tokens.get("gpt-4o-mini", 0),
    )
    return sc


def estimate_probes(limit: int | None) -> StageCost:
    """q1/q2 misalignment probes via PROBE_MODEL_NAME (see config.py)."""
    sc = StageCost(
        name="probes",
        detail=f"{PROBE_MODEL_NAME} on test generations (q1+q2 per task)",
    )
    probes_per_task = {"task_elections": 2, "task_sales": 1, "task_sm": 2}
    n_methods = 1 + len(TRAINED_METHODS)
    for task in TASKS:
        n = _apply_limit(_row_count(task, "test"), limit)
        tok = TASK_TOKENS[task]
        calls = probes_per_task[task] * n * len(MODELS) * n_methods
        in_per_call = tok["prompt"] + 2 * tok["completion"] + 250  # rubric overhead
        out_per_call = 250
        sc.openai_input_tokens[PROBE_MODEL_NAME] = (
            sc.openai_input_tokens.get(PROBE_MODEL_NAME, 0) + calls * in_per_call
        )
        sc.openai_output_tokens[PROBE_MODEL_NAME] = (
            sc.openai_output_tokens.get(PROBE_MODEL_NAME, 0) + calls * out_per_call
        )
    sc.openai_cost = _openai_cost(
        PROBE_MODEL_NAME,
        sc.openai_input_tokens.get(PROBE_MODEL_NAME, 0),
        sc.openai_output_tokens.get(PROBE_MODEL_NAME, 0),
    )
    return sc


# ---------- Driver ----------
def fmt_money(x: float) -> str:
    return f"${x:>9,.2f}"


def fmt_tok(x: int) -> str:
    return f"{x/1e6:>7.2f}M" if x else f"{'-':>8}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Same --limit you'd pass to run_experiments.sh; caps n per task.")
    p.add_argument("--json", action="store_true", help="Output JSON instead of a table.")
    args = p.parse_args()

    stages: List[StageCost] = [
        estimate_generate1(args.limit),
        estimate_train(),
        estimate_generate2(args.limit),
        estimate_compete(args.limit),
        estimate_probes(args.limit),
    ]

    if args.json:
        print(json.dumps([asdict(s) for s in stages], indent=2))
        return

    print("=" * 110)
    print(f"new_experiments cost estimate  (models: {', '.join(MODELS)})")
    print(f"  voters: train={NUM_VOTERS_TRAIN}  compete={NUM_VOTERS_COMPETE}  "
          f"bio_mode={VOTER_BIO_MODE} (~{BIO_TOKENS} tokens/bio)  "
          f"probe_model={PROBE_MODEL_NAME}")
    if args.limit:
        print(f"  --limit {args.limit}  (capped to first {args.limit} prompts per task)")
    print("=" * 110)

    print(f"{'stage':14} {'detail':62} {'prefill':>9} {'sample':>9} {'train':>9}")
    print("-" * 110)
    for s in stages:
        print(
            f"{s.name:14} {s.detail[:62]:62} "
            f"{fmt_tok(s.tinker_prefill_tokens):>9} "
            f"{fmt_tok(s.tinker_sample_tokens):>9} "
            f"{fmt_tok(s.tinker_train_tokens):>9}"
        )

    print()
    print(f"{'stage':14} " + "  ".join(f"{m.split('/')[-1][:24]:>24}" for m in MODELS) + f"  {'openai':>10}  {'total':>10}")
    print("-" * 110)
    grand_total = 0.0
    per_model_totals = {m: 0.0 for m in MODELS}
    openai_total = 0.0
    for s in stages:
        per_model_str = "  ".join(fmt_money(s.by_model_cost.get(m, 0.0)).rjust(24) for m in MODELS)
        for m in MODELS:
            per_model_totals[m] += s.by_model_cost.get(m, 0.0)
        openai_total += s.openai_cost
        grand_total += s.total
        print(f"{s.name:14} {per_model_str}  {fmt_money(s.openai_cost):>10}  {fmt_money(s.total):>10}")
    print("-" * 110)
    totals_str = "  ".join(fmt_money(per_model_totals[m]).rjust(24) for m in MODELS)
    print(f"{'TOTAL':14} {totals_str}  {fmt_money(openai_total):>10}  {fmt_money(grand_total):>10}")
    print("=" * 110)
    print()
    print("Notes:")
    print("  * Tinker prices snapshotted from https://tinker-docs.thinkingmachines.ai/tinker/models/")
    print("    Llama-3.3-70B is billed at the Llama-3.1-70B tier (no separate listing).")
    print("  * Token counts use heuristics derived from artsco's Qwen3-8B step1 outputs,")
    print("    inflated 1.3x for chat-template overhead and 70B-class output length.")
    print("  * OpenAI voter and probe costs scale with both models (voters score per-model output).")
    print("  * Use --limit 64 for a cheap smoke test before committing to the full run.")


if __name__ == "__main__":
    main()
