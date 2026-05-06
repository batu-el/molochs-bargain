"""Step 2 — baseline generation with audience feedback (train split).

Mirrors `artsco/src/generate1.py`: for every training prompt we sample
`NUM_PLAYERS=2` independent completions from the *base* model, then ask
`NUM_VOTERS_TRAIN` simulated voters (GPT-4o-mini) which one they prefer.
Output JSONL has one record per prompt with the same schema as the original
step1 file, so `build_train_data.py` can reuse the existing logic verbatim.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import List

from datasets import load_dataset, Dataset

# Patch openai timeout/retries BEFORE importing artsco.voter which constructs
# the client at module-import time.
from new_experiments.src import openai_patch  # noqa: F401

# Reuse the original voter system unchanged.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from artsco.voter.voters import Voters  # noqa: E402

from new_experiments.src.config import (  # noqa: E402
    MODELS,
    NUM_PLAYERS,
    NUM_VOTERS_TRAIN,
    TASKS,
    VOTER_BIO_MODE,
    VOTER_MODEL_NAME,
    VOTER_SAMPLE_SEED,
    step1_path,
    templated_split_path,
)
from new_experiments.src.data_utils import extract_answer, extract_think  # noqa: E402
from new_experiments.src.format_adapters import get_adapter  # noqa: E402
from new_experiments.src.personas import load_voter_bios  # noqa: E402
from new_experiments.src.tinker_utils import (  # noqa: E402
    get_base_sampling_client,
    get_tokenizer,
    sample_many,
)


async def _run(task: str, model_name: str, max_concurrency: int, limit: int | None) -> None:
    src = templated_split_path(task, model_name, "train")
    dst = step1_path(task, model_name, "train")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    ds = load_dataset("json", data_files=src, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    prompts: List[str] = list(ds["prompt"])

    print(f"[generate1] task={task} model={model_name} prompts={len(prompts)} -> {dst}")

    tokenizer = get_tokenizer(model_name)
    sampling_client = get_base_sampling_client(model_name)
    adapter = get_adapter(model_name)

    sampling_results = await sample_many(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        prompts=prompts,
        num_samples=NUM_PLAYERS,
        max_concurrency=max_concurrency,
        skip_special_tokens=adapter.skip_special_tokens_on_decode,
    )

    completions: List[List[str]] = [r.completions for r in sampling_results]
    player_thinks = [[extract_think(c, model_name) for c in row] for row in completions]
    player_candidates = [[extract_answer(c, task, model_name) for c in row] for row in completions]

    # ----- Voter feedback (GPT-4o-mini) -----
    # Sample NUM_VOTERS_TRAIN of the 800 train personas (in `subjects/`)
    # without replacement using a fixed seed -> same voters across all
    # (model, task, prompt). bio_mode controls which surface form is fed
    # to the voter (persona text by default; "demographics" for the
    # ablation, see config.VOTER_BIO_MODE).
    bios = load_voter_bios(
        "train", n=NUM_VOTERS_TRAIN, seed=VOTER_SAMPLE_SEED, bio_mode=VOTER_BIO_MODE,
    )
    voters = Voters(bios=bios, task=task, model_name=VOTER_MODEL_NAME)
    voter_votes, voter_thinks, _voter_choices = voters.get_votes_list(player_candidates)

    records = []
    for p, c, pc, pt, vv, vt in zip(
        prompts, completions, player_candidates, player_thinks, voter_votes, voter_thinks
    ):
        records.append({
            "prompt": [p] * NUM_PLAYERS,
            "completion": c,
            "player_candidates": pc,
            "player_thinks": pt,
            "voter_votes": vv,
            "voter_thinks": vt,
        })

    if os.path.exists(dst):
        os.remove(dst)
    Dataset.from_list(records).to_json(dst)
    print(f"[generate1] wrote {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--max_concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of prompts (debugging only).")
    args = p.parse_args()

    tasks = [args.task] if args.task else TASKS
    models = [args.model] if args.model else MODELS

    for task in tasks:
        for model_name in models:
            asyncio.run(_run(task, model_name, args.max_concurrency, args.limit))


if __name__ == "__main__":
    main()
