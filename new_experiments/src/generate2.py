"""Step 6 — sample from a trained LoRA on the test split.

Mirrors `artsco/src/generate2.py`. Loads the checkpoint URI saved by
`train.py`, recreates a Tinker SamplingClient, and writes an
`{split}_step2.json` file with the same schema as the original output so the
downstream `compete.py` and `probes.py` can consume it directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from datasets import Dataset, load_dataset

from new_experiments.src.config import (
    MODELS,
    NUM_PLAYERS,
    TASKS,
    TRAINED_METHODS,
    model_state_path,
    step2_path,
    templated_split_path,
)
from new_experiments.src.data_utils import extract_answer, extract_think
from new_experiments.src.format_adapters import get_adapter
from new_experiments.src.tinker_utils import (
    get_sampling_client_from_uri,
    get_tokenizer,
    sample_many,
)


async def _run(task: str, model_name: str, method: str, split: str, max_concurrency: int, limit: int | None) -> None:
    state_path = model_state_path(task, model_name, method)
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing trained model state: {state_path}. Run train.py first.")
    with open(state_path) as f:
        state = json.load(f)
    checkpoint_uri = state["checkpoint_uri"]

    src = templated_split_path(task, model_name, split)
    dst = step2_path(task, model_name, method, split)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    ds = load_dataset("json", data_files=src, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    prompts = list(ds["prompt"])

    print(f"[generate2] task={task} model={model_name} method={method} split={split} prompts={len(prompts)}")
    print(f"[generate2] checkpoint={checkpoint_uri}")

    tokenizer = get_tokenizer(model_name)
    sampling_client = get_sampling_client_from_uri(checkpoint_uri)
    adapter = get_adapter(model_name)

    sampling_results = await sample_many(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        prompts=prompts,
        num_samples=NUM_PLAYERS,
        max_concurrency=max_concurrency,
        skip_special_tokens=adapter.skip_special_tokens_on_decode,
    )

    records = []
    for prompt, res in zip(prompts, sampling_results):
        completions = res.completions
        thinks = [extract_think(c, model_name) for c in completions]
        candidates = [extract_answer(c, task, model_name) for c in completions]
        records.append({
            "prompt": [prompt] * NUM_PLAYERS,
            "completion": completions,
            "player_candidates": candidates,
            "player_thinks": thinks,
        })

    if os.path.exists(dst):
        os.remove(dst)
    Dataset.from_list(records).to_json(dst)
    print(f"[generate2] wrote {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--method", choices=TRAINED_METHODS, default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--max_concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    tasks = [args.task] if args.task else TASKS
    models = [args.model] if args.model else MODELS
    methods = [args.method] if args.method else TRAINED_METHODS

    for task in tasks:
        for model_name in models:
            for method in methods:
                asyncio.run(_run(task, model_name, method, args.split, args.max_concurrency, args.limit))


if __name__ == "__main__":
    main()
