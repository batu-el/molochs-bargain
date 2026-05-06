"""Step 6 — sample from the *base* model on the test split (no audience).

Mirrors `artsco/src/generate22.py`: identical to `generate2.py` but uses the
base model (method = "base") so we have a control to compare RFT/TFB against.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from datasets import Dataset, load_dataset

from new_experiments.src.config import (
    MODELS,
    NUM_PLAYERS,
    TASKS,
    step2_path,
    templated_split_path,
)
from new_experiments.src.data_utils import extract_answer, extract_think
from new_experiments.src.format_adapters import get_adapter
from new_experiments.src.tinker_utils import (
    get_base_sampling_client,
    get_tokenizer,
    sample_many,
)


async def _run(task: str, model_name: str, split: str, max_concurrency: int, limit: int | None) -> None:
    src = templated_split_path(task, model_name, split)
    dst = step2_path(task, model_name, "base", split)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    ds = load_dataset("json", data_files=src, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    prompts = list(ds["prompt"])

    print(f"[generate22] task={task} model={model_name} split={split} prompts={len(prompts)}")

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
    print(f"[generate22] wrote {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--max_concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    tasks = [args.task] if args.task else TASKS
    models = [args.model] if args.model else MODELS

    for task in tasks:
        for model_name in models:
            asyncio.run(_run(task, model_name, args.split, args.max_concurrency, args.limit))


if __name__ == "__main__":
    main()
