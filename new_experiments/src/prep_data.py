"""Step 1.1 — apply chat templates per (model, task, split).

Reads the model-agnostic raw data from `artsco/data/{task}/{split}.json` and
writes per-model JSONL files to
`new_experiments/data/{task}/{model}/{split}.json` with an extra `prompt`
field that contains the chat-templated string for that model.
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset

from new_experiments.src.config import (
    MODELS,
    SPLITS,
    TASKS,
    raw_split_path,
    templated_split_path,
)
from new_experiments.src.data_utils import process_dataset
from new_experiments.src.tinker_utils import get_tokenizer


def prep_one(task: str, model_name: str, split: str) -> None:
    src = raw_split_path(task, split)
    dst = templated_split_path(task, model_name, split)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    ds = load_dataset("json", data_files=src, split="train")
    tokenizer = get_tokenizer(model_name)
    ds = ds.map(
        lambda ex: {"prompt": process_dataset(ex, tokenizer=tokenizer, ds_name=task, model_name=model_name)},
        load_from_cache_file=False,
    )

    if os.path.exists(dst):
        os.remove(dst)
    ds.to_json(dst)
    print(f"[prep_data] wrote {dst} ({len(ds)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--split", choices=SPLITS, default=None)
    args = p.parse_args()

    tasks = [args.task] if args.task else TASKS
    models = [args.model] if args.model else MODELS
    splits = [args.split] if args.split else SPLITS

    for task in tasks:
        for model_name in models:
            for split in splits:
                prep_one(task, model_name, split)


if __name__ == "__main__":
    main()
