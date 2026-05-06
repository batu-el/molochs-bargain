"""Step 2.1 — turn step1 output into RFT and TFB SFT datasets.

Direct port of `artsco/step2.1.ipynb`:

- RFT picks the player with more voter support per prompt and keeps the
  (prompt, completion) pair, then replicates 3x and shuffles.
- TFB constructs (tfb_prompt, "<think> voter_think </think>") pairs using a
  voter-think prompt builder, then concatenates with the RFT data.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter

from datasets import Dataset, load_dataset

from new_experiments.src.config import (
    MODELS,
    TASKS,
    raw_split_path,
    step1_path,
    train_data_path,
)
from new_experiments.src.data_utils import build_tfb_prompt
from new_experiments.src.format_adapters import render_think_only_completion
from new_experiments.src.tinker_utils import get_tokenizer


def get_rft(dataset):
    rft = []
    for idx in range(len(dataset)):
        votes = dataset[idx]["voter_votes"]
        counts = Counter(votes)
        diff = counts.get(0, 0) - counts.get(1, 0)
        if diff > 0:
            entry = {k: v[0] for k, v in dataset[idx].items() if k in ("prompt", "completion")}
        elif diff < 0:
            entry = {k: v[1] for k, v in dataset[idx].items() if k in ("prompt", "completion")}
        else:
            continue
        rft.append(entry)
    return rft


def get_tfb(dataset, dataset_base, tokenizer, task: str, model_name: str):
    tfb = []
    for idx in range(len(dataset)):
        candidates = dataset[idx]["player_candidates"]
        thinks = dataset[idx]["voter_thinks"]
        example = dataset_base[idx]
        prompt = build_tfb_prompt(example, candidates, tokenizer, task, model_name)
        for think in thinks:
            if not think:
                continue
            completion = render_think_only_completion(model_name, think)
            tfb.append({"prompt": prompt, "completion": completion})
    return tfb


def build_one(task: str, model_name: str, split: str = "train", seed: int = 0) -> None:
    rng = random.Random(seed)

    step1_ds = load_dataset("json", data_files=step1_path(task, model_name, split), split="train")
    base_ds = load_dataset("json", data_files=raw_split_path(task, split), split="train")
    tokenizer = get_tokenizer(model_name)

    # ----- RFT -----
    rft_list = get_rft(step1_ds)
    rft_list = rft_list * 3
    rng.shuffle(rft_list)
    rft_path = train_data_path(task, model_name, split, "rft")
    os.makedirs(os.path.dirname(rft_path), exist_ok=True)
    if os.path.exists(rft_path):
        os.remove(rft_path)
    Dataset.from_list(rft_list).to_json(rft_path)
    print(f"[build_train_data] wrote {rft_path} ({len(rft_list)} rows)")

    # ----- TFB -----
    tfb_list = get_tfb(step1_ds, base_ds, tokenizer, task=task, model_name=model_name)
    rng.shuffle(tfb_list)
    tfb_list = tfb_list[: len(rft_list)] + rft_list
    rng.shuffle(tfb_list)
    tfb_path = train_data_path(task, model_name, split, "tfb")
    if os.path.exists(tfb_path):
        os.remove(tfb_path)
    Dataset.from_list(tfb_list).to_json(tfb_path)
    print(f"[build_train_data] wrote {tfb_path} ({len(tfb_list)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tasks = [args.task] if args.task else TASKS
    models = [args.model] if args.model else MODELS

    for task in tasks:
        for model_name in models:
            build_one(task, model_name, "train", seed=args.seed)


if __name__ == "__main__":
    main()
