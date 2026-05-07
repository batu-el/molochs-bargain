"""Step 2.1 — turn step1 output into RFT, TFB, DPO, and KTO SFT datasets.

For every train prompt the step1 file already has two model completions and
their per-voter votes. From that we build four parallel training corpora:

| Method | Examples per prompt | Loss                                                     |
|--------|--------------------|----------------------------------------------------------|
| RFT    | (prompt, winner) only                          | SFT NLL on winner                              |
| TFB    | (prompt, winner) + (tfb_prompt, voter_think)   | SFT NLL warm-up on voter chains-of-thought     |
| DPO    | (prompt, winner, loser) paired                 | pairwise contrastive vs reference (DPO loss)  |
| KTO    | (prompt, winner, des=True) + (prompt, loser, des=False) | per-example KTO loss vs reference     |

Notes:
- Ties (winner_votes == loser_votes) are skipped for *all* four methods so each
  prompt contributes deterministically.
- The TFB augmentation is *only* about reusing voter chains-of-thought as warm-
  up data on top of RFT-style targets; it does NOT interact with DPO or KTO.
- We deliberately do NOT replicate the dataset (no `* 3` like the original
  pipeline). NUM_EPOCHS=3 in config.py is the explicit, comparable knob across
  all four methods.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import List

from datasets import Dataset

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


def _load_json_records(path: str):
    """Load JSON array or JSONL without routing large nested strings through PyArrow."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _winner_loser_idx(votes: List[int]) -> tuple[int, int] | None:
    """Return (winner_idx, loser_idx) in {0,1} or None on ties / empty votes."""
    counts = Counter(votes)
    diff = counts.get(0, 0) - counts.get(1, 0)
    if diff > 0:
        return 0, 1
    if diff < 0:
        return 1, 0
    return None


def get_rft(dataset):
    rft = []
    for idx in range(len(dataset)):
        wl = _winner_loser_idx(dataset[idx]["voter_votes"])
        if wl is None:
            continue
        winner = wl[0]
        rft.append({
            "prompt": dataset[idx]["prompt"][winner],
            "completion": dataset[idx]["completion"][winner],
        })
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


def get_dpo(dataset):
    """Pairs of (prompt, chosen, rejected). Skip ties."""
    out = []
    for idx in range(len(dataset)):
        wl = _winner_loser_idx(dataset[idx]["voter_votes"])
        if wl is None:
            continue
        winner, loser = wl
        ex = dataset[idx]
        # Both candidates were sampled from the same prompt, so prompt[0]==prompt[1].
        out.append({
            "prompt": ex["prompt"][winner],
            "chosen": ex["completion"][winner],
            "rejected": ex["completion"][loser],
        })
    return out


def get_kto(dataset):
    """Per-example (prompt, completion, desirable). Each non-tie prompt yields
    one desirable=True (winner) and one desirable=False (loser) entry.
    """
    out = []
    for idx in range(len(dataset)):
        wl = _winner_loser_idx(dataset[idx]["voter_votes"])
        if wl is None:
            continue
        winner, loser = wl
        ex = dataset[idx]
        out.append({
            "prompt": ex["prompt"][winner],
            "completion": ex["completion"][winner],
            "desirable": True,
        })
        out.append({
            "prompt": ex["prompt"][loser],
            "completion": ex["completion"][loser],
            "desirable": False,
        })
    return out


def _write_dataset(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    Dataset.from_list(rows).to_json(path)


def build_one(task: str, model_name: str, split: str = "train", seed: int = 0) -> None:
    rng = random.Random(seed)

    step1_ds = _load_json_records(step1_path(task, model_name, split))
    base_ds = _load_json_records(raw_split_path(task, split))
    tokenizer = get_tokenizer(model_name)

    # ----- RFT (winner only) -----
    rft_list = get_rft(step1_ds)
    rng.shuffle(rft_list)
    rft_path = train_data_path(task, model_name, split, "rft")
    _write_dataset(rft_list, rft_path)
    print(f"[build_train_data] wrote {rft_path} ({len(rft_list)} rows)")

    # ----- TFB (RFT + voter-think warm-up; sized symmetrically to RFT) -----
    tfb_voter = get_tfb(step1_ds, base_ds, tokenizer, task=task, model_name=model_name)
    rng.shuffle(tfb_voter)
    tfb_list = tfb_voter[: len(rft_list)] + rft_list
    rng.shuffle(tfb_list)
    tfb_path = train_data_path(task, model_name, split, "tfb")
    _write_dataset(tfb_list, tfb_path)
    print(f"[build_train_data] wrote {tfb_path} ({len(tfb_list)} rows)")

    # ----- DPO (paired chosen/rejected) -----
    dpo_list = get_dpo(step1_ds)
    rng.shuffle(dpo_list)
    dpo_path = train_data_path(task, model_name, split, "dpo")
    _write_dataset(dpo_list, dpo_path)
    print(f"[build_train_data] wrote {dpo_path} ({len(dpo_list)} pairs)")

    # ----- KTO (unpaired (prompt, completion, desirable)) -----
    kto_list = get_kto(step1_ds)
    # Don't shuffle: train.py interleaves desirable/undesirable in the batch
    # (every consecutive pair is the same prompt's winner+loser) so the per-
    # batch KL anchor stays balanced and the loss has a stable mean signal.
    kto_path = train_data_path(task, model_name, split, "kto")
    _write_dataset(kto_list, kto_path)
    print(f"[build_train_data] wrote {kto_path} ({len(kto_list)} entries)")


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
