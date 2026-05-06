"""Step 7 — pairwise voter competition between methods.

Mirrors `artsco/step2.2{election,sales,sm}.ipynb`:
For every (model, task) and every method pair (base, rft), (base, tfb),
(rft, tfb), build N test-set duels using the player-0 candidate from each
method and ask `NUM_VOTERS_COMPETE` simulated voters to choose. Report mean
preference for the second method (>0.5 means the second won).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset

# Patch openai timeout/retries BEFORE importing artsco.voter which constructs
# the client at module-import time.
from new_experiments.src import openai_patch  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from artsco.voter.voters import Voters  # noqa: E402

from new_experiments.src.config import (  # noqa: E402
    METHODS,
    MODELS,
    NUM_VOTERS_COMPETE,
    TASKS,
    VOTER_BIO_MODE,
    VOTER_MODEL_NAME,
    VOTER_SAMPLE_SEED,
    competition_path,
    step2_path,
)
from new_experiments.src.personas import load_voter_bios  # noqa: E402


METHOD_PAIRS: List[Tuple[str, str]] = [("base", "rft"), ("base", "tfb"), ("rft", "tfb")]


def _load_player0(task: str, model_name: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for method in METHODS:
        path = step2_path(task, model_name, method, "test")
        if not os.path.exists(path):
            print(f"[compete] WARNING: missing {path}")
            out[method] = []
            continue
        ds = load_dataset("json", data_files=path, split="train")
        out[method] = [c[0] for c in ds["player_candidates"]]
    return out


def _run_one(task: str, limit: int | None) -> dict:
    # Use the *test* persona split so the compete-time voter pool is disjoint
    # from the train-time audience used in generate1.py. Sample
    # NUM_VOTERS_COMPETE of the 200 test personas without replacement using
    # a fixed seed; bio_mode follows config.VOTER_BIO_MODE so persona vs
    # demographics ablations stay aligned across train and test stages.
    bios = load_voter_bios(
        "test", n=NUM_VOTERS_COMPETE, seed=VOTER_SAMPLE_SEED, bio_mode=VOTER_BIO_MODE,
    )
    voters = Voters(bios=bios, task=task, model_name=VOTER_MODEL_NAME)

    results: dict = {"mean": {}, "std": {}}
    for model_name in MODELS:
        results["mean"][model_name] = {}
        results["std"][model_name] = {}
        cands = _load_player0(task, model_name)

        for first, second in METHOD_PAIRS:
            if not cands.get(first) or not cands.get(second):
                print(f"[compete] skip {model_name} {first}-{second} (missing data)")
                continue
            n = min(len(cands[first]), len(cands[second]))
            if limit is not None:
                n = min(n, limit)
            duels = list(zip(cands[first][:n], cands[second][:n]))

            votes_2d, _thoughts_2d, _choices_2d = voters.get_votes_list(duels)
            second_wins = []
            for v in votes_2d:
                c = Counter(v)
                total = c.get(0, 0) + c.get(1, 0)
                if total == 0:
                    continue
                second_wins.append(round(c.get(1, 0) / total, 2))
            mean = float(np.mean(second_wins)) if second_wins else float("nan")
            std = float(np.std(second_wins)) if second_wins else float("nan")
            key = f"{first}-{second}"
            results["mean"][model_name][key] = mean
            results["std"][model_name][key] = std
            print(f"[compete] {model_name} {key}: mean={mean:.3f} std={std:.3f}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    tasks = [args.task] if args.task else TASKS
    for task in tasks:
        out_path = competition_path(task)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        results = _run_one(task, args.limit)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[compete] wrote {out_path}")


if __name__ == "__main__":
    main()
