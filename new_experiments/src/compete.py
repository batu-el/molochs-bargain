"""Step 7 — pairwise voter competition between methods.

Mirrors `artsco/step2.2{election,sales,sm}.ipynb`:
For every (model, task) and every method pair (base, rft), (base, tfb),
(rft, tfb), build N test-set duels using the player-0 candidate from each
method and ask `NUM_VOTERS_COMPETE` simulated voters to choose. Report mean
preference for the second method (>0.5 means the second won).
"""

from __future__ import annotations

import argparse
import glob
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
    RES_DIR,
    TASKS,
    VOTER_BIO_MODE,
    VOTER_MODEL_NAME,
    VOTER_SAMPLE_SEED,
    competition_path,
    step2_path,
)
from new_experiments.src.personas import load_voter_bios  # noqa: E402


METHOD_PAIRS: List[Tuple[str, str]] = [("base", "rft"), ("base", "tfb"), ("rft", "tfb")]


def _competition_part_path(task: str, model_name: str) -> str:
    """Per-model intermediate file for a single (task, model) shard."""
    return os.path.join(RES_DIR, task, "competition_parts", f"{model_name}.json")


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


def _run_one(task: str, limit: int | None, models: List[str]) -> dict:
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
    for model_name in models:
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


def _merge_parts(task: str) -> None:
    """Consolidate every res/{task}/competition_parts/**/*.json into competition.json."""
    parts_dir = os.path.join(RES_DIR, task, "competition_parts")
    if not os.path.isdir(parts_dir):
        print(f"[compete-merge] {task}: no parts dir at {parts_dir}; skipping")
        return
    part_files = sorted(glob.glob(os.path.join(parts_dir, "**", "*.json"), recursive=True))
    if not part_files:
        print(f"[compete-merge] {task}: no part files under {parts_dir}; skipping")
        return

    merged: dict = {"mean": {}, "std": {}}
    for path in part_files:
        with open(path) as f:
            part = json.load(f)
        for k in ("mean", "std"):
            for model_name, pair_to_val in part.get(k, {}).items():
                if model_name in merged[k]:
                    print(
                        f"[compete-merge] WARNING: {task} model={model_name} "
                        f"already merged; overwriting from {path}"
                    )
                merged[k][model_name] = pair_to_val

    out_path = competition_path(task)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[compete-merge] wrote {out_path}  (merged {len(part_files)} part(s))")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Restrict to a single model and write to a per-model part file under "
            "res/{task}/competition_parts/. Use --merge afterwards to consolidate."
        ),
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="Consolidate per-model part files into the final competition.json (no API calls).",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if args.model is not None and args.model not in MODELS:
        raise SystemExit(f"--model must be one of {MODELS}, got {args.model!r}")

    tasks = [args.task] if args.task else TASKS

    if args.merge:
        for task in tasks:
            _merge_parts(task)
        return

    models = [args.model] if args.model else list(MODELS)
    for task in tasks:
        if args.model:
            out_path = _competition_part_path(task, args.model)
        else:
            out_path = competition_path(task)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        results = _run_one(task, args.limit, models)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[compete] wrote {out_path}")


if __name__ == "__main__":
    main()
