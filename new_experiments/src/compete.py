"""Step 7 — pairwise voter competition between methods.

For every (model, task) and every method pair we build N test-set duels using
the player-0 candidate from each method, then ask the fixed train-audience
voters (`NUM_VOTERS_TRAIN` people from `subjects/train_{persona,demographic}_{N}.json`)
to choose. The *same* audience evaluates training rollouts in generate1 and
the test-set duels here, so the comparison is end-to-end consistent.

Method pairs (5 methods total: base, rft, tfb, dpo, kto -> 4 base-vs-trained pairs):

    (base, rft), (base, tfb), (base, dpo), (base, kto)

(`> 0.5` means the trained method won against base.)

Output JSON schema (per (task, model)):

    {
      "audiences": {
        "train": {"mean": {model: {pair: float}}, "std": {model: {pair: float}}}
      }
    }

The `audiences` wrapper is kept (with a single "train" key) so re-introducing
a held-out test audience later only requires adding "test" to
`config.AUDIENCES` -- no schema migration. To enable that, also confirm the
matching `subjects/test_{persona,demographic}_{N}.json` file exists.

When run with `--model X` it writes to `res/{task}/competition_parts/{X}.json`;
the consolidating `--merge` pass writes the canonical `res/{task}/competition.json`.
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
    AUDIENCES,
    METHODS,
    MODELS,
    RES_DIR,
    TASKS,
    VOTER_BIO_MODE,
    VOTER_MODEL_NAME,
    competition_path,
    step2_path,
)
from new_experiments.src.personas import load_voter_bios  # noqa: E402


# Base-vs-trained pairs only. Each trained method is scored against the
# untrained base model, so result["mean"][model]["base-X"] > 0.5 means
# method X beats base for that (model, task).
METHOD_PAIRS: List[Tuple[str, str]] = [("base", m) for m in ("rft", "tfb", "dpo", "kto")]


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


def _empty_results() -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    return {
        "audiences": {a: {"mean": {}, "std": {}} for a in AUDIENCES}
    }


def _run_one(task: str, limit: int | None, models: List[str]) -> dict:
    """Run pairwise voter competition for `models` on `task` under each audience."""
    # Build a Voters instance per audience once and reuse across (model, pair).
    voters_by_audience: Dict[str, Voters] = {}
    for aud in AUDIENCES:
        bios = load_voter_bios(audience=aud, bio_mode=VOTER_BIO_MODE)
        voters_by_audience[aud] = Voters(bios=bios, task=task, model_name=VOTER_MODEL_NAME)
        print(f"[compete] task={task} audience={aud} (bios={len(bios)})")

    results = _empty_results()
    for model_name in models:
        for aud in AUDIENCES:
            results["audiences"][aud]["mean"][model_name] = {}
            results["audiences"][aud]["std"][model_name] = {}

        cands = _load_player0(task, model_name)

        for first, second in METHOD_PAIRS:
            if not cands.get(first) or not cands.get(second):
                print(f"[compete] skip {model_name} {first}-{second} (missing data)")
                continue
            n = min(len(cands[first]), len(cands[second]))
            if limit is not None:
                n = min(n, limit)
            duels = list(zip(cands[first][:n], cands[second][:n]))

            for aud in AUDIENCES:
                votes_2d, _thoughts_2d, _choices_2d = voters_by_audience[aud].get_votes_list(duels)
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
                results["audiences"][aud]["mean"][model_name][key] = mean
                results["audiences"][aud]["std"][model_name][key] = std
                print(
                    f"[compete] aud={aud} {model_name} {key}: "
                    f"mean={mean:.3f} std={std:.3f}"
                )

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

    merged = _empty_results()
    for path in part_files:
        with open(path) as f:
            part = json.load(f)
        # Backwards-compat: old per-model parts may still use the flat
        # {"mean":..., "std":...} layout (single audience). Lift them under the
        # train-audience bucket so they don't break the merge.
        if "audiences" not in part:
            part = {"audiences": {"train": {"mean": part.get("mean", {}), "std": part.get("std", {})}}}
        for aud, bucket in part["audiences"].items():
            if aud not in merged["audiences"]:
                merged["audiences"][aud] = {"mean": {}, "std": {}}
            for k in ("mean", "std"):
                for model_name, pair_to_val in bucket.get(k, {}).items():
                    if model_name in merged["audiences"][aud][k]:
                        print(
                            f"[compete-merge] WARNING: {task} aud={aud} "
                            f"model={model_name} already merged; overwriting from {path}"
                        )
                    merged["audiences"][aud][k][model_name] = pair_to_val

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
