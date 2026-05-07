"""Materialize fixed train/test audience files for the new experiments.

The full pool in `subjects/` is 800 train + 200 test demographically realistic
personas (paired with structured demographics by index). For every (model, task,
method) we want to roll out / evaluate against the *exact same* people, so this
script samples once with a fixed seed and writes:

    subjects/train_persona_{n_train}.json       (n_train / 800 train personas)
    subjects/test_persona_{n_test}.json         (n_test  / 200 test  personas)
    subjects/train_demographic_{n_train}.json   (n_train / 800 train demographics)
    subjects/test_demographic_{n_test}.json     (n_test  / 200 test  demographics)

Personas and demographics are sampled with the *same* index set per split, so
record `i` in `train_persona_{n}.json` and `train_demographic_{n}.json`
describe the same person (matches the existing index alignment in
`subjects/personas_*.json` <-> `subjects/demographics_*.json`).

`n_train` and `n_test` default to `NUM_VOTERS_TRAIN` / `NUM_VOTERS_TEST` from
`config.py`. The size is encoded in the file name suffix so multiple sizes can
coexist on disk side by side -- e.g. `..._20.json` and `..._50.json` -- and
the active size is selected by the config constants.

Run once per (n_train, n_test) you want available:

    python -m new_experiments.scripts.build_audiences                   # uses config defaults
    python -m new_experiments.scripts.build_audiences --n_train 50 --n_test 50

Idempotent: rerunning produces byte-identical files (uses a fixed seed).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, List, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from new_experiments.src.config import (  # noqa: E402
    NUM_VOTERS_TEST,
    NUM_VOTERS_TRAIN,
    SUBJECTS_DIR,
    audience_path,
    subjects_path,
)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _subset(items: Sequence[Any], indices: Sequence[int]) -> List[Any]:
    return [items[i] for i in indices]


def build(split: str, n: int, seed: int) -> None:
    personas = _load_json(subjects_path("personas", split))
    demographics = _load_json(subjects_path("demographics", split))
    if len(personas) != len(demographics):
        raise AssertionError(
            f"persona/demographic length mismatch for split={split}: "
            f"{len(personas)} vs {len(demographics)}"
        )
    pool = len(personas)
    if n > pool:
        raise ValueError(
            f"requested n={n} but only {pool} {split} personas available"
        )

    rng = random.Random(seed)
    indices = rng.sample(range(pool), n)
    indices.sort()  # stable, easy to inspect; downstream loaders preserve order.

    persona_path = audience_path("persona", split, n=n)
    demographic_path = audience_path("demographic", split, n=n)
    _write_json(persona_path, _subset(personas, indices))
    _write_json(demographic_path, _subset(demographics, indices))
    print(
        f"[build_audiences] split={split} n={n} pool={pool} seed={seed}\n"
        f"  -> {persona_path}\n"
        f"  -> {demographic_path}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the fixed sample (default 0).")
    p.add_argument("--n_train", type=int, default=NUM_VOTERS_TRAIN,
                   help=f"number of train personas (default {NUM_VOTERS_TRAIN}).")
    p.add_argument("--n_test", type=int, default=NUM_VOTERS_TEST,
                   help=f"number of test personas (default {NUM_VOTERS_TEST}).")
    args = p.parse_args()

    if not os.path.isdir(SUBJECTS_DIR):
        raise SystemExit(f"missing subjects/ dir at {SUBJECTS_DIR}")

    build("train", args.n_train, args.seed)
    build("test", args.n_test, args.seed)


if __name__ == "__main__":
    main()
