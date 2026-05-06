"""Step 8 — misalignment probes (q1/q2 per task).

Mirrors `run_analysis_*_q*.ipynb`. The probe logic itself lives in the
existing `trends/*.py` modules (model-agnostic). This script feeds the
new-experiment generations into those probes and writes per-task CSVs.

Probes:
- task_elections: q1 (fabricated info), q2 (populist language)
- task_sales:     q1 (product misrepresentation)
- task_sm:        q1 (fabricated info), q2 (unsafe content)

The `trends/*.py` modules hardcode `model = "gpt-4o"` for the probe LLM. To
keep `trends/` shared with the original artsco pipeline unchanged while still
letting the new-experiments runs use a cheaper probe model, we monkey-patch
`trends.utils.get_response_from_openai` BEFORE importing the per-task probe
modules so they pick up the patched binding via their
`from trends.utils import get_response_from_openai` import. The forced model
is `config.PROBE_MODEL_NAME` (currently gpt-4o-mini, ~94% cheaper than gpt-4o).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import pandas as pd

# Patch openai timeout/retries BEFORE importing trends.* which constructs the
# client at module-import time.
from new_experiments.src import openai_patch  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from new_experiments.src.config import (  # noqa: E402
    METHODS,
    MODELS,
    PROBE_MODEL_NAME,
    TASKS,
    probes_path,
    step2_path,
)

# ---------------------------------------------------------------------------
# Force every trends/*_q*.py probe to use config.PROBE_MODEL_NAME instead of
# the hardcoded "gpt-4o". Each `trends/*_q*.py` module does
#     from trends.utils import get_response_from_openai
# at import time, so we have to (a) patch trends.utils.get_response_from_openai
# BEFORE we import the per-task probe modules, and (b) re-bind the symbol on
# any probe module that was already imported earlier in this process.
# ---------------------------------------------------------------------------
import trends.utils as _trends_utils  # noqa: E402

_orig_get_response_from_openai = _trends_utils.get_response_from_openai


def _force_probe_model(query: str, model: str, output_field: str) -> str:
    return _orig_get_response_from_openai(
        query=query, model=PROBE_MODEL_NAME, output_field=output_field
    )


_trends_utils.get_response_from_openai = _force_probe_model
for _mod_name in (
    "trends.elections_q1",
    "trends.elections_q2",
    "trends.sales_q1",
    "trends.sm_q1",
    "trends.sm_q2",
):
    if _mod_name in sys.modules:
        setattr(sys.modules[_mod_name], "get_response_from_openai", _force_probe_model)
print(f"[probes] using probe model: {PROBE_MODEL_NAME} (overrides trends/*.py 'gpt-4o' default)")

from trends.elections_q1 import check_batch as check_elections_q1  # noqa: E402
from trends.elections_q2 import check_batch as check_elections_q2  # noqa: E402
from trends.sales_q1 import check_batch as check_sales_q1  # noqa: E402
from trends.sm_q1 import check_batch as check_sm_q1  # noqa: E402
from trends.sm_q2 import check_batch as check_sm_q2  # noqa: E402


PROBES = {
    "task_elections": {"q1": check_elections_q1, "q2": check_elections_q2},
    "task_sales": {"q1": check_sales_q1},
    "task_sm": {"q1": check_sm_q1, "q2": check_sm_q2},
}


def _load_step2(task: str, model_name: str, method: str) -> List[dict]:
    path = step2_path(task, model_name, method, "test")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out.append({
                "prompt": row["prompt"][0],
                "completion": row["completion"][0],
                "player_candidates": row["player_candidates"][0],
                "player_thinks": row["player_thinks"][0],
            })
    return out


def _make_df(task: str, model_name: str, method: str, check_fn) -> pd.DataFrame:
    data = _load_step2(task, model_name, method)
    if not data:
        return pd.DataFrame()
    prompts = [item["prompt"] for item in data]
    generations = [item["player_candidates"] for item in data]
    thinks, answers = check_fn(prompts, generations)
    return pd.DataFrame({
        "model": model_name,
        "method": method,
        "prompt": prompts,
        "generation": generations,
        "think": thinks,
        "label": answers,
    })


def run_task(task: str, qid: str, limit: int | None) -> None:
    if qid not in PROBES.get(task, {}):
        print(f"[probes] no probe {qid} for task {task}; skipping")
        return
    check_fn = PROBES[task][qid]
    out_path = probes_path(task, qid)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    frames = []
    for model_name in MODELS:
        for method in METHODS:
            print(f"[probes] {task}/{qid} model={model_name} method={method}")
            data = _load_step2(task, model_name, method)
            if not data:
                print(f"[probes] - no data; skipping")
                continue
            if limit is not None:
                data = data[:limit]
            prompts = [item["prompt"] for item in data]
            generations = [item["player_candidates"] for item in data]
            thinks, answers = check_fn(prompts, generations)
            df = pd.DataFrame({
                "model": model_name,
                "method": method,
                "prompt": prompts,
                "generation": generations,
                "think": thinks,
                "label": answers,
            })
            frames.append(df)

    if not frames:
        print(f"[probes] no frames produced for {task}/{qid}")
        return

    full = pd.concat(frames, ignore_index=True)
    full["label"] = pd.to_numeric(full["label"], errors="coerce")
    full.to_csv(out_path, index=False)
    print(f"[probes] wrote {out_path}")
    print(full.groupby(["model", "method"])["label"].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--qid", default=None, help="q1 or q2")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    tasks = [args.task] if args.task else TASKS
    for task in tasks:
        qids = [args.qid] if args.qid else list(PROBES.get(task, {}).keys())
        for qid in qids:
            run_task(task, qid, args.limit)


if __name__ == "__main__":
    main()
