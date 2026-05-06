"""Format-compliance smoke test for all 5 models.

For each (model x task), sample N=2 base completions on a handful of prompts
and verify the model's *native* response format is correctly produced and
parsed by `format_adapters`. Reports per-cell empty-rate and an aggregate
verdict.

Usage:
  python -m new_experiments.src.test_format
  python -m new_experiments.src.test_format --limit 4 --model openai/gpt-oss-20b
  python -m new_experiments.src.test_format --task task_elections
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import textwrap
from typing import List

from new_experiments.src import openai_patch  # noqa: F401  (no-op for sampling-only)
from new_experiments.src.config import (
    MODELS,
    NUM_PLAYERS,
    TASKS,
    NEW_EXP_DIR,
)
from new_experiments.src.data_utils import extract_answer, extract_think
from new_experiments.src.format_adapters import get_adapter
from new_experiments.src.tinker_utils import (
    get_base_sampling_client,
    get_tokenizer,
    sample_many,
)


def _load_json_records(path: str, limit: int) -> List[dict]:
    """Read a JSONL file (datasets library's native format) without going
    through `load_dataset`, which insists on writing lockfiles into the HF
    cache directory."""
    out: List[dict] = []
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            # Plain JSON array
            data = json.load(f)
            out.extend(data[:limit])
        else:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    return out


async def _run_one(task: str, model_name: str, limit: int, max_concurrency: int):
    # Always rebuild prompts from raw so we exercise the *current* format adapter,
    # not whatever was cached on disk by a previous prep_data run.
    from new_experiments.src.config import raw_split_path
    from new_experiments.src.data_utils import build_prompt_content
    from new_experiments.src.format_adapters import apply_chat_template_for

    raw = _load_json_records(raw_split_path(task, "test"), limit)
    tokenizer_for_prompt = get_tokenizer(model_name)
    prompts: List[str] = [
        apply_chat_template_for(
            tokenizer_for_prompt,
            model_name=model_name,
            task=task,
            user_content=build_prompt_content(ex, task),
        )
        for ex in raw
    ]

    tokenizer = get_tokenizer(model_name)
    adapter = get_adapter(model_name)
    client = get_base_sampling_client(model_name)

    results = await sample_many(
        sampling_client=client,
        tokenizer=tokenizer,
        prompts=prompts,
        num_samples=NUM_PLAYERS,
        max_concurrency=max_concurrency,
        skip_special_tokens=adapter.skip_special_tokens_on_decode,
    )

    rows = []
    for prompt, res in zip(prompts, results):
        for completion in res.completions:
            think = extract_think(completion, model_name)
            answer = extract_answer(completion, task, model_name)
            rows.append({
                "prompt": prompt,
                "completion": completion,
                "think": think,
                "answer": answer,
            })

    n = len(rows)
    n_empty_answer = sum(1 for r in rows if not r["answer"])
    n_empty_think  = sum(1 for r in rows if not r["think"])
    median_answer_len = sorted(len(r["answer"]) for r in rows)[n // 2] if rows else 0
    median_think_len  = sorted(len(r["think"]) for r in rows)[n // 2] if rows else 0

    return {
        "task": task,
        "model": model_name,
        "style": adapter.style,
        "n": n,
        "empty_answer_rate": n_empty_answer / n if n else 0.0,
        "empty_think_rate": n_empty_think / n if n else 0.0,
        "median_answer_chars": median_answer_len,
        "median_think_chars": median_think_len,
        "rows": rows,
    }


def _print_row(stat):
    print(
        f"  {stat['task']:18} {stat['model']:42} {stat['style']:11} "
        f"n={stat['n']:>3}  ans_empty={stat['empty_answer_rate']:>5.0%}  "
        f"think_empty={stat['empty_think_rate']:>5.0%}  "
        f"median_ans={stat['median_answer_chars']:>4}c  "
        f"median_think={stat['median_think_chars']:>4}c"
    )


def _print_sample(stat, max_chars: int = 600):
    if not stat["rows"]:
        return
    print("\n  --- sample completion ---")
    r = stat["rows"][0]
    raw = r["completion"][:max_chars].replace("\n", "\\n")
    print(f"  raw       : {raw}")
    if len(r["completion"]) > max_chars:
        print(f"              ... (+{len(r['completion']) - max_chars} chars)")
    print(f"  think     : {textwrap.shorten(r['think'] or '(none)', width=240)}")
    print(f"  answer    : {textwrap.shorten(r['answer'] or '(NONE!)', width=240)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=4,
                   help="Prompts per (task, model) cell.")
    p.add_argument("--max_concurrency", type=int, default=8)
    p.add_argument("--show_samples", action="store_true",
                   help="Print one raw + parsed completion per cell.")
    p.add_argument("--save_dir", default=os.path.join(NEW_EXP_DIR, "logs", "format_test"))
    args = p.parse_args()

    tasks = [args.task] if args.task else TASKS
    models = [args.model] if args.model else MODELS

    os.makedirs(args.save_dir, exist_ok=True)
    print("=" * 120)
    print(f"  Format-compliance smoke test  (limit={args.limit} per cell)")
    print("=" * 120)

    summary = []
    for model in models:
        print(f"\n----- {model} -----")
        for task in tasks:
            try:
                stat = asyncio.run(
                    _run_one(task, model, args.limit, args.max_concurrency)
                )
            except Exception as e:
                print(f"  {task:18} {model:42}  ERROR: {type(e).__name__}: {e}")
                continue
            _print_row(stat)
            if args.show_samples:
                _print_sample(stat)
            summary.append({k: v for k, v in stat.items() if k != "rows"})
            with open(os.path.join(args.save_dir, f"{model.replace('/', '__')}__{task}.json"), "w") as f:
                json.dump(stat, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 120)
    print("  AGGREGATE")
    print("=" * 120)
    print(f"  {'task':18} {'model':42} {'style':11} {'n':>4} {'ans_empty':>10} {'think_empty':>12}")
    print("  " + "-" * 116)
    for s in summary:
        print(
            f"  {s['task']:18} {s['model']:42} {s['style']:11} "
            f"{s['n']:>4} {s['empty_answer_rate']:>10.0%} {s['empty_think_rate']:>12.0%}"
        )
    bad = [s for s in summary if s["empty_answer_rate"] > 0.25]
    if bad:
        print("\n  ! Cells with >25% empty answers (format extraction broken):")
        for s in bad:
            print(f"    - {s['model']} / {s['task']} -> {s['empty_answer_rate']:.0%}")
        raise SystemExit(2)
    print("\n  PASS: all cells have <25% empty answer rate.")


if __name__ == "__main__":
    main()
