"""Step 5 — LoRA SFT through the Tinker API.

Mirrors `artsco/src/train.py`:
- LoRA rank 16 (alpha 32 — Tinker uses fixed alpha=2*rank by default).
- 1 epoch, lr=2e-4, cosine_with_min_lr (warmup 3%, min_lr_ratio 10%).
- Batch size 16, full sequence (max 4096 tokens).
- Cross-entropy loss with weights=0 over the prompt tokens.

After training, the resulting LoRA checkpoint URI is saved to
`new_experiments/models/{task}/{model}/{method}/state.json` so that
`generate2.py` can reload it for inference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import List

import numpy as np
import tinker
from datasets import load_dataset

from new_experiments.src.config import (
    LEARNING_RATE,
    LORA_RANK,
    MAX_SEQ_LENGTH,
    MIN_LR_RATIO,
    MODELS,
    NUM_EPOCHS,
    PER_DEVICE_BATCH,
    TASKS,
    TRAINED_METHODS,
    WARMUP_RATIO,
    model_state_path,
    train_data_path,
)
from new_experiments.src.tinker_utils import (
    build_sft_datum,
    cosine_with_min_lr,
    create_lora_training_client,
    get_tokenizer,
)


def _checkpoint_name(task: str, model_name: str, method: str) -> str:
    safe_model = model_name.replace("/", "__")
    return f"sft__{task}__{safe_model}__{method}"


async def _train_one(task: str, model_name: str, method: str, batch_size: int, save_every: int) -> None:
    data_path = train_data_path(task, model_name, "train", method)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing training data: {data_path}. Run build_train_data first.")

    out_path = model_state_path(task, model_name, method)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[train] task={task} model={model_name} method={method} -> {out_path}")
    ds = load_dataset("json", data_files=data_path, split="train")

    tokenizer = get_tokenizer(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[train] encoding {len(ds)} examples ...")
    data: List[tinker.types.Datum] = []
    for ex in ds:
        prompt = ex["prompt"]
        completion = ex["completion"] or ""
        datum = build_sft_datum(
            tokenizer=tokenizer,
            prompt_text=prompt,
            completion_text=completion,
            max_length=MAX_SEQ_LENGTH,
        )
        if datum is not None:
            data.append(datum)

    if not data:
        raise RuntimeError(f"No training data after encoding for {data_path}")
    print(f"[train] kept {len(data)}/{len(ds)} examples after encoding")

    rng = np.random.default_rng(0)
    n = len(data)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"[train] {n} datums, batch_size={batch_size}, total_steps={total_steps}")

    training_client = create_lora_training_client(model_name=model_name, rank=LORA_RANK)

    global_step = 0
    last_checkpoint_uri: str | None = None
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        order = rng.permutation(n)
        for batch_start in range(0, n, batch_size):
            batch_idx = order[batch_start : batch_start + batch_size]
            batch = [data[i] for i in batch_idx]

            lr = cosine_with_min_lr(
                step=global_step,
                total_steps=total_steps,
                base_lr=LEARNING_RATE,
                warmup_ratio=WARMUP_RATIO,
                min_lr_ratio=MIN_LR_RATIO,
            )
            fwd_future = await training_client.forward_backward_async(batch, "cross_entropy")
            optim_future = await training_client.optim_step_async(
                tinker.AdamParams(learning_rate=lr)
            )
            fwd_result = await fwd_future.result_async()
            await optim_future.result_async()

            logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_result.loss_fn_outputs])
            weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
            loss = float(-np.dot(logprobs, weights) / max(weights.sum(), 1.0))
            global_step += 1
            elapsed = time.time() - t0

            print(
                f"[train] epoch={epoch} step={global_step}/{total_steps} "
                f"loss={loss:.4f} lr={lr:.2e} elapsed={elapsed:.1f}s"
            )

            if save_every > 0 and global_step % save_every == 0:
                ckpt_name = f"{_checkpoint_name(task, model_name, method)}__step{global_step}"
                interim_future = await training_client.save_weights_for_sampler_async(name=ckpt_name)
                interim = await interim_future.result_async()
                last_checkpoint_uri = interim.path
                print(f"[train] interim checkpoint -> {last_checkpoint_uri}")

    final_name = f"{_checkpoint_name(task, model_name, method)}__final"
    final_future = await training_client.save_weights_for_sampler_async(name=final_name)
    final = await final_future.result_async()
    last_checkpoint_uri = final.path
    print(f"[train] final checkpoint -> {last_checkpoint_uri}")

    state = {
        "task": task,
        "model_name": model_name,
        "method": method,
        "checkpoint_uri": last_checkpoint_uri,
        "checkpoint_name": final_name,
        "lora_rank": LORA_RANK,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": batch_size,
        "total_steps": global_step,
    }
    with open(out_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[train] saved state -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--method", choices=TRAINED_METHODS, default=None)
    p.add_argument("--batch_size", type=int, default=PER_DEVICE_BATCH)
    p.add_argument("--save_every", type=int, default=0)
    args = p.parse_args()

    tasks = [args.task] if args.task else TASKS
    models = [args.model] if args.model else MODELS
    methods = [args.method] if args.method else TRAINED_METHODS

    for task in tasks:
        for model_name in models:
            for method in methods:
                asyncio.run(_train_one(task, model_name, method, args.batch_size, args.save_every))


if __name__ == "__main__":
    main()
