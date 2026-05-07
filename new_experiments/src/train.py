"""Step 5 — LoRA fine-tuning through the Tinker API.

Four trained methods, one entry point. The dispatch is:

    rft / tfb : standard SFT NLL on (prompt, winner_completion) pairs via
                 `forward_backward_async(data, "cross_entropy")`. The TFB
                 dataset additionally includes (tfb_prompt, voter_think)
                 warm-up entries built by `build_train_data.py`.
    dpo       : pairwise contrastive loss vs a frozen base-model reference,
                 trained with `forward_backward_custom_async`. Each gradient
                 step processes (chosen, rejected) interleaved pairs.
    kto       : per-example KTO loss vs a frozen base-model reference, with
                 a batch-mean KL anchor. Each gradient step processes
                 (desirable, undesirable) interleaved entries.

For DPO/KTO we precompute reference logprob sums once per (task, model, method)
and cache them under `models/{task}/{model}/{method}/ref_logprobs.json`.
Hyperparameters (LoRA rank, lr, schedule, batch size, num epochs, beta) all
live in `config.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import List, Tuple

import numpy as np
import tinker
from datasets import load_dataset

from new_experiments.src.config import (
    DPO_BETA,
    KTO_BETA,
    KTO_DESIRABLE_WEIGHT,
    KTO_UNDESIRABLE_WEIGHT,
    LEARNING_RATE,
    LORA_RANK,
    MAX_SEQ_LENGTH,
    MIN_LR_RATIO,
    MODELS,
    NUM_EPOCHS,
    PER_DEVICE_BATCH,
    PREF_METHODS,
    TASKS,
    TRAINED_METHODS,
    WARMUP_RATIO,
    model_state_path,
    ref_logprob_cache_path,
    train_data_path,
)
from new_experiments.src.tinker_utils import (
    TokenizedExample,
    _encode_pair,
    build_pref_datum,
    build_sft_datum,
    compute_ref_logprob_sums,
    cosine_with_min_lr,
    create_lora_training_client,
    get_base_sampling_client,
    get_tokenizer,
    load_ref_cache,
    make_dpo_loss,
    make_kto_loss,
    save_ref_cache,
    truncate_to_max_length,
)


def _checkpoint_name(task: str, model_name: str, method: str) -> str:
    safe_model = model_name.replace("/", "__")
    return f"sft__{task}__{safe_model}__{method}"


# ---------------------------------------------------------------------------
# RFT / TFB (standard cross-entropy SFT)
# ---------------------------------------------------------------------------
def _build_sft_data(ds, tokenizer) -> List[tinker.types.Datum]:
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
    return data


async def _train_sft(
    task: str,
    model_name: str,
    method: str,
    batch_size: int,
    save_every: int,
) -> str:
    data_path = train_data_path(task, model_name, "train", method)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing training data: {data_path}. Run build_train_data first.")

    ds = load_dataset("json", data_files=data_path, split="train")
    tokenizer = get_tokenizer(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[train/sft] encoding {len(ds)} examples ...")
    data = _build_sft_data(ds, tokenizer)
    if not data:
        raise RuntimeError(f"No training data after encoding for {data_path}")
    print(f"[train/sft] kept {len(data)}/{len(ds)} examples after encoding")

    n = len(data)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"[train/sft] {n} datums, batch_size={batch_size}, total_steps={total_steps}")

    rng = np.random.default_rng(0)
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
                f"[train/sft] method={method} epoch={epoch} step={global_step}/{total_steps} "
                f"loss={loss:.4f} lr={lr:.2e} elapsed={elapsed:.1f}s"
            )

            if save_every > 0 and global_step % save_every == 0:
                ckpt_name = f"{_checkpoint_name(task, model_name, method)}__step{global_step}"
                interim_future = await training_client.save_weights_for_sampler_async(name=ckpt_name)
                interim = await interim_future.result_async()
                last_checkpoint_uri = interim.path
                print(f"[train/sft] interim checkpoint -> {last_checkpoint_uri}")

    final_name = f"{_checkpoint_name(task, model_name, method)}__final"
    final_future = await training_client.save_weights_for_sampler_async(name=final_name)
    final = await final_future.result_async()
    last_checkpoint_uri = final.path
    print(f"[train/sft] final checkpoint -> {last_checkpoint_uri}")
    return last_checkpoint_uri


# ---------------------------------------------------------------------------
# DPO / KTO (custom torch loss with frozen base reference)
# ---------------------------------------------------------------------------
def _build_pref_examples(
    ds,
    tokenizer,
    method: str,
) -> Tuple[List[TokenizedExample], List[tinker.types.Datum]]:
    """Tokenize the dataset into TokenizedExample (for ref logprobs) AND a
    parallel list of training Datums (for the LoRA forward/backward).

    For DPO: each dataset row -> 2 datums (chosen, rejected) interleaved AND
    one TokenizedExample carrying both completion token lists.
    For KTO: each dataset row -> 1 datum AND one TokenizedExample.
    """
    examples: List[TokenizedExample] = []
    datums: List[tinker.types.Datum] = []

    if method == "dpo":
        for ex in ds:
            prompt = ex["prompt"]
            chosen = ex["chosen"]
            rejected = ex["rejected"]
            ptoks, ctoks = _encode_pair(tokenizer, prompt, chosen)
            _, rtoks = _encode_pair(tokenizer, prompt, rejected)
            ptoks_c, ctoks_t = truncate_to_max_length(ptoks, ctoks, MAX_SEQ_LENGTH)
            ptoks_r, rtoks_t = truncate_to_max_length(ptoks, rtoks, MAX_SEQ_LENGTH)
            if not ctoks_t or not rtoks_t:
                continue
            chosen_datum = build_pref_datum(tokenizer, ptoks_c, ctoks_t, MAX_SEQ_LENGTH)
            rejected_datum = build_pref_datum(tokenizer, ptoks_r, rtoks_t, MAX_SEQ_LENGTH)
            if chosen_datum is None or rejected_datum is None:
                continue
            examples.append(TokenizedExample(
                prompt_tokens=ptoks_c,
                completion_tokens=ctoks_t,
                rejected_completion_tokens=rtoks_t,
            ))
            datums.append(chosen_datum)
            datums.append(rejected_datum)
        return examples, datums

    if method == "kto":
        for ex in ds:
            prompt = ex["prompt"]
            completion = ex["completion"]
            desirable = bool(ex["desirable"])
            ptoks, ctoks = _encode_pair(tokenizer, prompt, completion)
            ptoks_t, ctoks_t = truncate_to_max_length(ptoks, ctoks, MAX_SEQ_LENGTH)
            if not ctoks_t:
                continue
            datum = build_pref_datum(tokenizer, ptoks_t, ctoks_t, MAX_SEQ_LENGTH)
            if datum is None:
                continue
            examples.append(TokenizedExample(
                prompt_tokens=ptoks_t,
                completion_tokens=ctoks_t,
                desirable=desirable,
            ))
            datums.append(datum)
        return examples, datums

    raise ValueError(f"_build_pref_examples only supports dpo/kto, got {method!r}")


async def _ensure_ref_cache(
    task: str,
    model_name: str,
    method: str,
    examples: List[TokenizedExample],
    max_concurrency: int,
) -> List[dict]:
    cache_path = ref_logprob_cache_path(task, model_name, method)
    cached = load_ref_cache(cache_path)
    if cached is not None and len(cached) == len(examples):
        print(f"[train/{method}] using cached reference logprobs ({len(cached)} entries) -> {cache_path}")
        return cached
    if cached is not None:
        print(
            f"[train/{method}] cache size mismatch (cached={len(cached)}, "
            f"now={len(examples)}); recomputing"
        )
    print(f"[train/{method}] computing base-model reference logprobs for {len(examples)} entries ...")
    sampling_client = get_base_sampling_client(model_name)
    t0 = time.time()
    ref = await compute_ref_logprob_sums(
        sampling_client=sampling_client,
        examples=examples,
        method=method,
        max_concurrency=max_concurrency,
    )
    save_ref_cache(cache_path, ref)
    print(f"[train/{method}] cached {len(ref)} ref entries -> {cache_path}  "
          f"(elapsed {time.time()-t0:.1f}s)")
    return ref


async def _train_pref(
    task: str,
    model_name: str,
    method: str,
    batch_size: int,
    save_every: int,
    max_concurrency: int,
) -> str:
    data_path = train_data_path(task, model_name, "train", method)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing training data: {data_path}. Run build_train_data first.")

    ds = load_dataset("json", data_files=data_path, split="train")
    tokenizer = get_tokenizer(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[train/{method}] encoding {len(ds)} dataset rows ...")
    examples, datums = _build_pref_examples(ds, tokenizer, method)
    print(
        f"[train/{method}] kept {len(examples)} examples / {len(datums)} datums "
        f"after encoding (truncate to {MAX_SEQ_LENGTH} tokens)"
    )
    if not examples:
        raise RuntimeError(f"No training data after encoding for {data_path}")

    ref_data = await _ensure_ref_cache(task, model_name, method, examples, max_concurrency)

    # Build per-method loss closure with the (possibly recomputed) reference data.
    if method == "dpo":
        ref_chosen = [r["chosen"] for r in ref_data]
        ref_rejected = [r["rejected"] for r in ref_data]
        # Each pair index maps to two datums (2i, 2i+1); the shuffle below must
        # therefore permute *pairs*, not individual datums.
        n_pairs = len(examples)
        # Effective batch size in PAIRS: each fwd-bwd processes batch_size datums =
        # batch_size/2 pairs.
        if batch_size % 2 != 0:
            raise ValueError(f"DPO batch_size must be even, got {batch_size}")
        pairs_per_batch = batch_size // 2

        rng = np.random.default_rng(0)
        steps_per_epoch = (n_pairs + pairs_per_batch - 1) // pairs_per_batch
        total_steps = steps_per_epoch * NUM_EPOCHS
        print(f"[train/dpo] {n_pairs} pairs, pairs_per_batch={pairs_per_batch}, total_steps={total_steps}")

        training_client = create_lora_training_client(model_name=model_name, rank=LORA_RANK)
        global_step = 0
        last_uri: str | None = None
        t0 = time.time()
        for epoch in range(NUM_EPOCHS):
            order = rng.permutation(n_pairs)
            for batch_start in range(0, n_pairs, pairs_per_batch):
                pair_idx = order[batch_start : batch_start + pairs_per_batch]
                batch_datums: List[tinker.types.Datum] = []
                batch_ref_chosen: List[float] = []
                batch_ref_rejected: List[float] = []
                for pi in pair_idx:
                    batch_datums.append(datums[2 * pi])
                    batch_datums.append(datums[2 * pi + 1])
                    batch_ref_chosen.append(ref_chosen[pi])
                    batch_ref_rejected.append(ref_rejected[pi])

                lr = cosine_with_min_lr(
                    step=global_step, total_steps=total_steps, base_lr=LEARNING_RATE,
                    warmup_ratio=WARMUP_RATIO, min_lr_ratio=MIN_LR_RATIO,
                )
                loss_fn = make_dpo_loss(
                    beta=DPO_BETA,
                    ref_chosen=batch_ref_chosen,
                    ref_rejected=batch_ref_rejected,
                )
                fwd_future = await training_client.forward_backward_custom_async(
                    batch_datums, loss_fn, loss_type_input="logprobs",
                )
                optim_future = await training_client.optim_step_async(
                    tinker.AdamParams(learning_rate=lr)
                )
                fwd_result = await fwd_future.result_async()
                await optim_future.result_async()

                metrics = fwd_result.metrics
                global_step += 1
                elapsed = time.time() - t0
                print(
                    f"[train/dpo] epoch={epoch} step={global_step}/{total_steps} "
                    f"loss={metrics.get('dpo_loss', float('nan')):.4f} "
                    f"acc={metrics.get('dpo_accuracy', float('nan')):.3f} "
                    f"margin={metrics.get('dpo_margin', float('nan')):+.3f} "
                    f"lr={lr:.2e} elapsed={elapsed:.1f}s"
                )

                if save_every > 0 and global_step % save_every == 0:
                    ckpt_name = f"{_checkpoint_name(task, model_name, method)}__step{global_step}"
                    interim_future = await training_client.save_weights_for_sampler_async(name=ckpt_name)
                    interim = await interim_future.result_async()
                    last_uri = interim.path

        final_name = f"{_checkpoint_name(task, model_name, method)}__final"
        final_future = await training_client.save_weights_for_sampler_async(name=final_name)
        final = await final_future.result_async()
        last_uri = final.path
        print(f"[train/dpo] final checkpoint -> {last_uri}")
        return last_uri

    if method == "kto":
        ref_completion = [r["completion"] for r in ref_data]
        desirable_mask = [r["desirable"] for r in ref_data]
        n = len(examples)

        rng = np.random.default_rng(0)
        steps_per_epoch = (n + batch_size - 1) // batch_size
        total_steps = steps_per_epoch * NUM_EPOCHS
        print(f"[train/kto] {n} entries, batch_size={batch_size}, total_steps={total_steps}")

        training_client = create_lora_training_client(model_name=model_name, rank=LORA_RANK)
        global_step = 0
        last_uri: str | None = None
        t0 = time.time()

        # We want every batch to have a balanced mix of desirable/undesirable
        # so the batch-mean KL anchor (z_ref) is meaningful. The KTO dataset is
        # written by build_train_data.py as interleaved (winner, loser) pairs,
        # so we shuffle at the *pair* level (every 2 entries stay together).
        if n % 2 != 0:
            print(f"[train/kto] WARNING: dataset has odd length {n}; dropping last entry")
            n -= 1
        n_pairs = n // 2
        for epoch in range(NUM_EPOCHS):
            pair_order = rng.permutation(n_pairs)
            # Flatten back to entry indices, keeping winner+loser adjacent.
            order = np.empty(n, dtype=np.int64)
            for k, p in enumerate(pair_order):
                order[2 * k] = 2 * p
                order[2 * k + 1] = 2 * p + 1

            for batch_start in range(0, n, batch_size):
                idx = order[batch_start : batch_start + batch_size]
                batch_datums = [datums[i] for i in idx]
                batch_ref = [ref_completion[i] for i in idx]
                batch_des = [desirable_mask[i] for i in idx]

                lr = cosine_with_min_lr(
                    step=global_step, total_steps=total_steps, base_lr=LEARNING_RATE,
                    warmup_ratio=WARMUP_RATIO, min_lr_ratio=MIN_LR_RATIO,
                )
                loss_fn = make_kto_loss(
                    beta=KTO_BETA,
                    ref_completion=batch_ref,
                    desirable_mask=batch_des,
                    desirable_weight=KTO_DESIRABLE_WEIGHT,
                    undesirable_weight=KTO_UNDESIRABLE_WEIGHT,
                )
                fwd_future = await training_client.forward_backward_custom_async(
                    batch_datums, loss_fn, loss_type_input="logprobs",
                )
                optim_future = await training_client.optim_step_async(
                    tinker.AdamParams(learning_rate=lr)
                )
                fwd_result = await fwd_future.result_async()
                await optim_future.result_async()

                metrics = fwd_result.metrics
                global_step += 1
                elapsed = time.time() - t0
                print(
                    f"[train/kto] epoch={epoch} step={global_step}/{total_steps} "
                    f"loss={metrics.get('kto_loss', float('nan')):.4f} "
                    f"z_ref={metrics.get('kto_z_ref', float('nan')):+.3f} "
                    f"h_des={metrics.get('kto_h_desirable', float('nan')):+.3f} "
                    f"h_und={metrics.get('kto_h_undesirable', float('nan')):+.3f} "
                    f"lr={lr:.2e} elapsed={elapsed:.1f}s"
                )

                if save_every > 0 and global_step % save_every == 0:
                    ckpt_name = f"{_checkpoint_name(task, model_name, method)}__step{global_step}"
                    interim_future = await training_client.save_weights_for_sampler_async(name=ckpt_name)
                    interim = await interim_future.result_async()
                    last_uri = interim.path

        final_name = f"{_checkpoint_name(task, model_name, method)}__final"
        final_future = await training_client.save_weights_for_sampler_async(name=final_name)
        final = await final_future.result_async()
        last_uri = final.path
        print(f"[train/kto] final checkpoint -> {last_uri}")
        return last_uri

    raise ValueError(f"_train_pref only supports dpo/kto, got {method!r}")


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------
async def _train_one(
    task: str,
    model_name: str,
    method: str,
    batch_size: int,
    save_every: int,
    max_concurrency: int,
) -> None:
    out_path = model_state_path(task, model_name, method)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[train] task={task} model={model_name} method={method} -> {out_path}")

    if method in PREF_METHODS:
        checkpoint_uri = await _train_pref(
            task=task,
            model_name=model_name,
            method=method,
            batch_size=batch_size,
            save_every=save_every,
            max_concurrency=max_concurrency,
        )
        beta = DPO_BETA if method == "dpo" else KTO_BETA
        extra = {"beta": beta}
    else:
        checkpoint_uri = await _train_sft(
            task=task,
            model_name=model_name,
            method=method,
            batch_size=batch_size,
            save_every=save_every,
        )
        extra = {}

    state = {
        "task": task,
        "model_name": model_name,
        "method": method,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_name": f"{_checkpoint_name(task, model_name, method)}__final",
        "lora_rank": LORA_RANK,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": batch_size,
        **extra,
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
    p.add_argument("--max_concurrency", type=int, default=32,
                   help="Max concurrent base-model forward calls when computing "
                        "DPO/KTO reference logprobs.")
    args = p.parse_args()

    tasks = [args.task] if args.task else TASKS
    models = [args.model] if args.model else MODELS
    methods = [args.method] if args.method else TRAINED_METHODS

    for task in tasks:
        for model_name in models:
            for method in methods:
                asyncio.run(
                    _train_one(
                        task=task,
                        model_name=model_name,
                        method=method,
                        batch_size=args.batch_size,
                        save_every=args.save_every,
                        max_concurrency=args.max_concurrency,
                    )
                )


if __name__ == "__main__":
    main()
