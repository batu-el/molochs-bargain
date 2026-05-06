"""Tinker SDK helpers.

Two kinds of work happen against the Tinker API in these experiments:

1. Sampling — both the *base* model (for generate1 / generate22) and the
   *trained* LoRA (for generate2). We use `service_client.create_sampling_client`
   for base models and `training_client.save_weights_and_get_sampling_client`
   for LoRA checkpoints.
2. Training — LoRA SFT via `service_client.create_lora_training_client` plus a
   manual `forward_backward` / `optim_step` loop on (prompt, completion) pairs.

Sampling is fully async and uses `asyncio.gather` with a `Semaphore` to keep
the inflight request count bounded.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import tinker
from tinker import types

from new_experiments.src.config import LORA_RANK, MAX_NEW_TOKENS, TEMPERATURE


# ---------- Service client (singleton per process) ----------
_SERVICE: Optional[tinker.ServiceClient] = None


def get_service_client() -> tinker.ServiceClient:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = tinker.ServiceClient()
    return _SERVICE


# ---------- Tokenizer (Tinker first, HF fallback) ----------
# Cache per-model so we don't keep recreating SamplingClients for the same model
# across a single process.
_TOKENIZER_CACHE: dict = {}


def get_tokenizer(model_name: str):
    """Return a `PreTrainedTokenizer` for `model_name`.

    Strategy:
    1. Try HuggingFace first (fast, works for non-gated models). Honor the
       `tokenizer_for(...)` alias table from `config.py` so gated models can
       redirect to a public mirror (e.g., `unsloth/Llama-3.3-70B-Instruct`).
    2. If that fails (e.g., gated repo, no internet), fall back to Tinker's
       internal mirror via `service_client.create_sampling_client(...).get_tokenizer()`.

    HF-first ordering avoids a known issue where Tinker's `get_tokenizer()`
    can hang indefinitely for some newer models (e.g. Qwen/Qwen3-32B).
    """
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]

    from transformers import AutoTokenizer
    from new_experiments.src.config import tokenizer_for
    repo = tokenizer_for(model_name)
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        _TOKENIZER_CACHE[model_name] = tok
        return tok
    except Exception as e:
        print(f"[tinker_utils] HuggingFace tokenizer fetch failed for {repo}: "
              f"{type(e).__name__}: {e}; falling back to Tinker mirror.")

    # Tinker fallback (slower, can hang for some models; only use when HF
    # is gated or unavailable).
    client = get_service_client().create_sampling_client(base_model=model_name)
    tok = client.get_tokenizer()
    _TOKENIZER_CACHE[model_name] = tok
    return tok


# ---------- Sampling ----------
@dataclass
class SamplingResult:
    completions: List[str]  # decoded text per sample, length = num_samples


async def _sample_one(
    sampling_client: tinker.SamplingClient,
    tokenizer,
    prompt_text: str,
    num_samples: int,
    sampling_params: types.SamplingParams,
    semaphore: asyncio.Semaphore,
    skip_special_tokens: bool = True,
) -> SamplingResult:
    async with semaphore:
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt = types.ModelInput.from_ints(tokens=prompt_tokens)
        result = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )
    completions = [
        tokenizer.decode(seq.tokens, skip_special_tokens=skip_special_tokens)
        for seq in result.sequences
    ]
    return SamplingResult(completions=completions)


async def sample_many(
    sampling_client: tinker.SamplingClient,
    tokenizer,
    prompts: Sequence[str],
    num_samples: int = 2,
    max_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    stop: Optional[Sequence[str]] = None,
    max_concurrency: int = 32,
    skip_special_tokens: bool = True,
) -> List[SamplingResult]:
    """Sample `num_samples` completions for each prompt, concurrently."""
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=list(stop) if stop else [],
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        _sample_one(
            sampling_client, tokenizer, p, num_samples, sampling_params, semaphore,
            skip_special_tokens=skip_special_tokens,
        )
        for p in prompts
    ]
    return await asyncio.gather(*tasks)


def get_base_sampling_client(model_name: str) -> tinker.SamplingClient:
    return get_service_client().create_sampling_client(base_model=model_name)


def get_sampling_client_from_uri(checkpoint_uri: str) -> tinker.SamplingClient:
    return get_service_client().create_sampling_client(model_path=checkpoint_uri)


# ---------- SFT Datum construction ----------
def build_sft_datum(
    tokenizer,
    prompt_text: str,
    completion_text: str,
    eos_token_id: Optional[int] = None,
    max_length: int = 4096,
) -> Optional[types.Datum]:
    """Encode a (prompt, completion) pair into a Tinker Datum.

    `weights` is 0 over the prompt portion and 1 over the completion portion,
    matching the semantics TRL's SFT trainer uses with `assistant_only_loss`.

    Returns `None` if there is nothing to learn (empty completion or prompt
    already longer than `max_length`).
    """
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)

    if not completion_tokens:
        return None

    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and completion_tokens[-1] != eos_token_id:
        completion_tokens = completion_tokens + [eos_token_id]

    full = prompt_tokens + completion_tokens
    if len(full) > max_length:
        full = full[:max_length]
        completion_len = max_length - len(prompt_tokens)
    else:
        completion_len = len(completion_tokens)

    if completion_len <= 0 or len(full) < 2:
        return None

    input_tokens = full[:-1]
    target_tokens = full[1:]

    weights = [0] * len(target_tokens)
    for i in range(len(target_tokens) - completion_len, len(target_tokens)):
        if 0 <= i < len(weights):
            weights[i] = 1

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=np.array(weights, dtype=np.float32),
            target_tokens=np.array(target_tokens, dtype=np.int64),
        ),
    )


# ---------- Cosine-with-min-lr schedule (mirrors original training) ----------
def cosine_with_min_lr(step: int, total_steps: int, base_lr: float, warmup_ratio: float, min_lr_ratio: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


# ---------- Training client factory ----------
def create_lora_training_client(model_name: str, rank: int = LORA_RANK):
    return get_service_client().create_lora_training_client(base_model=model_name, rank=rank)
