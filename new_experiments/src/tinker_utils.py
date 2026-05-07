"""Tinker SDK helpers.

Two kinds of work happen against the Tinker API in these experiments:

1. Sampling — both the *base* model (for generate1 / generate22) and the
   *trained* LoRA (for generate2). We use `service_client.create_sampling_client`
   for base models and `training_client.save_weights_and_get_sampling_client`
   for LoRA checkpoints.
2. Training — LoRA SFT via `service_client.create_lora_training_client` plus a
   manual `forward_backward` / `optim_step` loop on (prompt, completion) pairs.
   For the contrastive preference methods (DPO, KTO) we additionally call
   `compute_logprobs` on a frozen base sampling client to get reference logprob
   sums, then run training with `forward_backward_custom_async` so the loss is
   a function of (current logprobs - reference logprobs).

Sampling is fully async and uses `asyncio.gather` with a `Semaphore` to keep
the inflight request count bounded.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

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


# ===========================================================================
# Reference logprobs for DPO / KTO
# ===========================================================================
#
# Both DPO and KTO compare the *current* policy's logprob of a (prompt,
# completion) sequence to the same logprob under a frozen *reference* policy.
# We use the base model as the reference (no separate SFT warm-start needed).
#
# For each (prompt, completion) we send `prompt + completion` to the base
# sampling client's `compute_logprobs`, which returns one logprob per position
# (logp of token_t given tokens_<t). The sum over the COMPLETION region is the
# scalar `log pi_ref(completion | prompt)` we plug into the loss.
#
# Computing these forwards once per training entry is wasteful, so we cache the
# result to disk under `models/{task}/{model}/{method}/ref_logprobs.json`.
# ===========================================================================


@dataclass
class TokenizedExample:
    """Token-aligned view of one training entry needed for both ref-logprob
    computation and per-step custom-loss training.

    For RFT/TFB (cross_entropy): `completion_tokens` is `winner_tokens`, the
    other fields are unused.

    For DPO: `winner_tokens` and `loser_tokens` are both filled in (chosen and
    rejected completions).

    For KTO: `completion_tokens` is the single completion; `desirable` flags
    whether it's the winner (True) or the loser (False).
    """
    prompt_tokens: List[int]
    completion_tokens: List[int]
    desirable: Optional[bool] = None
    # DPO-only fields (None for RFT/TFB/KTO).
    rejected_completion_tokens: Optional[List[int]] = None


def _encode_pair(tokenizer, prompt_text: str, completion_text: str) -> Tuple[List[int], List[int]]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
    if eos_token_id is not None and (not completion_tokens or completion_tokens[-1] != eos_token_id):
        completion_tokens = completion_tokens + [eos_token_id]
    return prompt_tokens, completion_tokens


def truncate_to_max_length(
    prompt_tokens: List[int],
    completion_tokens: List[int],
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """Right-truncate the joint sequence to `max_length` tokens.

    Truncates from the end (drops trailing completion tokens first; if the
    prompt itself already exceeds `max_length` the completion becomes empty
    and the caller should drop the example).
    """
    if len(prompt_tokens) >= max_length:
        return prompt_tokens[:max_length], []
    budget = max_length - len(prompt_tokens)
    return prompt_tokens, completion_tokens[:budget]


async def _compute_one_ref_sum(
    sampling_client: tinker.SamplingClient,
    prompt_tokens: List[int],
    completion_tokens: List[int],
    semaphore: asyncio.Semaphore,
) -> float:
    """Sum of base-model log p(token_t | tokens_<t) over the completion region."""
    if not completion_tokens:
        return 0.0
    full = prompt_tokens + completion_tokens
    prompt = types.ModelInput.from_ints(tokens=full)
    async with semaphore:
        logprobs = await sampling_client.compute_logprobs_async(prompt)
    # `logprobs[t]` is logp of token at position t; logprobs[0] is None
    # (no preceding context), so we sum positions [len(prompt_tokens), len(full)).
    completion_lp = logprobs[len(prompt_tokens):]
    return float(sum(lp for lp in completion_lp if lp is not None))


async def compute_ref_logprob_sums(
    sampling_client: tinker.SamplingClient,
    examples: Sequence[TokenizedExample],
    method: str,
    max_concurrency: int = 32,
) -> List[dict]:
    """Compute base-model logp sums for every (prompt, completion) we'll train on.

    Returns a list parallel to `examples` whose schema depends on `method`:
      - "dpo": [{"chosen": float, "rejected": float}]
      - "kto": [{"completion": float, "desirable": bool}]

    For RFT/TFB this function should not be called; they don't need reference
    logprobs.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    if method == "dpo":
        async def _one(ex: TokenizedExample) -> dict:
            chosen_sum = await _compute_one_ref_sum(
                sampling_client, ex.prompt_tokens, ex.completion_tokens, semaphore
            )
            rejected_sum = await _compute_one_ref_sum(
                sampling_client,
                ex.prompt_tokens,
                ex.rejected_completion_tokens or [],
                semaphore,
            )
            return {"chosen": chosen_sum, "rejected": rejected_sum}
        return await asyncio.gather(*[_one(ex) for ex in examples])

    if method == "kto":
        async def _one(ex: TokenizedExample) -> dict:
            ref_sum = await _compute_one_ref_sum(
                sampling_client, ex.prompt_tokens, ex.completion_tokens, semaphore
            )
            return {"completion": ref_sum, "desirable": bool(ex.desirable)}
        return await asyncio.gather(*[_one(ex) for ex in examples])

    raise ValueError(f"compute_ref_logprob_sums only supports dpo/kto, got {method!r}")


def load_ref_cache(cache_path: str) -> Optional[List[dict]]:
    """Return cached reference logprobs, or None if cache doesn't exist."""
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "r") as f:
        return json.load(f)


def save_ref_cache(cache_path: str, ref_data: List[dict]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(ref_data, f)


# ===========================================================================
# Custom-loss data builders (DPO / KTO Datums for training)
# ===========================================================================
def build_pref_datum(
    tokenizer,
    prompt_tokens: List[int],
    completion_tokens: List[int],
    max_length: int = 4096,
) -> Optional[types.Datum]:
    """Encode a (prompt, completion) into a Datum with weights=1 over the
    completion region.

    Like `build_sft_datum` but takes pre-tokenized inputs and skips the
    EOS-injection logic (the caller is expected to have done that already in
    `_encode_pair`).
    """
    prompt_tokens, completion_tokens = truncate_to_max_length(
        prompt_tokens, completion_tokens, max_length
    )
    if not completion_tokens:
        return None

    full = prompt_tokens + completion_tokens
    if len(full) < 2:
        return None

    input_tokens = full[:-1]
    target_tokens = full[1:]
    completion_len = len(completion_tokens)

    weights = [0.0] * len(target_tokens)
    start = len(target_tokens) - completion_len
    for i in range(max(0, start), len(weights)):
        weights[i] = 1.0

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=np.array(weights, dtype=np.float32),
            target_tokens=np.array(target_tokens, dtype=np.int64),
        ),
    )


# ===========================================================================
# Custom torch losses for DPO / KTO
# ===========================================================================
# Both losses run inside `training_client.forward_backward_custom_async`. The
# Tinker backend gives us a `logprobs_list[i]` shaped (seq_len_i - 1,) per
# datum -- one logp value per *target* position. We multiply by the same
# `weights` vector we sent in (1.0 over completion, 0.0 over prompt) and sum
# to get `log pi_theta(completion | prompt)` for each datum.
def _completion_logp_theta(data, logprobs_list):
    """Per-datum scalar tensor: sum over completion positions of weights * logp.

    Output is a 1-D torch tensor of shape (B,), preserving autograd back to
    `logprobs_list`.
    """
    import torch  # local import keeps tinker_utils torch-optional for non-train uses
    out = []
    for d, lp in zip(data, logprobs_list):
        weights_np = d.loss_fn_inputs["weights"]
        if hasattr(weights_np, "data"):  # tinker TensorData wrapper
            weights_np = np.asarray(weights_np.data, dtype=np.float32)
        else:
            weights_np = np.asarray(weights_np, dtype=np.float32)
        weights_t = torch.as_tensor(weights_np, dtype=lp.dtype, device=lp.device)
        out.append((lp * weights_t).sum())
    return torch.stack(out)


def make_dpo_loss(beta: float, ref_chosen: Sequence[float], ref_rejected: Sequence[float]):
    """Return a closure with the DPO loss matching tinker's CustomLossFnV1.

    Expects the batch to contain interleaved (chosen, rejected) datums in pairs:
        [chosen_0, rejected_0, chosen_1, rejected_1, ...]
    `ref_chosen[i]` and `ref_rejected[i]` are the matching reference logp sums
    for pair `i` (i.e. corresponds to datums 2i and 2i+1).
    """
    import torch

    ref_chosen_t = torch.tensor(list(ref_chosen), dtype=torch.float32)
    ref_rejected_t = torch.tensor(list(ref_rejected), dtype=torch.float32)

    def _loss(data, logprobs_list):
        if len(data) % 2 != 0:
            raise ValueError(
                f"DPO batch size must be even (chosen+rejected interleaved); "
                f"got {len(data)}"
            )
        n_pairs = len(data) // 2
        if n_pairs > len(ref_chosen_t):
            raise ValueError(
                f"DPO ref cache size mismatch: pairs={n_pairs}, "
                f"ref_chosen={len(ref_chosen_t)}"
            )
        log_pi = _completion_logp_theta(data, logprobs_list)  # (2*n_pairs,)
        log_pi_chosen = log_pi[0::2]   # (n_pairs,)
        log_pi_rejected = log_pi[1::2]

        ref_c = ref_chosen_t[:n_pairs].to(log_pi.device)
        ref_r = ref_rejected_t[:n_pairs].to(log_pi.device)

        logits = beta * ((log_pi_chosen - ref_c) - (log_pi_rejected - ref_r))
        loss = -torch.nn.functional.logsigmoid(logits).mean()

        with torch.no_grad():
            chosen_reward = beta * (log_pi_chosen - ref_c)
            rejected_reward = beta * (log_pi_rejected - ref_r)
            margin = chosen_reward - rejected_reward
            accuracy = (margin > 0).float().mean()

        metrics = {
            "dpo_loss": float(loss.item()),
            "dpo_chosen_reward": float(chosen_reward.mean().item()),
            "dpo_rejected_reward": float(rejected_reward.mean().item()),
            "dpo_margin": float(margin.mean().item()),
            "dpo_accuracy": float(accuracy.item()),
        }
        return loss, metrics

    return _loss


def make_kto_loss(
    beta: float,
    ref_completion: Sequence[float],
    desirable_mask: Sequence[bool],
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
):
    """Return a closure with the KTO loss matching tinker's CustomLossFnV1.

    Per-example KTO loss with a batch-mean KL anchor (z_ref):

        h_w = log pi_theta(y_w | x) - log pi_ref(y_w | x)
        z_ref = stop_grad(max(0, mean(h_w)))   # batch-level KL upper bound
        loss_desirable   = lambda_D * (1 - sigmoid(beta * (h_w - z_ref)))
        loss_undesirable = lambda_U * (1 - sigmoid(beta * (z_ref - h_w)))
        loss = mean(per-example loss)

    Datum order in the batch determines which example uses which label and
    which reference logprob; both lists must align by index with `data`.
    """
    import torch

    ref_t = torch.tensor(list(ref_completion), dtype=torch.float32)
    des_t = torch.tensor([1.0 if d else 0.0 for d in desirable_mask], dtype=torch.float32)

    def _loss(data, logprobs_list):
        n = len(data)
        if n > len(ref_t):
            raise ValueError(
                f"KTO ref cache size mismatch: batch={n}, ref={len(ref_t)}"
            )
        log_pi = _completion_logp_theta(data, logprobs_list)   # (n,)
        ref = ref_t[:n].to(log_pi.device)
        des = des_t[:n].to(log_pi.device)

        h = log_pi - ref
        # KL upper bound, no gradient. Clamped to 0 so the anchor is non-negative
        # (matches the reference KTO implementation).
        z_ref = torch.clamp(h.detach().mean(), min=0.0)

        loss_desirable = desirable_weight * (1.0 - torch.sigmoid(beta * (h - z_ref)))
        loss_undesirable = undesirable_weight * (1.0 - torch.sigmoid(beta * (z_ref - h)))
        per_example = des * loss_desirable + (1.0 - des) * loss_undesirable
        loss = per_example.mean()

        with torch.no_grad():
            n_des = des.sum().clamp(min=1.0)
            n_und = (1.0 - des).sum().clamp(min=1.0)
            mean_h_des = (h * des).sum() / n_des
            mean_h_und = (h * (1.0 - des)).sum() / n_und

        metrics = {
            "kto_loss": float(loss.item()),
            "kto_z_ref": float(z_ref.item()),
            "kto_h_desirable": float(mean_h_des.item()),
            "kto_h_undesirable": float(mean_h_und.item()),
            "kto_n_desirable": int(des.sum().item()),
            "kto_n_undesirable": int((1 - des).sum().item()),
        }
        return loss, metrics

    return _loss
