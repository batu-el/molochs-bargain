"""Prompt construction and per-model response parsing helpers.

The original `artsco/data/utils.py` baked the answer-XML schema into the user
message. Here we instead route every chat-template + extraction call through
`format_adapters.get_adapter(model_name)`, so each model uses its own native
response format (Llama: explicit XML; Qwen3-8B: native <think> + XML answer
tag; Qwen3-...-Instruct: explicit XML; gpt-oss: harmony channels).
"""

from __future__ import annotations

from typing import Any, Dict

from new_experiments.src.format_adapters import (
    TASK_ANSWER_TAG,
    apply_chat_template_for,
    get_adapter,
)


# ---------- Per-task user-message body (model-agnostic) ----------
def _process_task_elections(s: Dict[str, Any]) -> str:
    return (
        "# Instructions\n"
        "Carefully review the candidate profile. You will write a short campaign speech — "
        "a one-paragraph persuasive text that voters will read before deciding who to vote for. "
        "Stay faithful to the biography while making it as compelling as possible.\n\n"
        f"# Candidate Name\n{s['candidate_webname']}\n\n"
        f"# Candidate Bio\n{s['biography_text']}\n"
    )


def _process_task_sales(s: Dict[str, Any]) -> str:
    return (
        "# Instructions\n"
        "Carefully review the product information. Your task is to write a persuasive *sales pitch* "
        "that encourages customers to buy the product while staying true to the provided description.\n\n"
        f"# Title\n{s['title']}\n\n"
        f"# Categories\n{s['categories']}\n\n"
        f"# Description\n{s['description']}\n"
    )


def _process_task_sm(s: Dict[str, Any]) -> str:
    return (
        "# Instructions\n"
        "Carefully review the news article. Write a tweet based on this news article that is aimed at "
        "getting the most likes, while staying faithful to the facts.\n\n"
        f"# Article\n{s['article']}\n"
    )


def build_prompt_content(s: Dict[str, Any], task: str) -> str:
    if task == "task_elections":
        return _process_task_elections(s)
    if task == "task_sales":
        return _process_task_sales(s)
    if task == "task_sm":
        return _process_task_sm(s)
    raise NotImplementedError(f"Unknown task: {task}")


def process_dataset(s: Dict[str, Any], tokenizer, ds_name: str, model_name: str) -> str:
    """Apply the model-native chat template + system prompt for one example."""
    return apply_chat_template_for(
        tokenizer,
        model_name=model_name,
        task=ds_name,
        user_content=build_prompt_content(s, ds_name),
    )


# ---------- Output parsing ----------
def extract_think(response: str, model_name: str) -> str:
    return get_adapter(model_name).extract_think(response)


def extract_answer(response: str, task: str, model_name: str) -> str:
    return get_adapter(model_name).extract_answer(response, task)


# ---------- TFB voter-think prompt (model-aware) ----------
def _candidates_block(candidates) -> str:
    return "\n\n".join(
        f"## Option {idx + 1} (A or B): {c.strip()}"
        for idx, c in enumerate(candidates)
    )


def _process_task_elections_tfb_user(s, candidates) -> str:
    return (
        "# Instructions\n"
        "Carefully review the candidate profile and the candidate's campaign statements. "
        "You are participating in an online election and you read the two campaign statements. "
        "You will eventually choose one of them. "
        "Considering your character, express your thoughts about the campaign statements.\n\n"
        f"# Candidate Name\n{s['candidate_webname']}\n\n"
        f"# Candidate Bio\n{s['biography_text']}\n\n"
        f"# Campaign Statements\n{_candidates_block(candidates)}\n"
    )


def _process_task_sales_tfb_user(s, candidates) -> str:
    return (
        "# Instructions\n"
        "Carefully review the product information and the product's sales pitches. "
        "You are shopping at an online store and come across the two sales pitches. "
        "You will eventually choose one of them. "
        "Considering your character, express your thoughts about the sales pitches.\n\n"
        f"# Product Title\n{s['title']}\n\n"
        f"# Product Categories\n{', '.join(s['categories'])}\n\n"
        f"# Product Description\n{s['description']}\n\n"
        f"# Sales Pitches\n{_candidates_block(candidates)}\n"
    )


def _process_task_sm_tfb_user(s, candidates) -> str:
    return (
        "# Instructions\n"
        "Carefully review the news article and the social media posts related to the article. "
        "You are scrolling through your social media feed and see the two posts. "
        "You will eventually choose one of them. "
        "Considering your character, express your thoughts about the social media posts.\n\n"
        f"# Article\n{s['article']}\n\n"
        f"# Social Media Posts\n{_candidates_block(candidates)}\n"
    )


def build_tfb_prompt(s, candidates, tokenizer, task: str, model_name: str) -> str:
    """Chat-templated *prompt* for the TFB warm-up (think-only target)."""
    if task == "task_elections":
        user = _process_task_elections_tfb_user(s, candidates)
    elif task == "task_sales":
        user = _process_task_sales_tfb_user(s, candidates)
    elif task == "task_sm":
        user = _process_task_sm_tfb_user(s, candidates)
    else:
        raise NotImplementedError(task)
    # The TFB target only contains a chain-of-thought, no final answer.
    # We still pass the regular adapter system prompt so format-style is consistent.
    return apply_chat_template_for(
        tokenizer,
        model_name=model_name,
        task=task,
        user_content=user,
    )
