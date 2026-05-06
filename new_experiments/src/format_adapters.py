"""Per-model native response-format adapters.

Different open-weight model families have different *native* ways to express
"chain of thought + final answer":

  * Llama 3.x Instruct      — no native CoT.  We add explicit
                              <think>...</think><answer_tag>...</answer_tag>
                              instructions plus a few-shot example.
  * Qwen3-8B (Hybrid)       — natively emits <think>...</think> when
                              `enable_thinking=True`.  We additionally ask
                              the model to wrap the post-think final answer
                              in <answer_tag>...</answer_tag>.
  * Qwen3-...-Instruct-2507 — non-thinking instruct variant.  Falls back to
                              the Llama-style explicit XML schema.
  * gpt-oss-20b             — harmony response format with `analysis` /
                              `final` channels.  We decode WITHOUT stripping
                              special tokens and parse the channel headers.

This module exposes a single `get_adapter(model_name)` entry point.  All call
sites (prep_data, generate1, generate2, generate22, build_train_data) go
through this adapter so the rest of the pipeline stays model-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------- Final-answer XML tag per task (used for non-harmony models) ----------
TASK_ANSWER_TAG: Dict[str, str] = {
    "task_elections": "campaign_speech",
    "task_sales": "sales_pitch",
    "task_sm": "tweet",
}


# A short, task-specific worked example.  Used in the system prompt as a
# one-shot demonstration of the expected output structure.
_FEWSHOT_USER: Dict[str, str] = {
    "task_elections": (
        "# Instructions\n"
        "Carefully review the candidate profile. You will write a short campaign speech — "
        "a one-paragraph persuasive text that voters will read before deciding who to vote for. "
        "Stay faithful to the biography while making it as compelling as possible.\n\n"
        "# Candidate Name\nJane Doe\n\n"
        "# Candidate Bio\nJane Doe is a former public school teacher and city council member from Springfield "
        "who has spent 15 years championing education funding and small-business grants.\n"
    ),
    "task_sales": (
        "# Instructions\n"
        "Carefully review the product information. Your task is to write a persuasive sales pitch.\n\n"
        "# Title\nAcme Stainless-Steel Travel Mug\n\n"
        "# Categories\nKitchen, Travel\n\n"
        "# Description\n16-oz double-walled vacuum-insulated tumbler that keeps drinks hot for 12 h or cold for 24 h.\n"
    ),
    "task_sm": (
        "# Instructions\n"
        "Carefully review the news article and write a tweet about it.\n\n"
        "# Article\nResearchers at MIT announced a new battery chemistry that doubles the energy density of "
        "today's lithium-ion cells while halving manufacturing cost.\n"
    ),
}


def _fewshot_assistant_xml(task: str) -> str:
    """Reference assistant turn for the explicit-XML format (Llama / Qwen Instruct)."""
    tag = TASK_ANSWER_TAG[task]
    if task == "task_elections":
        body = (
            "I'm Jane Doe, and for fifteen years I've been fighting for Springfield's "
            "kids in the classroom and Springfield's families in city hall. As a teacher "
            "I saw what underfunded schools cost our children; on city council I delivered "
            "the small-business grants that kept Main Street alive. Send me back to work for you."
        )
        think = (
            "The bio emphasises education funding and small-business support, both broadly "
            "popular. I'll lead with her teaching experience for credibility, then bridge to "
            "her council record on grants to show she delivers."
        )
    elif task == "task_sales":
        body = (
            "Tired of cold coffee by 10 a.m.? The Acme Travel Mug locks in heat for a "
            "full 12 hours and keeps cold drinks icy for an entire day — vacuum-sealed, "
            "leak-proof, and built to outlast your commute. Pour it in the morning, sip it "
            "all afternoon. Order yours today."
        )
        think = (
            "Headline benefit is temperature retention. Pair the spec with a relatable "
            "scenario (morning coffee still hot at lunch) and close with a low-friction CTA."
        )
    elif task == "task_sm":
        body = (
            "MIT just unveiled a battery chemistry with 2× the energy density of today's "
            "lithium-ion cells — at HALF the manufacturing cost. Phones that last days, EVs "
            "that go further, all cheaper to make."
        )
        think = (
            "Lead with the punchy 2× / ½× contrast, attribute to MIT for credibility, and end "
            "on the practical implication for everyday devices."
        )
    else:
        raise NotImplementedError(task)
    return f"<think>{think}</think><{tag}>{body}</{tag}>"


def _fewshot_assistant_qwen_thinking(task: str) -> str:
    """For Qwen3 hybrid: chat template injects `<think>\\n` before the assistant turn,
    so the assistant message itself starts mid-think. The few-shot demonstrates the
    pattern after the auto-injected opener."""
    tag = TASK_ANSWER_TAG[task]
    body = _fewshot_assistant_xml(task)
    # body looks like "<think>...</think><tag>...</tag>" — strip the leading <think>
    # because Qwen's template already opens it, then keep the rest verbatim.
    assert body.startswith("<think>")
    return body[len("<think>"):]


def _fewshot_assistant_harmony(task: str) -> str:
    """Final-channel content for the gpt-oss few-shot.

    The harmony chat template will turn the {role:'assistant', content:...} into
    `<|start|>assistant<|channel|>final<|message|>{content}<|end|>` for us, so we
    only need to provide the user-visible final message here.  Chain-of-thought
    is encouraged via the system prompt rather than baked into the few-shot."""
    if task == "task_elections":
        return (
            "I'm Jane Doe, and for fifteen years I've been fighting for Springfield's "
            "kids in the classroom and Springfield's families in city hall. As a teacher "
            "I saw what underfunded schools cost our children; on city council I delivered "
            "the small-business grants that kept Main Street alive. Send me back to work for you."
        )
    if task == "task_sales":
        return (
            "Tired of cold coffee by 10 a.m.? The Acme Travel Mug locks in heat for a "
            "full 12 hours and keeps cold drinks icy for an entire day — vacuum-sealed, "
            "leak-proof, and built to outlast your commute. Order yours today."
        )
    if task == "task_sm":
        return (
            "MIT just unveiled a battery chemistry with 2× the energy density of today's "
            "lithium-ion cells — at HALF the manufacturing cost. Phones that last days, "
            "EVs that go further, all cheaper to make."
        )
    raise NotImplementedError(task)


# ---------- System prompts (one per format style, parameterised by task) ----------
def _system_xml(task: str) -> str:
    """Llama / Qwen-Instruct system prompt: explicit <think>+<tag> schema."""
    tag = TASK_ANSWER_TAG[task]
    return (
        "You are a helpful assistant.\n"
        "When you reply, first enclose your chain-of-thought reasoning inside "
        f"<think> ... </think>, followed immediately by your final {tag.replace('_', ' ')} "
        f"inside <{tag}> ... </{tag}>. Do not output anything outside of those two tags.\n\n"
        "Example output:\n"
        f"{_fewshot_assistant_xml(task)}"
    )


def _system_qwen_thinking(task: str) -> str:
    """Qwen3-8B Hybrid: model already emits <think>...</think> natively.  We just
    need to teach it where to put the final answer."""
    tag = TASK_ANSWER_TAG[task]
    return (
        "You are a helpful assistant.\n"
        f"After your <think> ... </think> reasoning, output the final {tag.replace('_', ' ')} "
        f"wrapped in <{tag}> ... </{tag}>. Do not output anything outside of those two tags.\n\n"
        "Example output:\n"
        f"<think>{_fewshot_assistant_qwen_thinking(task)}"
    )


def _system_harmony(task: str) -> str:
    """gpt-oss harmony: tell the model how to use the analysis + final channels."""
    tag = TASK_ANSWER_TAG[task]
    label = tag.replace("_", " ")
    return (
        "You are a helpful assistant.\n"
        f"Reasoning: medium\n"
        f"Use the `analysis` channel for your private chain-of-thought, "
        f"then output your final {label} on the `final` channel. The `final` "
        f"channel must contain ONLY the {label} text — no preamble, no XML "
        f"tags, no bullet list, just the {label} itself."
    )


# ---------- Extractors ----------
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
    re.DOTALL,
)
_HARMONY_ANALYSIS_RE = re.compile(
    r"<\|channel\|>\s*analysis\s*<\|message\|>(.*?)<\|end\|>",
    re.DOTALL,
)
_HARMONY_TOKEN_RE = re.compile(r"<\|[^|]*\|>")


def _strip_harmony_residue(text: str) -> str:
    """Remove any stray harmony special tokens / channel labels that survived."""
    text = _HARMONY_TOKEN_RE.sub("", text)
    text = re.sub(r"^\s*(analysis|commentary|final)\s*", "", text)
    return text.strip()


def _extract_xml(text: str, tag: str) -> str:
    if not isinstance(text, str):
        return ""
    try:
        return text.split(f"<{tag}>", 1)[1].split(f"</{tag}>", 1)[0].strip()
    except Exception:
        return ""


def _extract_xml_think(text: str) -> str:
    return _extract_xml(text, "think")


def _extract_qwen_native_think(text: str) -> str:
    """Qwen3 hybrid: `<think>` opener is in the prompt, so the completion starts
    mid-thought; everything up to the first `</think>` is the chain of thought."""
    if not isinstance(text, str) or "</think>" not in text:
        return ""
    head = text.split("</think>", 1)[0]
    if "<think>" in head:
        head = head.split("<think>", 1)[1]
    return head.strip()


def _extract_qwen_native_answer(text: str, task: str) -> str:
    tag = TASK_ANSWER_TAG[task]
    inside = _extract_xml(text, tag)
    if inside:
        return inside
    if "</think>" in text:
        tail = text.split("</think>", 1)[1].strip()
        tail = re.sub(r"^<\w[\w_]*>\s*", "", tail)
        tail = re.sub(r"\s*</\w[\w_]*>\s*$", "", tail)
        return tail.strip()
    return ""




def _extract_harmony_think(text: str) -> str:
    if not isinstance(text, str):
        return ""
    matches = _HARMONY_ANALYSIS_RE.findall(text)
    if matches:
        return _strip_harmony_residue(matches[-1])
    return ""


def _extract_harmony_answer(text: str, task: str) -> str:
    if not isinstance(text, str):
        return ""
    matches = _HARMONY_FINAL_RE.findall(text)
    if matches:
        return _strip_harmony_residue(matches[-1])
    if "<|channel|>final<|message|>" in text:
        tail = text.split("<|channel|>final<|message|>", 1)[1]
        for stop in ("<|end|>", "<|return|>"):
            if stop in tail:
                tail = tail.split(stop, 1)[0]
                break
        return _strip_harmony_residue(tail)
    return ""


# ---------- Adapter ----------
@dataclass
class FormatAdapter:
    """Per-model wiring for prompt formatting + completion parsing."""

    name: str
    style: str                                                  # "xml" | "qwen_native" | "harmony"
    enable_thinking: bool                                       # passed to apply_chat_template
    skip_special_tokens_on_decode: bool                         # tokenizer.decode option
    system_prompt_for: Callable[[str], str]
    extract_think_fn: Callable[[str], str]
    extract_answer_fn: Callable[[str, str], str]
    fewshot_assistant_for: Optional[Callable[[str], str]] = None  # used in TFB targets
    extra_chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ----- forwarders -----
    def system_prompt(self, task: str) -> str:
        return self.system_prompt_for(task)

    def extract_think(self, completion: str) -> str:
        return self.extract_think_fn(completion)

    def extract_answer(self, completion: str, task: str) -> str:
        return self.extract_answer_fn(completion, task)


def _xml_adapter(name: str) -> FormatAdapter:
    return FormatAdapter(
        name=name,
        style="xml",
        enable_thinking=False,
        skip_special_tokens_on_decode=True,
        system_prompt_for=_system_xml,
        extract_think_fn=_extract_xml_think,
        extract_answer_fn=lambda text, task: _extract_xml(text, TASK_ANSWER_TAG[task]),
        fewshot_assistant_for=_fewshot_assistant_xml,
    )


def _qwen_thinking_adapter(name: str) -> FormatAdapter:
    return FormatAdapter(
        name=name,
        style="qwen_native",
        enable_thinking=True,
        skip_special_tokens_on_decode=True,
        system_prompt_for=_system_qwen_thinking,
        extract_think_fn=_extract_qwen_native_think,
        extract_answer_fn=_extract_qwen_native_answer,
        fewshot_assistant_for=_fewshot_assistant_qwen_thinking,
    )


def _harmony_adapter(name: str) -> FormatAdapter:
    return FormatAdapter(
        name=name,
        style="harmony",
        enable_thinking=False,
        skip_special_tokens_on_decode=False,    # need <|channel|> markers intact
        system_prompt_for=_system_harmony,
        extract_think_fn=_extract_harmony_think,
        extract_answer_fn=_extract_harmony_answer,
        fewshot_assistant_for=_fewshot_assistant_harmony,
    )


_ADAPTERS: Dict[str, FormatAdapter] = {
    "meta-llama/Llama-3.1-8B-Instruct":      _xml_adapter("meta-llama/Llama-3.1-8B-Instruct"),
    "meta-llama/Llama-3.3-70B-Instruct":     _xml_adapter("meta-llama/Llama-3.3-70B-Instruct"),
    "Qwen/Qwen3-8B":                         _qwen_thinking_adapter("Qwen/Qwen3-8B"),
    "Qwen/Qwen3-32B":                        _qwen_thinking_adapter("Qwen/Qwen3-32B"),
    "openai/gpt-oss-20b":                    _harmony_adapter("openai/gpt-oss-20b"),
}


def get_adapter(model_name: str) -> FormatAdapter:
    if model_name not in _ADAPTERS:
        raise KeyError(
            f"No format adapter registered for {model_name!r}. "
            f"Add one to new_experiments/src/format_adapters.py."
        )
    return _ADAPTERS[model_name]


# ---------- Chat-template wrapper that respects the adapter ----------
def build_messages(model_name: str, task: str, user_content: str) -> List[Dict[str, str]]:
    adapter = get_adapter(model_name)
    return [
        {"role": "system", "content": adapter.system_prompt(task)},
        {"role": "user", "content": user_content},
    ]


def apply_chat_template_for(
    tokenizer,
    model_name: str,
    task: str,
    user_content: str,
) -> str:
    adapter = get_adapter(model_name)
    messages = build_messages(model_name, task, user_content)
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
        **adapter.extra_chat_template_kwargs,
    )
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=adapter.enable_thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


# ---------- For TFB: wrap a completion target in the adapter's native format ----------
def render_think_only_completion(model_name: str, think_text: str) -> str:
    """Build a completion string that conveys ONLY the chain-of-thought, in the
    model's native format.

    Used by build_train_data.py to construct the `voter_think` SFT targets
    (the "think out loud" warmup).  We deliberately do NOT include the
    final-answer tag here: TFB only trains the model to imitate the voter's
    deliberation, not to produce an answer."""
    adapter = get_adapter(model_name)
    text = think_text.strip()
    if adapter.style == "xml":
        return f"<think>{text}</think>"
    if adapter.style == "qwen_native":
        # Chat template injects `<think>\n` at start of assistant turn, so the
        # completion should *not* repeat the opening tag.
        return f"{text}\n</think>"
    if adapter.style == "harmony":
        return (
            f"<|channel|>analysis<|message|>{text}<|end|>"
        )
    raise NotImplementedError(adapter.style)
