"""Patch the global `openai.OpenAI` client used by `artsco.voter` and `trends.*`.

Default OpenAI SDK behaviour is `timeout=600s, max_retries=2`. A single
hung request therefore blocks the whole `as_completed` loop for up to
~30 min. Both `artsco.voter.utils` and `trends.utils` instantiate the
client at import time, so we patch the constructor before any of those
modules import.

Usage: `import new_experiments.src.openai_patch  # noqa: F401`
"""

from __future__ import annotations

import os

import openai

_OPENAI_TIMEOUT = float(os.environ.get("OPENAI_TIMEOUT", "30"))         # seconds per request
_OPENAI_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "3"))         # retries on transient errors

_orig_init = openai.OpenAI.__init__


def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("timeout", _OPENAI_TIMEOUT)
    kwargs.setdefault("max_retries", _OPENAI_RETRIES)
    _orig_init(self, *args, **kwargs)


if getattr(openai.OpenAI, "_artsco_patched", False) is False:
    openai.OpenAI.__init__ = _patched_init  # type: ignore[assignment]
    openai.OpenAI._artsco_patched = True    # type: ignore[attr-defined]
    print(
        f"[openai_patch] timeout={_OPENAI_TIMEOUT}s, max_retries={_OPENAI_RETRIES} "
        f"(override with OPENAI_TIMEOUT / OPENAI_MAX_RETRIES env vars)"
    )
