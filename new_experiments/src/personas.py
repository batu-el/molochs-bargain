"""Voter persona loader for the new experiments.

Loads paired (persona text, demographics) records from `subjects/`:

- `subjects/personas_{split}.json`     : list[str]  -- free-form persona descriptions
- `subjects/demographics_{split}.json` : list[dict] -- structured demographic fields

Personas and demographics are aligned by index (record `i` in personas_train.json
describes the same person as record `i` in demographics_train.json).

The voter biography fed to the OpenAI roleplay prompt is selected by `bio_mode`:

- "persona"      (DEFAULT): free-form persona paragraph only.
- "demographics" (ABLATION): structured demographic list only -- used to test
                  the contribution of the rich personality text by isolating
                  what a voter can do with demographics alone.
- "both": demographics header + persona paragraph (combined view).

This replaces the older `artsco/voter/utils.load_persona100()` (100 historical-
figure dicts) with 800 train + 200 test demographically realistic personas.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Sequence

from new_experiments.src.config import subjects_path

BioMode = str  # one of {"persona", "demographics", "both"}
_VALID_BIO_MODES = ("persona", "demographics", "both")


# Pretty labels for demographic keys (everything else falls back to title-case).
_DEMOGRAPHIC_LABELS: Dict[str, str] = {
    "geographic_region": "Region",
    "gender": "Gender",
    "age": "Age",
    "education_level": "Education",
    "race": "Race / ethnicity",
    "us_citizen": "US citizen",
    "marital_status": "Marital status",
    "religion": "Religion",
    "religious_attendance": "Religious attendance",
    "political_affiliation": "Political affiliation",
    "income": "Household income",
    "political_views": "Political views",
    "household_size": "Household size",
    "employment_status": "Employment status",
}


def _label(key: str) -> str:
    return _DEMOGRAPHIC_LABELS.get(key, key.replace("_", " ").capitalize())


def _format_demographics(d: Dict[str, Any]) -> str:
    lines: List[str] = []
    for k, v in d.items():
        if v is None or v == "":
            continue
        lines.append(f"- {_label(k)}: {v}")
    return "\n".join(lines)


def _format_voter_bio(
    persona_text: str,
    demographics: Dict[str, Any],
    bio_mode: BioMode,
) -> str:
    """Build a single bio string per voter according to `bio_mode`."""
    if bio_mode == "persona":
        return persona_text.strip()
    if bio_mode == "demographics":
        return "## Demographics\n" + _format_demographics(demographics)
    if bio_mode == "both":
        return (
            "## Demographics\n"
            f"{_format_demographics(demographics)}\n\n"
            "## Personality, Values, and Background\n"
            f"{persona_text.strip()}"
        )
    raise ValueError(
        f"bio_mode must be one of {_VALID_BIO_MODES}, got {bio_mode!r}"
    )


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_voter_bios(
    split: str,
    n: int | None = None,
    seed: int | None = 0,
    bio_mode: BioMode = "persona",
) -> List[str]:
    """Return a list of voter biographies.

    Args:
        split: "train" or "test" (selects `personas_{split}.json` and
            `demographics_{split}.json`).
        n: number of personas to return. If None, returns the full pool
            (800 train / 200 test) in file order.
        seed: random seed for sampling without replacement when ``n`` is
            smaller than the pool. Pass ``None`` to fall back to the
            deterministic ``[:n]`` head of the file (legacy behavior).
            Default 0 -> reproducible random subset.
        bio_mode: which view of the voter to return (default "persona"):
            - "persona"      : free-form persona paragraph only (DEFAULT).
            - "demographics" : structured demographic list only (ABLATION).
            - "both"         : demographics header + persona paragraph.

    Raises:
        ValueError on bad arguments / asking for more voters than available.
        AssertionError if persona/demographic files have mismatched lengths.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if bio_mode not in _VALID_BIO_MODES:
        raise ValueError(
            f"bio_mode must be one of {_VALID_BIO_MODES}, got {bio_mode!r}"
        )

    personas: Sequence[str] = _load_json(subjects_path("personas", split))
    demographics: Sequence[Dict[str, Any]] = _load_json(subjects_path("demographics", split))

    if len(personas) != len(demographics):
        raise AssertionError(
            f"persona/demographics length mismatch for split={split}: "
            f"{len(personas)} personas vs {len(demographics)} demographics"
        )

    bios = [_format_voter_bio(p, d, bio_mode) for p, d in zip(personas, demographics)]
    total = len(bios)

    if n is None:
        return bios
    if n > total:
        raise ValueError(
            f"requested n={n} voters but only {total} {split} personas available"
        )

    if seed is None or n == total:
        return bios[:n]

    # Reproducible sample without replacement. Using a private Random instance
    # so we don't disturb the global RNG state (callers may set their own).
    # NB: the *index set* is identical across bio_modes for a given (n, seed),
    # so the persona / demographics / both views always describe the same 25
    # underlying people -- ablations only change the bio surface form.
    rng = random.Random(seed)
    indices = rng.sample(range(total), n)
    indices.sort()  # keep the original order for stable iteration / logging
    return [bios[i] for i in indices]


def available_voter_count(split: str) -> int:
    """Number of personas available in `subjects/personas_{split}.json`."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return len(_load_json(subjects_path("personas", split)))


if __name__ == "__main__":
    for split in ("train", "test"):
        for mode in _VALID_BIO_MODES:
            sampled = load_voter_bios(split, n=25, seed=0, bio_mode=mode)
            print(f"[personas] split={split} mode={mode} count={len(sampled)} (seed=0)")
            print("-" * 80)
            print(sampled[0])
            print("-" * 80)
