"""Voter persona loader for the new experiments.

Loads paired (persona text, demographics) records from the *fixed* train/test
audiences in `subjects/`:

- `subjects/{audience}_persona_{N}.json`     : list[str]  -- N free-form persona paragraphs
- `subjects/{audience}_demographic_{N}.json` : list[dict] -- N structured demographic dicts

where `audience` is "train" or "test" and N comes from `NUM_VOTERS_TRAIN` /
`NUM_VOTERS_TEST` in `config.py`. Both audiences are sampled once (with a
fixed seed) by `new_experiments/scripts/build_audiences.py` from the larger
800-train / 200-test pool that lives next to them in `subjects/`. We commit to
these people across every (model, task, prompt, method) so all comparisons
share the same evaluators.

Personas and demographics are aligned by index (record `i` in
`train_persona_{N}.json` describes the same person as record `i` in
`train_demographic_{N}.json`). The voter biography fed to the OpenAI roleplay
prompt is selected by `bio_mode`:

- "persona"      (DEFAULT): free-form persona paragraph only.
- "demographics" (ABLATION): structured demographic list only -- isolates what a
                  voter can do with demographics alone.
- "both": demographics header + persona paragraph (combined view).

This replaces the older `artsco/voter/utils.load_persona100()` (100 historical-
figure dicts) with the demographically realistic audiences.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from new_experiments.src.config import (
    AUDIENCES,
    NUM_VOTERS_TEST,
    NUM_VOTERS_TRAIN,
    audience_path,
)

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


_EXPECTED_SIZE = {"train": NUM_VOTERS_TRAIN, "test": NUM_VOTERS_TEST}


def load_voter_bios(
    audience: str = "train",
    bio_mode: BioMode = "persona",
) -> List[str]:
    """Return the fixed list of voter biographies for `audience`.

    The on-disk file is selected via `audience_path(...)`, which uses the
    `NUM_VOTERS_TRAIN` / `NUM_VOTERS_TEST` constants from `config.py` to pick
    the size suffix (e.g. `subjects/train_persona_20.json`).

    Args:
        audience: "train" or "test".
        bio_mode: which view of the voter to return (default "persona"):
            - "persona"      : free-form persona paragraph only (DEFAULT).
            - "demographics" : structured demographic list only (ABLATION).
            - "both"         : demographics header + persona paragraph.

    Raises:
        ValueError on bad arguments.
        AssertionError if persona/demographic files have mismatched lengths.
    """
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}, got {audience!r}")
    if bio_mode not in _VALID_BIO_MODES:
        raise ValueError(
            f"bio_mode must be one of {_VALID_BIO_MODES}, got {bio_mode!r}"
        )

    personas: List[str] = _load_json(audience_path("persona", audience))
    demographics: List[Dict[str, Any]] = _load_json(audience_path("demographic", audience))

    if len(personas) != len(demographics):
        raise AssertionError(
            f"persona/demographic length mismatch for audience={audience}: "
            f"{len(personas)} personas vs {len(demographics)} demographics"
        )

    expected = _EXPECTED_SIZE[audience]
    if len(personas) != expected:
        # Soft warning so non-default audience sizes still work; the fixed-N
        # assumption only matters for cost estimates and reproducibility.
        print(
            f"[personas] WARNING: audience={audience} has {len(personas)} "
            f"people, expected {expected}. Re-run "
            "new_experiments/scripts/build_audiences.py to refresh."
        )

    return [_format_voter_bio(p, d, bio_mode) for p, d in zip(personas, demographics)]


def available_voter_count(audience: str = "train") -> int:
    """Number of people in the configured `audience` file."""
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}, got {audience!r}")
    return len(_load_json(audience_path("persona", audience)))


if __name__ == "__main__":
    for aud in AUDIENCES:
        for mode in _VALID_BIO_MODES:
            sampled = load_voter_bios(audience=aud, bio_mode=mode)
            print(f"[personas] audience={aud} mode={mode} count={len(sampled)}")
            print("-" * 80)
            print(sampled[0])
            print("-" * 80)
