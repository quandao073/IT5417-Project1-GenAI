"""The query template: the one string the text encoder reads for a parsed query.

`render` is the only place a template is built, for search now and for the benchmark and the
fine-tuning captions later, so a query is phrased the way the encoder is scored and trained:
"cardiomegaly and pulmonary edema and no pleural effusion". Each concept is its
`concepts.yaml:<concept>.en[0]`, except No Finding, which reads "normal chest" ("no finding"
reads as a negation). Concepts follow CHEXPERT_LABELS, so "A and B" and "B and A" are one
string. No "chest x-ray showing" prefix: the template reads like impression text.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.data.labels import CHEXPERT_LABELS

NO_FINDING = "No Finding"
NO_FINDING_PHRASE = "normal chest"
_ORDER = {name: i for i, name in enumerate(CHEXPERT_LABELS)}


def phrase(concept: str, concepts_cfg: dict) -> str:
    if concept == NO_FINDING:
        return NO_FINDING_PHRASE
    return concepts_cfg["concepts"][concept]["en"][0]


def _ordered(concepts: Sequence[str]) -> list[str]:
    return sorted(dict.fromkeys(concepts), key=_ORDER.__getitem__)


def render(include: Sequence[str], exclude: Sequence[str], concepts_cfg: dict) -> str:
    """'a and b and no c' in CHEXPERT_LABELS order; 'no c' alone when nothing is included.
    Refuses an empty template, a negated No Finding and a concept both wanted and refused."""
    if not include and not exclude:
        raise ValueError("a template needs at least one concept")
    if NO_FINDING in exclude:
        raise ValueError("No Finding cannot be negated in a template")
    if both := set(include) & set(exclude):
        raise ValueError(f"concepts both included and excluded: {sorted(both)}")
    parts = [phrase(c, concepts_cfg) for c in _ordered(include)]
    parts += [f"no {phrase(c, concepts_cfg)}" for c in _ordered(exclude)]
    return " and ".join(parts)
