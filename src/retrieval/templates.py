"""Template strings for the image modes (H1, WORKLOG D30-D32).

Every sentence the image-mode encoder is trained on, benchmarked with or queried with is
rendered here from concept names: "cardiomegaly and pulmonary edema and no pleural effusion".
The LLM parser never writes these strings; it only produces the SearchQuery they are rendered
from, so a query is in the training distribution by construction. Concepts are ordered by
CHEXPERT_LABELS so "A and B" and "B and A" are one string. No "chest x-ray showing" prefix:
the templates read like impression text, as in CXR-RETRIEVE.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.data.labels import CHEXPERT_LABELS
from src.query.schema import Constraint, SearchQuery

NO_FINDING = "No Finding"
KINDS = ("single", "pair", "negation", "paraphrase", "triple", "pair_negation")
_ORDER = {name: i for i, name in enumerate(CHEXPERT_LABELS)}


def phrase(concept: str, concepts_cfg: dict) -> str:
    """concepts.yaml:<concept>.en[0], the one surface form every template uses."""
    return concepts_cfg["concepts"][concept]["en"][0]


def _ordered(concepts: Sequence[str]) -> list[str]:
    return sorted(dict.fromkeys(concepts), key=_ORDER.__getitem__)


def _check(include: Sequence[str], exclude: Sequence[str]) -> None:
    if not include and not exclude:
        raise ValueError("a template needs at least one concept")
    if NO_FINDING in exclude:
        raise ValueError("No Finding cannot be negated in a template")
    if both := set(include) & set(exclude):
        raise ValueError(f"concepts both included and excluded: {sorted(both)}")


def render(include: Sequence[str], exclude: Sequence[str], concepts_cfg: dict) -> str:
    """'a and b and no c' in CHEXPERT_LABELS order; 'no c' alone when nothing is included."""
    _check(include, exclude)
    parts = [phrase(c, concepts_cfg) for c in _ordered(include)]
    parts += [f"no {phrase(c, concepts_cfg)}" for c in _ordered(exclude)]
    return " and ".join(parts)


def render_paraphrase(include: Sequence[str], exclude: Sequence[str], concepts_cfg: dict) -> str:
    """The paper's paraphrase: 'an image with a but without b' / 'an image with a and b'."""
    _check(include, exclude)
    if not include:
        raise ValueError("the paraphrase template needs an included concept")
    text = "an image with " + " and ".join(phrase(c, concepts_cfg) for c in _ordered(include))
    if exclude:
        text += " but without " + " and ".join(phrase(c, concepts_cfg) for c in _ordered(exclude))
    return text


@dataclass(frozen=True)
class TemplateQuery:
    id: str
    kind: str
    text: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    gold: SearchQuery


def _query(kind: str, k: int, include: Sequence[str], exclude: Sequence[str],
           concepts_cfg: dict) -> TemplateQuery:
    include, exclude = _ordered(include), _ordered(exclude)
    text = (render_paraphrase if kind == "paraphrase" else render)(include, exclude, concepts_cfg)
    gold = SearchQuery(include=[Constraint(concept=c, status="PRESENT") for c in include],
                       exclude=list(exclude), views=[], language="en")
    return TemplateQuery(f"{kind}-{k:03d}", kind, text, tuple(include), tuple(exclude), gold)


def generate(concepts: Sequence[str], kinds: Sequence[str], concepts_cfg: dict, *,
             seed: int = 20260101, n_multi: int = 60) -> list[TemplateQuery]:
    """Template queries over `concepts`: every single; every unordered pair and every ordered
    (a, no b) over the pathologies (No Finding only ever stands alone); paraphrases of those
    pairs; and `n_multi` sampled three-concept queries (half 'a and b and c', half 'a and b
    and no c'), drawn without repetition by `np.random.default_rng(seed)`."""
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown template kinds {sorted(unknown)}; known: {KINDS}")
    concepts = _ordered(concepts)
    pathologies = [c for c in concepts if c != NO_FINDING]
    pairs = [(a, b) for i, a in enumerate(pathologies) for b in pathologies[i + 1:]]
    negations = [(a, b) for a in pathologies for b in pathologies if a != b]
    out: list[TemplateQuery] = []
    for kind in KINDS:
        if kind not in kinds:
            continue
        if kind == "single":
            specs = [([c], []) for c in concepts]
        elif kind == "pair":
            specs = [([a, b], []) for a, b in pairs]
        elif kind == "negation":
            specs = [([a], [b]) for a, b in negations]
        elif kind == "paraphrase":
            specs = [([a, b], []) for a, b in pairs] + [([a], [b]) for a, b in negations]
        else:
            specs = _sample_multi(pathologies, kind, n_multi, seed)
        out.extend(_query(kind, k, inc, exc, concepts_cfg) for k, (inc, exc) in enumerate(specs))
    return out


def _sample_multi(pathologies: Sequence[str], kind: str, n_multi: int, seed: int) -> list:
    """Distinct three-concept specs, in a fixed order given the seed."""
    rng = np.random.default_rng(seed)
    n = n_multi // 2 if kind == "triple" else n_multi - n_multi // 2
    seen: set[tuple] = set()
    specs: list = []
    while len(specs) < n:
        a, b, c = (pathologies[i] for i in rng.choice(len(pathologies), 3, replace=False))
        spec = (_ordered([a, b, c]), []) if kind == "triple" else (_ordered([a, b]), [c])
        key = (tuple(spec[0]), tuple(spec[1]))
        if key in seen:
            continue
        seen.add(key)
        specs.append(spec)
    return specs


def training_concepts(training_cfg: dict) -> list[str]:
    """configs/training.yaml:concepts, checked against the 14 labels and their order."""
    concepts = list(training_cfg["concepts"])
    if unknown := [c for c in concepts if c not in _ORDER]:
        raise ValueError(f"training.yaml names unknown concepts {unknown}")
    if concepts != _ordered(concepts):
        raise ValueError("training.yaml:concepts must follow CHEXPERT_LABELS order")
    return concepts
