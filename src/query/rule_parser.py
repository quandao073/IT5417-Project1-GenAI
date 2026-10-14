"""Deterministic query parser over configs/concepts.yaml (plan 6.4, 10.3).

Longest token-sequence match over the English and Vietnamese surface forms, a
negation cue within a window before (5 tokens) or after (3 tokens) each hit, an
uncertainty cue in the same windows (the nearer of the two cues wins, a tie goes
to uncertainty), AP/PA views, and language detection. Tokens are split on
whitespace; "," ";" "." are kept as clause tokens, "? ! : ( ) [ ] \" ' /" are
separators that are dropped, and "-" stays inside a token ("x-ray"). It
never raises: a request it does not understand parses to an empty SearchQuery,
which the API reports with a warning. Vietnamese is not word-segmented; the
multi-token surface forms carry that load, so "tràn khí màng phổi" beats
"tràn khí" because it is longer, not because of any grammar.

semantic_query is NOT set here; QueryParser (parser.py) builds it with the one
rule both parsers share (spec 4.2).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.query.schema import Constraint, SearchQuery

NEGATION_WINDOW_BEFORE = 5
CUE_WINDOW_AFTER = 3
# A cue's scope ends at these. "and"/"và" are deliberately absent so that
# "no effusion or pneumothorax" and "không có tràn dịch và tràn khí" negate both.
TERMINATORS = frozenset({"but", "with", "showing", "nhưng", "kèm", ",", ";", "."})
_VI_CHARS = frozenset(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
_TOKEN = re.compile(r"""[,;.]|[^\s,;.?!:()\[\]"'/]+""")

Form = tuple[tuple[str, ...], str, str]   # (tokens, key, language)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


@dataclass(frozen=True)
class Hit:
    start: int        # token index, inclusive
    end: int          # token index, exclusive
    key: str          # concept name, view name, or the cue text
    language: str     # "en" | "vi": which list the matched form came from


def _forms(section: dict, keys: Iterable[str] | None = None) -> list[Form]:
    out: list[Form] = []
    for key, by_lang in section.items():
        if keys is not None and key not in keys:
            continue
        for lang in ("en", "vi"):
            out.extend((tuple(tokenize(f)), key, lang) for f in by_lang[lang])
    return sorted(out, key=lambda f: -len(f[0]))      # longest first; stable, en before vi


def _cue_forms(section: dict) -> list[Form]:
    forms = [(tuple(tokenize(f)), f, lang) for lang in ("en", "vi") for f in section[lang]]
    return sorted(forms, key=lambda f: -len(f[0]))


def match(tokens: Sequence[str], forms: Sequence[Form]) -> list[Hit]:
    """Greedy left-to-right, longest form first, no overlaps."""
    hits: list[Hit] = []
    i = 0
    while i < len(tokens):
        for form, key, lang in forms:
            n = len(form)
            if n and tuple(tokens[i:i + n]) == form:
                hits.append(Hit(i, i + n, key, lang))
                i += n
                break
        else:
            i += 1
    return hits


class RuleParser:
    def __init__(self, concepts_cfg: dict):
        self.concepts = _forms(concepts_cfg["concepts"])
        self.views = _forms(concepts_cfg["views"], keys=("AP", "PA"))
        self.negation = _cue_forms(concepts_cfg["negation_cues"])
        self.negation_post = _cue_forms(concepts_cfg["negation_cues_post"])
        self.uncertainty = _cue_forms(concepts_cfg["uncertainty_cues"])
        self.en_first = {c: v["en"][0] for c, v in concepts_cfg["concepts"].items()}

    def parse(self, text: str) -> SearchQuery:
        tokens = tokenize(text)
        concepts = match(tokens, self.concepts)
        views = match(tokens, self.views)
        taken = [(h.start, h.end) for h in concepts + views]

        def free(hits: list[Hit]) -> list[Hit]:
            # A cue inside a matched form ("no" in "no finding") is part of the form.
            return [h for h in hits if not any(h.start < e and h.end > s for s, e in taken)]

        neg = free(match(tokens, self.negation))
        neg_post = free(match(tokens, self.negation_post))
        unc = free(match(tokens, self.uncertainty))

        polarity: dict[str, str] = {}      # concept -> PRESENT | UNCERTAIN | EXCLUDED
        for h in concepts:
            # The nearest applicable cue wins; on a tie uncertainty does.
            u = _nearest(_before(tokens, unc, h), _after(tokens, unc, h))
            n = _nearest(_before(tokens, neg, h), _after(tokens, neg_post, h))
            if u is not None and (n is None or u <= n):
                polarity[h.key] = "UNCERTAIN"
            elif n is not None:
                polarity[h.key] = "EXCLUDED"
            else:
                polarity[h.key] = "PRESENT"
        # Insertion order is first-mention order; a later mention of the same
        # concept overwrote its polarity above ("effusion ... no effusion").
        include = [Constraint(concept=c, status=s) for c, s in polarity.items()
                   if s != "EXCLUDED"]
        exclude = [c for c, s in polarity.items() if s == "EXCLUDED"]
        hits = concepts + views + neg + neg_post + unc
        language = "vi" if _is_vietnamese(text, hits) else "en"
        return SearchQuery(include=include, exclude=exclude,
                           views=list(dict.fromkeys(h.key for h in views)), language=language)


def _nearest(*gaps: int | None) -> int | None:
    found = [g for g in gaps if g is not None]
    return min(found) if found else None


def _before(tokens: Sequence[str], cues: list[Hit], h: Hit) -> int | None:
    """Smallest token gap to an applicable cue before the hit, or None."""
    return _nearest(*(h.start - c.end for c in cues
                      if c.end <= h.start and h.start - c.end <= NEGATION_WINDOW_BEFORE
                      and not TERMINATORS & set(tokens[c.end:h.start])))


def _after(tokens: Sequence[str], cues: list[Hit], h: Hit) -> int | None:
    """Smallest token gap to an applicable cue after the hit, or None."""
    return _nearest(*(c.start - h.end for c in cues
                      if c.start >= h.end and c.start - h.end <= CUE_WINDOW_AFTER
                      and not TERMINATORS & set(tokens[h.end:c.start])))


def _is_vietnamese(text: str, hits: list[Hit]) -> bool:
    return any(h.language == "vi" for h in hits) or bool(_VI_CHARS & set(normalize(text)))
