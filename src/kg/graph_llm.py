"""LLM steps of V2-4 (D58): a first review of the strongest SUGGESTS edges and Vietnamese alias
proposals, both by one Anthropic model through `llm_select` (cache, refusals as answers). Each
writes a review CSV that Claude in the chat app adjudicates; nothing here decides alone.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal

import polars as pl
from pydantic import BaseModel, ValidationError

from src.kg import llm_select


class SuggestReview(BaseModel):
    decision: Literal["keep", "drop"]
    reason: str


class AliasProposal(BaseModel):
    aliases: list[str]
    reason: str


SUGGEST_SCHEMA = llm_select._schema(SuggestReview)
ALIAS_SCHEMA = llm_select._schema(AliasProposal)
REFUSED_REVIEW = {"decision": "refused", "reason": "refused"}
REFUSED_ALIASES = {"aliases": [], "reason": "refused"}


def _parser(model: type[BaseModel]):
    def parse(content: str, item: dict) -> dict:
        try:
            return model.model_validate_json(content).model_dump()
        except ValidationError as exc:
            raise llm_select.LLMSelectError(
                f"reply does not fit the schema: {exc.errors()[:1]}") from exc
    return parse


parse_review, parse_aliases = _parser(SuggestReview), _parser(AliasProposal)


def _concept(info: dict, cui: str) -> dict:
    name, types = info.get(cui, (cui, ""))
    return {"cui": cui, "name": name, "semantic_types": types}


def suggest_items(edges: pl.DataFrame, info: dict, top: int) -> list[dict]:
    """The `top` strongest SUGGESTS edges (n, then p) as items: both concepts and the counts."""
    strongest = (edges.filter(pl.col("rel") == "SUGGESTS")
                 .sort(["n", "p", "src", "dst"], descending=[True, True, False, False]).head(top))
    return [{"kind": "SUGGESTS", "string": f"{s}>{d}", "input": {
        "source": _concept(info, s), "target": _concept(info, d), "n_reports_with_both": n,
        "n_reports_with_source": n_src, "p": round(p, 4)}}
        for s, d, n, n_src, p in strongest.select("src", "dst", "n", "n_src", "p").iter_rows()]


def alias_items(nodes: pl.DataFrame, info: dict, english: dict[str, list[str]],
                existing_vi: dict[str, list[str]], top: int) -> list[dict]:
    """The 14 labels and the `top` other concepts by n_pool as items: name, semantic types,
    English names, and the Vietnamese forms the label already has (configs/concepts.yaml)."""
    labels = nodes.filter(pl.col("scorable")).sort("cui")
    others = (nodes.filter(~pl.col("scorable")).sort(["n_pool", "cui"], descending=[True, False])
              .head(top))
    items = []
    for cui, label, n in pl.concat([labels, others]).select("cui", "chexpert_label",
                                                           "n_pool").iter_rows():
        items.append({"kind": "alias", "string": cui, "input": {
            **_concept(info, cui), "chexpert_label": label, "n_pool_reports": n,
            "english_names": english.get(cui, [])[:8],
            "vietnamese_forms_already_used": existing_vi.get(label, []) if label else []}})
    return items


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def _inside(alias: str, phrase: str) -> bool:
    a, p = _tokens(alias), _tokens(phrase)
    return a != p and any(p[i:i + len(a)] == a for i in range(len(p) - len(a) + 1))


def alias_flags(rows: Sequence[tuple[str, str]], blocked: Collection[str],
                existing_vi: dict[str, list[str]], label_of: dict[str, str]) -> list[str]:
    """Per (cui, alias), the blocker's flags (T20), `;`-joined: single_syllable (one Vietnamese
    syllable, easily found inside other words), inside_blocked (sits inside a blocked or common
    phrase, like "phù" in "phù hợp"), shared (proposed for another concept too, or already the
    form of another label)."""
    owners: dict[str, set[str]] = {}
    for cui, alias in rows:
        owners.setdefault(alias.lower(), set()).add(cui)
    for label, forms in existing_vi.items():
        for form in forms:
            owners.setdefault(form.lower(), set()).add(f"label:{label}")
    out = []
    for cui, alias in rows:
        found = set()
        if len(_tokens(alias)) == 1:
            found.add("single_syllable")
        if any(_inside(alias, phrase) for phrase in blocked):
            found.add("inside_blocked")
        others = owners[alias.lower()] - {cui, f"label:{label_of.get(cui)}"}
        if others:
            found.add("shared")
        out.append(";".join(sorted(found)))
    return out
