"""LLM steps of V2-4 (D58): a first review of the strongest SUGGESTS edges and Vietnamese alias
proposals, both by one Anthropic model through `llm_select` (cache, refusals as answers); and,
after V2-6 (D68), a route proposal for each core disease that reaches no score column. Each
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


class RouteProposal(BaseModel):
    decision: Literal["is_a", "suggests", "none"]
    columns: list[str]
    reason: str


SUGGEST_SCHEMA = llm_select._schema(SuggestReview)
ALIAS_SCHEMA = llm_select._schema(AliasProposal)
ROUTE_SCHEMA = llm_select._schema(RouteProposal)
REFUSED_REVIEW = {"decision": "refused", "reason": "refused"}
REFUSED_ALIASES = {"aliases": [], "reason": "refused"}
REFUSED_ROUTE = {"decision": "refused", "columns": [], "reason": "refused"}


def _parser(model: type[BaseModel]):
    def parse(content: str, item: dict) -> dict:
        try:
            return model.model_validate_json(content).model_dump()
        except ValidationError as exc:
            raise llm_select.LLMSelectError(
                f"reply does not fit the schema: {exc.errors()[:1]}") from exc
    return parse


parse_review, parse_aliases = _parser(SuggestReview), _parser(AliasProposal)


def parse_route(content: str, item: dict) -> dict:
    """A route proposal names only columns of the item's list: at least one for is_a and
    suggests (one for is_a), none for none."""
    answer = _parser(RouteProposal)(content, item)
    allowed = {c["cui"] for c in item["input"]["columns"]}
    columns = list(dict.fromkeys(answer["columns"]))
    if set(columns) - allowed:
        raise llm_select.LLMSelectError(f"columns outside the list: {set(columns) - allowed}")
    wanted = {"is_a": (1, 1), "suggests": (1, len(allowed)), "none": (0, 0)}[answer["decision"]]
    if not wanted[0] <= len(columns) <= wanted[1]:
        raise llm_select.LLMSelectError(
            f"{answer['decision']} with {len(columns)} columns")
    return {**answer, "columns": columns}


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


def route_items(pending: Sequence[dict], info: dict, columns: Sequence[str],
                min_n: int) -> list[dict]:
    """The core diseases without a route (scripts/kg_disease_routes.py) with at least `min_n`
    pool mentions as items: the disease and every score column, each a concept."""
    listed = [_concept(info, c) for c in columns]
    return [{"kind": "route", "string": row["cui"], "input": {
        "disease": _concept(info, row["cui"]), "n_pool_reports": row["n_pool"],
        "columns": listed}} for row in pending if row["n_pool"] >= min_n]


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
