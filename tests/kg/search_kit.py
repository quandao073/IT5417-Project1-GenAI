"""Pieces to build a KGSearch over the tiny KG with scripted language-model clients."""

from __future__ import annotations

import json
import time

from src.common.config import load_config
from src.kg.guard import Guard, GuardedClient
from src.kg.llm_query import ConceptChooser, Explainer, GroundingIndex
from src.kg.pipeline import KGSearch
from src.kg.scoring import ScoreMatrix
from src.kg.store import StoreError
from src.query.parser import QueryParser
from src.query.rule_parser import RuleParser
from tests.kg import tiny_kg
from tests.kg.fake_store import FakeStore

LABEL_IDS = {label: cui for cui, label in tiny_kg.LABEL_OF.items()}


class Chat:
    """`chat` returns the next scripted item: a str (the reply), an Exception (raised) or a
    number (seconds to hang, then an LLMError-free empty object)."""

    model = "scripted"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, schema, stream=False, meta=None):
        self.calls.append(messages)
        if meta is not None:
            meta["first_token_ms"] = 12.0
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, int | float):
            time.sleep(item)
            item = "{}"
        return (item if isinstance(item, str) else json.dumps(item)), 1.0

    def probe(self):
        return "ok"

    def version(self):
        return None


class DownStore(FakeStore):
    """A store whose every query fails, like a stopped Neo4j."""

    def __getattribute__(self, name):
        if name in {"build", "aliases_exact", "aliases_fulltext", "concepts", "scorable_below",
                    "scorable_above", "suggesting", "anatomy_findings"}:
            def fail(*a, **k):
                raise StoreError("neo4j unreachable")
            return fail
        return super().__getattribute__(name)


def parse_reply(include=(), exclude=(), terms=(), views=()):
    return {"findings": "", "include": [{"concept": c, "status": "PRESENT"} for c in include],
            "exclude": list(exclude), "views": list(views), "language": "vi",
            "terms": [{"text": t, "polarity": p} for t, p in terms]}


def matrix() -> ScoreMatrix:
    scores, images = tiny_kg.scores(), tiny_kg.images()
    cols = [c for c in scores.columns if c not in ("study_id", "image_id")]
    views = images["view"].to_numpy().astype("U2")
    return ScoreMatrix(cols, scores.select(cols).to_numpy().T, scores["study_id"].to_list(),
                       scores["image_id"].to_list(), views)


def index() -> GroundingIndex:
    return GroundingIndex.build(tiny_kg.nodes(), tiny_kg.aliases(), load_config("kg")["explain"])


def make_search(*, parser_chat=None, choose_chat=None, explain_chat=None, store="fake",
                guard=None, timeout=1.0) -> KGSearch:
    guard = guard or Guard(3, 60.0)
    rule = RuleParser(load_config("concepts"))
    pclient = GuardedClient(parser_chat, guard, timeout) if parser_chat else None
    parser = QueryParser(rule, template="x {}", llm_client=pclient, system_prompt="p",
                         max_retries=0, with_terms=True)
    chooser = ConceptChooser(choose_chat, guard, "choose", timeout) if choose_chat else None
    explainer = (Explainer(explain_chat, guard, "explain", timeout, index())
                 if explain_chat else None)
    store = FakeStore() if store == "fake" else store
    nodes = tiny_kg.nodes()
    return KGSearch(cfg=load_config("kg"), matrix=matrix(), parser=parser, store=store,
                    quality=tiny_kg.quality(), label_ids=LABEL_IDS,
                    names=dict(nodes.select("cui", "name").iter_rows()), chooser=chooser,
                    explainer=explainer, versions={"kg_build_id": "kg_test",
                                                   "matrix_version": "m@0", "build_id": "b"})

