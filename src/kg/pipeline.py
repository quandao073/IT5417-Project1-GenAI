"""The KG search (docs/v2-kg-design.md, "Truy vấn online"): question -> parser -> concepts ->
score columns -> images -> explanation.

    parse        QueryParser (KG prompt: include/exclude on the 14 labels, `terms` beyond them);
                 a failing language model leaves the rule parser's labels
    labels       each label of include/exclude is expanded like any concept (D55, D63)
    terms        lookup_concept, the model chooses a candidate with the whole question as context,
                 expand_concepts; the terms run in parallel
    translate    the English of a Vietnamese question (src/kg/translate.py), beside the parser
    score        ScoreMatrix in RAM fused with the cosine of the English (D74, D75), top-k, view
                 filter; nothing wanted understood: the text route (D73, T44)
    explain      one call for the page; separate from `search`, so the results can be sent first

When Neo4j fails the labels are scored directly from the matrix and the terms are dropped with a
warning (N8); when the language model fails the parser, the chooser and the explainer each fall
back at once (N3). Every step is timed and the trace says what each step decided.
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from src.kg import expand, lookup
from src.kg.expand import ABSENT, PRESENT, Column, Intent
from src.kg.llm_query import Choice, build_facts
from src.kg.messages import Warn, render_all
from src.kg.scoring import ScoreMatrix
from src.kg.store import StoreError
from src.kg.text_fallback import fuse
from src.kg.translate import is_vietnamese
from src.query.schema import SearchQuery

logger = logging.getLogger("cxr.kg")


@dataclass
class KGResult:
    request_id: str
    question: str
    language: str
    query: SearchQuery
    intents: list[Intent]
    warnings: list[Warn]
    rows: list[dict]                       # rank, row, study_id, image_id, view, score
    trace: dict
    timings: dict
    degraded: list[str] = field(default_factory=list)

    def warnings_text(self) -> list[str]:
        everything = [*self.warnings, *[w for i in self.intents for w in i.warnings]]
        return render_all(everything, self.language)


class KGSearch:
    def __init__(self, *, cfg: dict, matrix: ScoreMatrix, parser, store, quality: dict,
                 label_ids: dict[str, str], names: dict[str, str], chooser=None,
                 explainer=None, versions: dict | None = None, semantic=None,
                 text_fallback=None, translator=None):
        self.cfg, self.matrix, self.parser, self.store = cfg, matrix, parser, store
        self.quality, self.label_ids, self.names = quality, label_ids, names
        self.chooser, self.explainer, self.versions = chooser, explainer, versions or {}
        self.semantic = semantic        # sapbert.AliasIndex or None
        self.text_fallback = text_fallback  # text_fallback.TextFallback or None (D73)
        self.translator = translator        # translate.Translator or None: the question as written
        self._terms_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kg-terms")

    # -- intents ---------------------------------------------------------------------------

    def _direct_label(self, label: str, polarity: str) -> Intent:
        """A label scored straight from its matrix column: no graph (the 14-label route)."""
        cui = self.label_ids[label]
        q = self.quality[cui]
        col = Column(cui=cui, column=q["column"], weight=1.0, quality=q["auroc_lenient"],
                     weak=bool(q["weak"]), borderline=bool(q["borderline"]),
                     name=self.names.get(cui, label))
        intent = Intent(label, cui, col.name, polarity, "direct", columns=[col], path=[col.name])
        if col.weak:
            intent.warnings.append(Warn("weak_quality", {"name": col.name,
                                                         "auroc": f"{col.quality:.2f}"}))
        return intent

    def _label(self, label: str, polarity: str, kg_down: list[str]) -> Intent:
        """A label of include/exclude, expanded through the graph unless it is down (`kg_down`
        collects the reason, and every later label goes straight to its column)."""
        if self.store is not None and not kg_down:
            try:
                return expand.expand_concepts(self.store, self.label_ids[label], polarity,
                                              self.cfg["query"]["expand"], label)
            except StoreError as exc:
                kg_down.append(str(exc))
        return self._direct_label(label, polarity)

    def _term(self, question: str, term, qcfg: dict) -> tuple[lookup.Lookup, Choice, Intent]:
        t0 = time.perf_counter()
        found = lookup.lookup_concept(self.store, term.text, qcfg, self.semantic)
        t1 = time.perf_counter()
        if self.chooser is not None:
            choice = self.chooser.choose(question, term.text, term.polarity, found)
        else:
            top = found.candidates[0].cui if found.candidates else None
            choice = Choice(top, "deterministic" if top else "none", "language model disabled")
        t2 = time.perf_counter()
        if choice.cui is None:
            intent = Intent(term.text, "", term.text, term.polarity, "none",
                            warnings=[Warn("unrecognized", {"term": term.text})])
        else:
            intent = expand.expand_concepts(self.store, choice.cui, term.polarity,
                                            qcfg["expand"], term.text)
        t3 = time.perf_counter()
        choice.ms = choice.ms or (t2 - t1) * 1000
        found.timings = {"lookup_ms": (t1 - t0) * 1000, "choose_ms": (t2 - t1) * 1000,
                         "expand_ms": (t3 - t2) * 1000}
        return found, choice, intent

    def _sides(self, question: str, query: SearchQuery) -> list[str]:
        """The sides the question names: the parser's, else words that cannot mean anything else."""
        found = list(query.sides)
        words = f" {lookup.normalize(question)} "
        for side, forms in self.cfg["query"]["question_sides"].items():
            if side not in found and any(f" {lookup.normalize(f)} " in words for f in forms):
                found.append(side)
        return found

    # -- the search ------------------------------------------------------------------------

    def search(self, question: str, top_k: int = 10, views: tuple[str, ...] = ()) -> KGResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[Warn] = []
        degraded: list[str] = []
        qcfg = self.cfg["query"]

        english = (self._terms_pool.submit(self.translator.english, question)
                   if self.translator is not None and self.text_fallback is not None else None)
        t = time.perf_counter()
        parsed = self.parser.parse(question)
        timings["parse_ms"] = (time.perf_counter() - t) * 1000
        query = parsed.query
        if parsed.fallback_used or (not getattr(self.parser, "llm", None)):
            reason = next((w for w in parsed.warnings if "llm" in w.lower()),
                          "language model disabled")
            warnings.append(Warn("llm_fallback", {"reason": reason[:100]}))
            degraded.append("parser: rule parser")
        sides = self._sides(question, query)
        if sides:
            warnings.append(Warn("side_ignored"))
        trace: dict = {"parser": {"parser": parsed.parser, "fallback": parsed.fallback_used,
                                  "llm_ms": parsed.llm_ms, "cached": parsed.cached,
                                  "terms": [t.model_dump() for t in query.terms],
                                  "sides": sides},
                       "terms": []}

        kg_down: list[str] = [] if self.store is not None else ["no Neo4j connection"]
        intents: list[Intent] = []
        t = time.perf_counter()
        for c in query.include:
            intents.append(self._label(c.concept, PRESENT, kg_down))
            if c.status == "UNCERTAIN":
                warnings.append(Warn("uncertain_as_present", {"name": intents[-1].name}))
        for label in query.exclude:
            intents.append(self._label(label, ABSENT, kg_down))
        timings["labels_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        modifiers: list[str] = []             # terms that only qualify: one warning lists them
        if query.terms and not kg_down:
            try:
                futures = [self._terms_pool.submit(self._term, question, term, qcfg)
                           for term in query.terms]
                for term, future in zip(query.terms, futures, strict=True):
                    found, choice, intent = future.result()
                    if not found.core or found.modifier:   # a side or a modifier: no concept
                        trace["terms"].append({"term": term.text, "skipped": (
                            "modifier" if found.modifier else "side only")})
                        if found.modifier:
                            modifiers.append(term.text)
                        continue
                    intents.append(intent)
                    if (choice.via == "deterministic" and len(found.candidates) > 1
                            and not found.single_exact):
                        warnings.append(Warn("chosen_deterministic", {
                            "term": term.text, "name": intent.name}))
                        degraded.append("chooser: first candidate")
                    trace["terms"].append({
                        "term": term.text, "polarity": term.polarity, **found.to_dict(),
                        "timings_ms": {k: round(v, 1) for k, v in found.timings.items()},
                        "choice": {"cui": choice.cui, "via": choice.via, "reason": choice.reason,
                                   "ms": round(choice.ms, 1), "error": choice.error,
                                   "cached": choice.cached},
                        "intent": intent.to_dict()})
            except StoreError as exc:
                kg_down.append(str(exc))
                intents = [i for i in intents if i.term in set(query.constrained_concepts())]
        if modifiers:
            warnings.append(Warn("not_considered", {"words": ", ".join(modifiers)}))
        if kg_down:
            degraded.append(f"knowledge graph unavailable: {kg_down[0]}")
            if query.terms:
                warnings += [Warn("terms_dropped", {"term": t.text}) for t in query.terms]
            warnings.append(Warn("kg_unavailable"))
        timings["terms_ms"] = (time.perf_counter() - t) * 1000

        # a location beside findings is not scored: the findings decide, the location only warns
        located = [i for i in intents if i.kind == "anatomy"]
        if located and any(i.columns and i.kind != "anatomy" for i in intents):
            intents = [i for i in intents if i.kind != "anatomy"]
            warnings += [Warn("location_ignored", {"region": i.name}) for i in located]
            trace["location_ignored"] = [i.name for i in located]

        # one intent per (concept, polarity); a repeated phrase must not count twice
        seen, unique = set(), []
        for it in intents:
            key = (it.cui, it.polarity) if it.columns else None
            if key is None or key not in seen:
                unique.append(it)
                seen.add(key)
        intents = unique
        trace["intents"] = [i.to_dict() for i in intents]

        if english is not None:
            t = time.perf_counter()
            translation = english.result()
            timings["translate_wait_ms"] = (time.perf_counter() - t) * 1000
            timings["translate_ms"] = translation.ms
            trace["translation"] = {"text": translation.text, "called": translation.called,
                                    "error": translation.error}
            english = translation.text
        else:                               # no translator: only an English question has one
            english = None if is_vietnamese(question) else question
        trace["english"] = english

        t = time.perf_counter()
        rows: list[dict] = []
        weight = qcfg["fusion"]["weight"]
        if any(i.columns for i in intents):
            scores = self.matrix.score(intents)
            if weight and self.text_fallback is not None and english:
                scores = fuse(scores, self.text_fallback.cosine(english), weight)
                trace["fusion"] = {"weight": weight, "text": english}
            top = self.matrix.top_k(scores, top_k, tuple(views or query.views))
            rows = [{"rank": r, "row": int(row), "study_id": self.matrix.study_ids[row],
                     "image_id": self.matrix.image_ids[row], "view": str(self.matrix.views[row]),
                     "score": float(scores[row])} for r, row in enumerate(top, start=1)]
        # nothing wanted was understood: rank by text like B1 (D73); a resolved negation alone
        # keeps the KG ranking
        wanted = [i for i in intents if i.polarity == PRESENT]
        if (self.text_fallback is not None and not any(i.columns for i in wanted)
                and (wanted or not rows)):
            text = self.text_fallback.text(query, english, question)
            top = self.text_fallback.top(text, top_k, tuple(views or query.views))
            rows = [{"rank": r, "row": row, "study_id": self.matrix.study_ids[row],
                     "image_id": self.matrix.image_ids[row], "view": str(self.matrix.views[row]),
                     "score": score} for r, (row, score) in enumerate(top, start=1)]
            warnings.append(Warn("text_fallback"))
            trace["fallback"] = {"route": "text", "text": text}
        timings["score_ms"] = (time.perf_counter() - t) * 1000
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        timings = {k: round(v, 1) for k, v in timings.items()}
        return KGResult(uuid.uuid4().hex, question, query.language, query, intents, warnings,
                        rows, trace, timings, degraded)

    def explain(self, result: KGResult, top: int | None = None):
        """The explanation of the page, or None when no explainer is configured or there is
        nothing to explain (no results, or results of the text route: no KG facts)."""
        if self.explainer is None or not result.rows or "fallback" in result.trace:
            return None
        rows = result.rows[:top or self.cfg["explain"]["top_results"]]
        probs = self.matrix.probabilities([r["row"] for r in rows], result.intents)
        base = getattr(self.explainer, "knowledge", None)     # None: prompt v3, no knowledge
        items = None if base is None else base.items(result.intents, result.language)
        facts = build_facts(result.question, result.language, result.intents, rows, probs,
                            self.explainer.index.names(result.intents), items)
        return self.explainer.explain(facts, result.intents, items)
