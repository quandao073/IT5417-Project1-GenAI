"""The language-model steps of the KG search (docs/v2-kg-design.md, "Truy vấn online" steps 2 and
5, "Rào chắn"): choosing a concept for a phrase, and explaining a page of results.

`ConceptChooser` gives the model the whole question and the candidates of `lookup_concept` and
accepts only a cui from that list; anything else, an error, a timeout or an open circuit falls
back to the first candidate (deterministic, the ablation row of the design). `Explainer` makes ONE
call for the whole page: a summary and one line per image. What it writes is checked sentence by
sentence against the facts it was given (`GroundingIndex.check`): a sentence that names a finding
or disease of the graph the facts do not contain, quotes a number they do not contain, or uses
diagnostic wording is dropped; when nothing survives the caller shows the raw path instead.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from src.kg.expand import ABSENT, NOISY_OR, WEIGHTED, WEIGHTED_OR, Intent
from src.kg.guard import Guard
from src.kg.lookup import Lookup, normalize
from src.query.llm_parser import LLMError, LLMInvalidOutput

CHOICE_SCHEMA = {"type": "object", "properties": {
    "choice": {"type": "string"}, "reason": {"type": "string"}}, "required": ["choice", "reason"]}
EXPLAIN_SCHEMA = {"type": "object", "properties": {
    "summary": {"type": "string"},
    "images": {"type": "array", "items": {"type": "object", "properties": {
        "rank": {"type": "integer"}, "line": {"type": "string"}},
        "required": ["rank", "line"]}}},
    "required": ["summary", "images"]}
# prompt v4: the same plus the "related knowledge" part, every sentence citing items [K#]
EXPLAIN_KNOWLEDGE_SCHEMA = {**EXPLAIN_SCHEMA, "properties": {
    **EXPLAIN_SCHEMA["properties"], "knowledge": {"type": "string"}},
    "required": ["summary", "images", "knowledge"]}
logger = logging.getLogger("cxr.kg")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_CITATION = re.compile(r"\[(K\d+)\]")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Choice:
    cui: str | None
    via: str                 # llm | single | deterministic | none
    reason: str = ""
    ms: float = 0.0
    error: str | None = None
    cached: bool = False


class ConceptChooser:
    """`cache_file`: the model's choices as JSONL ({key, cui, reason}), read at startup and
    appended to, so a question gets the same concept across restarts (WORKLOG D68). A file that
    cannot be written leaves the cache in memory only."""

    def __init__(self, client, guard: Guard, system_prompt: str, timeout_s: float,
                 gloss: dict[str, str] | None = None, cache_file: Path | None = None):
        self.client, self.guard, self.timeout_s = client, guard, timeout_s
        self.system_prompt, self.gloss = system_prompt, gloss or {}
        self.prompt_sha256 = sha256(system_prompt)
        self._cache: OrderedDict[str, Choice] = OrderedDict()   # what the model chose, per input
        self._lock = threading.Lock()
        self.cache_file = cache_file
        if cache_file is not None and cache_file.is_file():
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(ValueError, KeyError):
                    row = json.loads(line)
                    self._cache[row["key"]] = Choice(row["cui"], "llm", row.get("reason", ""))

    def _remember(self, key: str, choice: Choice) -> None:
        with self._lock:
            self._cache[key] = choice
            if self.cache_file is None:
                return
            try:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "cui": choice.cui,
                                             "reason": choice.reason}, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning("chooser cache not written (%s); memory only", exc)
                self.cache_file = None

    def choose(self, question: str, term: str, polarity: str, lookup: Lookup) -> Choice:
        cands = lookup.candidates
        if not cands:
            return Choice(None, "none", "no candidate")
        if lookup.single_exact:
            return Choice(cands[0].cui, "single", "one exact candidate")
        started = time.perf_counter()
        payload = {"question": question, "term": term, "polarity": polarity, "candidates": [
            {"cui": c.cui, "name": c.name, "label": c.label, "parents": c.parents,
             "matched_alias": c.alias, "match": c.match, "scored_directly": c.scorable,
             "weak": bool(c.weak), **({"when_to_use": self.gloss[c.cui]}
                                      if c.cui in self.gloss else {})}
            for c in cands]}
        body = json.dumps(payload, ensure_ascii=False)
        # the key is everything the model sees, so the same question gets the same answer
        key = sha256("\n".join([getattr(self.client, "model", ""), self.prompt_sha256, body]))
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return replace(hit, ms=0.0, cached=True)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": body}]
        try:
            content, _ = self.guard.call(lambda: self.client.chat(messages, CHOICE_SCHEMA),
                                         self.timeout_s)
            answer = json.loads(content)
            choice = str(answer["choice"]).strip()
        except (LLMError, LLMInvalidOutput, ValueError, KeyError) as exc:
            return Choice(cands[0].cui, "deterministic", "", _ms(started), str(exc)[:120])
        ms = _ms(started)
        if choice != "none" and choice not in {c.cui for c in cands}:
            return Choice(cands[0].cui, "deterministic", "", ms,      # outside the list: never
                          f"the model chose {choice!r}, not a candidate")
        result = Choice(None if choice == "none" else choice, "llm",
                        str(answer.get("reason", "")), ms)
        self._remember(key, result)
        return result


# ---- the explanation -------------------------------------------------------------------------

def _percent(p: float) -> int:
    return int(round(100 * p))


def build_facts(question: str, language: str, intents: list[Intent], rows: list[dict],
                probabilities: list[dict], names: dict[str, list[str]] | None = None,
                knowledge: list[dict] | None = None) -> dict:
    """What the explanation may say. `rows` are the results (rank, view), `probabilities` the
    ScoreMatrix.probabilities of those rows. Numbers are whole percents and ranks only. `names`
    (GroundingIndex.names) gives every concept of the path its EN and VI aliases, so the text can
    name a concept in the question's language with words the grounding check accepts.
    `knowledge` (KnowledgeBase.items) adds the numbered items the knowledge part cites."""
    names = names or {}

    def called(cui: str) -> dict:
        return {"aliases": names[cui]} if names.get(cui) else {}

    spec = []
    for i, it in enumerate(intents, start=1):
        if not it.columns:
            continue
        spec.append({"id": i, "asked": it.term, "concept": it.name, **called(it.cui),
                     "kind": it.kind,
                     "combine": {NOISY_OR: "any_of", WEIGHTED: "weighted",
                                 WEIGHTED_OR: "any_of_weighted"}.get(it.combine,
                                                                               "single"),
                     "excluded": it.polarity == ABSENT,
                     "uses": [{"concept": c.name, **called(c.cui),
                               "weight_percent": _percent(c.weight)} for c in it.columns]})
    results = []
    for row, prob in zip(rows, probabilities, strict=True):
        match = []
        for i, it in enumerate(intents):
            if i not in prob:
                continue
            match.append({"intent": i + 1, "percent": _percent(prob[i]["intent"]),
                          "concepts": [{"concept": c.name,
                                        "percent": _percent(prob[i]["columns"][c.cui])}
                                       for c in it.columns]})
        results.append({"rank": row["rank"], "view": row["view"], "match": match})
    facts = {"question": question, "language": language, "intents": spec, "results": results}
    if knowledge is not None:
        facts["knowledge"] = [{"id": k["id"], "text": k["text"]} for k in knowledge]
    return facts


def facts_numbers(facts: dict) -> set[str]:
    """Every whole number the facts contain, as text."""
    return set(re.findall(r"\d+", json.dumps(facts)))


class GroundingIndex:
    """Aliases of the Finding and Disease concepts of the graph, to spot a mention of a concept
    that is not in the facts. Built once at startup from aliases.parquet and nodes.parquet."""

    def __init__(self, aliases: dict[str, set[str]], by_cui: dict[str, set[str]], cfg: dict,
                 shown: dict[str, list[str]] | None = None):
        self.aliases, self.by_cui, self.cfg = aliases, by_cui, cfg
        self.shown = shown or {}         # cui -> the few aliases the facts show, VI first
        self.diagnostic = [d.casefold() for d in cfg["diagnostic"]]

    def names(self, intents: list[Intent]) -> dict[str, list[str]]:
        """The aliases the facts show for every concept of the intents' paths."""
        cuis = {it.cui for it in intents} | {c.cui for it in intents for c in it.columns}
        return {c: self.shown[c] for c in cuis if c in self.shown}

    @classmethod
    def build(cls, nodes, aliases, cfg: dict) -> GroundingIndex:
        """`nodes`, `aliases`: the polars tables of data/kg/graph (`aliases` with `lang` and
        `source` gives `shown`: per language the hand-made forms first, then the shortest)."""
        keep = set(nodes.filter(nodes["in_core"], nodes["labels"].is_in(["Finding", "Disease"]))
                   ["cui"])
        by_cui: dict[str, set[str]] = {}
        for cui, norm in aliases.select("cui", "norm").iter_rows():
            by_cui.setdefault(cui, set()).add(norm)
        index: dict[str, set[str]] = {}
        stop = {w.casefold() for w in cfg["generic_words"]}
        for cui in keep:
            for norm in by_cui.get(cui, ()):
                words = norm.split()
                if not words or norm in stop or (len(words) == 1
                                                 and len(norm) < cfg["min_single_word_chars"]):
                    continue
                index.setdefault(norm, set()).add(cui)
        shown: dict[str, list[str]] = {}
        if "lang" in aliases.columns and "source" in aliases.columns:
            k = cfg["aliases_shown"]
            for (cui,), rows in aliases.group_by("cui"):
                rows = rows.with_columns((rows["source"] == "mrconso").alias("umls"),
                                         rows["norm"].str.len_chars().alias("len"))
                pick = []
                for lang in ("vi", "en"):
                    pick += rows.filter(rows["lang"] == lang).sort("umls", "len", "norm")[
                        "norm"].to_list()[:k]
                shown[cui] = pick
        return cls(index, by_cui, cfg, shown)

    def allowed(self, facts: dict, intents: list[Intent]) -> tuple[set[str], set[str]]:
        """(cuis, normalised phrases) the explanation may mention: the intents' concepts and
        columns, every alias of those, and the words of the question and of the terms."""
        cuis = {it.cui for it in intents} | {c.cui for it in intents for c in it.columns}
        phrases = {p for c in cuis for p in self.by_cui.get(c, ())}
        text = " ".join([facts["question"], *[json.dumps(i, ensure_ascii=False)
                                              for i in facts["intents"]]])
        words = normalize(text).split()
        phrases |= {" ".join(words[i:i + n]) for n in range(1, 5)
                    for i in range(len(words) - n + 1)}
        return cuis, phrases

    def check_knowledge(self, sentence: str, items: dict[str, dict]) -> tuple[str | None, int]:
        """(None or why the sentence is not grounded, citations): a sentence of the knowledge
        part cites at least one item, only items that exist, and names only concepts and numbers
        of the items it cites."""
        cited = _CITATION.findall(sentence)
        if not cited:
            return "no citation", 0
        unknown = [c for c in cited if c not in items]
        if unknown:
            return f"citation not in the knowledge: {unknown[0]}", 0
        text = _CITATION.sub(" ", sentence)
        sources = " ".join(items[c]["text"] for c in cited)
        given = {n.replace(",", ".") for n in _NUMBER.findall(sources)}
        for number in _NUMBER.findall(text):
            if number.replace(",", ".") not in given:
                return f"number not in the cited items: {number}", 0
        cuis = {c for k in cited for c in items[k]["cuis"]}
        phrases = {p for c in cuis for p in self.by_cui.get(c, ())}
        words = normalize(sources).split()
        phrases |= {" ".join(words[i:i + n]) for n in range(1, 5)
                    for i in range(len(words) - n + 1)}
        why = self.check(text, cuis, phrases, set(re.findall(r"\d+", text)))
        return (why.replace("in the facts", "in the cited items") if why else None,
                0 if why else len(cited))

    def check(self, sentence: str, cuis: set[str], phrases: set[str],
              numbers: set[str]) -> str | None:
        """None when the sentence is grounded, else why it is not."""
        low = sentence.casefold()
        if "<" in sentence or ">" in sentence:      # a tool-call fragment that leaked into the text
            return "markup in the text"
        for word in self.diagnostic:
            if word in low:
                return f"diagnostic wording: {word}"
        for number in _NUMBER.findall(sentence):
            if number.replace(",", ".").split(".")[0] not in numbers:
                return f"number not in the facts: {number}"
        words = normalize(sentence).split()
        covered: set[int] = set()        # words inside an allowed mention, longest first
        for n in range(4, 0, -1):
            for i in range(len(words) - n + 1):
                span = range(i, i + n)
                if covered.intersection(span):
                    continue
                gram = " ".join(words[i:i + n])
                owners = self.aliases.get(gram)
                if owners and owners & cuis or gram in phrases:
                    covered.update(span)
                elif owners:
                    return f"concept not in the facts: {gram}"
        return None


@dataclass
class Explanation:
    status: str                       # ok | partial | ungrounded | failed
    summary: str = ""
    images: dict[int, str] = field(default_factory=dict)
    dropped: list[dict] = field(default_factory=list)
    first_token_ms: float | None = None
    total_ms: float = 0.0
    error: str | None = None
    knowledge: str = ""                         # prompt v4 only: the kept sentences, with [K#]
    knowledge_items: list[dict] | None = None   # the items given (id, text); None for prompt v3
    citations: int = 0                          # [K#] in the kept knowledge sentences

    def to_dict(self) -> dict:
        return asdict(self)


class Explainer:
    def __init__(self, client, guard: Guard, system_prompt: str, timeout_s: float,
                 index: GroundingIndex, knowledge=None):
        """`knowledge`: a knowledge.KnowledgeBase when the prompt is v4 (explain.knowledge),
        else None."""
        self.client, self.guard, self.timeout_s, self.index = client, guard, timeout_s, index
        self.system_prompt, self.knowledge = system_prompt, knowledge
        self.prompt_sha256 = sha256(system_prompt)

    def explain(self, facts: dict, intents: list[Intent],
                knowledge: list[dict] | None = None) -> Explanation:
        """Facts with a "knowledge" list are prompt v4 (summary, images and the knowledge part);
        without it, prompt v3. `knowledge`: the items of those facts with their `cuis`."""
        started = time.perf_counter()
        with_knowledge = "knowledge" in facts
        schema = EXPLAIN_KNOWLEDGE_SCHEMA if with_knowledge else EXPLAIN_SCHEMA
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(facts, ensure_ascii=False,
                                                         separators=(",", ":"))}]
        meta: dict = {}
        try:
            content, _ = self.guard.call(
                lambda: self.client.chat(messages, schema, stream=True, meta=meta),
                self.timeout_s)
            answer = json.loads(content)
            summary, images = str(answer["summary"]), list(answer["images"])
            told = str(answer["knowledge"]) if with_knowledge else ""
        except (LLMError, LLMInvalidOutput, ValueError, KeyError, TypeError) as exc:
            return Explanation("failed", total_ms=_ms(started), error=str(exc)[:160],
                               first_token_ms=meta.get("first_token_ms"))
        cuis, phrases = self.index.allowed(facts, intents)
        numbers = facts_numbers(facts)
        dropped, kept = [], []
        for sentence in _SENTENCE.split(summary.strip()):
            why = self.index.check(sentence, cuis, phrases, numbers) if sentence else None
            (dropped.append({"text": sentence, "why": why}) if why else kept.append(sentence))
        ranks = {r["rank"] for r in facts["results"]}
        lines: dict[int, str] = {}
        for item in images:
            try:
                rank, line = int(item["rank"]), str(item["line"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            why = ("rank not in the results" if rank not in ranks
                   else self.index.check(line, cuis, phrases, numbers))
            (dropped.append({"text": line, "why": why, "rank": rank}) if why
             else lines.__setitem__(rank, line))
        known, citations = [], 0
        items = {k["id"]: {"cuis": [], **k} for k in knowledge or facts.get("knowledge", [])}
        for sentence in _SENTENCE.split(told.strip()):
            if not sentence:
                continue
            why, n = self.index.check_knowledge(sentence, items)
            if why:
                dropped.append({"text": sentence, "why": why, "part": "knowledge"})
            else:
                known.append(sentence)
                citations += n
        status = ("ungrounded" if not kept and not lines else "partial" if dropped else "ok")
        return Explanation(status, " ".join(kept), lines, dropped,
                           meta.get("first_token_ms"), _ms(started), knowledge=" ".join(known),
                           knowledge_items=(facts["knowledge"] if with_knowledge else None),
                           citations=citations)


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
