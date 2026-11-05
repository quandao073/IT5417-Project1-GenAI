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

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field

from src.kg.expand import ABSENT, NOISY_OR, WEIGHTED, Intent
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
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Choice:
    cui: str | None
    via: str                 # llm | single | deterministic | none
    reason: str = ""
    ms: float = 0.0
    error: str | None = None


class ConceptChooser:
    def __init__(self, client, guard: Guard, system_prompt: str, timeout_s: float):
        self.client, self.guard, self.timeout_s = client, guard, timeout_s
        self.system_prompt = system_prompt
        self.prompt_sha256 = sha256(system_prompt)

    def choose(self, question: str, term: str, polarity: str, lookup: Lookup) -> Choice:
        cands = lookup.candidates
        if not cands:
            return Choice(None, "none", "no candidate")
        if len(cands) == 1 and cands[0].match == "exact":
            return Choice(cands[0].cui, "single", "one exact candidate")
        started = time.perf_counter()
        payload = {"question": question, "term": term, "polarity": polarity, "candidates": [
            {"cui": c.cui, "name": c.name, "label": c.label, "parents": c.parents,
             "matched_alias": c.alias, "match": c.match, "scored_directly": c.scorable}
            for c in cands]}
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        try:
            content, _ = self.guard.call(lambda: self.client.chat(messages, CHOICE_SCHEMA),
                                         self.timeout_s)
            answer = json.loads(content)
            choice = str(answer["choice"]).strip()
        except (LLMError, LLMInvalidOutput, ValueError, KeyError) as exc:
            return Choice(cands[0].cui, "deterministic", "", _ms(started), str(exc)[:120])
        ms = _ms(started)
        if choice == "none":
            return Choice(None, "llm", str(answer.get("reason", "")), ms)
        if choice not in {c.cui for c in cands}:           # outside the list: never accepted
            return Choice(cands[0].cui, "deterministic", "", ms,
                          f"the model chose {choice!r}, not a candidate")
        return Choice(choice, "llm", str(answer.get("reason", "")), ms)


# ---- the explanation -------------------------------------------------------------------------

def _percent(p: float) -> int:
    return int(round(100 * p))


def build_facts(question: str, language: str, intents: list[Intent], rows: list[dict],
                probabilities: list[dict]) -> dict:
    """What the explanation may say. `rows` are the results (rank, view), `probabilities` the
    ScoreMatrix.probabilities of those rows. Numbers are whole percents and ranks only."""
    spec = []
    for i, it in enumerate(intents, start=1):
        if not it.columns:
            continue
        spec.append({"id": i, "asked": it.term, "concept": it.name, "kind": it.kind,
                     "combine": {NOISY_OR: "any_of", WEIGHTED: "weighted"}.get(it.combine,
                                                                               "single"),
                     "excluded": it.polarity == ABSENT,
                     "uses": [{"concept": c.name, "weight_percent": _percent(c.weight)}
                              for c in it.columns]})
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
    return {"question": question, "language": language, "intents": spec, "results": results}


def facts_numbers(facts: dict) -> set[str]:
    """Every whole number the facts contain, as text."""
    return set(re.findall(r"\d+", json.dumps(facts)))


class GroundingIndex:
    """Aliases of the Finding and Disease concepts of the graph, to spot a mention of a concept
    that is not in the facts. Built once at startup from aliases.parquet and nodes.parquet."""

    def __init__(self, aliases: dict[str, set[str]], by_cui: dict[str, set[str]], cfg: dict):
        self.aliases, self.by_cui, self.cfg = aliases, by_cui, cfg
        self.diagnostic = [d.casefold() for d in cfg["diagnostic"]]

    @classmethod
    def build(cls, nodes, aliases, cfg: dict) -> GroundingIndex:
        """`nodes`, `aliases`: the polars tables of data/kg/graph."""
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
                if (len(words) == 1 and (len(norm) < cfg["min_single_word_chars"]
                                         or norm in stop)) or not words:
                    continue
                index.setdefault(norm, set()).add(cui)
        return cls(index, by_cui, cfg)

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

    def check(self, sentence: str, cuis: set[str], phrases: set[str],
              numbers: set[str]) -> str | None:
        """None when the sentence is grounded, else why it is not."""
        low = sentence.casefold()
        for word in self.diagnostic:
            if word in low:
                return f"diagnostic wording: {word}"
        for number in _NUMBER.findall(sentence):
            if number.replace(",", ".").split(".")[0] not in numbers:
                return f"number not in the facts: {number}"
        words = normalize(sentence).split()
        for n in range(4, 0, -1):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                owners = self.aliases.get(gram)
                if owners and not owners & cuis and gram not in phrases:
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

    def to_dict(self) -> dict:
        return asdict(self)


class Explainer:
    def __init__(self, client, guard: Guard, system_prompt: str, timeout_s: float,
                 index: GroundingIndex):
        self.client, self.guard, self.timeout_s, self.index = client, guard, timeout_s, index
        self.system_prompt = system_prompt
        self.prompt_sha256 = sha256(system_prompt)

    def explain(self, facts: dict, intents: list[Intent]) -> Explanation:
        started = time.perf_counter()
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]
        try:
            content, _ = self.guard.call(
                lambda: self.client.chat(messages, EXPLAIN_SCHEMA, stream=True), self.timeout_s)
            answer = json.loads(content)
            summary, images = str(answer["summary"]), list(answer["images"])
        except (LLMError, LLMInvalidOutput, ValueError, KeyError, TypeError) as exc:
            return Explanation("failed", total_ms=_ms(started), error=str(exc)[:160],
                               first_token_ms=getattr(self.client, "last_first_token_ms", None))
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
        status = ("ungrounded" if not kept and not lines else "partial" if dropped else "ok")
        return Explanation(status, " ".join(kept), lines, dropped,
                           getattr(self.client, "last_first_token_ms", None), _ms(started))


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
