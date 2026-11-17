"""The English of a question, for the text route and score fusion of the KG search (WORKLOG D74,
T44): what B0 encodes. A Vietnamese question goes to the model (prompt `kg_b0_translate_v1`); an
English one is its own English. `ask(question) -> str` makes the call; the search runs it beside
the parser. A failed call gives None, and the search then uses the question as written.

Vietnamese typed without tone marks ("suy tim", "tran dich mang phoi") is told by the KG's own
Vietnamese aliases (`ascii_phrases`: unmarked form -> "vi form = English name") and goes to
`ask_unaccented` (prompt `kg_translate_unaccented_v1`) with the meanings of the phrases it holds:
the plain prompt read "suy tim" as "find", and an unmarked prompt alone still read "tran dich mang
phoi" as "lung infiltrate" (demo review 2026-11-17). Every benchmark question has tone marks, so
its translation is the evaluation's.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass

from src.query.rule_parser import _VI_CHARS, normalize

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["text"],
          "properties": {"text": {"type": "string"}}}


def _words(question: str) -> str:
    return f" {' '.join(re.findall(r'[a-z0-9]+', normalize(question)))} "


def is_vietnamese(question: str, ascii_phrases: Collection[str] = ()) -> bool:
    """A letter only Vietnamese uses (the rule parser's test; it matches the language of all 172
    benchmark questions), or one of `ascii_phrases`: Vietnamese forms written without such a
    letter ("suy tim"), as whole words of the normalised question."""
    if _VI_CHARS & set(normalize(question)):
        return True
    padded = _words(question)
    return any(f" {p} " in padded for p in ascii_phrases)


def strip_marks(text: str) -> str:
    """`text` without Vietnamese tone marks and with đ as d: "tràn dịch" -> "tran dich"."""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def usable(text: str | None) -> str:
    """`text` without blanks and wrapping quotes; "" when nothing is left (T44: `""`)."""
    return (text or "").strip().strip("\"'“”").strip()


@dataclass
class Translation:
    text: str | None            # None: the call failed
    called: bool
    ms: float = 0.0
    error: str | None = None


class Translator:
    def __init__(self, ask: Callable[[str], str], ascii_phrases: Mapping[str, str] | None = None,
                 ask_unaccented: Callable[[str], str] | None = None):
        """`ascii_phrases`: unmarked Vietnamese form -> its meaning (runtime.vi_ascii_phrases).
        Without `ask_unaccented`, an unmarked question goes to `ask` as written."""
        self.ask, self.ascii_phrases = ask, dict(ascii_phrases or {})
        self.ask_unaccented = ask_unaccented

    def is_vietnamese(self, question: str) -> bool:
        return is_vietnamese(question, self.ascii_phrases)

    def _ask(self, question: str) -> str:
        if self.ask_unaccented is None or _VI_CHARS & set(normalize(question)):
            return self.ask(question)
        return self.ask_unaccented(_unaccented_message(question, self.ascii_phrases))

    def english(self, question: str) -> Translation:
        if not self.is_vietnamese(question):
            return Translation(question, False)
        started = time.perf_counter()
        try:
            text = usable(self._ask(question)) or None
            error = None if text else "empty translation"
        except Exception as exc:  # noqa: BLE001 - an outage or a bad reply: the question as written
            text, error = None, f"{type(exc).__name__}: {exc}"[:200]
        return Translation(text, True, (time.perf_counter() - started) * 1000, error)


def _unaccented_message(question: str, meanings: Mapping[str, str]) -> str:
    """The question, then one "Term: vi = en" line per KG phrase it holds."""
    padded = _words(question)
    known = [m for p, m in sorted(meanings.items()) if f" {p} " in padded]
    return question + "".join(f"\nTerm: {m}" for m in known)


def model_ask(client, guard, prompt: str, timeout_s: float) -> Callable[[str], str]:
    """`ask` over a parser-contract client under the search's guard."""
    def ask(question: str) -> str:
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
        content, _ = guard.call(lambda: client.chat(messages, SCHEMA), timeout_s)
        return json.loads(content)["text"]
    return ask
