"""The English of a question, for the text route and score fusion of the KG search (WORKLOG D74,
T44): what B0 encodes. A Vietnamese question goes to the model (prompt `kg_b0_translate_v1`); an
English one is its own English. `ask(question) -> str` makes the call; the search runs it beside
the parser. A failed call gives None, and the search then uses the question as written.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from src.query.rule_parser import _VI_CHARS, normalize

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["text"],
          "properties": {"text": {"type": "string"}}}


def is_vietnamese(question: str, ascii_phrases: frozenset[str] = frozenset()) -> bool:
    """A letter only Vietnamese uses (the rule parser's test; it matches the language of all 172
    benchmark questions), or one of `ascii_phrases`: Vietnamese forms written without such a
    letter ("suy tim"), as whole words of the normalised question."""
    text = normalize(question)
    if _VI_CHARS & set(text):
        return True
    padded = f" {' '.join(re.findall(r'[a-z0-9]+', text))} "
    return any(f" {p} " in padded for p in ascii_phrases)


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
    def __init__(self, ask: Callable[[str], str], ascii_phrases: frozenset[str] = frozenset()):
        self.ask, self.ascii_phrases = ask, ascii_phrases

    def is_vietnamese(self, question: str) -> bool:
        return is_vietnamese(question, self.ascii_phrases)

    def english(self, question: str) -> Translation:
        if not self.is_vietnamese(question):
            return Translation(question, False)
        started = time.perf_counter()
        try:
            text = usable(self.ask(question)) or None
            error = None if text else "empty translation"
        except Exception as exc:  # noqa: BLE001 - an outage or a bad reply: the question as written
            text, error = None, f"{type(exc).__name__}: {exc}"[:200]
        return Translation(text, True, (time.perf_counter() - started) * 1000, error)


def model_ask(client, guard, prompt: str, timeout_s: float) -> Callable[[str], str]:
    """`ask` over a parser-contract client under the search's guard."""
    def ask(question: str) -> str:
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
        content, _ = guard.call(lambda: client.chat(messages, SCHEMA), timeout_s)
        return json.loads(content)["text"]
    return ask
