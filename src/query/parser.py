"""One parse() for every mode: cache, then the LLM (one retry), then the rule parser.

Also the single rule that builds semantic_query for both parsers (spec 4.2), so
the vector leg sees the same text whichever parser answered; the ablation then
measures constraints, not phrasing. `auto` and `rule` never raise; `llm` (used
by the evaluation to measure the LLM alone) raises ParseError instead of
falling back.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
import time
import unicodedata
from collections import OrderedDict
from typing import Literal

from pydantic import ValidationError

from src.query import llm_parser as llm
from src.query.rule_parser import RuleParser
from src.query.schema import ParseError, ParseResult, SearchQuery

Mode = Literal["auto", "llm", "rule"]
SEMANTIC_MAX = 300
RETRY_MESSAGE = ("That reply was invalid: {error}. Reply again with one JSON object that "
                 "follows the rules and uses only the 14 concept names.")


def clean(text: str) -> str:
    """NFC and whitespace only; case is kept because the LLM reads it ('PA')."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def _error_text(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        return f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"[:200]
    return str(exc)[:200]


class QueryParser:
    def __init__(self, rule: RuleParser, *, template: str, llm_client=None,
                 system_prompt: str = "", max_retries: int = 1, cache_size: int = 1024,
                 startup_warning: str | None = None):
        self.rule = rule
        self.template = template
        self.llm = llm_client
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.startup_warning = startup_warning
        self.prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        self._cache: OrderedDict[str, ParseResult] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def parse(self, text: str, *, mode: Mode = "auto") -> ParseResult:
        text = clean(text)
        key = self._key(text, mode)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return dataclasses.replace(hit, cached=True, llm_ms=None)
        result = self._parse(text, mode)
        if result.fallback_used:
            return result         # a transient LLM failure must not stick to this text
        with self._lock:
            self._cache[key] = result
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    def _key(self, text: str, mode: str) -> str:
        model = getattr(self.llm, "model", "")
        raw = f"{mode}\n{model}\n{self.prompt_sha256}\n{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _parse(self, text: str, mode: Mode) -> ParseResult:
        warnings: list[str] = []
        if mode != "rule" and self.llm is not None:
            query, llm_ms, failure = self._ask_llm(text, warnings)
            if query is not None:
                query = self._finalize(query, "llm", text, warnings)
                return ParseResult(query, "llm", warnings=tuple(warnings), llm_ms=llm_ms)
            if mode == "llm":
                raise ParseError(failure)
            warnings.append(failure)
            query = self._finalize(self.rule.parse(text), "rule", text, warnings)
            return ParseResult(query, "rule", fallback_used=True, warnings=tuple(warnings),
                               llm_ms=llm_ms)
        if mode == "llm":
            raise ParseError("llm parser is not configured")
        if mode == "auto" and self.startup_warning:
            warnings.append(self.startup_warning)
        query = self._finalize(self.rule.parse(text), "rule", text, warnings)
        return ParseResult(query, "rule", warnings=tuple(warnings))

    def _ask_llm(self, text: str, warnings: list[str]) -> tuple[SearchQuery | None, float, str]:
        """(query, wall ms over every attempt, failure reason). Transport errors never
        retry; invalid output retries `max_retries` times with the error fed back."""
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}]
        schema = llm.output_schema()
        started = time.perf_counter()
        attempts = 0
        while True:
            try:
                content, _ = self.llm.chat(messages, schema)
            except llm.LLMError as exc:
                return None, _ms(started), f"llm unreachable: {exc}"
            except llm.LLMInvalidOutput as exc:
                attempts += 1
                error = _error_text(exc)
                if attempts > self.max_retries:
                    return None, _ms(started), (
                        f"llm output invalid after {attempts} attempts: {error}")
                continue          # retry the same messages; there is no reply to feed back
            try:
                out = llm.parse_output(content)
                query = SearchQuery(semantic_query=out.findings[:SEMANTIC_MAX],
                                    include=out.include, exclude=out.exclude,
                                    views=out.views, language=out.language)
            except (llm.LLMInvalidOutput, ValidationError) as exc:
                attempts += 1
                error = _error_text(exc)
                if attempts > self.max_retries:
                    return None, _ms(started), (
                        f"llm output invalid after {attempts} attempts: {error}")
                messages = [*messages, {"role": "assistant", "content": content},
                            {"role": "user", "content": RETRY_MESSAGE.format(error=error)}]
                continue
            if attempts:
                warnings.append(f"llm retried {attempts}x")
            return query, _ms(started), ""

    def _finalize(self, query: SearchQuery, parser: str, text: str,
                  warnings: list[str]) -> SearchQuery:
        """The semantic_query rule of spec 4.2, identical for both parsers."""
        if query.include:
            phrase = " and ".join(self.rule.en_first[c.concept] for c in query.include)
            semantic = self.template.format(phrase)
        elif parser == "llm":
            semantic = query.semantic_query          # the LLM's `findings` phrase
            if not semantic:
                warnings.append("no positive finding; vector leg skipped")
        elif query.language == "en":
            semantic = text[:SEMANTIC_MAX]
            what = "no positive finding" if query.exclude else "no known concept"
            warnings.append(f"{what}; vector leg encodes the raw text")
        else:
            semantic = ""
            warnings.append("no concept recognised in a Vietnamese query; vector leg skipped")
        return query.model_copy(update={"semantic_query": semantic})


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
