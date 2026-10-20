"""LLM adapters for the query parser (plan 6.4, 10.3; WORKLOG D20).

One chat call per query with a JSON schema whose concept fields are enums of the
14 labels, so the server constrains decoding and the model cannot name a concept
that does not exist. Pydantic validates anyway: an older server ignores the
schema, and the retry/fallback path has to work regardless. Transport failures
raise LLMError (the caller falls back at once); bad content raises
LLMInvalidOutput (the caller retries once with the error fed back).

Two adapters share that contract: `OllamaClient` (the local default, D5) and
`OpenAICompatClient`, which speaks the OpenAI chat-completions API and so also
serves Gemini through its OpenAI-compatible endpoint. Both receive the same
prompt and schema, so swapping the model measures the model and nothing else.
"""

from __future__ import annotations

import json
import re
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.common.config import default_config_dir
from src.data.labels import CHEXPERT_LABELS
from src.query.schema import Constraint, Language, View, canonical_concept

ProbeStatus = Literal["ok", "missing_model", "unreachable"]
_EXAMPLE = re.compile(r"^Request: (.+)$", re.MULTILINE)


class LLMError(RuntimeError):
    """Timeout, connection or HTTP failure. Not retried."""


class LLMInvalidOutput(ValueError):
    """The reply was not the JSON we asked for. Retried once."""


class LLMOutput(BaseModel):
    findings: str = ""
    include: list[Constraint] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    views: list[View] = Field(default_factory=list)
    language: Language

    @field_validator("exclude")
    @classmethod
    def _canonical_exclude(cls, value: list[str]) -> list[str]:
        return [canonical_concept(name) for name in value]


def output_schema() -> dict:
    """Hand-built (not model_json_schema) so there are no $refs for the grammar builder."""
    concept = {"type": "string", "enum": list(CHEXPERT_LABELS)}
    return {
        "type": "object",
        "properties": {
            "findings": {"type": "string"},
            "include": {"type": "array", "items": {
                "type": "object",
                "properties": {"concept": concept,
                               "status": {"type": "string", "enum": ["PRESENT", "UNCERTAIN"]}},
                "required": ["concept", "status"]}},
            "exclude": {"type": "array", "items": concept},
            "views": {"type": "array", "items": {"type": "string", "enum": ["AP", "PA"]}},
            "language": {"type": "string", "enum": ["en", "vi"]},
        },
        "required": ["findings", "include", "exclude", "views", "language"],
    }


def parse_output(content: str) -> LLMOutput:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMInvalidOutput(f"not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMInvalidOutput(f"expected a JSON object, got {type(data).__name__}")
    try:
        return LLMOutput.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise LLMInvalidOutput(f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}") from exc


def render_prompt(template: str, concepts_cfg: dict) -> str:
    lines = [f"- {concept}: {', '.join(forms['en'])} | {', '.join(forms['vi'])}"
             for concept, forms in concepts_cfg["concepts"].items()]
    return template.replace("{{CONCEPTS}}", "\n".join(lines))


def load_prompt(relative_path: str, concepts_cfg: dict) -> str:
    """`relative_path` is relative to configs/ (models.yaml:llm.prompt_file)."""
    text = (default_config_dir() / relative_path).read_text(encoding="utf-8")
    return render_prompt(text, concepts_cfg)


def prompt_examples(prompt: str) -> list[str]:
    """The few-shot requests, so a test can keep them out of the held-out benchmark."""
    return _EXAMPLE.findall(prompt)


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: float, *, num_predict: int = 300,
                 seed: int = 20260101, keep_alive: str = "30m", transport=None):
        import httpx

        self.model = model
        self.options = {"temperature": 0, "seed": seed, "num_predict": num_predict}
        self.keep_alive = keep_alive
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def probe(self) -> ProbeStatus:
        try:
            response = self._http.get("/api/tags", timeout=3.0)
            response.raise_for_status()
            names = {m.get("name") for m in response.json().get("models", [])}
        except Exception:  # noqa: BLE001 - any failure means "not usable", which is the answer
            return "unreachable"
        return "ok" if self.model in names else "missing_model"

    def version(self) -> str | None:
        try:
            response = self._http.get("/api/version", timeout=3.0)
            response.raise_for_status()
            return response.json().get("version")
        except Exception:  # noqa: BLE001
            return None

    def chat(self, messages: list[dict], schema: dict) -> tuple[str, float]:
        """One non-streaming chat completion. Returns (content, server-side ms)."""
        import httpx

        payload = {"model": self.model, "messages": messages, "stream": False,
                   "format": schema, "options": self.options, "keep_alive": self.keep_alive}
        try:
            response = self._http.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        try:
            body = response.json()
            content = body["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMInvalidOutput(f"malformed chat response: {exc}") from exc
        return content, body.get("total_duration", 0) / 1e6


def _strict(schema):
    """A copy of `schema` with `additionalProperties: false` on every object.

    OpenAI's strict structured output refuses a schema without it. Nothing else
    changes, so the 14-concept enums constrain decoding exactly as they do for
    Ollama, and the shared `output_schema()` the Ollama path was measured with
    stays untouched.
    """
    if isinstance(schema, dict):
        out = {key: _strict(value) for key, value in schema.items()}
        if out.get("type") == "object":
            out["additionalProperties"] = False
        return out
    if isinstance(schema, list):
        return [_strict(value) for value in schema]
    return schema


class OpenAICompatClient:
    """OpenAI chat completions with strict JSON-schema output.

    Also serves Gemini: its OpenAI-compatible endpoint takes the same request at
    base URL https://generativelanguage.googleapis.com/v1beta/openai.
    """

    def __init__(self, base_url: str, model: str, timeout_s: float, *, api_key: str,
                 max_tokens: int = 2048, options: dict | None = None, transport=None):
        import httpx

        self.model = model
        # Model-specific request fields (models.yaml:llm.hosted_request). Families differ:
        # gpt-5-nano rejects temperature 0 and needs reasoning_effort=minimal to stop
        # spending 1,536 reasoning tokens on one parse; gpt-5.4 takes temperature 0 and
        # reasoning_effort=none. Temperature 0 is the default for everything else.
        self.options = dict(options) if options is not None else {"temperature": 0}
        # Not Ollama's num_predict (300): a thinking model counts its reasoning against
        # this cap. gemini-2.5-flash spent 212-221 thinking tokens on one parse and was
        # cut off (finish_reason "length", JSON missing its closing brace) 2/6 times at 300.
        self.max_tokens = max_tokens
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport,
                                  headers={"Authorization": f"Bearer {api_key}"})

    def probe(self) -> ProbeStatus:
        try:
            response = self._http.get("models", timeout=10.0)
            response.raise_for_status()
            # Gemini lists ids as "models/<name>"; OpenAI as "<name>".
            ids = {str(m.get("id", "")).removeprefix("models/")
                   for m in response.json().get("data", [])}
        except Exception:  # noqa: BLE001 - a bad key or a dead endpoint both mean "not usable"
            return "unreachable"
        return "ok" if self.model in ids else "missing_model"

    def version(self) -> str | None:
        return None   # a hosted API has no server version to record

    def chat(self, messages: list[dict], schema: dict) -> tuple[str, float]:
        """One non-streaming completion. Returns (content, wall ms)."""
        import httpx

        # No `seed`: Gemini's OpenAI endpoint rejects it with a 400 (2026-10-20), and on a
        # hosted API it is best-effort anyway. `max_completion_tokens`, not `max_tokens`:
        # the GPT-5 family rejects the old name.
        payload = {
            "model": self.model, "messages": messages,
            "max_completion_tokens": self.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "search_query", "strict": True, "schema": _strict(schema)}},
            **self.options,
        }
        started = time.perf_counter()
        try:
            response = self._http.post("chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # The body says why (bad schema, quota, unknown model); the key is never in it.
            raise LLMError(f"{exc} {exc.response.text[:300]}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMInvalidOutput(f"malformed chat response: {exc}") from exc
        if not content:
            raise LLMInvalidOutput("reply has no content (refusal or empty)")
        return content, (time.perf_counter() - started) * 1000


def request_options(llm_cfg: dict, model: str) -> dict:
    """The hosted-request fields for `model` (models.yaml:llm.hosted_request), else default."""
    table = llm_cfg["hosted_request"]
    return dict(table.get(model, table["default"]))


def make_client(provider: str, base_url: str, model: str, timeout_s: float, *,
                api_key: str | None, num_predict: int, seed: int, keep_alive: str,
                options: dict | None = None):
    """The adapter `LLM_PROVIDER` names. `openai` covers any OpenAI-compatible API."""
    if provider == "ollama":
        return OllamaClient(base_url, model, timeout_s, num_predict=num_predict, seed=seed,
                            keep_alive=keep_alive)
    if provider == "openai":
        if not api_key:
            raise ValueError("LLM_PROVIDER=openai needs LLM_API_KEY")
        return OpenAICompatClient(base_url, model, timeout_s, api_key=api_key, options=options)
    raise ValueError(f"unknown LLM provider {provider!r}; expected ollama or openai")
