"""LLM adapters for the query parser (plan 6.4, 10.3; WORKLOG D20).

One chat call per query with a JSON schema whose concept fields are enums of the
14 labels, so the server constrains decoding and the model cannot name a concept
that does not exist. Pydantic validates anyway: an older server ignores the
schema, and the retry/fallback path has to work regardless. Transport failures
raise LLMError (the caller falls back at once); bad content raises
LLMInvalidOutput (the caller retries once with the error fed back).

Three adapters share that contract: `AnthropicClient` (the default, D61), `OllamaClient`
(local) and `OpenAICompatClient`, which speaks the OpenAI chat-completions API and so
also serves Gemini through its OpenAI-compatible endpoint. All receive the same prompt
and schema, so swapping the model measures the model and nothing else.
"""

from __future__ import annotations

import json
import re
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.common.config import default_config_dir
from src.data.labels import CHEXPERT_LABELS
from src.query.schema import Constraint, Language, Term, View, canonical_concept

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
    terms: list[Term] = Field(default_factory=list)
    language: Language

    @field_validator("exclude")
    @classmethod
    def _canonical_exclude(cls, value: list[str]) -> list[str]:
        return [canonical_concept(name) for name in value]


def output_schema(with_terms: bool = False) -> dict:
    """Hand-built (not model_json_schema) so there are no $refs for the grammar builder.
    `with_terms` adds the `terms` array of the KG parser prompt (query_parser_v2)."""
    concept = {"type": "string", "enum": list(CHEXPERT_LABELS)}
    schema = {
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
    if with_terms:
        schema["properties"]["terms"] = {"type": "array", "items": {
            "type": "object",
            "properties": {"text": {"type": "string"},
                           "polarity": {"type": "string", "enum": ["PRESENT", "ABSENT"]}},
            "required": ["text", "polarity"]}}
        schema["required"] = [*schema["required"], "terms"]
    return schema


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


def render_prompt(template: str, concepts_cfg: dict, drop_forms: tuple[str, ...] = ()) -> str:
    """The prompt with the concept list filled in; `drop_forms` are forms left out of the list
    (kg.yaml query.ambiguous_forms: a form that names two concepts)."""
    drop = {f.casefold() for f in drop_forms}

    def forms(names: list[str]) -> str:
        return ", ".join(n for n in names if n.casefold() not in drop)

    lines = [f"- {concept}: {forms(f['en'])} | {forms(f['vi'])}"
             for concept, f in concepts_cfg["concepts"].items()]
    return template.replace("{{CONCEPTS}}", "\n".join(lines))


def load_prompt(relative_path: str, concepts_cfg: dict, drop_forms: tuple[str, ...] = ()) -> str:
    """`relative_path` is relative to configs/ (models.yaml:llm.prompt_file)."""
    text = (default_config_dir() / relative_path).read_text(encoding="utf-8")
    return render_prompt(text, concepts_cfg, drop_forms)


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


class AnthropicClient:
    """Claude through the Anthropic SDK; the default since the switch away from OpenAI (D61).

    The schema is a strict tool the model must call, so the reply is the tool input. The
    system prompt carries a cache breakpoint: it is the same ~2.6k tokens for every query.
    Sampling parameters are not sent (current Claude models reject non-default values);
    `effort` and `tool_choice` come from models.yaml:llm.hosted_request.
    """

    TOOL = "submit_query"

    def __init__(self, model: str, timeout_s: float, *, api_key: str,
                 max_tokens: int = 2048, options: dict | None = None, client=None,
                 max_retries: int = 1, tool: tuple[str, str] | None = None,
                 connect_timeout_s: float | None = None):
        """`max_retries` is the SDK's own (0 on the KG query path, where a failure falls back at
        once, N3); `tool` is (name, description) of the schema tool when it is not the parser's."""
        import anthropic

        options = dict(options or {})
        self.tool_name, self.tool_description = tool or (
            self.TOOL, "Submit the parsed search query.")
        self.last_first_token_ms: float | None = None
        self.model, self.max_tokens = model, max_tokens
        self.effort = options.get("effort")
        self.tool_choice = options.get("tool_choice", "tool")
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        timeout = (timeout_s if connect_timeout_s is None
                   else anthropic.Timeout(timeout_s, connect=connect_timeout_s))
        self._client = client or anthropic.Anthropic(api_key=api_key, timeout=timeout,
                                                     max_retries=max_retries)

    def _stream(self, request: dict, started: float):
        first = None
        with self._client.messages.stream(**request) as events:
            for event in events:
                delta = getattr(event, "delta", None)
                if first is None and getattr(delta, "type", "") == "input_json_delta":
                    first = (time.perf_counter() - started) * 1000
            response = events.get_final_message()
        self.last_first_token_ms = first
        return response

    def probe(self) -> ProbeStatus:
        import anthropic

        try:
            self._client.models.retrieve(self.model)
        except anthropic.NotFoundError:
            return "missing_model"
        except Exception:  # noqa: BLE001 - a bad key or no network both mean "not usable"
            return "unreachable"
        return "ok"

    def version(self) -> str | None:
        return None   # a hosted API has no server version to record

    def chat(self, messages: list[dict], schema: dict, stream: bool = False
             ) -> tuple[str, float]:
        """One call; returns (the tool input as JSON, wall ms). With `stream` the reply is read
        as it arrives and `last_first_token_ms` holds the time of its first input delta."""
        import anthropic

        system = [{"type": "text", "text": messages[0]["content"],
                   "cache_control": {"type": "ephemeral"}}]
        tool = {"name": self.tool_name, "description": self.tool_description,
                "input_schema": _strict(schema), "strict": True}
        choice = ({"type": "tool", "name": self.tool_name} if self.tool_choice == "tool"
                  else {"type": "auto"})
        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        started = time.perf_counter()
        request = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                   "messages": messages[1:], "tools": [tool], "tool_choice": choice, **extra}
        try:
            if stream:
                response = self._stream(request, started)
            else:
                response = self._client.messages.create(**request)
        except anthropic.APIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        self.usage["calls"] += 1
        for name in list(self.usage)[1:]:
            self.usage[name] += getattr(response.usage, name, 0) or 0
        if response.stop_reason == "refusal":
            raise LLMError("the model declined (stop_reason refusal)")
        call = next((b for b in response.content if b.type == "tool_use"), None)
        if call is None:
            raise LLMInvalidOutput(
                f"no {self.tool_name} call (stop_reason {response.stop_reason})")
        return json.dumps(call.input), (time.perf_counter() - started) * 1000


def request_options(llm_cfg: dict, model: str) -> dict:
    """The hosted-request fields for `model` (models.yaml:llm.hosted_request), else default."""
    table = llm_cfg["hosted_request"]
    return dict(table.get(model, table["default"]))


def make_client(provider: str, base_url: str, model: str, timeout_s: float, *,
                api_key: str | None, num_predict: int, seed: int, keep_alive: str,
                options: dict | None = None):
    """The adapter `LLM_PROVIDER` names: `anthropic` (default), `ollama`, or `openai`, which
    covers any OpenAI-compatible API and stays behind that flag."""
    if provider == "anthropic":
        if not api_key:
            raise ValueError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY")
        return AnthropicClient(model, timeout_s, api_key=api_key, options=options)
    if provider == "ollama":
        return OllamaClient(base_url, model, timeout_s, num_predict=num_predict, seed=seed,
                            keep_alive=keep_alive)
    if provider == "openai":
        if not api_key:
            raise ValueError("LLM_PROVIDER=openai needs LLM_API_KEY")
        return OpenAICompatClient(base_url, model, timeout_s, api_key=api_key, options=options)
    raise ValueError(f"unknown LLM provider {provider!r}; expected anthropic, ollama or openai")
