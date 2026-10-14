"""Ollama adapter for the query parser (plan 6.4, 10.3; WORKLOG D20).

One chat call per query with `format` set to a JSON schema whose concept fields
are enums of the 14 labels, so the server constrains decoding and the model
cannot name a concept that does not exist. Pydantic validates anyway: an older
server ignores `format`, and the retry/fallback path has to work regardless.
Transport failures raise LLMError (the caller falls back at once); bad content
raises LLMInvalidOutput (the caller retries once with the error fed back).
"""

from __future__ import annotations

import json
import re
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
