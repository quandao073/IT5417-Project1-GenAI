"""Selection step of the V2-3 linker (D56): two Anthropic models, run independently with the
same prompt and input, each choose for a vocabulary string one CUI among its candidates, or skip,
out_of_scope, none. Offline and run once: every answer is cached in
data/kg/link/llm_cache_<name>.jsonl under the sha256 of model, prompt and input. Only the string,
its counts, the strings it is located_at with and the UMLS candidates go out; never a report
sentence (D45). A refusal is an answer: `none`, reason "refused".

The same client serves the silver-standard judge of the gate sample (`GRADE_SCHEMA`).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ValidationError

INPUTS_FILE = "llm_inputs.jsonl"
CONTEXT_SIZE = 5


class LLMSelectError(RuntimeError):
    """A call that returned no usable answer: HTTP error, truncation, a reply off the schema."""


class Refused(LLMSelectError):
    """The model declined (stop_reason refusal); the call's usage is still billed."""

    def __init__(self, usage: dict):
        super().__init__("refused")
        self.usage = usage


REFUSED_CHOICE = {"decision": "none", "cui": "", "reason": "refused"}
REFUSED_GRADE = {"grade": "unsure", "reason": "refused"}


class Choice(BaseModel):
    decision: Literal["cui", "skip", "out_of_scope", "none"]
    cui: str
    reason: str


class Grade(BaseModel):
    grade: Literal["correct", "incorrect", "unsure"]
    reason: str


def _schema(model: type[BaseModel]) -> dict:
    """The model's JSON schema in the strict form both APIs accept: every field required, no
    extra field, no titles."""
    properties = {name: {k: v for k, v in field.items() if k != "title"}
                  for name, field in model.model_json_schema()["properties"].items()}
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


CHOICE_SCHEMA, GRADE_SCHEMA = _schema(Choice), _schema(Grade)


def load_prompt(path: Path) -> tuple[str, str]:
    """(text, sha256) of a versioned prompt file under configs/prompts/ (D20)."""
    text = path.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def located_context(occurrences: pl.DataFrame) -> dict[tuple[str, str], list[list]]:
    """For an Observation string, the Anatomy strings it is most often located_at; for an Anatomy
    string, the Observation strings most often located at it. At most CONTEXT_SIZE, with counts."""
    counts = occurrences.group_by("obs", "anat").len().sort(["len", "obs", "anat"],
                                                           descending=[True, False, False])
    out: dict[tuple[str, str], list[list]] = {}
    for obs, anat, n in counts.iter_rows():
        for key, other in ((("Observation", obs), anat), (("Anatomy", anat), obs)):
            if len(out.setdefault(key, [])) < CONTEXT_SIZE:
                out[key].append([other, n])
    return out


def inputs(links: pl.DataFrame, cands: pl.DataFrame, info: pl.DataFrame,
           context: dict[tuple[str, str], list[list]]) -> list[dict]:
    """One item per string of the selection step: kind, string and the `input` sent as is."""
    meta = {c: (n, t, s) for c, n, t, s in
            info.select("cui", "name", "semantic_types", "sabs").iter_rows()}
    by_string: dict[tuple[str, str], list[dict]] = {}
    for kind, string, cui, cosine, exact in (cands.sort("rank").select(
            "kind", "string", "cui", "cosine", "exact").iter_rows()):
        name, types, sabs = meta[cui]
        by_string.setdefault((kind, string), []).append(
            {"cui": cui, "name": name, "semantic_types": types, "sources": sabs,
             "cosine": round(cosine, 3), "exact_name": exact})
    items = []
    for kind, string, mentions in (links.filter(pl.col("route").is_not_null())
                                   .select("kind", "string", "mentions").iter_rows()):
        located = "located_at" if kind == "Observation" else "observations_located_here"
        items.append({"kind": kind, "string": string, "input": {
            "string": string, "kind": kind, "mentions": mentions,
            located: context.get((kind, string), []),
            "candidates": by_string.get((kind, string), [])}})
    return items


def write_inputs(items: Sequence[dict], path: Path) -> None:
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
                    encoding="utf-8")


def read_inputs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cache_key(model: str, prompt_sha: str, payload: dict) -> str:
    blob = json.dumps({"model": model, "prompt_sha256": prompt_sha, "input": payload},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate(content: str, allowed: set[str]) -> dict:
    """{decision, cui, reason} from a reply. A CUI outside the candidates makes the decision
    `rejected` (D56); any other decision drops the cui."""
    try:
        choice = Choice.model_validate_json(content)
    except ValidationError as exc:
        raise LLMSelectError(f"reply does not fit the schema: {exc.errors()[:1]}") from exc
    if choice.decision != "cui":
        return {"decision": choice.decision, "cui": "", "reason": choice.reason}
    if choice.cui not in allowed:
        return {"decision": "rejected", "cui": choice.cui,
                "reason": f"not among the candidates; {choice.reason}"}
    return choice.model_dump()


class Cache:
    """Append-only JSON lines; a key's latest answered record wins, errors never do."""

    def __init__(self, path: Path):
        self.path, self._lock, self.answers = path, threading.Lock(), {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record.get("answer") is not None:
                    self.answers[record["key"]] = record

    def put(self, record: dict) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record.get("answer") is not None:
                self.answers[record["key"]] = record


def choices(items: Sequence[dict], cache: Cache, model: str,
            prompt_sha: str) -> dict[tuple[str, str], dict]:
    """The cached choice of every item this model answered under this prompt."""
    out = {}
    for item in items:
        record = cache.answers.get(cache_key(model, prompt_sha, item["input"]))
        if record is not None:
            out[(item["kind"], item["string"])] = record["answer"]
    return out


class AnthropicChooser:
    """One Anthropic model through the SDK's Messages API with JSON-schema output
    (`output_config.format`). No fallback model (D56): a refusal raises `Refused`."""

    def __init__(self, name: str, cfg: dict, api_key: str, client=None):
        import anthropic

        self.name, self.model = name, cfg["model"]
        self.effort, self.max_tokens = cfg["effort"], cfg["max_tokens"]
        self._client = client or anthropic.Anthropic(api_key=api_key, max_retries=3,
                                                     timeout=cfg["timeout_seconds"])

    def ask(self, system: str, user: str, schema: dict) -> tuple[str, dict, str]:
        """(content, usage, model that answered)."""
        import anthropic

        try:
            response = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": schema}})
        except anthropic.APIError as exc:
            raise LLMSelectError(f"{type(exc).__name__}: {exc}") from exc
        usage = {"input_tokens": response.usage.input_tokens,
                 "output_tokens": response.usage.output_tokens}
        if response.stop_reason == "refusal":
            raise Refused(usage)
        if response.stop_reason != "end_turn":
            raise LLMSelectError(f"stop_reason {response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return text, usage, response.model


def api_key(name: str, env_file: Path = Path(".env")) -> str | None:
    """A key from the environment, else from .env (never logged)."""
    if os.environ.get(name):
        return os.environ[name]
    from dotenv import dotenv_values

    return dotenv_values(env_file).get(name) if env_file.is_file() else None


def parse_choice(content: str, item: dict) -> dict:
    return validate(content, {c["cui"] for c in item["input"]["candidates"]})


def parse_grade(content: str, item: dict) -> dict:
    try:
        return Grade.model_validate_json(content).model_dump()
    except ValidationError as exc:
        raise LLMSelectError(f"reply does not fit the schema: {exc.errors()[:1]}") from exc


def run(items: Sequence[dict], chooser, system: str, prompt_sha: str, cache: Cache,
        limit: int | None, workers: int, schema: dict = CHOICE_SCHEMA,
        parse: Callable[[str, dict], dict] = parse_choice, refused: dict = REFUSED_CHOICE,
        clock: Callable[[], float] = time.perf_counter) -> dict:
    """Ask `chooser` for every item it has not answered under this prompt, at most `limit`;
    a refusal is cached as `refused`. Returns counts, the summed usage and the wall time."""
    todo = [i for i in items
            if cache_key(chooser.model, prompt_sha, i["input"]) not in cache.answers][:limit]

    def one(item: dict) -> dict:
        key = cache_key(chooser.model, prompt_sha, item["input"])
        record = {"key": key, "name": chooser.name, "model": chooser.model,
                  "prompt_sha256": prompt_sha, "kind": item["kind"], "string": item["string"],
                  "answer": None, "error": None, "usage": None, "served_model": None,
                  "at": datetime.now(UTC).isoformat(timespec="seconds")}
        try:
            content, record["usage"], record["served_model"] = chooser.ask(
                system, json.dumps(item["input"], ensure_ascii=False), schema)
            record["answer"] = parse(content, item)
        except Refused as exc:
            record["answer"], record["usage"] = dict(refused), exc.usage
        except LLMSelectError as exc:
            record["error"] = str(exc)
        cache.put(record)
        return record

    started = clock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = list(pool.map(one, todo))
    usage = [r["usage"] for r in done if r["usage"]]
    return {"called": len(done), "answered": sum(r["answer"] is not None for r in done),
            "rejected": sum((r["answer"] or {}).get("decision") == "rejected" for r in done),
            "refused": sum((r["answer"] or {}).get("reason") == "refused" for r in done),
            "errors": sorted({r["error"] for r in done if r["error"]})[:5],
            "input_tokens": sum(u["input_tokens"] or 0 for u in usage),
            "output_tokens": sum(u["output_tokens"] or 0 for u in usage),
            "seconds": round(clock() - started, 1),
            "remaining": len(items) - len(cache.answers.keys() & {
                cache_key(chooser.model, prompt_sha, i["input"]) for i in items})}
