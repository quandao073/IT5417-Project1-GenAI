"""The language-model calls of the KG evaluation (V2-7): one ledger that adds up what every call
really cost (from its `usage`) and stops the run cleanly at the budget, and a cached JSON call on
top of `llm_select.AnthropicChooser` (structured output, refusal as an error).

Prices are per million tokens in configs/eval_kg.yaml (`pricing`). The ledger is append-only
JSON lines under the evaluation work dir; a rerun reads it back, so the budget covers the whole
session, not one process.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from src.kg import llm_select


class BudgetExceeded(RuntimeError):
    """The ledger reached the budget: nothing more is sent."""


def usd(model: str, usage: dict, pricing: dict) -> float:
    price = pricing[model]
    return (usage.get("input_tokens", 0) * price["input"]
            + usage.get("output_tokens", 0) * price["output"]
            + usage.get("cache_read_input_tokens", 0) * price["cache_read"]
            + usage.get("cache_creation_input_tokens", 0) * price["cache_write"]) / 1e6


class Ledger:
    def __init__(self, path: Path, pricing: dict, budget: float):
        self.path, self.pricing, self.budget = path, pricing, budget
        self._lock = threading.Lock()
        self.records: list[dict] = []
        if path.is_file():
            self.records = [json.loads(line) for line in
                            path.read_text(encoding="utf-8").splitlines() if line]

    @property
    def total(self) -> float:
        return sum(r["usd"] for r in self.records)

    def by(self, field: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.records:
            out[r[field]] = out.get(r[field], 0.0) + r["usd"]
        return {k: round(v, 4) for k, v in sorted(out.items())}

    def check(self) -> None:
        if self.total >= self.budget:
            raise BudgetExceeded(f"spent {self.total:.2f} of {self.budget:.2f} USD")

    def add(self, step: str, model: str, usage: dict, calls: int = 1) -> float:
        cost = usd(model, usage, self.pricing)
        record = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "step": step,
                  "model": model, "calls": calls, "usd": round(cost, 6),
                  **{k: int(usage.get(k, 0) or 0) for k in (
                      "input_tokens", "output_tokens", "cache_read_input_tokens",
                      "cache_creation_input_tokens")}}
        with self._lock:
            self.records.append(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        return cost

    def add_client_delta(self, step: str, client, before: dict) -> dict:
        """Charge what a pipeline client (`AnthropicClient.usage`) spent since `before`."""
        now = dict(client.usage)
        delta = {k: now[k] - before.get(k, 0) for k in now}
        if delta.get("calls"):
            self.add(step, client.model, delta, calls=delta["calls"])
        return now


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedCaller:
    """A JSON-schema call whose answers are cached by (model, prompt sha256, key payload)."""

    def __init__(self, name: str, cfg: dict, api_key: str, cache_path: Path, ledger: Ledger,
                 chooser=None):
        self.name, self.model = name, cfg["model"]
        self.chooser = chooser or llm_select.AnthropicChooser(name, cfg, api_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = llm_select.Cache(cache_path)
        self.ledger = ledger

    def key(self, prompt_sha: str, payload: dict) -> str:
        return llm_select.cache_key(self.model, prompt_sha, payload)

    def cached(self, prompt_sha: str, payload: dict) -> dict | None:
        record = self.cache.answers.get(self.key(prompt_sha, payload))
        return None if record is None else record["answer"]

    def call(self, step: str, system: str, prompt_sha: str, payload: dict, user: str,
             schema: dict, parse=json.loads) -> dict:
        """The cached answer, else one call: its usage goes to the ledger before the reply is
        parsed, so a reply that fails to parse is still paid for."""
        hit = self.cached(prompt_sha, payload)
        if hit is not None:
            return hit
        self.ledger.check()
        record = {"key": self.key(prompt_sha, payload), "name": self.name, "model": self.model,
                  "prompt_sha256": prompt_sha, "kind": step, "string": "", "answer": None,
                  "error": None, "usage": None,
                  "at": datetime.now(UTC).isoformat(timespec="seconds")}
        try:
            content, usage, record["served_model"] = self.chooser.ask(system, user, schema)
            record["usage"] = usage
            self.ledger.add(step, self.model, usage)
            record["answer"] = parse(content)
        except llm_select.Refused as exc:
            record["usage"], record["error"] = exc.usage, "refused"
            self.ledger.add(step, self.model, exc.usage)
        except (llm_select.LLMSelectError, ValueError) as exc:
            record["error"] = str(exc)[:300]
        self.cache.put(record)
        if record["answer"] is None:
            raise llm_select.LLMSelectError(f"{self.name}: {record['error']}")
        return record["answer"]
