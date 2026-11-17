"""Scope check of the KG parser (WORKLOG D79): how often it calls a request out of scope.

Only the parser runs (prompt `models.yaml: llm.kg_prompt_file`, the KG search's client and guard),
on three sets: the out-of-scope requests of benchmarks/out_of_scope_v1.txt (refusal rate), the dev
questions of benchmarks/kg_queries_v1.jsonl and docs/v2-6-smoke.txt (false refusals; the goal is
0). Heldout is never read. A request is refused by the search when the parser says out_of_scope and
no phrase names a concept of the KG; here "refused for sure" counts out_of_scope with nothing to
look up (no include, exclude or term), "out_of_scope" counts every out_of_scope. A parse that fell
back to the rule parser is counted apart: the rule parser never refuses.

    uv run python scripts/kg_scope_check.py --runs 3 --out data/eval/scope_check_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.common.config import Settings, load_config
from src.kg.guard import Guard, GuardedClient
from src.query import llm_parser as llm
from src.query.parser import QueryParser
from src.query.rule_parser import RuleParser


def kg_parser(timeout_s: float) -> QueryParser:
    """The KG search's parser as `runtime.build_kg_search` builds it, with `timeout_s`."""
    settings = Settings()
    if settings.anthropic_api_key is None:
        raise SystemExit("STOP: ANTHROPIC_API_KEY is not set")
    cfg, models, concepts = load_config("kg"), load_config("models"), load_config("concepts")
    kl = models["kg_llm"]
    client = llm.AnthropicClient(kl["parser"]["model"], timeout_s,
                                 api_key=settings.anthropic_api_key.get_secret_value(),
                                 max_tokens=2048, max_retries=0,
                                 connect_timeout_s=kl["connect_timeout_seconds"],
                                 options=llm.request_options(models["llm"], kl["parser"]["model"]))
    system = llm.load_prompt(models["llm"]["kg_prompt_file"], concepts,
                             tuple(cfg["query"]["ambiguous_forms"]))
    template = models["image_encoder"]["text_prompt_template"]
    return QueryParser(RuleParser(concepts), template=template,
                       llm_client=GuardedClient(client, Guard(10, 1.0), timeout_s),
                       system_prompt=system, max_retries=0, cache_size=0, with_terms=True)


def sets() -> dict[str, list[str]]:
    oos = [line.split("\t")[0] for line in Path("benchmarks/out_of_scope_v1.txt")
           .read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    dev = [json.loads(line)["question"] for line in Path("benchmarks/kg_queries_v1.jsonl")
           .read_text(encoding="utf-8").splitlines() if json.loads(line)["split"] == "dev"]
    smoke = [line.strip() for line in Path("docs/v2-6-smoke.txt").read_text(encoding="utf-8")
             .splitlines() if line.strip() and not line.startswith("#")]
    return {"out_of_scope": oos, "dev": dev, "smoke": smoke}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("data/eval/scope_check_v1.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    parser, questions = kg_parser(args.timeout), sets()
    runs = []
    for _ in range(args.runs):
        run: dict = {}
        for name, qs in questions.items():
            rows = []
            for q in qs:
                r = parser.parse(q, mode="auto")
                query = r.query
                nothing = not (query.include or query.exclude or query.terms)
                rows.append({"q": q, "parser": r.parser, "scope": query.scope,
                             "reason": query.scope_reason, "nothing_to_look_up": nothing})
            run[name] = rows
        runs.append(run)
    summary = []
    for i, run in enumerate(runs, start=1):
        line = {"run": i}
        for name, rows in run.items():
            llm_rows = [r for r in rows if r["parser"] == "llm"]
            line[name] = {"n": len(rows), "rule_fallback": len(rows) - len(llm_rows),
                          "out_of_scope": sum(r["scope"] == "out_of_scope" for r in llm_rows),
                          "refused_for_sure": sum(r["scope"] == "out_of_scope"
                                                  and r["nothing_to_look_up"] for r in llm_rows),
                          "flagged": [r["q"] for r in llm_rows if r["scope"] == "out_of_scope"]
                          if name != "out_of_scope" else
                          [r["q"] for r in llm_rows if r["scope"] != "out_of_scope"]}
        summary.append(line)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"prompt": load_config("models")["llm"]["kg_prompt_file"],
                                    "prompt_sha256": parser.prompt_sha256, "summary": summary,
                                    "runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"prompt_sha256": parser.prompt_sha256, "summary": summary},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
