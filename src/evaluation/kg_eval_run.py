"""Orchestration of `cxr eval-kg` (docs/kg-evaluation.md): run the systems on one split, pool and
judge, score. Three phases, each resumable from data/eval/:

    run     every system and the search ablations, in one process (the B2 ablations that search
            again share the parser's in-process cache, so they see the same parse); the runs,
            B2's intents, warnings, explanations and timings go to runs_<split>.json
    judge   the pool (union of every run's top 10) graded by the judge, a seeded share graded
            again by the cross judge, the human sample written
    score   layers 2-4, cost and latency, the ablation tables, the error analysis

Nothing here reads the gold to rank: REF reads labels and says so; the gold is only the answer
key of layer 2 and the view rule of layer 3.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from src.evaluation import kg_eval as ke
from src.kg.expand import PRESENT, Column, Intent
from src.kg.messages import Warn

TRANSLATE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["text"],
                    "properties": {"text": {"type": "string"}}}
SEARCH_ABLATIONS = ("lookup_deterministic", "sapbert_off")


def intent_from_dict(d: dict) -> Intent:
    return Intent(d["term"], d["cui"], d["name"], d["polarity"], d["kind"], d["combine"],
                  [Column(**c) for c in d["columns"]], [Warn(w) for w in d["warnings"]],
                  list(d.get("path", [])))


def warning_codes(result) -> set[str]:
    return {w.code for w in result.warnings} | {w.code for i in result.intents
                                                for w in i.warnings}


class Runner:
    def __init__(self, *, rows, split, ecfg, work: Path, ledger, corpus, encoder, b1_parser,
                 search, translator, t_text, t_sha, label_matrix, label_of_cui, labels_cfg,
                 concepts_cfg, zero_shot_matrix=None):
        self.rows, self.split, self.ecfg, self.work, self.ledger = rows, split, ecfg, work, ledger
        self.corpus, self.encoder, self.b1_parser, self.search = corpus, encoder, b1_parser, search
        self.translator, self.t_text, self.t_sha = translator, t_text, t_sha
        self.label_matrix, self.label_of_cui = label_matrix, label_of_cui
        self.labels_cfg, self.concepts_cfg = labels_cfg, concepts_cfg
        self.zero_shot_matrix = zero_shot_matrix       # probe_off (D72): the caption columns
        self.k = ecfg["eval"]["top_k"]
        self._vecs: dict[str, np.ndarray] = {}
        self.path = work / f"runs_{split}.json"
        self.runs: dict = json.loads(self.path.read_text(encoding="utf-8")) if (
            self.path.is_file()) else {}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.runs, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    # -- the systems -------------------------------------------------------------------------

    def _encode(self, text: str) -> tuple[np.ndarray, float]:
        t = time.perf_counter()
        vec = self.encoder.encode_texts([text])[0]
        return np.asarray(vec, dtype=np.float32), (time.perf_counter() - t) * 1000

    def _studies(self, idx: list[int]) -> list[str]:
        return [self.corpus.study_ids[i] for i in idx]

    def run_b0(self, row: dict) -> dict:
        t = time.perf_counter()
        text = row["question"]
        if row["lang"] != "en":
            answer = self.translator.call("b0_translate", self.t_text, self.t_sha,
                                          {"question": text}, text, TRANSLATE_SCHEMA)
            text = answer["text"]
        ms_translate = (time.perf_counter() - t) * 1000
        vec, ms_encode = self._encode(text)
        t = time.perf_counter()
        idx = self.corpus.top(vec, self.k)
        return {"studies": self._studies(idx), "text": text,
                "ms": {"translate_ms": ms_translate, "encode_ms": ms_encode,
                       "search_ms": (time.perf_counter() - t) * 1000}}

    def run_b1(self, row: dict) -> dict:
        from src.retrieval.pipeline import query_text

        before = dict(self.b1_parser.llm.usage)
        t = time.perf_counter()
        parsed = self.b1_parser.parse(row["question"])
        ms_parse = (time.perf_counter() - t) * 1000
        self.ledger.add_client_delta("b1_parse", self.b1_parser.llm, before)
        warnings: list[str] = []
        text = query_text(parsed.query, self.concepts_cfg, warnings)
        out = {"studies": [], "text": text, "parser": parsed.parser,
               "fallback": parsed.fallback_used, "views": list(parsed.query.views),
               "include": [c.model_dump() for c in parsed.query.include],
               "exclude": list(parsed.query.exclude), "warnings": warnings,
               "ms": {"parse_ms": ms_parse}}
        if text is not None:
            vec, out["ms"]["encode_ms"] = self._encode(text)
            t = time.perf_counter()
            out["studies"] = self._studies(self.corpus.top(vec, self.k, parsed.query.views))
            out["ms"]["search_ms"] = (time.perf_counter() - t) * 1000
        return out

    def run_ref(self, row: dict) -> dict:
        """Reads labels (a ceiling only): the gold's CheXpert-label constraints filter the
        studies by resolved label, the gold English rendering ranks them."""
        from src.query.schema import Constraint
        from src.retrieval import label_filter as lf

        include = [self.label_of_cui[c["cui"]] for c in row["expected_scorable"]
                   if c["cui"] in self.label_of_cui]
        exclude = [self.label_of_cui[c] for c in row["exclude"] if c in self.label_of_cui]
        allowed = None
        if include or exclude:
            mask = lf.filter_studies_by_labels(
                self.label_matrix, [Constraint(concept=c, status="PRESENT")
                                    for c in dict.fromkeys(include)],
                list(dict.fromkeys(exclude)),
                absence_policy=self.labels_cfg["absence_policy_default"],
                uncertainty_policy=self.labels_cfg["uncertainty_policy_default"])
            by_study = dict(zip(self.label_matrix.study_ids, mask, strict=True))
            allowed = np.array([bool(by_study.get(s, False)) for s in self.corpus.study_ids])
        vec, _ = self._encode(row["query_en"])
        idx = self.corpus.top(vec, self.k, row["views"], allowed)
        return {"studies": self._studies(idx), "filter_labels": {"include": include,
                                                                  "exclude": exclude}}

    def _clients(self) -> dict:
        s = self.search
        out = {}
        if s.parser.llm is not None:
            out["b2_parse"] = getattr(s.parser.llm, "inner", s.parser.llm)
        if s.chooser is not None:
            out["b2_choose"] = s.chooser.client
        if s.explainer is not None:
            out["b2_explain"] = s.explainer.client
        return out

    def run_b2(self, row: dict, explain: bool, tag: str = "b2") -> dict:
        clients = self._clients()
        before = {k: dict(c.usage) for k, c in clients.items()}
        result = self.search.search(row["question"], top_k=self.k)
        out = {"studies": [r["study_id"] for r in result.rows], "ms": dict(result.timings),
               "intents": [i.to_dict() for i in result.intents],
               "warnings": sorted(warning_codes(result)), "degraded": result.degraded,
               "views": list(result.query.views),
               "terms": [{"term": t["term"], "choice": (t.get("choice") or {}).get("cui"),
                          "via": (t.get("choice") or {}).get("via"),
                          "candidates": [c["cui"] for c in t.get("candidates", [])]}
                         for t in result.trace["terms"]],
               "labels": [c.concept for c in result.query.include] + list(result.query.exclude),
               "fallback": result.trace.get("fallback")}
        if self.search.text_fallback is not None:     # what the text route would encode (D73)
            out["fallback_text"] = self.search.text_fallback.text(
                result.query, result.trace.get("english"), row["question"])
        if explain:
            from src.api.routes.kg import explanation_event

            keys = ("status", "summary", "images", "dropped", "first_token_ms", "total_ms",
                    "error")
            t = time.perf_counter()
            event = explanation_event(result, self.search.explain(result))
            out["explanation"] = {k: event.get(k) for k in keys}
            out["ms"]["explain_ms"] = (time.perf_counter() - t) * 1000
            v3 = self._explainer_v3()
            if v3 is not None:      # layer 4 without the knowledge part, on the same results
                v4, self.search.explainer = self.search.explainer, v3
                try:
                    event = explanation_event(result, self.search.explain(result))
                finally:
                    self.search.explainer = v4
                out["explanation_v3"] = {k: event.get(k) for k in keys}
        for name, client in clients.items():
            step = name if tag == "b2" else f"{tag}:{name}"
            self.ledger.add_client_delta(step, client, before[name])
        return out

    def _explainer_v3(self):
        """The explainer with prompt v3 and no knowledge base, when B2 runs v4; else None."""
        e = self.search.explainer
        if e is None or e.knowledge is None:
            return None
        if not hasattr(self, "_v3"):
            from src.common.config import default_config_dir, load_config
            from src.kg.llm_query import Explainer

            name = load_config("models")["kg_llm"]["explain"]["prompt_file"]
            text = (default_config_dir() / name).read_text(encoding="utf-8")
            self._v3 = Explainer(e.client, e.guard, text, e.timeout_s, e.index, None)
        return self._v3

    def run_all(self, systems, rerun: bool) -> None:
        todo = [s for s in systems if rerun or s not in self.runs]
        b2_group = {"B2", *SEARCH_ABLATIONS}
        if b2_group & set(todo):        # one process: the search ablations share B2's parse
            todo = list(dict.fromkeys([*todo, *b2_group]))
        for system in todo:
            self.runs[system] = {}
        # warm the language-model paths off the scored rows (N1: cold first call)
        self.b1_parser.parse("chest x-ray")
        self.search.search("pleural effusion", top_k=1)
        for row in self.rows:
            qid = row["id"]
            for system in todo:
                self.ledger.check()
                if system == "B0":
                    self.runs[system][qid] = self.run_b0(row)
                elif system == "B1":
                    self.runs[system][qid] = self.run_b1(row)
                elif system == "REF":
                    self.runs[system][qid] = self.run_ref(row)
                elif system == "B2":
                    self.runs[system][qid] = self.run_b2(row, explain=True)
                elif system == "lookup_deterministic":
                    chooser, self.search.chooser = self.search.chooser, None
                    try:
                        self.runs[system][qid] = self.run_b2(row, False, system)
                    finally:
                        self.search.chooser = chooser
                elif system == "sapbert_off":
                    semantic, self.search.semantic = self.search.semantic, None
                    try:
                        self.runs[system][qid] = self.run_b2(row, False, system)
                    finally:
                        self.search.semantic = semantic
            print(f"  {qid}: " + ", ".join(f"{s} {len(self.runs[s][qid]['studies'])}"
                                           for s in todo) + f"; {self.ledger.total:.2f} USD")
            self.save()

    def _vec(self, text: str) -> np.ndarray:
        if text not in self._vecs:
            self._vecs[text] = self._encode(text)[0]
        return self._vecs[text]

    def run_transforms(self) -> None:
        """The ablations that only change how B2's intents are combined or scored: rescored, no
        search. One that leaves nothing wanted with a column takes the text route, as the search
        would (D73). probe_off rescores B2's intents on the zero-shot matrix (D72)."""
        matrix, fb = self.search.matrix, self.search.text_fallback
        for name in self.ecfg["eval"]["ablations"]:
            if name in SEARCH_ABLATIONS:
                continue
            self.runs[name] = {}
            for row in self.rows:
                b2 = self.runs["B2"][row["id"]]
                vec = self._vec(b2["fallback_text"]) if b2.get("fallback_text") else None
                if name == "probe_off":
                    intents = [intent_from_dict(d) for d in b2["intents"]]
                    used = self.zero_shot_matrix
                else:
                    intents = ke.transform([intent_from_dict(d) for d in b2["intents"]], name)
                    used = matrix
                self.runs[name][row["id"]] = {
                    "studies": ke.rescore(used, intents, self.k, b2["views"], fb, vec),
                    "intents": [i.to_dict() for i in intents]}
        self.save()

    def run_fusion(self) -> None:
        """The declared grid (eval.fusion): B2's KG score fused with the CLIP cosine of B1's
        caption or of B0's English text; a question the KG scores nothing, or that B2 ranked by
        text (the search's rule), keeps B2's studies, a question without that text keeps them
        too (w = 0)."""
        fcfg = self.ecfg["eval"]["fusion"]
        matrix, vectors = self.search.matrix, self.search.text_fallback.vectors
        for source in fcfg["sources"]:
            for w in fcfg["weights"]:
                name = f"fuse_{source}_w{w:g}"
                self.runs[name] = {}
                for row in self.rows:
                    b2 = self.runs["B2"][row["id"]]
                    text = self.runs[{"b1": "B1", "b0": "B0"}[source]][row["id"]].get("text")
                    got = None
                    if text and not b2.get("fallback"):
                        got = ke.fuse(matrix, [intent_from_dict(d) for d in b2["intents"]],
                                      vectors, self._vec(text), w, self.k, b2["views"])
                    self.runs[name][row["id"]] = {
                        "studies": b2["studies"] if got is None else got,
                        "fused": got is not None, "text": text}
        self.save()


# -- judging ---------------------------------------------------------------------------------

def pool(runs: dict, rows, k: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        seen: dict[str, None] = {}
        for system in sorted(runs):
            for s in runs[system].get(row["id"], {}).get("studies", [])[:k]:
                seen.setdefault(s, None)
        out[row["id"]] = list(seen)
    return out


def judge_pool(judge: ke.Judge, rows, pooled: dict, reports: dict) -> int:
    n = 0
    for row in rows:
        n += judge.grade(row["id"], row["question"], row["query_en"], pooled[row["id"]], reports)
    return n


def cross_pairs(rows, pooled: dict, fraction: float, seed: int) -> list[tuple[str, str]]:
    pairs = [(r["id"], s) for r in rows for s in pooled[r["id"]]]
    return sorted(ke.seeded_sample(pairs, fraction, seed))


def write_human_sample(path: Path, key_path: Path, rows, judge: ke.Judge, pooled: dict,
                       reports: dict, n: int, seed: int) -> dict:
    """`n` pairs balanced by the judge's grade, shuffled; the grades go to a separate key file
    so the person grading does not see them."""
    import random

    by_id = {r["id"]: r for r in rows}
    buckets: dict[int, list] = {0: [], 1: [], 2: []}
    for qid, studies in pooled.items():
        for s in studies:
            j = judge.get(qid, s)
            if j is not None:
                buckets[j["grade"]].append((qid, s))
    rng = random.Random(seed)
    per = {g: min(len(b), n // 3 + (1 if g < n % 3 else 0)) for g, b in buckets.items()}
    short = n - sum(per.values())
    for g in (2, 1, 0):                   # fill from the bigger buckets when one is too small
        extra = min(short, len(buckets[g]) - per[g])
        per[g] += extra
        short -= extra
    chosen = [p for g, b in buckets.items() for p in rng.sample(b, per[g])]
    rng.shuffle(chosen)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as h, \
            open(key_path, "w", encoding="utf-8", newline="") as hk:
        w, wk = csv.writer(h), csv.writer(hk)
        w.writerow(["pair", "question", "query_en", "findings", "impression", "human_grade",
                    "human_violates_exclusion", "note"])
        wk.writerow(["pair", "qid", "study", "judge_grade", "judge_violates_exclusion"])
        for i, (qid, s) in enumerate(chosen, start=1):
            r, rep, j = by_id[qid], reports[s], judge.get(qid, s)
            w.writerow([i, r["question"], r["query_en"], rep["findings"], rep["impression"],
                        "", "", ""])
            wk.writerow([i, qid, s, j["grade"], j["violates_exclusion"]])
    return {g: per[g] for g in per}


# -- scoring ---------------------------------------------------------------------------------

def score_runs(runs: dict, rows, judge: ke.Judge, view_of: dict, k: int, view_rule: bool,
               pooled: dict) -> dict:
    """{system: {qid: metrics}}; the ideal ranking of a question is over its whole judged pool."""
    out: dict = {}
    for row in rows:
        qid = row["id"]
        views = row["views"]

        def view_ok(s, views=views):
            return (not view_rule) or (not views) or view_of[s] in views

        def judged(s, qid=qid):
            return judge.get(qid, s)

        ideal = ke.gains(pooled[qid], judged, view_ok)
        has_excl = bool(row["exclude"]) or any(t["polarity"] != PRESENT for t in row["terms"])
        for system, by_q in runs.items():
            if qid in by_q:
                out.setdefault(system, {})[qid] = ke.query_metrics(
                    by_q[qid]["studies"], judged, view_ok, ideal, k, has_excl)
    return out


METRICS = ("ndcg10", "p10", "p10_lenient", "excl_viol10", "no_result")


def aggregate(per_q: dict, rows, keep=lambda r: True) -> dict:
    """{system: {group or 'all': {metric: mean}}} over the rows `keep` accepts."""
    out: dict = {}
    for system, by_q in per_q.items():
        table: dict = {}
        for scope in ["all", *dict.fromkeys(r["group"] for r in rows)]:
            chosen = [r["id"] for r in rows if keep(r) and r["id"] in by_q
                      and (scope == "all" or r["group"] == scope)]
            cell = {"n": len(chosen)}
            for m in METRICS:
                vals = [float(by_q[q][m]) for q in chosen if by_q[q][m] is not None]
                cell[m] = round(float(np.mean(vals)), 4) if vals else None
            table[scope] = cell
        out[system] = table
    return out


def diffs(per_q: dict, rows, pairs, n_boot: int, seed: int, keep=lambda r: True) -> dict:
    out = {}
    for a, b in pairs:
        if a not in per_q or b not in per_q:
            continue
        ids = [r["id"] for r in rows if keep(r) and r["id"] in per_q[a] and r["id"] in per_q[b]]
        out[f"{a}-{b}"] = {m: ke.bootstrap_mean_diff(
            [per_q[a][q][m] for q in ids], [per_q[b][q][m] for q in ids], n_boot, seed)
            for m in ("ndcg10", "p10")}
    return out


def error_step(row: dict, b2: dict) -> str:
    """Where B2 went wrong for a question it lost: parse/lookup (a gold concept never reached a
    candidate list), choice (it was a candidate, not chosen), expansion (the concepts are right,
    the columns are not), ranking (the columns match the gold)."""
    intents = [intent_from_dict(d) for d in b2["intents"]]
    chosen = {(i.cui, i.polarity) for i in intents}
    candidates = {c for t in b2["terms"] for c in t["candidates"]}
    missing = [(t["cui"], t["polarity"]) for t in row["terms"]
               if t["cui"] and (t["cui"], t["polarity"]) not in chosen]
    if missing:
        return "choice" if any(c in candidates for c, _ in missing) else "parse_or_lookup"
    if ke.signed_columns(intents) != ke.gold_columns(row):
        return "expansion"
    return "ranking"
