"""Write the new heldout questions of the KG benchmark as text blocks for the chat's gold review.

    uv run python scripts/kg_bench_export_review.py

Reads benchmarks/kg_queries_v1.jsonl, data/eval/bench_extend.json (the new ids), the notes and
review reasons of data/eval/bench_notes.jsonl and the concept names of the graph. Concept names
are UMLS content, so the output stays under data/ (CLAUDE.md section 5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

from src.common.config import Settings
from src.evaluation import kg_bench as kb

OUT = Path("data/eval/heldout_new_gold_review.txt")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = {r["id"]: r for r in kb.read_jsonl(Path("benchmarks/kg_queries_v1.jsonl"))}
    new = json.loads(Path("data/eval/bench_extend.json").read_text(encoding="utf-8"))["new_ids"]
    notes = {n["id"]: n for n in kb.read_jsonl(Path("data/eval/bench_notes.jsonl"))}
    graph_dir = Path(Settings().kg_graph_dir)
    names = dict(pl.read_parquet(graph_dir / "nodes.parquet").select("cui", "name").iter_rows())
    quality = json.loads((graph_dir / "quality.json").read_text(encoding="utf-8"))

    def name(cui: str) -> str:
        return f"{cui} {names.get(cui, '?')}" + (" [weak]" if quality.get(cui, {}).get("weak")
                                                  else "")

    blocks = []
    picked = [rows[i] for i in new if rows[i]["split"] == "heldout"]
    for n, r in enumerate(picked, start=1):
        a = r["auto_check"]
        lines = [f"=== {n}. {r['id']}  ({r['group']}, {r['lang']}, pair {r['pair'] or '-'})",
                 f"Câu hỏi : {r['question']}", f"EN      : {r['query_en']}", "Terms   :"]
        lines += [f"  - \"{t['text']}\" -> {name(t['cui']) if t['cui'] else '(không có CUI)'} "
                  f"[{t['polarity']}]" for t in r["terms"]]
        lines.append("Cột kỳ vọng:")
        lines += [f"  - {name(c['cui'])}" for c in r["expected_scorable"]] or ["  (không có)"]
        lines.append("Exclude : " + (", ".join(name(c) for c in r["exclude"]) or "-"))
        lines.append(f"Views   : {', '.join(r['views']) or '-'}")
        lines.append(f"Cảnh báo: {', '.join(r['expect_warnings']) or '-'}")
        if a["columns_match"]:
            lines.append("auto_check: khớp gold")
        else:
            def listed(cuis):
                return ", ".join(name(c) for c in cuis) or "-"
            lines.append(f"auto_check lệch: graph chọn {listed(a['graph_include'])}; thiếu "
                         f"{listed(a['missing'])}; thừa {listed(a['extra'])}; kiểu "
                         f"{', '.join(a['kinds']) or '-'}")
        if a.get("weak_flags") or a.get("unknown_cuis"):
            lines.append(f"  cờ: weak {a.get('weak_flags')} ; CUI lạ {a.get('unknown_cuis')}")
        note = notes.get(r["id"], {})
        lines.append(f"Ghi chú Opus: {note.get('note', '')}")
        lines.append(f"Sonnet: {r['review']['verdict']} — {note.get('review_reason', '')}")
        blocks.append("\n".join(lines))
    OUT.write_text(f"{len(picked)} câu heldout mới của benchmarks/kg_queries_v1.jsonl\n\n"
                   + "\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"{len(picked)} blocks written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
