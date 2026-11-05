"""Read-only: which forms of the 14 labels (configs/concepts.yaml, en and vi) also name a scorable
concept below the label (IS_A) that has its own, usually better, score column. Such a form must
reach `lookup_concept` as a term instead of being read as the label (WORKLOG D67); the result is
`configs/kg.yaml: query.ambiguous_forms`. Needs data/kg/graph/; writes label_shadows.json there.

    uv run python scripts/kg_label_shadows.py
"""
import json
from pathlib import Path

import polars as pl
import yaml

from src.common.config import load_config
from src.kg.lookup import normalize
from src.kg.vocab import singular_candidate

graph = Path("data/kg/graph")
nodes = pl.read_parquet(graph / "nodes.parquet", columns=["cui", "name", "scorable"])
edges = pl.read_parquet(graph / "edges.parquet").filter(pl.col("rel") == "IS_A")
aliases = pl.read_parquet(graph / "aliases.parquet", columns=["cui", "norm"])
chexpert = Path("configs/kg/chexpert_cuis.yaml").read_text(encoding="utf-8")
mapping = yaml.safe_load(chexpert)["labels"]
names = dict(nodes.select("cui", "name").iter_rows())
scorable = set(nodes.filter("scorable")["cui"])
parents: dict[str, set[str]] = {}
for src, dst in edges.select("src", "dst").iter_rows():
    parents.setdefault(src, set()).add(dst)


def below(cui: str, root: str) -> bool:
    seen, todo = set(), [cui]
    while todo:
        for p in parents.get(todo.pop(), ()):
            if p == root:
                return True
            if p not in seen:
                seen.add(p)
                todo.append(p)
    return False


by_norm: dict[str, set[str]] = {}
for cui, norm in aliases.iter_rows():
    by_norm.setdefault(norm, set()).add(cui)

found = []
for label, forms in load_config("concepts")["concepts"].items():
    root = mapping[label]["id"]
    for lang in ("en", "vi"):
        for form in forms[lang]:
            norm = normalize(form)
            cuis = by_norm.get(norm, set()) | by_norm.get(singular_candidate(norm) or norm, set())
            kids = sorted(c for c in cuis if c in scorable and c != root and below(c, root))
            if kids:
                found.append({"label": label, "form": form, "lang": lang,
                              "children": [{"cui": c, "name": names[c]} for c in kids]})
Path(graph / "label_shadows.json").write_text(json.dumps(found, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
for row in found:
    print(f"{row['label']:<28} {row['lang']}  {row['form']!r:<28} -> "
          f"{[c['cui'] for c in row['children']]}")
print(f"{len(found)} forms")
