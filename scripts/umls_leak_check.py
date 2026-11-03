"""Read-only (WORKLOG T38): find UMLS names in committed text before a PR. For every CUI that a
file changed between two revisions mentions, the MRCONSO names of that CUI (normalised, two words
or more) that also occur in the same file. Needs the local filtered UMLS (data/umls/2026AA/).
Prints file, CUI and the matched name; a name is UMLS content, so the output stays on screen and is
not written anywhere. A match that is report vocabulary or a CheXpert label name is allowed; a
source heading name copied into a comment is not (CLAUDE.md, "Dữ liệu và license").

    uv run python scripts/umls_leak_check.py main HEAD
"""
import re
import subprocess
import sys

import polars as pl

base, tip = sys.argv[1], sys.argv[2]
files = subprocess.run(["git", "diff", "--name-only", f"{base}..{tip}"], capture_output=True,
                       text=True, check=True).stdout.split()
norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()  # noqa: E731
texts = {}
for f in files:
    shown = subprocess.run(["git", "show", f"{tip}:{f}"], capture_output=True)
    if shown.returncode == 0:
        texts[f] = shown.stdout.decode("utf-8", errors="ignore")
cuis = {c for t in texts.values() for c in re.findall(r"C\d{7}", t)}
names = (pl.scan_parquet("data/umls/2026AA/mrconso.parquet").filter(pl.col("CUI").is_in(list(cuis)))
         .select("CUI", "STR").unique().collect())
by_cui: dict[str, set[str]] = {}
for cui, text in names.iter_rows():
    n = norm(text)
    if len(n.split()) >= 2:
        by_cui.setdefault(cui, set()).add(n)
hits = 0
for f, t in texts.items():
    body = f" {norm(t)} "
    for cui in sorted(set(re.findall(r"C\d{7}", t))):
        found = sorted(n for n in by_cui.get(cui, ()) if f" {n} " in body)
        if found:
            hits += 1
            print(f"{f}: {cui}: {found[:3]}")
print(f"{hits} (file, CUI) pairs with a multi-word UMLS name of that CUI in the file")
