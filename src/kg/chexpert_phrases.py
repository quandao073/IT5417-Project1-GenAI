"""Partial gold for the V2-3 linker from the CheXpert labeler's mention phrases (D56).

stanfordmlgroup/chexpert-labeler (MIT), `phrases/mention/<label>.txt`, one phrase per line, pinned
to a commit in configs/kg.yaml (linker.chexpert_phrases) and fetched into data/. A head string
equal to a phrase of label X must link to X's CUI in configs/kg/chexpert_cuis.yaml, to one of its
`children`, or to a CUI below them (MRREL CHD, T39). No Finding has no CUI and is not checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.kg import vocab

LABEL_FILES = ["atelectasis", "cardiomegaly", "consolidation", "edema",
               "enlarged_cardiomediastinum", "fracture", "lung_lesion", "lung_opacity",
               "no_finding", "pleural_effusion", "pleural_other", "pneumonia", "pneumothorax",
               "support_devices"]


def fetch(cfg: dict, out_dir: Path, transport=None) -> Path:
    """Download the mention phrase files at the pinned commit into out_dir/<commit>/."""
    import httpx

    target = out_dir / cfg["commit"]
    target.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{cfg['repo']}/{cfg['commit']}/phrases/mention"
    with httpx.Client(timeout=30, transport=transport) as client:
        for stem in LABEL_FILES:
            response = client.get(f"{base}/{stem}.txt")
            response.raise_for_status()
            (target / f"{stem}.txt").write_text(response.text, encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps(
        {"repo": cfg["repo"], "commit": cfg["commit"], "license": cfg["license"],
         "files": [f"phrases/mention/{s}.txt" for s in LABEL_FILES]}, indent=2), encoding="utf-8")
    return target


def load(directory: Path) -> dict[str, set[str]]:
    """{CheXpert label: normalised phrases}, No Finding left out."""
    out = {}
    for stem in LABEL_FILES:
        label = stem.replace("_", " ").title()
        if label == "No Finding":
            continue
        lines = (directory / f"{stem}.txt").read_text(encoding="utf-8").splitlines()
        out[label] = {p for line in lines if (p := vocab.normalize(line))}
    return out


def expected_cuis(mapping: dict, umls_dir: Path, devices: set[str],
                  device_label: str) -> dict[str, set[str]]:
    """{label: the CUIs a phrase of the label may link to}: its CUI id or its children, and every
    CUI below them along MRREL CHD rows of any source (T39); for `device_label`
    (PRJ:support_devices), every device CUI of the domain, `devices`."""
    from src.kg.linker import _descendants

    edges = (pl.scan_parquet(umls_dir / "mrrel.parquet").filter(pl.col("REL") == "CHD")
             .select("CUI1", "CUI2").collect())
    out = {}
    for label, entry in mapping.items():
        roots = [c for c in [entry["id"], *entry.get("children", [])] if c.startswith("C")]
        if entry["id"] == device_label:
            out[label] = _descendants(edges, roots) | devices
        elif roots:
            out[label] = _descendants(edges, roots)
    return out


def mismatches(links: pl.DataFrame, phrases: dict[str, set[str]],
               expected: dict[str, set[str]], column: str = "cui") -> tuple[dict, set]:
    """(per label {strings, linked, ok}, the (kind, string) pairs whose `column` CUI is not
    expected for any label whose phrase the string is). Observation strings only."""
    stats: dict[str, dict[str, int]] = {}
    bad: set[tuple[str, str]] = set()
    for string, cui in (links.filter(pl.col("kind") == "Observation")
                        .select("string", column).iter_rows()):
        labels = [label for label, ps in phrases.items() if string in ps and label in expected]
        for label in labels:
            s = stats.setdefault(label, {"strings": 0, "linked": 0, "ok": 0})
            s["strings"] += 1
            s["linked"] += cui is not None
            s["ok"] += cui is not None and cui in expected[label]
        if labels and cui is not None and not any(cui in expected[lb] for lb in labels):
            bad.add(("Observation", string))
    return dict(sorted(stats.items())), bad


def rate(stats: dict[str, dict[str, int]]) -> dict[str, int | float | None]:
    linked = sum(s["linked"] for s in stats.values())
    ok = sum(s["ok"] for s in stats.values())
    return {"strings": sum(s["strings"] for s in stats.values()), "linked": linked, "ok": ok,
            "rate": round(ok / linked, 3) if linked else None}
