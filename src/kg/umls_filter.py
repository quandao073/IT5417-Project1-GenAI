"""Filter the UMLS Metathesaurus down to what the KG build reads (docs/v2-kg-design.md, step 1).

The NLM release is read only: UMLS_ZIP for its MD5, UMLS_META_DIR for the RRF files. One Parquet
per RRF file and a manifest go to data/umls/<version>/. Both places are gitignored and never enter
a Docker image (WORKLOG D49, D50).

RRF files are '|'-separated, unquoted and headerless, and every line ends with a '|', so a parsed
line has one empty field more than MRFILES.RRF lists. Column names come from MRFILES.RRF.
Everything is checked before anything is written: a broken release leaves no output behind.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from pydantic_settings import BaseSettings, SettingsConfigDict

FILES = ("MRCONSO", "MRSTY", "MRREL", "MRSAB")
SNOMED = "SNOMEDCT_US"
_END = "_end"   # the empty field after each line's trailing '|'


class UmlsFilterError(RuntimeError):
    """The release on disk is not what the filter expects; nothing was written."""


class UmlsPaths(BaseSettings):
    """UMLS_ZIP and UMLS_META_DIR from the environment or .env; the defaults follow D50."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    umls_zip: Path = Path("data/umls/raw/umls-2026AA-metathesaurus-level0.zip")
    umls_meta_dir: Path = Path("data/umls/raw/2026AA/META")


def read_mrfiles(meta: Path) -> dict[str, dict]:
    """{"MRCONSO": {"columns": [...], "rows": n}, ...} for every file MRFILES.RRF lists."""
    path = meta / "MRFILES.RRF"
    if not path.is_file():
        raise UmlsFilterError(f"{path} not found")
    listed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, _description, fmt, _n_columns, rows, *_ = line.split("|")
        listed[name.removesuffix(".RRF")] = {"columns": fmt.split(","), "rows": int(rows)}
    return listed


def scan_rrf(meta: Path, name: str, columns: list[str]) -> pl.LazyFrame:
    """`name`.RRF as an all-string lazy frame named by MRFILES; refuses a column mismatch."""
    path = meta / f"{name}.RRF"
    if not path.is_file():
        raise UmlsFilterError(f"{path} not found")
    with open(path, encoding="utf-8") as handle:
        first = handle.readline().rstrip("\r\n").split("|")
    if len(first) != len(columns) + 1 or first[-1] != "":
        raise UmlsFilterError(f"{name}.RRF has {len(first) - 1} fields per line, "
                              f"MRFILES.RRF lists {len(columns)}")
    schema = {column: pl.String for column in [*columns, _END]}
    return pl.scan_csv(path, separator="|", has_header=False, quote_char=None,
                       schema=schema).drop(_END)


def _count(frame: pl.LazyFrame) -> int:
    return frame.select(pl.len()).collect(engine="streaming").item()


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def filter_umls(meta: Path, zip_path: Path, out_dir: Path, cfg: dict) -> dict:
    """Write the filtered MRCONSO, MRSTY, MRREL, MRSAB and manifest.json; return the manifest.

    `cfg` is configs/kg.yaml:umls. MRCONSO keeps English, unsuppressed names up to `max_srl`;
    MRREL keeps unsuppressed rows of `rel_sabs`; MRSTY (with STN) and MRSAB are kept whole.
    `include_snomed` adds SNOMEDCT_US names and relations, and requires it in the subset."""
    listed = read_mrfiles(meta)
    if missing := [name for name in FILES if name not in listed]:
        raise UmlsFilterError(f"MRFILES.RRF does not list {missing}")
    frames = {name: scan_rrf(meta, name, listed[name]["columns"]) for name in FILES}

    in_subset = set(frames["MRSAB"].filter(pl.col("SABIN") == "Y")
                    .select("RSAB").collect()["RSAB"].to_list())
    if cfg["include_snomed"] and SNOMED not in in_subset:
        raise UmlsFilterError(f"include_snomed is true but this subset has no {SNOMED}; "
                              "download the Full Subset after MLDS registration, or set "
                              "umls.include_snomed: false")
    if missing := sorted(set(cfg["rel_sabs"]) - in_subset):
        raise UmlsFilterError(f"rel_sabs not in this subset: {missing}")
    if not zip_path.is_file():
        raise UmlsFilterError(f"{zip_path} not found")
    rows_before = {name: _count(frame) for name, frame in frames.items()}
    for name, n in rows_before.items():
        if n != listed[name]["rows"]:
            raise UmlsFilterError(f"{name}.RRF has {n} rows, MRFILES.RRF lists "
                                  f"{listed[name]['rows']}: the extraction is incomplete")

    snomed = [SNOMED] if cfg["include_snomed"] else []
    srl_ok = pl.col("SRL").cast(pl.Int64) <= cfg["max_srl"]
    if snomed:
        srl_ok = srl_ok | (pl.col("SAB") == SNOMED)
    keep = {
        "MRCONSO": (pl.col("LAT") == "ENG") & (pl.col("SUPPRESS") == "N") & srl_ok,
        "MRREL": (pl.col("SUPPRESS") == "N") & pl.col("SAB").is_in([*cfg["rel_sabs"], *snomed]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, frame in frames.items():
        path = out_dir / f"{name.lower()}.parquet"
        (frame.filter(keep[name]) if name in keep else frame).sink_parquet(path)
        files[name.lower()] = {
            "source": f"{name}.RRF", "rows_before": rows_before[name],
            "rows_after": _count(pl.scan_parquet(path)), "bytes": path.stat().st_size,
            "sha256": _digest(path, "sha256"),
        }
    manifest = {
        "umls_version": cfg["version"], "zip": zip_path.as_posix(),
        "zip_md5": _digest(zip_path, "md5"), "meta_dir": meta.as_posix(), "config": cfg,
        "files": files, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    # Written last: a directory without manifest.json is an interrupted run, not an output.
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
