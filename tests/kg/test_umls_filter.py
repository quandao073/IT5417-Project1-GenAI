"""`cxr kg-filter-umls` on a few invented RRF lines; no real UMLS content is used here (T38)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from src.cli.main import main
from src.kg import umls_filter as uf

CONSO = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI",
         "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"]
STY = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]
REL = ["CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2", "RELA", "RUI", "SRUI", "SAB",
       "SL", "RG", "DIR", "SUPPRESS", "CVF"]
SAB = ["VCUI", "RCUI", "VSAB", "RSAB", "SON", "SF", "SVER", "VSTART", "VEND", "IMETA", "RMETA",
       "SLC", "SCC", "SRL", "TFR", "CFR", "CXTY", "TTYL", "ATNL", "LAT", "CENC", "CURVER",
       "SABIN", "SSN", "SCIT"]
COLUMNS = {"MRCONSO": CONSO, "MRSTY": STY, "MRREL": REL, "MRSAB": SAB}
CFG = {"version": "2099AA", "max_srl": 0, "include_snomed": False,
       "rel_sabs": ["MSH", "NCI", "FMA", "HPO"]}


def conso(cui, lat, sab, srl, suppress, text):
    return {"CUI": cui, "LAT": lat, "SAB": sab, "SRL": srl, "SUPPRESS": suppress, "STR": text,
            "AUI": f"A{cui[1:]}{lat}{sab}"}


def rel(cui1, cui2, sab, rela, suppress="N"):
    return {"CUI1": cui1, "REL": "PAR", "CUI2": cui2, "RELA": rela, "SAB": sab,
            "SUPPRESS": suppress}


def source(rsab, sabin="Y"):
    return {"RSAB": rsab, "VSAB": f"{rsab}2099", "SABIN": sabin}


CONSO_ROWS = [
    conso("C9000001", "ENG", "MSH", "0", "N", "invented finding one"),        # kept
    conso("C9000001", "FRE", "MSH", "0", "N", "trouvaille inventee"),         # not English
    conso("C9000002", "ENG", "NCI", "0", "O", "invented obsolete name"),      # suppressed
    conso("C9000003", "ENG", "XYZ", "3", "N", "invented restricted name"),    # SRL 3
    conso("C9000004", "ENG", "SNOMEDCT_US", "9", "N", "invented snomed name"),  # SRL 9
    conso("C9000005", "ENG", "FMA", "0", "N", 'invented "quoted" part'),      # kept, quote kept
]
STY_ROWS = [{"CUI": "C9000001", "TUI": "T033", "STN": "A2.2", "STY": "Finding"},
            {"CUI": "C9000005", "TUI": "T023", "STN": "A1.2.3.1", "STY": "Body Part"}]
REL_ROWS = [
    rel("C9000001", "C9000006", "MSH", "isa"),                # kept
    rel("C9000005", "C9000007", "FMA", "part_of"),            # kept
    rel("C9000001", "C9000008", "HPO", ""),                   # kept
    rel("C9000001", "C9000009", "NCI", "isa", suppress="O"),  # suppressed
    rel("C9000001", "C9000010", "XYZ", "isa"),                # source not listed
    rel("C9000004", "C9000011", "SNOMEDCT_US", "isa"),        # SNOMED, off by default
]
SAB_ROWS = [source("MSH"), source("NCI"), source("FMA"), source("HPO"), source("XYZ"),
            source("SNOMEDCT_US", sabin="N")]


def line(columns, row):
    return "|".join(row.get(c, "") for c in columns) + "|"


def write_meta(root: Path, *, sab_rows=SAB_ROWS, rows_listed=None) -> tuple[Path, Path]:
    """A META directory with MRFILES.RRF and the four files the filter reads, and a fake zip."""
    meta = root / "META"
    meta.mkdir(parents=True)
    data = {"MRCONSO": CONSO_ROWS, "MRSTY": STY_ROWS, "MRREL": REL_ROWS, "MRSAB": sab_rows}
    listing = []
    for name, rows in data.items():
        text = "".join(line(COLUMNS[name], r) + "\n" for r in rows)
        (meta / f"{name}.RRF").write_text(text, encoding="utf-8")
        n = (rows_listed or {}).get(name, len(rows))
        listing.append(f"{name}.RRF|{name} file|{','.join(COLUMNS[name])}|"
                       f"{len(COLUMNS[name])}|{n}|{len(text.encode())}|")
    (meta / "MRFILES.RRF").write_text("\n".join(listing) + "\n", encoding="utf-8")
    zip_path = root / "release.zip"
    zip_path.write_bytes(b"not a real release")
    return meta, zip_path


@pytest.fixture
def release(tmp_path):
    return write_meta(tmp_path / "raw")


def run(release, tmp_path, cfg=CFG):
    meta, zip_path = release
    out = tmp_path / "out"
    return uf.filter_umls(meta, zip_path, out, cfg), out


def test_mrfiles_gives_the_columns_and_row_count_of_each_file(release):
    listed = uf.read_mrfiles(release[0])
    assert listed["MRCONSO"] == {"columns": CONSO, "rows": len(CONSO_ROWS)}
    assert listed["MRSTY"]["columns"] == STY


def test_mrconso_keeps_english_unsuppressed_rows_up_to_max_srl(release, tmp_path):
    _, out = run(release, tmp_path)
    kept = pl.read_parquet(out / "mrconso.parquet")
    assert kept.columns == CONSO   # MRFILES names, SRL kept, the empty last field dropped
    assert kept["CUI"].to_list() == ["C9000001", "C9000005"]
    assert kept["STR"].to_list()[1] == 'invented "quoted" part'


def test_mrrel_keeps_unsuppressed_rows_of_the_listed_sources(release, tmp_path):
    _, out = run(release, tmp_path)
    kept = pl.read_parquet(out / "mrrel.parquet")
    assert kept["SAB"].to_list() == ["MSH", "FMA", "HPO"]


def test_mrsty_and_mrsab_are_kept_whole_with_stn(release, tmp_path):
    _, out = run(release, tmp_path)
    sty = pl.read_parquet(out / "mrsty.parquet")
    assert sty.columns == STY and sty["STN"].to_list() == ["A2.2", "A1.2.3.1"]
    assert pl.read_parquet(out / "mrsab.parquet").height == len(SAB_ROWS)


def test_include_snomed_without_snomed_in_the_subset_stops(release, tmp_path):
    with pytest.raises(uf.UmlsFilterError, match="SNOMEDCT_US"):
        run(release, tmp_path, {**CFG, "include_snomed": True})
    assert not (tmp_path / "out").exists()


def test_include_snomed_keeps_snomed_names_and_relations(tmp_path):
    sabs = [s for s in SAB_ROWS if s["RSAB"] != "SNOMEDCT_US"] + [source("SNOMEDCT_US")]
    manifest, out = run(write_meta(tmp_path / "raw", sab_rows=sabs), tmp_path,
                        {**CFG, "include_snomed": True})
    assert "C9000004" in pl.read_parquet(out / "mrconso.parquet")["CUI"].to_list()
    assert "SNOMEDCT_US" in pl.read_parquet(out / "mrrel.parquet")["SAB"].to_list()
    assert manifest["config"]["include_snomed"] is True


def test_a_listed_relation_source_missing_from_the_subset_stops(tmp_path):
    sabs = [s for s in SAB_ROWS if s["RSAB"] != "FMA"]
    with pytest.raises(uf.UmlsFilterError, match="FMA"):
        run(write_meta(tmp_path / "raw", sab_rows=sabs), tmp_path)


def test_columns_that_do_not_match_mrfiles_stop(release, tmp_path):
    meta, _ = release
    path = meta / "MRSTY.RRF"
    path.write_text(path.read_text(encoding="utf-8").replace("|\n", "|extra|\n"),
                    encoding="utf-8")
    with pytest.raises(uf.UmlsFilterError, match="MRSTY"):
        run(release, tmp_path)


def test_a_row_count_that_does_not_match_mrfiles_stops(tmp_path):
    release = write_meta(tmp_path / "raw", rows_listed={"MRREL": len(REL_ROWS) + 5})
    with pytest.raises(uf.UmlsFilterError, match="MRREL"):
        run(release, tmp_path)
    assert not (tmp_path / "out").exists()   # validated before anything is written


def test_a_missing_meta_file_stops(release, tmp_path):
    (release[0] / "MRREL.RRF").unlink()
    with pytest.raises(uf.UmlsFilterError, match="MRREL"):
        run(release, tmp_path)


def test_manifest_records_version_zip_md5_config_hashes_and_counts(release, tmp_path):
    manifest, out = run(release, tmp_path)
    assert manifest == json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["umls_version"] == "2099AA"
    assert manifest["zip_md5"] == hashlib.md5(b"not a real release").hexdigest()
    assert manifest["config"] == CFG
    conso_file = manifest["files"]["mrconso"]
    assert (conso_file["rows_before"], conso_file["rows_after"]) == (len(CONSO_ROWS), 2)
    for name, entry in manifest["files"].items():
        assert entry["sha256"] == hashlib.sha256((out / f"{name}.parquet").read_bytes()).hexdigest()


def test_cli_reads_the_inputs_from_the_environment(release, tmp_path, monkeypatch, capsys):
    meta, zip_path = release
    monkeypatch.setenv("UMLS_META_DIR", str(meta))
    monkeypatch.setenv("UMLS_ZIP", str(zip_path))
    out = tmp_path / "cli_out"
    assert main(["kg-filter-umls", "--out-dir", str(out)]) == 0
    assert (out / "manifest.json").is_file()
    assert "mrconso" in capsys.readouterr().out


def test_cli_reports_a_broken_release_without_a_traceback(release, tmp_path, monkeypatch, capsys):
    meta, zip_path = release
    monkeypatch.setenv("UMLS_META_DIR", str(meta))
    monkeypatch.setenv("UMLS_ZIP", str(tmp_path / "missing.zip"))
    assert main(["kg-filter-umls", "--out-dir", str(tmp_path / "cli_out")]) == 1
    assert "missing.zip" in capsys.readouterr().err
