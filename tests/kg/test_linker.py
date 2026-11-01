"""`cxr kg-link` (TIP V2-3, D56) on invented MRCONSO, MRSTY, MRREL rows and a fake SapBERT."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from src.cli.main import main
from src.common.config import default_config_dir, load_config
from src.kg import linker as lk
from src.kg import vocab
from tests.kg.test_vocab import world  # noqa: F401  (the pool, Plus CSV and RadGraph fixture)

REAL = load_config("kg")["linker"]
CFG = {**REAL, "domain": {"code_sabs": ["LNC"], "relas": REAL["domain"]["relas"],
                          "anatomy": {"positive": ["C1000020"], "negative": []},
                          "observation": {"positive": ["C1000021"], "negative": ["C1000022"]}}}


def conso(cui, text, sab, preferred=False):
    return {"CUI": cui, "STR": text, "SAB": sab, "TS": "P" if preferred else "S",
            "ISPREF": "Y" if preferred else "N", "STT": "PF" if preferred else "VO"}


ROWS = [
    conso("C1000001", "Pleural effusion", "MSH", True), conso("C1000002", "Effusion", "NCI", True),
    conso("C1000003", "Pneumothorax", "MSH", True), conso("C1000003", "Collapsed lung", "MSH"),
    conso("C1000004", "Thickening", "NCI", True),
    conso("C1000005", "Pleural thickening", "MSH", True),
    conso("C1000006", "Pleura", "FMA", True), conso("C1000007", "Lung", "FMA", True),
    conso("C1000008", "Mass", "NCI", True), conso("C1000009", "Mass", "HPO"),
    conso("C1000009", "Lump", "MSH"), conso("C1000011", "Sternotomy", "MSH", True),
    conso("C1000012", "Left lower lobe", "FMA", True), conso("C1000013", "Lower lobe", "FMA", True),
    conso("C1000014", "Thickening pleura", "NCI", True),
    conso("C1000016", "Crowding", "NCI", True),              # below the negative root only
    conso("C1000017", "Hilum", "LNC", True),                 # named by a code system only
    conso("C1000020", "Thorax", "FMA", True), conso("C1000021", "Thoracic disease", "MSH", True),
    conso("C1000022", "Tooth disease", "MSH", True),
]
STY = {"C1000001": "T046", "C1000002": "T046", "C1000003": "T047", "C1000004": "T033",
       "C1000005": "T047", "C1000006": "T023", "C1000007": "T023", "C1000008": "T033",
       "C1000009": "T033", "C1000011": "T061", "C1000012": "T023", "C1000013": "T023",
       "C1000014": "T033", "C1000016": "T033", "C1000017": "T029", "C1000020": "T029",
       "C1000021": "T047", "C1000022": "T047"}


def rel(whole, rela, part, sab="FMA", kind="RO"):
    return {"CUI1": whole, "REL": kind, "CUI2": part, "SAB": sab, "RELA": rela}


REL = [rel("C1000012", "inverse_isa", "C1000013", kind="PAR"),
       rel("C1000013", "isa", "C1000012", kind="CHD"),
       rel("C1000020", "regional_part_of", "C1000007"), rel("C1000020", "regional_part_of",
                                                            "C1000006"),
       rel("C1000007", "constitutional_part_of", "C1000013"),
       rel("C1000020", "regional_part_of", "C1000017"),
       *[rel("C1000021", None, c, sab="MSH", kind="CHD") for c in
         ("C1000001", "C1000002", "C1000003", "C1000005", "C1000008", "C1000014")],
       rel("C1000008", "isa", "C1000009", sab="NCI", kind="CHD"),   # below a root, two steps down
       *[rel("C1000022", None, c, sab="MSH", kind="CHD") for c in ("C1000016", "C1000009")]]
OVERRIDES = {"Observation": {"effusion": "C1000001", "tip": "skip", "collapsed lung": "C9999999"},
             "Anatomy": {}}


@pytest.fixture
def umls(tmp_path):
    out = tmp_path / "umls"
    out.mkdir()
    pl.DataFrame(ROWS).write_parquet(out / "mrconso.parquet")
    pl.DataFrame([{"CUI": c, "TUI": t, "STY": t} for c, t in STY.items()]).write_parquet(
        out / "mrsty.parquet")
    pl.DataFrame(REL).write_parquet(out / "mrrel.parquet")
    return out


class FakeSap:
    """Dictionary names are one-hot; a query is a mix, so its nearest name is known."""

    revision, max_length = "fakerevision0000", 25
    NAMES = sorted({vocab.normalize(r["STR"]) for r in ROWS})
    MIXES = {"airspace process": {"effusion": 0.9, "lung": 0.2},
             "hilum": {"effusion": 0.9, "lung": 0.3}}   # Anatomy: "effusion" is not allowed

    def __init__(self, cfg=None):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), len(self.NAMES) + 1), dtype=np.float32)
        for row, text in enumerate(texts):
            mix = self.MIXES.get(text, {text: 1.0} if text in self.NAMES else {})
            for name, w in mix.items():
                out[row, self.NAMES.index(name)] = w
            out[row, -1] = 0.0 if mix else 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def heads(*rows):
    return pl.DataFrame([(k, i + 1, s, 10, 5, "80") for i, (k, s) in enumerate(rows)],
                        schema={"kind": pl.String, "rank": pl.UInt32, "string": pl.String,
                                "mentions": pl.UInt32, "studies": pl.UInt32, "band": pl.String},
                        orient="row")


def link(umls, tmp_path, *rows):
    found, outside = lk.link_heads(heads(*rows), umls, CFG, OVERRIDES, lk.domain(umls, CFG),
                                   FakeSap, tmp_path / "sapbert")
    return {(k, s): r for k, s, r in zip(found["kind"], found["string"],
                                         found.iter_rows(named=True), strict=True)}, outside


def test_rules_fire_in_order(umls, tmp_path):
    got, _ = link(umls, tmp_path, *[("Observation", s) for s in
                                    ("normal", "tip", "effusion", "collapsed lung", "thickening",
                                     "pneumothorax")])
    methods = {s: (r["method"], r["cui"]) for (_, s), r in got.items()}
    assert methods == {
        "normal": ("normal", None), "tip": ("skip", None),
        "effusion": ("override", "C1000001"),            # over the exact C1000002
        "collapsed lung": ("override", "C9999999"),      # over the exact Pneumothorax CUI (D21)
        "thickening": ("descriptive", None),             # alone, although an exact name exists
        "pneumothorax": ("exact", "C1000003")}


def test_exact_ranks_by_preferred_name_and_keeps_only_types_of_the_kind(umls, tmp_path):
    got, outside = link(umls, tmp_path, ("Observation", "mass"), ("Anatomy", "pneumothorax"),
                        ("Observation", "sternotomy"))
    mass = got[("Observation", "mass")]
    assert (mass["method"], mass["cui"], mass["alt1"], mass["n_candidates"]) == (
        "exact", "C1000008", "C1000009", 2)               # preferred beats more sources
    assert got[("Anatomy", "pneumothorax")]["method"] == "sapbert"   # T047 is no Anatomy type
    assert outside == {("Observation", "sternotomy")}     # a procedure (T061) only
    assert (got[("Observation", "sternotomy")]["method"],
            got[("Observation", "sternotomy")]["cui"]) == ("out_of_scope", None)
    hand, _ = lk.link_heads(heads(("Observation", "sternotomy")), umls, CFG,
                            {"Observation": {"sternotomy": "skip"}}, lk.domain(umls, CFG),
                            FakeSap, tmp_path / "sapbert")
    assert hand["method"].to_list() == ["skip"]                     # an override still wins


def test_the_domain_drops_code_system_names_and_cuis_below_a_negative_root(umls, tmp_path):
    dom = set(lk.domain(umls, CFG).iter_rows())
    assert {("C1000007", "Anatomy"), ("C1000012", "Anatomy"),       # thorax > lung > lobe > left
            ("C1000009", "Observation")} <= dom                     # below the positive root too
    assert ("C1000017", "Anatomy") not in dom                       # LNC names only
    assert ("C1000016", "Observation") not in dom                   # below the negative root only
    assert ("C1000022", "Observation") not in dom                   # the negative root itself
    assert ("C1000004", "Observation") in dom                       # below no root at all
    got, outside = link(umls, tmp_path, ("Observation", "crowding"), ("Anatomy", "hilum"))
    # on the label table, so not out_of_scope; out of the domain, so no exact link
    assert got[("Observation", "crowding")]["method"] == "sapbert" and not outside
    assert got[("Anatomy", "hilum")]["cui"] == "C1000007"           # FakeSap's mix, not the LNC CUI
    sizes = lk.domain_sizes(umls, CFG, lk.domain(umls, CFG))
    assert sizes["Anatomy"] == {"label_table": 6, "domain": 5}


def test_adjective_goes_through_its_noun(umls, tmp_path):
    got, _ = link(umls, tmp_path, ("Anatomy", "pleural"))
    pleural = got[("Anatomy", "pleural")]
    assert (pleural["method"], pleural["query"], pleural["cui"]) == ("adjective", "pleura",
                                                                      "C1000006")


def test_sapbert_takes_the_nearest_name_allowed_for_the_kind(umls, tmp_path):
    got, _ = link(umls, tmp_path, ("Observation", "airspace process"), ("Anatomy", "hilum"))
    obs, anat = got[("Observation", "airspace process")], got[("Anatomy", "hilum")]
    assert (obs["method"], obs["matched"], obs["cui"]) == ("sapbert", "effusion", "C1000002")
    assert (anat["matched"], anat["cui"]) == ("lung", "C1000007")
    assert 0 < anat["cosine"] < obs["cosine"] < 1
    cache = tmp_path / "sapbert" / FakeSap.revision[:12]
    assert (cache / "vectors.npy").is_file() and (cache / "meta.json").is_file()


def test_a_lateral_cui_moves_to_its_unlateralised_parent(umls, tmp_path):
    got, _ = link(umls, tmp_path, ("Anatomy", "left lower lobe"), ("Anatomy", "lung"))
    lobe = got[("Anatomy", "left lower lobe")]
    assert (lobe["cui"], lobe["lateral_from"], lobe["lateral_unresolved"]) == (
        "C1000013", "C1000012", False)
    assert got[("Anatomy", "lung")]["lateral_from"] is None


def test_review_takes_band_80_unsure_band_90_and_flagged_links_of_any_band():
    links = pl.DataFrame({
        "string": ["a", "b", "c", "d", "pleural", "f"],
        "band": ["80", "90", "90", "95", "95", "95"],
        "method": ["exact", "exact", "sapbert", "exact", "sapbert", "sapbert"],
        "n_candidates": [1, 1, 1, 1, 1, 1],
        "query": ["a", "b", "c", "d", "pleura", "f"],
        "lateral_unresolved": [False, False, False, True, False, False]})
    assert links.filter(lk.needs_review(links))["string"].to_list() == ["a", "c", "d", "pleural"]


def mentions(*rows):
    """(study, kind, entity, joined, located_at) -> the columns located_pairs reads."""
    return pl.DataFrame([(s, "impression", k, e, j, False, loc) for s, k, e, j, loc in rows],
                        schema={"study_id": pl.String, "section": pl.String, "kind": pl.String,
                                "entity": pl.String, "joined": pl.String,
                                "dropped_head": pl.Boolean, "located_at": pl.List(pl.String)},
                        orient="row")


PAIRS = mentions(
    *[(f"s{i}", "Anatomy", "1", "pleural", []) for i in range(10)],
    *[(f"s{i}", "Observation", "2", "effusion", ["1"]) for i in range(10)],
    ("s10", "Anatomy", "1", "lung", []), ("s10", "Observation", "2", "effusion", ["1"]),
    ("s11", "Observation", "2", "effusion", []),
    ("s12", "Anatomy", "1", "pleura", []), ("s12", "Observation", "2", "thickening", ["1"]))


def test_pairs_are_cut_at_their_coverage_and_tried_in_both_orders(umls):
    occurrences = lk.located_pairs(PAIRS, ["effusion", "thickening"])
    top = lk.top_pairs(occurrences, {"coverage": 0.9, "max_pairs": 10})
    # 10 pleural + 1 lung: lung starts at 10/11 > 0.9 of effusion's located mentions, so it is cut
    assert top.select("obs", "anat", "n").rows() == [("effusion", "pleural", 10),
                                                     ("thickening", "pleura", 1)]
    pairs = lk.pair_links(top, umls, CFG, lk.domain(umls, CFG))
    assert pairs.select("phrase", "cui").rows() == [("pleural effusion", "C1000001"),
                                                    ("thickening pleura", "C1000014")]
    share = lk.pair_share(PAIRS, occurrences, pairs, ["effusion"])
    assert share == {"effusion": {"mentions": 12, "located": 11, "pair_linked": 10}}
    assert lk.top_pairs(occurrences, {"coverage": 1.0, "max_pairs": 1})["anat"].to_list() == [
        "pleural", "pleura"]


def test_mentions_carry_their_entity_and_the_anatomy_heads_they_are_located_at():
    def entity(tokens, start, kind="Observation", relations=()):
        return {"tokens": tokens, "label": f"{kind}::definitely present", "start_ix": start,
                "end_ix": start, "relations": [list(r) for r in relations]}
    entities = {
        "0": entity("Small", 0, relations=[("modify", "2")]),
        "1": entity("pleural", 1, "Anatomy"),
        "2": entity("effusion", 2, relations=[("located_at", "1")]),
        "3": entity("left", 3, "Anatomy", [("modify", "4")]),
        "4": entity("lobe", 4, "Anatomy"),
        "5": entity("opacity", 5, relations=[("located_at", "3")]),       # a modifier: its head
        "6": entity("patchy", 6, relations=[("modify", "7"), ("located_at", "4")]),
        "7": entity("consolidation", 7),                                  # through its modifier
        "8": entity("tube", 8, relations=[("located_at", "2")]),          # not an Anatomy
    }
    got = {tokens[head]: (eid, located) for _, _, tokens, head, eid, located
           in vocab._mentions(entities)}
    assert got["effusion"] == ("2", ["1"])
    assert got["opacity"] == ("5", ["4"]) and got["consolidation"] == ("7", ["4"])
    assert got["tube"] == ("8", []) and got["lobe"] == ("4", [])


def test_overrides_hold_strings_and_codes_only(tmp_path):
    real = lk.load_overrides(default_config_dir() / "kg" / "linker_overrides.yaml")
    assert real["Observation"]["collapsed lung"] == "C0004144"
    bad = tmp_path / "overrides.yaml"
    for text in ("observation: {effusion: pleural effusion}\n", "observation: {Effusion: skip}\n",
                 "findings: {}\n"):
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(lk.LinkerError):
            lk.load_overrides(bad)


def test_cli_links_the_head_and_writes_the_review_file(world, umls, tmp_path,  # noqa: F811
                                                        monkeypatch, capsys):
    pool, csv, rg = world
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pool.write_parquet(pool_dir / "pool_images.parquet")
    (pool_dir / "pool.json").write_text(json.dumps({"pool_sha256": "abc"}), encoding="utf-8")
    common = ["--pool-dir", str(pool_dir), "--plus-csv", str(csv), "--radgraph-dir", str(rg),
              "--umls-dir", str(umls)]
    main(["kg-pool-vocab", *common, "--out", str(tmp_path / "vocab")])
    monkeypatch.setattr("src.kg.sapbert.SapBert", FakeSap)
    real_domain = lk.domain
    monkeypatch.setattr(lk, "domain", lambda umls_dir, _cfg: real_domain(umls_dir, CFG))
    out = tmp_path / "link"
    assert main(["kg-link", *common, "--vocab-dir", str(tmp_path / "vocab"),
                 "--sapbert-cache", str(tmp_path / "sapbert"), "--out", str(out)]) == 0
    links = pl.read_parquet(out / "links.parquet")
    table = pl.read_parquet(tmp_path / "vocab" / "vocab_joined.parquet")
    assert set(links["string"]) <= set(table["string"]) and links["method"].null_count() == 0
    review = pl.read_csv(out / "review.csv")
    assert review.height == links.filter(pl.col("band") == "80").height + links.filter(
        lk.needs_review(links) & (pl.col("band") == "90")).height
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pool_sha256"] == "abc" and manifest["tau"] is None
    assert "Pleural" not in capsys.readouterr().out      # counts only, no UMLS names


def test_cli_stops_when_the_vocabulary_differs_from_v2_2(world, umls, tmp_path,  # noqa: F811
                                                          capsys):
    pool, csv, rg = world
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    pool.write_parquet(pool_dir / "pool_images.parquet")
    (pool_dir / "pool.json").write_text(json.dumps({"pool_sha256": "abc"}), encoding="utf-8")
    vocab_dir = tmp_path / "vocab"
    vocab_dir.mkdir()
    pl.DataFrame({"kind": ["Observation"], "string": ["x"], "mentions": [1], "studies": [1]},
                 schema={"kind": pl.String, "string": pl.String, "mentions": pl.UInt32,
                         "studies": pl.UInt32}).write_parquet(vocab_dir / "vocab_joined.parquet")
    assert main(["kg-link", "--pool-dir", str(pool_dir), "--plus-csv", str(csv),
                 "--radgraph-dir", str(rg), "--umls-dir", str(umls), "--vocab-dir",
                 str(vocab_dir), "--out", str(tmp_path / "link")]) == 1
    assert "differs" in capsys.readouterr().err
